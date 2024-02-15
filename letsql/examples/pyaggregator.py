import pickle
from abc import (
    ABC,
    abstractmethod,
)
from collections.abc import (
    Iterable,
)

import dask
import ibis.expr.datatypes as dt
import pandas as pd
import pyarrow as pa
import xgboost as xgb
from datafusion import (
    Accumulator,
    udaf,
)
from ibis.backends.base.sqlglot import (
    F,
)
from ibis.backends.datafusion.compiler.values import (
    translate_val,
)
from ibis.legacy.udf.vectorized import (
    ReductionVectorizedUDF,
    reduction,
)


def train_xgboost_model(df, features, target, seed=0):
    if "rownum" in df:
        df = df.sort_values("rownum", ignore_index=True)
    param = {"max_depth": 4, "eta": 1, "objective": "binary:logistic", "seed": seed}
    num_round = 10
    X = df[list(features)]
    y = df[target]
    dtrain = xgb.DMatrix(X, y)
    bst = xgb.train(param, dtrain, num_boost_round=num_round)
    return bst


def calc_best_features(df, candidates, target, n):
    return (
        pd.Series(
            train_xgboost_model(df, candidates, target)
            .get_score()
        )
        .tail(n)
        .pipe(lambda s: tuple(
            {"feature": k, "score": v}
            for k, v in s.items()
        ))
    )


def make_pandas_udf(t, f, output_type):
    schema = t.schema()
    input_type = [schema[c] for c in t.columns]

    @reduction(
        input_type=input_type,
        output_type=output_type,
    )
    def f_on_series(*args):
        df = pd.DataFrame({series.name: series for series in args})
        return f(df)

    def udf_on_t(t):
        return f_on_series(*(t[col] for col in schema))

    return udf_on_t


class PyAggregator(Accumulator, ABC):
    """Variadic aggregator for udafs"""

    def __init__(self):
        self._states = []

    def pystate(self):
        struct_arr = pa.concat_arrays(map(pickle.loads, self._states))
        df = pa.Table.from_batches(
            [pa.RecordBatch.from_struct_array(struct_arr)]
        ).to_pandas()
        return df

    def state(self):
        value = pa.array(
            [self._states],
            type=self.state_type,
        )
        return value

    @abstractmethod
    def pyevaluate(self):
        pass

    def evaluate(self):
        return pa.scalar(
            self.pyevaluate(),
            type=self.return_type,
        )

    def update(self, *arrays) -> None:
        state = pa.StructArray.from_arrays(
            arrays,
            names=self.names,
        )
        self._states.append(pickle.dumps(state))

    def merge(self, states: pa.Array) -> None:
        for state in states.to_pylist():
            self._states.extend(state)

    def supports_retract_batch(self):
        return False

    @classmethod
    @property
    def names(cls):
        return tuple(field.name for field in cls.struct_type)

    @classmethod
    @property
    def input_type(cls):
        return list(field.type for field in cls.struct_type)

    @classmethod
    @property
    @abstractmethod
    def return_type(cls):
        pass

    @classmethod
    @property
    @abstractmethod
    def struct_type(cls):
        return pa.struct(())

    @classmethod
    @property
    def state_type(cls):
        return pa.list_(pa.binary())

    @classmethod
    @property
    def volatility(cls):
        return "stable"

    @classmethod
    @property
    def name(cls):
        return cls.__name__.lower()

    @classmethod
    def register_udaf(cls, ctx):
        if not isinstance(cls.input_type, Iterable) or not all(
            isinstance(el, pa.DataType) for el in cls.input_type
        ):
            raise ValueError(
                f"{cls.__name__}.input_type must be iterable of pa.DataType"
            )

        def register_ibis_reduction_f(cls):
            @translate_val.register(ReductionVectorizedUDF)
            def _fmt(op, **kw):
                return F[op.func.__name__](*kw["func_args"])

            class Klass(ReductionVectorizedUDF):
                pass

            @reduction(
                input_type=[dt.core.from_pyarrow(el) for el in cls.input_type],
                output_type=dt.core.from_pyarrow(cls.return_type),
            )
            def ibis_reduction_f(*args, where=None):
                return Klass(*args, where=where).to_expr()

            ibis_reduction_f.func.__name__ = cls.name

            def udf_on_t(t):
                return ibis_reduction_f(*(t[col] for col in cls.names))

            return udf_on_t

        ctx.register_udaf(
            udaf(
                cls,
                cls.input_type,
                cls.return_type,
                [cls.state_type],
                cls.volatility,
                cls.name,
            )
        )
        ibis_reduction_f = register_ibis_reduction_f(cls)

        return ibis_reduction_f


def make_struct_type(t):
    return pa.struct(
        (
            pa.field(
                field_name,
                t[field_name].type().to_pyarrow(),
            )
            for field_name in t.columns
        )
    )


def make_tokenized_name(*args, prefix="my_udaf_", length=8):
    return f"{prefix}{dask.base.tokenize(*args)[:length]}"


def make_datafusion_udaf(
    t, df_to_value, return_type, name=None,
):
    struct_type = make_struct_type(t)
    if name is None:
        name = make_tokenized_name(df_to_value, return_type, struct_type)

    class MyAggregator(PyAggregator):

        def pyevaluate(self):
            return df_to_value(self.pystate())

        @classmethod
        @property
        def return_type(cls):
            return return_type

        @classmethod
        @property
        def struct_type(cls):
            return struct_type

        @classmethod
        @property
        def name(cls):
            return name

    udaf = MyAggregator.register_udaf(t._find_backend().con)

    return udaf
