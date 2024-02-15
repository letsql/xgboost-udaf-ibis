import pathlib
import warnings

import ibis
import ibis.expr.datatypes as dt
import pandas as pd
import toolz
from ibis.formats.pyarrow import PyArrowType

from letsql.examples.pyaggregator import (
    calc_best_features,
    make_datafusion_udaf,
    make_pandas_udf,
)


def join_splat(df, col):
    return (
        df
        .drop(col, axis=1)
        .join(
            df
            [col]
            .apply(lambda x: pd.Series({
                f"{k}{i}": v
                for (i, dct) in enumerate(x)
                for k, v in dct.items()
            }))
        )
    )


def main():
    path = pathlib.Path(__file__).parent.parent.parent.joinpath("data.rownum.parquet")
    candidates = (
        "emp_length",
        "dti",
        "annual_inc",
        "loan_amnt",
        "fico_range_high",
        "cr_age_days",
    )
    by = "issue_y"
    target = "event_occurred"
    cols = list(candidates) + [by, target, "rownum"]
    curried_calc_best_features = toolz.curry(
        calc_best_features, candidates=candidates, target=target, n=2
    )
    ibis_output_type = dt.infer(({"feature": "feature", "score": 0.},))

    pandas_t = ibis.pandas.connect().read_parquet(path, "t")[cols]
    pandas_udaf = make_pandas_udf(
        pandas_t,
        curried_calc_best_features,
        ibis_output_type,
    )
    pandas_expr = (
        pandas_t
        .group_by(by)
        .agg(pandas_udaf(pandas_t).name("best_features"))
    )

    datafusion_t = ibis.datafusion.connect().read_parquet(path, "t")[cols]
    datafusion_udaf = make_datafusion_udaf(
        datafusion_t,
        curried_calc_best_features,
        PyArrowType.from_ibis(ibis_output_type),
    )
    datafusion_expr = (
        datafusion_t
        .group_by(by)
        .agg(datafusion_udaf(datafusion_t).name("best_features"))
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from_pandas_ibis = (
            pandas_expr
            .execute()
            .sort_values(by, ignore_index=True)
            .pipe(join_splat, "best_features")
        )
        from_datafusion_ibis = (
            datafusion_expr
            .execute()
            .sort_values(by, ignore_index=True)
            .pipe(join_splat, "best_features")
        )
        from_pandas = (
            pd.read_parquet(path)[cols]
            .groupby(by)
            .apply(curried_calc_best_features)
            .rename("best_features").reset_index()
            .pipe(join_splat, "best_features")
        )

    assert from_pandas.equals(from_pandas_ibis)
    assert from_pandas.equals(from_datafusion_ibis)
    return from_pandas
