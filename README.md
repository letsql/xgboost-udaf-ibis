# pandas style groupby-apply in DataFusion via Ibis

This repo contains the code described in [our blog post](https://letsql.dev/posts/xgboost-udaf-ibis/) that demonstrates how to do pandas style groupby-apply aggregation (as opposed to accumulation) in [DataFusion](https://arrow.apache.org/datafusion/).

# run

## via pip
```
python3.10 -m venv venv
source venv/bin/activate
pip install git+https://github.com/letsql/xgboost-udaf-ibis
xgboost-udaf-ibis
```
## via nix
```
nix run github:letsql/xgboost-udaf-ibis
```
