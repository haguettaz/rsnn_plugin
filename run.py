import math

import polars as pl

import rsnn_plugin as rp

df = pl.DataFrame(
    {
        "neuron": [0, 0, 0],
        "start": [1.0, 3.0, 8.0],
        "weight_0": [0.0, 1.0, 1.0],
        "weight_1": [math.e, -1.0, 1.0],
        "f_thresh": [1.0, 1.0, 1.0],
    }
)

# df = df.with_columns(c1=rp.scan_coef_1(pl.col("length"), pl.col("weight_1").shift()))
# df = df.with_columns(
#     c0=rp.scan_coef_0(
#         pl.col("weight_0"), pl.col("coef_1").shift(), pl.col("length").shift()
#     )
# )
# print(df)

print(
    df.group_by("neuron").agg(
        ftime=rp.first_ftime(
            pl.col("start"),
            pl.col("start").diff().shift(-1, fill_value=float("inf")),
            pl.col("f_thresh"),
            pl.col("weight_0"),
            pl.col("weight_1"),
        )
    )
)
