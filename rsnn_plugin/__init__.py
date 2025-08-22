from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from polars.plugins import register_plugin_function

from rsnn_plugin._internal import __version__ as __version__

if TYPE_CHECKING:
    from rsnn_plugin.typing import IntoExprColumn

LIB = Path(__file__).parent


def scan_coef_0(
    length: IntoExprColumn,
    coef_1: IntoExprColumn,
    weight_0: IntoExprColumn,
) -> pl.Expr:
    return register_plugin_function(
        args=[
            length,
            coef_1,
            weight_0,
        ],
        plugin_path=LIB,
        function_name="scan_coef_0",
        is_elementwise=False,
    )


def scan_coef_1(length: IntoExprColumn, weight_1: IntoExprColumn) -> pl.Expr:
    return register_plugin_function(
        args=[length, weight_1],
        plugin_path=LIB,
        function_name="scan_coef_1",
        is_elementwise=False,
    )


def first_ftime(
    start: IntoExprColumn,
    length: IntoExprColumn,
    f_thresh: IntoExprColumn,
    weight_0: IntoExprColumn,
    weight_1: IntoExprColumn,
) -> pl.Expr:
    return register_plugin_function(
        args=[start, length, f_thresh, weight_0, weight_1],
        plugin_path=LIB,
        function_name="first_ftime",
        is_elementwise=False,
        returns_scalar=True,
    )
