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
    prev_length: IntoExprColumn,
    prev_coef_1: IntoExprColumn,
    in_coef_0: IntoExprColumn,
) -> pl.Expr:
    return register_plugin_function(
        args=[
            prev_length,
            prev_coef_1,
            in_coef_0,
        ],
        plugin_path=LIB,
        function_name="scan_coef_0",
        is_elementwise=False,
    )


def scan_coef_1(prev_length: IntoExprColumn, in_coef_1: IntoExprColumn) -> pl.Expr:
    return register_plugin_function(
        args=[prev_length, in_coef_1],
        plugin_path=LIB,
        function_name="scan_coef_1",
        is_elementwise=False,
    )


def energy_syn_to_syn_metric(f_time, start_syn_i, start_syn_j) -> pl.Expr:
    return register_plugin_function(
        args=[
            f_time,
            start_syn_i,
            start_syn_j,
        ],
        plugin_path=LIB,
        function_name="energy_syn_to_syn_metric",
        is_elementwise=True,
    )


def energy_rec_to_syn_metric(f_time, start_rec, start_syn) -> pl.Expr:
    return register_plugin_function(
        args=[
            f_time,
            start_rec,
            start_syn,
        ],
        plugin_path=LIB,
        function_name="energy_rec_to_syn_metric",
        is_elementwise=True,
    )


def max_violation(
    start: IntoExprColumn,
    length: IntoExprColumn,
    coef_0: IntoExprColumn,
    coef_1: IntoExprColumn,
    *,
    vmax: float,
) -> pl.Expr:
    return register_plugin_function(
        args=[start, length, coef_0, coef_1],
        plugin_path=LIB,
        function_name="max_violation",
        kwargs={"vmax": vmax},
        is_elementwise=False,
        returns_scalar=True,
    )


def first_ftime(
    start: IntoExprColumn,
    length: IntoExprColumn,
    prev_length: IntoExprColumn,
    in_coef_0: IntoExprColumn,
    in_coef_1: IntoExprColumn,
    f_thresh: IntoExprColumn,
) -> pl.Expr:
    return register_plugin_function(
        args=[start, length, prev_length, in_coef_0, in_coef_1, f_thresh],
        plugin_path=LIB,
        function_name="first_ftime",
        is_elementwise=False,
        returns_scalar=True,
    )


def extend_periodically(
    time: IntoExprColumn,
    period: IntoExprColumn,
    tmin: IntoExprColumn,
    tmax: IntoExprColumn,
) -> pl.Expr:
    return register_plugin_function(
        args=[time, period, tmin, tmax],
        plugin_path=LIB,
        function_name="extend_periodically",
        is_elementwise=True,
    )
