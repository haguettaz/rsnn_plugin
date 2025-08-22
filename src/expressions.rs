#![allow(clippy::unused_unit)]

use itertools::izip;
use lambert_w::{lambert_w0, lambert_wm1};
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use pyo3_polars::export::polars_core::utils::CustomIterTools;

fn same_output_type(input_fields: &[Field]) -> PolarsResult<Field> {
    let field = &input_fields[0];
    Ok(field.clone())
}

#[polars_expr(output_type_func=same_output_type)]
fn scan_coef_0(inputs: &[Series]) -> PolarsResult<Series> {
    let length = inputs[0].f64()?;
    let coef_1 = inputs[1].f64()?;
    let weight_0 = inputs[2].f64()?;

    let out: Float64Chunked = izip!(weight_0.iter(), coef_1.iter(), length.iter())
        .scan(
            0_f64,
            |state: &mut f64, (w0, c1, dt): (Option<f64>, Option<f64>, Option<f64>)| match (
                w0, c1, dt,
            ) {
                (Some(w0), Some(c1), Some(dt)) => {
                    *state = w0 + (*state + c1 * dt) * (-dt).exp();
                    Some(Some(*state))
                },
                (Some(w0), None, None) => {
                    *state = w0;
                    Some(Some(*state))
                },
                (None, Some(c1), Some(dt)) => {
                    *state = (*state + c1 * dt) * (-dt).exp();
                    Some(Some(*state))
                },
                _ => Some(None),
            },
        )
        .collect_trusted();
    Ok(out.into_series())
}

#[polars_expr(output_type_func=same_output_type)]
fn scan_coef_1(inputs: &[Series]) -> PolarsResult<Series> {
    let length = inputs[0].f64()?;
    let weight_1 = inputs[1].f64()?;

    let out: Float64Chunked = izip!(weight_1.iter(), length.iter())
        .scan(
            0_f64,
            |state: &mut f64, (w1, dt): (Option<f64>, Option<f64>)| match (w1, dt) {
                (Some(w1), Some(dt)) => {
                    *state = w1 + *state * (-dt).exp();
                    Some(Some(*state))
                },
                (Some(w1), None) => {
                    *state = w1;
                    Some(Some(*state))
                },
                (None, Some(dt)) => {
                    *state *= (-dt).exp();
                    Some(Some(*state))
                },
                _ => Some(None),
            },
        )
        .collect_trusted();
    Ok(out.into_series())
}

#[polars_expr(output_type=Float64)]
fn first_ftime(inputs: &[Series]) -> PolarsResult<Series> {
    let start = inputs[0].f64()?;
    let length = inputs[1].f64()?;
    let f_thresh = inputs[2].f64()?;
    let weight_0 = inputs[3].f64()?;
    let weight_1 = inputs[4].f64()?;

    let mut c0 = 0_f64;
    let mut c1 = 0_f64;

    let ftime = izip!(
        weight_0.iter(),
        weight_1.iter(),
        start.iter(),
        length.iter(),
        f_thresh.iter(),
    )
    .find_map(|(w0, w1, start, len, th)| match (start, len, th) {
        (Some(start), Some(len), Some(th)) => {
            let w0 = w0.unwrap_or(0.0);
            let w1 = w1.unwrap_or(0.0);
            println!("w0={w0}, w1={w1}");
            c0 = w0 + (c0 + c1 * len) * (-len).exp();
            c1 = w1 + c1 * (-len).exp();
            println!("c0={c0}, c1={c1}");

            let dt = {
                if c0 < th {
                    if c1 > 0.0 {
                        -lambert_w0(-th / c1 * (-c0 / c1).exp()) - c0 / c1
                    } else if c1 < 0.0 {
                        -lambert_wm1(-th / c1 * (-c0 / c1).exp()) - c0 / c1
                    } else {
                        (c0 / th).ln()
                    }
                } else {
                    0.0
                }
            };

            if (dt >= 0.0) && (dt < len) {
                Some(start + dt)
            } else {
                None
            }
        },
        (None, _, _) | (_, None, _) | (_, _, None) => None,
    });
    Ok(Series::new(PlSmallStr::EMPTY, vec![ftime]))
}
