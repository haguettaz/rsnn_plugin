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
    let prev_delta = inputs[0].f64()?;
    let prev_coef_1 = inputs[1].f64()?;
    let weight_0 = inputs[2].f64()?;

    let out: Float64Chunked = izip!(
        prev_delta.iter(),
        prev_coef_1.iter(),
        weight_0.iter()
    )
    .scan(
        0_f64,
        |state: &mut f64,
         (prev_delta, prev_coef_1, weight_0): (Option<f64>, Option<f64>, Option<f64>)| {
            match prev_delta {
                Some(prev_delta) => {
                    *state = (*state + prev_coef_1.unwrap_or(0.0) * prev_delta)
                        * (-prev_delta).exp()
                        + weight_0.unwrap_or(0.0);
                    Some(Some(*state))
                },
                None => {
                    *state = weight_0.unwrap_or(0.0);
                    Some(Some(*state))
                },
            }
        },
    )
    .collect_trusted();

    Ok(out.into_series())
}

#[polars_expr(output_type_func=same_output_type)]
fn scan_coef_1(inputs: &[Series]) -> PolarsResult<Series> {
    let prev_delta = inputs[0].f64()?;
    let weight_1 = inputs[1].f64()?;

    let out: Float64Chunked = izip!(prev_delta.iter(), weight_1.iter())
        .scan(
            0_f64,
            |state: &mut f64, (prev_delta, weight_1): (Option<f64>, Option<f64>)| match prev_delta {
                Some(prev_delta) => {
                    *state = *state * (-prev_delta).exp() + weight_1.unwrap_or(0.0);
                    Some(Some(*state))
                },
                None => {
                    *state = weight_1.unwrap_or(0.0);
                    Some(Some(*state))
                },
            },
        )
        .collect_trusted();
    Ok(out.into_series())
}

#[polars_expr(output_type=Float64)]
fn first_ftime(inputs: &[Series]) -> PolarsResult<Series> {
    let start = inputs[0].f64()?;
    let delta = inputs[1].f64()?;
    let prev_delta = inputs[2].f64()?;
    let weight_0 = inputs[3].f64()?;
    let weight_1 = inputs[4].f64()?;
    let f_thresh = inputs[5].f64()?;

    let mut coef_0 = 0_f64;
    let mut coef_1 = 0_f64;

    let ftime = izip!(
        start.iter(),
        delta.iter(),
        prev_delta.iter(),
        weight_0.iter(),
        weight_1.iter(),
        f_thresh.iter(),
    )
    .find_map(
        |(start, delta, prev_delta, weight_0, weight_1, threshold)| match (start, threshold) {
            (Some(start), Some(threshold)) => {
                (coef_0, coef_1) = match prev_delta {
                    Some(prev_delta) => {
                        coef_0 = weight_0.unwrap_or(0.0)
                            + (coef_0 + coef_1 * prev_delta) * (-prev_delta).exp();
                        coef_1 = weight_1.unwrap_or(0.0) + coef_1 * (-prev_delta).exp();
                        (coef_0, coef_1)
                    },
                    None => (weight_0.unwrap_or(0.0), weight_1.unwrap_or(0.0)),
                };

                // println!(
                //     "start: {start} prev_delta: {prev_delta:?} delta: {delta:?} coef_0: {coef_0}, coef_1: {coef_1}"
                // );

                let f_time = {
                    if coef_0 < threshold {
                        if coef_1 > 0.0 {
                            start
                                - lambert_w0(-threshold / coef_1 * (-coef_0 / coef_1).exp())
                                - coef_0 / coef_1
                        } else if coef_1 < 0.0 {
                            start
                                - lambert_wm1(-threshold / coef_1 * (-coef_0 / coef_1).exp())
                                - coef_0 / coef_1
                        } else {
                            start + (coef_0 / threshold).ln()
                        }
                    } else {
                        start
                    }
                };

                match delta {
                    Some(delta) => {
                        if (f_time >= start) && (f_time < start + delta) {
                            Some(f_time)
                        } else {
                            None
                        }
                    },
                    None => {
                        if f_time >= start {
                            Some(f_time)
                        } else {
                            None
                        }
                    },
                }
            },
            (None, _) | (_, None) => None,
        },
    );
    Ok(Series::new(PlSmallStr::EMPTY, vec![ftime]))
}
