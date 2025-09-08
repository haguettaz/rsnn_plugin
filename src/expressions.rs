#![allow(clippy::unused_unit)]

use core::f64;

use itertools::izip;
use lambert_w::{lambert_w0, lambert_wm1};
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use pyo3_polars::export::polars_core::utils::CustomIterTools;
use serde::Deserialize;

// fn same_output_type(input_fields: &[Field]) -> PolarsResult<Field> {
//     let field = &input_fields[0];
//     Ok(field.clone())
// }

#[polars_expr(output_type=Float64)]
fn scan_coef_0(inputs: &[Series]) -> PolarsResult<Series> {
    let prev_length = inputs[0].f64()?;
    let prev_coef_1 = inputs[1].f64()?;
    let in_coef_0 = inputs[2].f64()?;

    let out: Float64Chunked =
        izip!(prev_length.iter(), prev_coef_1.iter(), in_coef_0.iter())
            .scan(
                0_f64,
                |coef_0: &mut f64,
                 (prev_length, prev_coef_1, in_coef_0): (
                    Option<f64>,
                    Option<f64>,
                    Option<f64>,
                )| {
                    match prev_length {
                        Some(prev_length) => {
                            *coef_0 = (*coef_0 + prev_coef_1.unwrap_or(0.0) * prev_length)
                                * (-prev_length).exp()
                                + in_coef_0.unwrap_or(0.0);
                            Some(Some(*coef_0))
                        },
                        None => {
                            *coef_0 = in_coef_0.unwrap_or(0.0);
                            Some(Some(*coef_0))
                        },
                    }
                },
            )
            .collect_trusted();

    Ok(out.into_series())
}

#[polars_expr(output_type=Float64)]
fn scan_coef_1(inputs: &[Series]) -> PolarsResult<Series> {
    let prev_length = inputs[0].f64()?;
    let in_coef_1 = inputs[1].f64()?;

    let out: Float64Chunked = izip!(prev_length.iter(), in_coef_1.iter())
        .scan(
            0_f64,
            |coef_1: &mut f64, (prev_length, in_coef_1): (Option<f64>, Option<f64>)| {
                match prev_length {
                    Some(prev_length) => {
                        *coef_1 = *coef_1 * (-prev_length).exp() + in_coef_1.unwrap_or(0.0);
                        Some(Some(*coef_1))
                    },
                    None => {
                        *coef_1 = in_coef_1.unwrap_or(0.0);
                        Some(Some(*coef_1))
                    },
                }
            },
        )
        .collect_trusted();
    Ok(out.into_series())
}

#[polars_expr(output_type=Float64)]
fn energy_syn_to_syn_metric(inputs: &[Series]) -> PolarsResult<Series> {
    let delta = inputs[0].f64()?;
    let duration = inputs[1].f64()?;

    let out: Float64Chunked = izip!(delta.iter(), duration.iter())
        .map(|(delta, duration)| match (delta, duration) {
            (Some(delta), Some(duration)) => Some(
                (1.0 + delta.abs()
                    - ((1.0 + 2.0 * duration * (1.0 + duration))
                        + delta.abs() * (1.0 + 2.0 * duration))
                        * (-2.0 * duration).exp())
                    * (-delta.abs()).exp()
                    / 4.0,
            ),
            (Some(delta), None) => Some((1.0 + delta.abs()) * (-delta.abs()).exp() / 4.0),
            _ => None,
        })
        .collect_trusted();

    Ok(out.into_series())
}

#[polars_expr(output_type=Float64)]
fn energy_rec_to_syn_metric(inputs: &[Series]) -> PolarsResult<Series> {
    let delta = inputs[0].f64()?;
    let duration = inputs[1].f64()?;

    let out: Float64Chunked = izip!(delta.iter(), duration.iter())
        .map(|(delta, duration)| match (delta, duration) {
            (Some(delta), Some(duration)) => Some(
                (1.0 - (1.0 + 2.0 * duration) * (-2.0 * duration).exp()) * (-delta.abs()).exp()
                    / 4.0,
            ),
            (Some(delta), None) => Some((-delta.abs()).exp() / 4.0),
            _ => None,
        })
        .collect_trusted();

    Ok(out.into_series())
}

fn f64_list_dtype(input_fields: &[Field]) -> PolarsResult<Field> {
    let field = Field::new(
        input_fields[0].name().clone(),
        DataType::List(Box::new(DataType::Float64)),
    );
    Ok(field.clone())
}

#[polars_expr(output_type_func=f64_list_dtype)]
fn extend_periodically(inputs: &[Series]) -> PolarsResult<Series> {
    let time = inputs[0].f64()?;
    let period = inputs[1].f64()?;
    let tmin = inputs[2].f64()?;
    let tmax = inputs[3].f64()?;

    let out: ListChunked = izip!(time.iter(), period.iter(), tmin.iter(), tmax.iter())
        .map(|(time, period, tmin, tmax)| {
            let mut res = Vec::new();
            if let (Some(time), Some(period), Some(tmin), Some(tmax)) = (time, period, tmin, tmax) {
                let kmin = -((time - tmin) / period).floor() as i64;
                let kmax = -((time - tmax) / period).ceil() as i64;
                for k in kmin..=kmax {
                    res.push(time + k as f64 * period);
                }
            }
            Series::new(PlSmallStr::EMPTY, res)
        })
        .collect_trusted();

    Ok(out.into_series())
}

#[derive(Deserialize)]
struct MaxViolationKwargs {
    vmax: f64,
}

#[polars_expr(output_type=Float64)]
fn max_violation(inputs: &[Series], kwargs: MaxViolationKwargs) -> PolarsResult<Series> {
    let start = inputs[0].f64()?;
    let length = inputs[1].f64()?;
    let coef_0 = inputs[2].f64()?;
    let coef_1 = inputs[3].f64()?;

    let res = izip!(start.iter(), length.iter(), coef_0.iter(), coef_1.iter())
        .filter_map(|(start, length, coef_0, coef_1)| match start {
            Some(start) => {
                let coef_0 = coef_0.unwrap_or(0.0);
                let coef_1 = coef_1.unwrap_or(0.0);

                match length {
                    Some(length) => {
                        let dt = 1_f64 - coef_0 / coef_1;
                        if (coef_1 > 0.0) & (dt >= 0.0) & (dt < length) {
                            let v = (coef_0 + coef_1 * dt) * (-dt).exp();
                            Some((start + dt, v))
                        } else {
                            let vstart = coef_0;
                            let vend = (coef_0 + coef_1 * length) * (-length).exp();
                            if vstart > vend {
                                Some((start, vstart))
                            } else {
                                Some(((start + length - 1e-12).max(start), vend))
                            }
                        }
                    },
                    None => {
                        let dt = 1_f64 - coef_0 / coef_1;
                        if (coef_1 > 0.0) & (dt >= 0.0) {
                            Some((start + dt, (coef_0 + coef_1 * dt) * (-dt).exp()))
                        } else {
                            if coef_0 >= 0.0 {
                                Some((start, coef_0))
                            } else {
                                Some((f64::INFINITY, 0.0))
                            }
                        }
                    },
                }
            },
            None => None,
        })
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    let tmax: Option<f64> = match res {
        Some((tmax, vmax)) if vmax > kwargs.vmax => Some(tmax),
        _ => None,
    };
    Ok(Series::new(PlSmallStr::EMPTY, vec![tmax]))
}

#[polars_expr(output_type=Float64)]
fn first_ftime(inputs: &[Series]) -> PolarsResult<Series> {
    let start = inputs[0].f64()?;
    let length = inputs[1].f64()?;
    let prev_length = inputs[2].f64()?;
    let in_coef_0 = inputs[3].f64()?;
    let in_coef_1 = inputs[4].f64()?;
    let f_thresh = inputs[5].f64()?;

    let mut coef_0 = 0_f64;
    let mut coef_1 = 0_f64;

    let ftime = izip!(
        start.iter(),
        length.iter(),
        prev_length.iter(),
        in_coef_0.iter(),
        in_coef_1.iter(),
        f_thresh.iter(),
    )
    .find_map(
        |(start, length, prev_length, in_coef_0, in_coef_1, threshold)| match (start, threshold) {
            (Some(start), Some(threshold)) => {
                let in_coef_0 = in_coef_0.unwrap_or(0.0);
                let in_coef_1 = in_coef_1.unwrap_or(0.0);

                (coef_0, coef_1) = match prev_length {
                    Some(prev_length) => (
                        in_coef_0 + (coef_0 + coef_1 * prev_length) * (-prev_length).exp(),
                        in_coef_1 + coef_1 * (-prev_length).exp(),
                    ),
                    None => (in_coef_0, in_coef_1),
                };

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

                match length {
                    Some(length) => {
                        if (f_time >= start) && (f_time < start + length) {
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

// fn struct_coef_dtype(input_fields: &[Field]) -> PolarsResult<Field> {
//     let field = &input_fields[0];
//     match field.dtype() {
//         DataType::Struct(fields) => Ok(Field::new(
//             "struct_coef".into(),
//             DataType::Struct(fields.clone()),
//         )),
//         dtype => polars_bail!(InvalidOperation: "Expected Struct type, got {dtype:?}"),
//     }
// }

// #[derive(Debug, Clone, Copy)]
// struct Coef {
//     c0: f64,
//     c1: f64,
// }

// #[polars_expr(output_type_func=struct_coef_dtype)]
// fn scan_coef(inputs: &[Series]) -> PolarsResult<Series> {
//     let prev_delta = inputs[0].f64()?;
//     let weight = inputs[1].struct_()?;

//     let coef_results: Vec<Coef> = prev_delta
//         .iter()
//         .zip(weight.iter())
//         .scan(
//             Coef { c0: 0.0, c1: 0.0 },
//             |state: &mut Coef, (prev_delta, weight): (Option<f64>, Option<Coef>)| match (
//                 prev_delta, weight,
//             ) {
//                 (Some(prev_delta), Some(weight)) => {
//                     state.c0 = weight.c0 + (state.c0 + state.c1 * prev_delta) * (-prev_delta).exp();
//                     state.c1 = weight.c1 + state.c1 * (-prev_delta).exp();
//                     Some(*state)
//                 },
//                 (None, Some(weight)) => {
//                     state.c0 = weight.c0;
//                     state.c1 = weight.c1;
//                     Some(*state)
//                 },
//                 (Some(prev_delta), None) => {
//                     state.c0 = (state.c0 + state.c1 * prev_delta) * (-prev_delta).exp();
//                     state.c1 = state.c1 * (-prev_delta).exp();
//                     Some(*state)
//                 },
//                 (None, None) => {
//                     state.c0 = 0.0;
//                     state.c1 = 0.0;
//                     Some(*state)
//                 },
//             },
//         )
//         .collect();

//     let c0_values: Vec<f64> = coef_results.iter().map(|c| c.c0).collect();
//     let c1_values: Vec<f64> = coef_results.iter().map(|c| c.c1).collect();

//     let c0_series = Series::new("c0".into(), c0_values);
//     let c1_series = Series::new("c1".into(), c1_values);

//     let out = StructChunked::from_series("struct_coef".into(), [c0_series, c1_series].iter())?;
//     Ok(out.into_series())
// }

// fn same_dtype(input_fields: &[Field]) -> PolarsResult<Field> {
//     let field = &input_fields[0];
//     Ok(field.clone())
// }
