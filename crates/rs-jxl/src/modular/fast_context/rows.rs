//! Basic's fused four-predictor scan plus optional standard predictor rows.

use super::{ChannelView, Predictor, PredictorSet, scan, weighted};
use crate::modular::{predict_simd, predictor::NeighborRow};
use crate::{Error, Result};

pub(super) struct Rows {
    pub basic: scan::Row,
    extra: Vec<(Predictor, Vec<i32>)>,
    wide: Vec<i64>,
}

impl Rows {
    pub(super) fn new(width: usize, predictors: PredictorSet) -> Self {
        Self {
            basic: scan::Row::new(width),
            extra: predictors.predictors()[4..]
                .iter()
                .map(|&predictor| (predictor, vec![0; width]))
                .collect(),
            wide: if predictors == PredictorSet::Full {
                vec![0; width]
            } else {
                Vec::new()
            },
        }
    }

    pub(super) fn residuals_at(&self, x: usize) -> impl Iterator<Item = i32> + '_ {
        self.basic
            .residuals
            .iter()
            .map(move |row| row[x])
            .chain(self.extra.iter().map(move |(_, row)| row[x]))
    }

    pub(super) fn fill(
        &mut self,
        view: &ChannelView<'_>,
        y: usize,
        backend: scan::Backend,
        state: Option<&mut weighted::State>,
    ) -> Result<()> {
        scan::fill(
            view.pixels,
            view.width,
            view.stride,
            y,
            backend,
            &mut self.basic,
        )?;
        let mut state = state;
        for (predictor, residuals) in &mut self.extra {
            if *predictor == Predictor::Weighted {
                let state = state.as_deref_mut().ok_or_else(|| {
                    Error::InvalidInput(
                        "fast-context Weighted requires native predictor state".into(),
                    )
                })?;
                let neighbors = NeighborRow::new(view.pixels, view.width, view.stride, y)
                    .map_err(|e| Error::InvalidInput(e.to_string()))?;
                for (x, residual) in residuals[..view.width].iter_mut().enumerate() {
                    let n = neighbors
                        .at(x)
                        .map_err(|e| Error::InvalidInput(e.to_string()))?;
                    let value = view.pixels[y * view.stride + x];
                    let (prediction, _) = state.predict_and_update(
                        value, x, y, n.top, n.left, n.topright, n.topleft, n.toptop,
                    )?;
                    *residual = (i64::from(value) - i64::from(prediction)) as i32;
                }
            } else {
                let kernel = if backend == scan::Backend::Scalar {
                    predict_simd::residual_row_scalar
                } else {
                    predict_simd::residual_row
                };
                kernel(
                    view.pixels,
                    view.width,
                    view.stride,
                    y,
                    *predictor,
                    &mut self.wide,
                )
                .map_err(|e| Error::InvalidInput(e.to_string()))?;
                for (to, &from) in residuals[..view.width].iter_mut().zip(&self.wide) {
                    *to = from as i32;
                }
            }
        }
        Ok(())
    }
}
