// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use jxl_simd::{F32SimdVec, simd_function};

use crate::render::{
    Channels, ChannelsMut, ErasedLocalState, RenderPipelineInOutStage, for_each_chunk,
};

/// Apply Gabor-like filter to the 3 XYB channels.
#[derive(Debug)]
pub struct GaborishStage {
    weight0: [f32; 3],
    weight1: [f32; 3],
    weight2: [f32; 3],
}

impl GaborishStage {
    pub fn new(weight1: [f32; 3], weight2: [f32; 3]) -> Self {
        let mut w0 = [0.0; 3];
        let mut w1 = [0.0; 3];
        let mut w2 = [0.0; 3];
        for i in 0..3 {
            // stage_gaborish.cc normalizes with one shared reciprocal.
            let weight_total = 1.0 + 4.0 * (weight1[i] + weight2[i]);
            w0[i] = 1.0 / weight_total;
            w1[i] = weight1[i] * w0[i];
            w2[i] = weight2[i] * w0[i];
        }
        Self {
            weight0: w0,
            weight1: w1,
            weight2: w2,
        }
    }
}

impl std::fmt::Display for GaborishStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Gaborish filter")
    }
}

simd_function!(
    gaborish_process_dispatch,
    d: D,
    fn gaborish_process(
        stage: &GaborishStage,
        xsize: usize,
        input_rows: &Channels<f32>,
        output_rows: &mut ChannelsMut<f32>,
    ) {
        let w0 = stage.weight0.map(|w| D::F32Vec::splat(d, w));
        let w1 = stage.weight1.map(|w| D::F32Vec::splat(d, w));
        let w2 = stage.weight2.map(|w| D::F32Vec::splat(d, w));

        for_each_chunk(
            d,
            xsize,
            input_rows.view::<3, 3, 1>(),
            output_rows.view::<3, 1, 1>(),
            |_x, inv, outv| {
                macro_rules! step {
                    ($ch:literal) => {{
                        let p00 = inv.load::<_, $ch>(d, -1, -1);
                        let p01 = inv.load::<_, $ch>(d, -1, 0);
                        let p02 = inv.load::<_, $ch>(d, -1, 1);
                        let p10 = inv.load::<_, $ch>(d, 0, -1);
                        let p11 = inv.load::<_, $ch>(d, 0, 0);
                        let p12 = inv.load::<_, $ch>(d, 0, 1);
                        let p20 = inv.load::<_, $ch>(d, 1, -1);
                        let p21 = inv.load::<_, $ch>(d, 1, 0);
                        let p22 = inv.load::<_, $ch>(d, 1, 1);

                        let sides = (p10 + p12) + (p01 + p21);
                        let corners = (p00 + p02) + (p20 + p22);
                        let sum = p11 * w0[$ch];
                        let sum = sides.mul_add(w1[$ch], sum);
                        let sum = corners.mul_add(w2[$ch], sum);
                        outv.store::<_, $ch>(d, 0, sum);
                    }};
                }
                step!(0);
                step!(1);
                step!(2);
            },
        );
    }
);

impl RenderPipelineInOutStage for GaborishStage {
    type InputT = f32;
    type OutputT = f32;
    const SHIFT: (u8, u8) = (0, 0);
    const BORDER: (u8, u8) = (1, 1);

    fn uses_channel(&self, c: usize) -> bool {
        c < 3
    }

    fn process_row_chunk(
        &self,
        _position: (usize, usize),
        xsize: usize,
        input_rows: &Channels<f32>,
        output_rows: &mut ChannelsMut<f32>,
        _state: Option<&mut ErasedLocalState>,
        _previous_call_was_previous_row: bool,
    ) {
        gaborish_process_dispatch(self, xsize, input_rows, output_rows);
    }
}

#[cfg(test)]
mod test {
    use jxl_simd::{SimdDescriptor, test_all_instruction_sets};
    use test_log::test;

    use super::*;
    use crate::error::Result;
    use crate::image::{DataTypeTag, Image};
    use crate::render::low_memory_pipeline::row_buffers::RowBuffer;
    use crate::render::test::make_and_run_simple_pipeline;
    use crate::tests::assert_close;

    #[test]
    fn normalization_source_order() {
        let tiny = 1.0f32 / 67_108_864.0;
        let weight1 = [tiny, 0.115169525, 0.75];
        let weight2 = [tiny, 0.061248592, 0.5];
        let stage = GaborishStage::new(weight1, weight2);
        for c in 0..3 {
            let reciprocal = 1.0 / (1.0 + 4.0 * (weight1[c] + weight2[c]));
            assert_eq!(stage.weight0[c].to_bits(), reciprocal.to_bits());
            assert_eq!(
                stage.weight1[c].to_bits(),
                (weight1[c] * reciprocal).to_bits()
            );
            assert_eq!(
                stage.weight2[c].to_bits(),
                (weight2[c] * reciprocal).to_bits()
            );
        }
        // Separate additions each lose the half-ULP term at 1.0.
        let old_total = (1.0 + 4.0 * tiny) + 4.0 * tiny;
        assert_ne!(stage.weight0[0].to_bits(), (1.0 / old_total).to_bits());
    }

    fn gaborish_pairwise_sums<D: SimdDescriptor>(d: D) {
        let x0 = RowBuffer::x0_offset::<f32>();
        let large = 16_777_216.0f32;
        let values = [[large, 1.0, 0.0], [large, 0.0, -large], [1.0, 0.0, -large]];
        let inputs: Vec<_> = (0..3)
            .map(|_| {
                let mut input = RowBuffer::new(DataTypeTag::F32, 1, 0, 0, 1).unwrap();
                for (y, row) in values.iter().enumerate() {
                    input.get_row_mut::<f32>(y)[x0 - 1..x0 + 2].copy_from_slice(row);
                }
                input
            })
            .collect();
        let mut outputs: Vec<_> = (0..3)
            .map(|_| RowBuffer::new(DataTypeTag::F32, 0, 0, 0, 1).unwrap())
            .collect();
        let refs: Vec<_> = inputs.iter().collect();
        let input_rows = Channels::from_row_buffers(&refs, x0, 1, 3, 1);
        let stage = GaborishStage {
            weight0: [0.0; 3],
            weight1: [1.0, 0.0, 1.0],
            weight2: [0.0, 1.0, 1.0],
        };
        {
            let mut output_rows = ChannelsMut::from_row_buffers(&mut outputs, x0, 0, 1);
            gaborish_process(d, &stage, 1, &input_rows, &mut output_rows);
        }
        // Both source pairwise sums are 1; the old serial sums are 0.
        for (output, expected) in outputs.iter().zip([1.0f32, 1.0, 2.0]) {
            assert_eq!(output.get_row::<f32>(0)[x0].to_bits(), expected.to_bits());
        }
    }

    test_all_instruction_sets!(gaborish_pairwise_sums);

    #[test]
    fn consistency() -> Result<()> {
        crate::render::test::test_stage_consistency(
            || GaborishStage::new([0.115169525; 3], [0.061248592; 3]),
            (500, 500),
            3,
        )
    }

    #[test]
    fn checkerboard() -> Result<()> {
        let mut images = [
            Image::new((2, 2))?,
            Image::new((2, 2))?,
            Image::new((2, 2))?,
        ];
        for image in &mut images {
            image.row_mut(0).copy_from_slice(&[0.0, 1.0]);
            image.row_mut(1).copy_from_slice(&[1.0, 0.0]);
        }

        let stage = GaborishStage::new([0.115169525; 3], [0.061248592; 3]);
        let output = make_and_run_simple_pipeline(stage, &images, (2, 2), 0, 256)?;

        for out_ch in output.iter().take(3) {
            assert_close!(all, out_ch.row(0), &[0.20686048, 0.7931395], 1e-6);
            assert_close!(all, out_ch.row(1), &[0.7931395, 0.20686048], 1e-6);
        }

        Ok(())
    }
}
