//! Single-encode experiments on top of the ordinary native e9 policy.
//!
//! Unlike the portfolio scout, these switches do not run probes or retain a
//! second encoded result. Explicit composition may trade individual file sizes
//! for aggregate throughput/size; it makes no no-growth guarantee.

use super::{CandidateParams, adapter};
use crate::modular::{encode::ModularOptions, palette::PaletteOrder};
use crate::tools::lz77::SourceQuickMode;
use crate::{EncodeOptions, Error, Image, Result};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Options {
    /// Replace the source quick seed with sparse matching, not the final DP.
    pub sparse_lz77: bool,
    /// Use one previous compatible transformed channel in MA training.
    pub previous_channel: bool,
    /// Screen run-based palette orders using index/table estimates.
    pub screened_runs: bool,
}

impl Options {
    /// Build one native e9 plan. Untouched fields retain ordinary projection
    /// and caller overrides. Enabling `previous_channel` explicitly selects
    /// prev1, including for a deferred Source Patch atlas. No pixels are encoded.
    pub fn build(
        self,
        params: &CandidateParams,
        image: &Image,
        base: &EncodeOptions,
    ) -> Result<EncodeOptions> {
        if params.speed_tier != 1 {
            return Err(Error::InvalidInput(
                "source tool experiment requires effort 9".into(),
            ));
        }
        let mut params = params.clone();
        if self.previous_channel {
            params.max_properties = 1;
        }
        let mut options = adapter::project_native(&params, image, base)?.exact_options()?;
        self.configure(&mut options.modular);
        if let Some(reference) = &mut options.source_patch_reference {
            self.configure(&mut reference.modular);
        }
        options.modular.validate()?;
        Ok(options)
    }

    fn configure(self, modular: &mut ModularOptions) {
        if self.sparse_lz77 {
            if let Some(config) = &mut modular.source_lz77 {
                config.quick = SourceQuickMode::Sparse;
            }
        }
        if self.screened_runs {
            if let Some(policy) = &mut modular.palette_policy {
                policy.tools.order = PaletteOrder::ScreenedRuns;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modular::{
        ma_train::{DescriptionCost, TreeLearning},
        refinement::OffsetMode,
    };
    use crate::scheduler::params::Override;
    use crate::{PixelData, PixelLayout};

    fn image() -> Image {
        Image::new(
            32,
            32,
            PixelLayout::Rgb,
            PixelData::U8(vec![23; 32 * 32 * 3]),
        )
        .unwrap()
    }

    #[test]
    fn tools_compose_without_replacing_the_current_ma_or_batch_policy() {
        let image = image();
        for sparse_lz77 in [false, true] {
            for previous_channel in [false, true] {
                for screened_runs in [false, true] {
                    let controls = Options {
                        sparse_lz77,
                        previous_channel,
                        screened_runs,
                    };
                    let params = CandidateParams::default().backend(9).unwrap();
                    let options = controls
                        .build(&params, &image, &EncodeOptions::default())
                        .unwrap();
                    let m = &options.modular;
                    assert_eq!(m.previous_channels, usize::from(previous_channel));
                    assert_eq!(
                        m.sampling.as_ref().unwrap().previous_channels,
                        m.previous_channels
                    );
                    assert_eq!(
                        m.source_lz77.as_ref().unwrap().quick,
                        if sparse_lz77 {
                            SourceQuickMode::Sparse
                        } else {
                            SourceQuickMode::Off
                        }
                    );
                    assert!(m.source_lz77.as_ref().unwrap().chain_batch);
                    assert_eq!(
                        m.palette_policy.as_ref().unwrap().tools.order,
                        if screened_runs {
                            PaletteOrder::ScreenedRuns
                        } else {
                            PaletteOrder::Default
                        }
                    );
                    assert_eq!(m.tree_learning, TreeLearning::Symmetric);
                    assert_eq!(m.source_ma_scoring.description, DescriptionCost::Half);
                    assert_eq!(
                        m.refinement.offsets.as_ref().unwrap().mode,
                        OffsetMode::DistributionSinglePass
                    );
                }
            }
        }
    }

    #[test]
    fn reference_uses_the_same_tools_but_its_own_geometry() {
        let image = image();
        let mut params = CandidateParams::default().backend(9).unwrap();
        params.patches = Override::On;
        let controls = Options {
            sparse_lz77: true,
            previous_channel: true,
            screened_runs: true,
        };
        let options = controls
            .build(&params, &image, &EncodeOptions::default())
            .unwrap();
        let policy = options.source_patch_reference.unwrap();
        assert_eq!(policy.sampling.max_previous_channels, 1);
        let atlas = Image::new(
            16,
            16,
            PixelLayout::Rgb,
            PixelData::U8(vec![4; 16 * 16 * 3]),
        )
        .unwrap();
        let reference = policy.resolve(&atlas).unwrap();
        assert_eq!(reference.modular.previous_channels, 1);
        assert_eq!(
            reference.modular.source_lz77.unwrap().quick,
            SourceQuickMode::Sparse
        );
        assert_eq!(
            reference.modular.palette_policy.unwrap().tools.order,
            PaletteOrder::ScreenedRuns
        );
    }

    #[test]
    fn disabled_tools_preserve_explicit_settings_and_lower_efforts_are_rejected() {
        let image = image();
        let mut params = CandidateParams::default().backend(9).unwrap();
        params.max_properties = 2;
        let options = Options::default()
            .build(&params, &image, &EncodeOptions::default())
            .unwrap();
        assert_eq!(options.modular.previous_channels, 2);
        for effort in [1, 5, 8, 10, 11] {
            assert!(
                Options::default()
                    .build(
                        &params.backend(effort).unwrap(),
                        &image,
                        &EncodeOptions::default()
                    )
                    .is_err()
            );
        }
    }
}
