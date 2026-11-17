//! Explicit encoder-only Block Context Map experiments, not a new default.
//!
//! Quantile policies propose cuts from exact borrowed luma `adjusted_dc` values,
//! including padded blocks. Activity policies weight each block by one plus its
//! nonzero AC count, without JPEG pixel decoding. This is an activity proxy, not
//! exact downstream AC statistics or entropy/container cost (including CfL,
//! coefficient orders, prediction, clustering and header overhead).
//!
//! Call `prepare` before DC/control preparation, and pass the returned model to
//! both DC preparation and AC tokenization/header writing. The caller regenerates
//! qdc from encoded integer DC channel samples with these same strict `dc > t`
//! thresholds. DC preparation requires encoded DC to equal adjusted DC, so this
//! histogram matches qdc's DC samples for accepted reversible inputs. The decoder
//! receives the identical thresholds. Activity weights remain a proxy, not AC
//! cost. Final experiments must compare actual complete container sizes.

use std::collections::HashMap;

use super::transcode_layout::{BlockView, DcContexts, NUM_COEFFICIENT_ORDERS, TranscodeLayout};
use crate::{Error, Result};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Policy {
    #[default]
    Source,
    ChannelOnly,
    SharedChroma,
    PairedLuma,
    Quantile4,
    Quantile5,
    Activity4,
    Activity5,
}

/// Prepare a frame-global model without modifying input coefficients or layout.
/// Only DCT8 entries are populated; map order is Y, X, B with 13 order classes.
pub fn prepare(layout: &TranscodeLayout<'_>, policy: Policy) -> Result<DcContexts> {
    match policy {
        Policy::Source => Ok(layout.dc_contexts()),
        Policy::PairedLuma => Ok(layout.dc_contexts().with_paired_luma()),
        Policy::ChannelOnly => independent_channels(Vec::new(), layout.is_grayscale()),
        Policy::SharedChroma => {
            let source = layout.dc_contexts();
            let mut thresholds = Vec::new();
            thresholds
                .try_reserve_exact(source.luma_thresholds().len())
                .map_err(|_| Error::Limit("block context thresholds allocation"))?;
            thresholds.extend_from_slice(source.luma_thresholds());
            let buckets = thresholds.len() + 1;
            let mut map = allocate_map(buckets)?;
            for i in 0..buckets {
                map[i] = i as u8;
                // Keep the source's paired chroma buckets, aliasing X/B only.
                let chroma = buckets + if layout.is_grayscale() { 0 } else { i / 2 };
                map[NUM_COEFFICIENT_ORDERS * buckets + i] = chroma as u8;
                map[2 * NUM_COEFFICIENT_ORDERS * buckets + i] = chroma as u8;
            }
            DcContexts::from_parts(thresholds, map)
        }
        Policy::Quantile4 | Policy::Quantile5 | Policy::Activity4 | Policy::Activity5 => {
            let bins = match policy {
                Policy::Quantile4 | Policy::Activity4 => 4,
                _ => 5,
            };
            let activity = matches!(policy, Policy::Activity4 | Policy::Activity5);
            let histogram = luma_histogram(layout, activity)?;
            independent_channels(
                quantile_thresholds(&histogram, bins)?,
                layout.is_grayscale(),
            )
        }
    }
}

fn allocate_map(buckets: usize) -> Result<Vec<u8>> {
    let count = (3 * NUM_COEFFICIENT_ORDERS)
        .checked_mul(buckets)
        .ok_or(Error::Limit("block context map size overflow"))?;
    let mut map = Vec::new();
    map.try_reserve_exact(count)
        .map_err(|_| Error::Limit("block context map allocation"))?;
    map.resize(count, 0);
    Ok(map)
}

fn independent_channels(thresholds: Vec<i32>, grayscale: bool) -> Result<DcContexts> {
    let buckets = thresholds.len() + 1;
    // At most five buckets: three independent color channels fit in 15 IDs.
    if buckets > 5 {
        return Err(Error::InvalidInput(
            "independent JPEG contexts require at most five DC bins".into(),
        ));
    }
    let mut map = allocate_map(buckets)?;
    for i in 0..buckets {
        map[i] = i as u8;
        map[NUM_COEFFICIENT_ORDERS * buckets + i] = (buckets + if grayscale { 0 } else { i }) as u8;
        map[2 * NUM_COEFFICIENT_ORDERS * buckets + i] =
            (buckets + if grayscale { 0 } else { buckets + i }) as u8;
    }
    DcContexts::from_parts(thresholds, map)
}

fn add_weight(histogram: &mut HashMap<i32, u64>, dc: i32, weight: u64) -> Result<()> {
    if let Some(count) = histogram.get_mut(&dc) {
        *count = count
            .checked_add(weight)
            .ok_or(Error::Limit("luma DC histogram weight overflow"))?;
    } else {
        histogram
            .try_reserve(1)
            .map_err(|_| Error::Limit("luma DC histogram allocation"))?;
        histogram.insert(dc, weight);
    }
    Ok(())
}

fn luma_histogram(layout: &TranscodeLayout<'_>, activity: bool) -> Result<Vec<(i32, u64)>> {
    let luma = &layout.planes()[1];
    let (width, height) = luma.dimensions_in_blocks();
    // Exact signed keys: no source 2048-bin clipping or arbitrary input cap.
    // Reserve fallibly per new key, rather than allocating per input block.
    let mut histogram = HashMap::new();
    for y in 0..height {
        for x in 0..width {
            let dc = luma
                .adjusted_dc(x, y)
                .ok_or_else(|| Error::InvalidInput("missing luma DC block".into()))?;
            let nonzeros = if activity {
                match luma
                    .block(x, y)
                    .ok_or_else(|| Error::InvalidInput("missing luma AC block".into()))?
                {
                    BlockView::Jpeg(block) => block[1..].iter().filter(|&&c| c != 0).count(),
                    BlockView::Zero => 0,
                }
            } else {
                0
            };
            add_weight(&mut histogram, dc, 1 + nonzeros as u64)?;
        }
    }
    let mut sorted = Vec::new();
    sorted
        .try_reserve_exact(histogram.len())
        .map_err(|_| Error::Limit("sorted luma DC histogram allocation"))?;
    sorted.extend(histogram);
    sorted.sort_unstable_by_key(|&(dc, _)| dc);
    Ok(sorted)
}

/// Weighted nearest-rank cuts: first observed value reaching k/bins of mass.
/// Repeated ranks collapse to one cut. Excluding the maximum guarantees an
/// observed value on both sides; strict comparison keeps ties in the lower bin.
fn quantile_thresholds(histogram: &[(i32, u64)], bins: u8) -> Result<Vec<i32>> {
    if !(1..=16).contains(&bins)
        || histogram.is_empty()
        || histogram.iter().any(|&(_, weight)| weight == 0)
        || histogram.windows(2).any(|pair| pair[0].0 >= pair[1].0)
    {
        return Err(Error::InvalidInput(
            "invalid luma quantile histogram".into(),
        ));
    }
    let total = histogram.iter().try_fold(0u64, |sum, &(_, weight)| {
        sum.checked_add(weight)
            .ok_or(Error::Limit("total luma DC histogram weight overflow"))
    })?;
    let mut thresholds = Vec::new();
    thresholds
        .try_reserve_exact(usize::from(bins - 1).min(histogram.len() - 1))
        .map_err(|_| Error::Limit("luma quantile thresholds allocation"))?;
    let mut cumulative = 0u64;
    let mut rank = 1u8;
    for &(dc, weight) in &histogram[..histogram.len() - 1] {
        cumulative += weight; // Bounded by the checked total above.
        if rank < bins
            && u128::from(cumulative) * u128::from(bins) >= u128::from(total) * u128::from(rank)
        {
            thresholds.push(dc);
            // A heavy single value may consume several target quantiles.
            while rank < bins
                && u128::from(cumulative) * u128::from(bins) >= u128::from(total) * u128::from(rank)
            {
                rank += 1;
            }
        }
    }
    Ok(thresholds)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jpeg::ac_tokens::BlockContextModel;
    use crate::jpeg::data::{
        BLOCK_SIZE, CodingProcess, Component, ComponentScanInfo, JpegData, QuantTable, ScanInfo,
    };

    const POLICIES: [Policy; 8] = [
        Policy::Source,
        Policy::ChannelOnly,
        Policy::SharedChroma,
        Policy::PairedLuma,
        Policy::Quantile4,
        Policy::Quantile5,
        Policy::Activity4,
        Policy::Activity5,
    ];

    fn fixture(blocks: &[(i16, usize)], grayscale: bool) -> JpegData {
        let count = if grayscale { 1 } else { 3 };
        JpegData {
            width: blocks.len() as u32 * 8,
            height: 8,
            precision: 8,
            coding_process: Some(CodingProcess::Baseline),
            components: (0..count)
                .map(|c| {
                    let mut coefficients = vec![0; blocks.len() * BLOCK_SIZE];
                    for (block, &(dc, nonzeros)) in
                        coefficients.chunks_exact_mut(BLOCK_SIZE).zip(blocks)
                    {
                        block[0] = dc;
                        block[1..1 + nonzeros].fill(1);
                    }
                    Component {
                        id: c as u32 + 1,
                        h_samp_factor: 1,
                        v_samp_factor: 1,
                        quant_index: 0,
                        width_in_blocks: blocks.len() as u32,
                        height_in_blocks: 1,
                        coefficients,
                    }
                })
                .collect(),
            quant: vec![QuantTable {
                values: [1; BLOCK_SIZE],
                ..QuantTable::default()
            }],
            scan_info: vec![ScanInfo {
                components: (0..count)
                    .map(|component_index| ComponentScanInfo {
                        component_index,
                        ..ComponentScanInfo::default()
                    })
                    .collect(),
                ss: 0,
                se: 63,
                quant_table_indices: [Some(0), None, None, None],
                ..ScanInfo::default()
            }],
            ..JpegData::default()
        }
    }

    fn model(contexts: &DcContexts) -> BlockContextModel<'_> {
        BlockContextModel::new(
            [&[], contexts.luma_thresholds(), &[]],
            &[],
            contexts.context_map(),
            usize::from(contexts.num_contexts()),
        )
        .unwrap()
    }

    #[test]
    fn all_policies_are_valid_dense_dct8_only_and_preserve_inputs() {
        let blocks: Vec<_> = (0..64).map(|i| (i * 31 - 1000, i as usize % 64)).collect();
        for grayscale in [false, true] {
            let jpeg = fixture(&blocks, grayscale);
            let before: Vec<_> = jpeg
                .components
                .iter()
                .map(|c| c.coefficients.clone())
                .collect();
            let layout = TranscodeLayout::new(&jpeg).unwrap();
            let source = layout.dc_contexts();
            assert_eq!(Policy::default(), Policy::Source);
            for policy in POLICIES {
                let contexts = prepare(&layout, policy).unwrap();
                let validated = model(&contexts);
                let buckets = contexts.num_dc_contexts();
                let mut used = [false; 16];
                for plane in contexts
                    .context_map()
                    .chunks_exact(NUM_COEFFICIENT_ORDERS * buckets)
                {
                    for &id in &plane[..buckets] {
                        used[usize::from(id)] = true;
                    }
                    assert!(plane[buckets..].iter().all(|&id| id == 0));
                }
                assert!(
                    used[..usize::from(contexts.num_contexts())]
                        .iter()
                        .all(|&b| b)
                );
                assert_eq!(validated.num_dc_contexts(), buckets);
                assert!(contexts.luma_thresholds().windows(2).all(|p| p[0] < p[1]));
                if policy == Policy::Source {
                    assert_eq!(contexts, source);
                } else if policy == Policy::PairedLuma {
                    assert_eq!(contexts, layout.dc_contexts().with_paired_luma());
                }
                if matches!(
                    policy,
                    Policy::Quantile4
                        | Policy::Quantile5
                        | Policy::Activity4
                        | Policy::Activity5
                        | Policy::ChannelOnly
                ) {
                    assert!(contexts.num_contexts() <= 15);
                    assert!(contexts.luma_thresholds().len() <= 4);
                    for dc in 0..buckets {
                        assert_eq!(validated.context(dc as u8, 1, 0, 1).unwrap(), dc);
                        assert_eq!(
                            validated.context(dc as u8, 1, 0, 0).unwrap(),
                            buckets + if grayscale { 0 } else { dc }
                        );
                        assert_eq!(
                            validated.context(dc as u8, 1, 0, 2).unwrap(),
                            buckets + if grayscale { 0 } else { buckets + dc }
                        );
                    }
                    assert_eq!(
                        usize::from(contexts.num_contexts()),
                        if grayscale { buckets + 1 } else { 3 * buckets }
                    );
                }
            }
            assert_eq!(layout.dc_contexts(), source);
            for (component, saved) in jpeg.components.iter().zip(before) {
                assert_eq!(component.coefficients, saved);
            }
        }
    }

    #[test]
    fn shared_chroma_preserves_source_thresholds_luma_and_paired_chroma_buckets() {
        let blocks: Vec<_> = (0..4096).map(|i| ((i % 2048) as i16 - 1024, 0)).collect();
        for grayscale in [false, true] {
            let jpeg = fixture(&blocks, grayscale);
            let layout = TranscodeLayout::new(&jpeg).unwrap();
            let source = layout.dc_contexts();
            let shared = prepare(&layout, Policy::SharedChroma).unwrap();
            assert_eq!(shared.luma_thresholds(), source.luma_thresholds());
            let buckets = shared.num_dc_contexts();
            assert!(buckets >= 3);
            assert_eq!(
                &shared.context_map()[..buckets],
                &source.context_map()[..buckets]
            );
            let validated = model(&shared);
            for i in 0..buckets {
                let expected = buckets + if grayscale { 0 } else { i / 2 };
                assert_eq!(validated.context(i as u8, 1, 0, 0).unwrap(), expected);
                assert_eq!(validated.context(i as u8, 1, 0, 2).unwrap(), expected);
            }
            assert_eq!(
                usize::from(shared.num_contexts()),
                buckets + if grayscale { 1 } else { buckets.div_ceil(2) }
            );
        }
    }

    #[test]
    fn quantiles_use_exact_nearest_rank_and_strict_dc_boundaries() {
        let histogram: Vec<_> = (0..20).map(|i| (i, 1)).collect();
        assert_eq!(quantile_thresholds(&histogram, 4).unwrap(), [4, 9, 14]);
        assert_eq!(quantile_thresholds(&histogram, 5).unwrap(), [3, 7, 11, 15]);
        // At exact cumulative equality the cut belongs after the current value.
        assert_eq!(quantile_thresholds(&[(10, 1), (20, 1)], 2).unwrap(), [10]);
        let contexts = independent_channels(vec![4, 9, 14], false).unwrap();
        let validated = model(&contexts);
        for (dc, bucket) in [(4, 0), (5, 1), (9, 1), (10, 2), (14, 2), (15, 3)] {
            assert_eq!(validated.dc_context([i32::MAX, dc, i32::MIN]), bucket);
        }
    }

    #[test]
    fn activity_uses_borrowed_luma_nonzero_ac_counts_and_changes_cuts() {
        let mut jpeg = fixture(
            &[
                (0, 0),
                (1, 0),
                (2, 0),
                (3, 63),
                (4, 0),
                (5, 0),
                (6, 0),
                (7, 0),
            ],
            false,
        );
        // Chroma activity must not influence luma weights.
        for component in &mut jpeg.components[1..] {
            component.coefficients.fill(99);
        }
        let layout = TranscodeLayout::new(&jpeg).unwrap();
        assert_eq!(
            luma_histogram(&layout, true).unwrap(),
            [
                (0, 1),
                (1, 1),
                (2, 1),
                (3, 64),
                (4, 1),
                (5, 1),
                (6, 1),
                (7, 1)
            ]
        );
        assert_eq!(
            prepare(&layout, Policy::Quantile4)
                .unwrap()
                .luma_thresholds(),
            [1, 3, 5]
        );
        assert_eq!(
            prepare(&layout, Policy::Activity4)
                .unwrap()
                .luma_thresholds(),
            [3]
        );
        assert_eq!(
            prepare(&layout, Policy::Quantile5)
                .unwrap()
                .luma_thresholds(),
            [1, 3, 4, 6]
        );
        assert_eq!(
            prepare(&layout, Policy::Activity5)
                .unwrap()
                .luma_thresholds(),
            [3]
        );
    }

    #[test]
    fn ties_extremes_and_heavy_maxima_never_manufacture_empty_bins() {
        for bins in [4, 5] {
            for histogram in [
                vec![(17, 100)],
                vec![(i32::MIN, 1), (i32::MAX, u64::MAX - 1)],
                vec![(i32::MIN, u64::MAX - 1), (i32::MAX, 1)],
                vec![(-9, 4), (0, 4), (9, 4)],
            ] {
                let thresholds = quantile_thresholds(&histogram, bins).unwrap();
                let mut occupancy = vec![0; thresholds.len() + 1];
                for &(dc, _) in &histogram {
                    occupancy[thresholds.iter().filter(|&&t| dc > t).count()] += 1;
                }
                assert!(occupancy.iter().all(|&n| n > 0));
                assert!(thresholds.len() < usize::from(bins));
                assert!(thresholds.windows(2).all(|pair| pair[0] < pair[1]));
            }
        }
        let jpeg = fixture(
            &[(i16::MIN, 0), (-1025, 0), (1024, 0), (i16::MAX, 0)],
            false,
        );
        let layout = TranscodeLayout::new(&jpeg).unwrap();
        assert_eq!(
            prepare(&layout, Policy::Quantile4)
                .unwrap()
                .luma_thresholds(),
            [i32::from(i16::MIN), -1025, 1024]
        );
        let jpeg = fixture(&[(27, 63); 20], true);
        let layout = TranscodeLayout::new(&jpeg).unwrap();
        for policy in [
            Policy::Quantile4,
            Policy::Quantile5,
            Policy::Activity4,
            Policy::Activity5,
        ] {
            let contexts = prepare(&layout, policy).unwrap();
            assert!(contexts.luma_thresholds().is_empty());
            assert_eq!(contexts.num_contexts(), 2);
        }
    }

    #[test]
    fn rgb_offsets_and_mcu_padding_participate_in_histogram() {
        let mut jpeg = fixture(&[(i16::MIN, 0), (-2, 0), (0, 0), (i16::MAX, 0)], false);
        for (component, id) in jpeg.components.iter_mut().zip(b"RGB") {
            component.id = u32::from(*id);
        }
        let layout = TranscodeLayout::new(&jpeg).unwrap();
        assert_eq!(
            prepare(&layout, Policy::Quantile4)
                .unwrap()
                .luma_thresholds(),
            [-31744, 1022, 1024]
        );
        let mut jpeg = fixture(&[(-7, 0), (-3, 0)], true);
        jpeg.width = 1;
        jpeg.components[0].h_samp_factor = 2;
        let layout = TranscodeLayout::new(&jpeg).unwrap();
        assert_eq!(luma_histogram(&layout, false).unwrap(), [(-7, 1), (-3, 1)]);
        assert_eq!(
            prepare(&layout, Policy::Quantile4)
                .unwrap()
                .luma_thresholds(),
            [-7]
        );
    }

    #[test]
    fn histogram_errors_report_invalid_inputs_and_real_weight_overflow() {
        assert!(matches!(
            quantile_thresholds(&[(0, u64::MAX), (1, 1)], 4),
            Err(Error::Limit(_))
        ));
        let mut histogram = HashMap::new();
        add_weight(&mut histogram, i32::MIN, u64::MAX).unwrap();
        assert!(matches!(
            add_weight(&mut histogram, i32::MIN, 1),
            Err(Error::Limit(_))
        ));
        for invalid in [
            vec![],
            vec![(0, 0)],
            vec![(1, 1), (0, 1)],
            vec![(0, 1), (0, 1)],
        ] {
            assert!(matches!(
                quantile_thresholds(&invalid, 4),
                Err(Error::InvalidInput(_))
            ));
        }
    }
}
