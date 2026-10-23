// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Derived from BSD-3-Clause libjxl; see docs/PORTING_KERNELS.md.

//! Decoder-arithmetic acceptance guard for integer Add, derived from frozen
//! b866b3d dec_modular.cc::SingleFromSingle, blending.cc::PerformBlending,
//! dec_external_image.cc::FloatToU32 and stage_write.cc::MakeUnsigned.
//!
//! enc_patch_dictionary.cc marks component pixels visited, NOT their complete
//! bounding boxes. Disconnected components can have overlapping rectangles;
//! exact integer SubtractFrom/compose alone cannot prove lossless f32 replay.
//! This guard checks original declared samples after decoder recovery, not
//! normalized-f32 equality or an epsilon. Row events avoid a full-frame float
//! allocation and skip all arithmetic on uncovered pixels.

use std::collections::BTreeSet;

use super::{
    ImageView, PatchBlendMode, PatchDictionary, SignedPatchImage, invalid, validate_add, zeros,
};
use crate::{Error, Result};

fn recovers(value: f32, maximum: f32, expected: i32, byte_output: bool) -> bool {
    if !value.is_finite() {
        return false;
    }
    // FloatToU32 clamps before multiplication; stage_write clamps afterwards.
    let external = (value.clamp(0.0, 1.0) * maximum).round_ties_even() as i32;
    let scaled = value * maximum;
    if external != expected || scaled.clamp(0.0, maximum).round_ties_even() as i32 != expected {
        return false;
    }
    if byte_output {
        // MakeUnsigned<u8> adds kDither in [-0.49219, +0.49219]. Checking both
        // extrema with the actual f32 operations proves EVERY table entry
        // recovers the same word, regardless of channel/position/orientation.
        // This is deliberately stronger than checking just one dither phase.
        for dither in [-0.49219f32, 0.49219f32] {
            if (scaled + dither).clamp(0.0, maximum).round_ties_even() as i32 != expected {
                return false;
            }
        }
    }
    true
}

/// Require exact original nominal samples after normalized decoder f32 Add.
/// Only covered color pixels are replayed. Extras and uncovered pixels are
/// unchanged by Add and are not numerically converted here. Also accepts only
/// results safe for every frozen-source u8 output dither phase when bits <= 8.
/// Failure is explicit, never an implicit fallback or approximate acceptance.
pub fn verify_add_replay(
    image: ImageView<'_>,
    residual: &SignedPatchImage,
    reference: &SignedPatchImage,
    dictionary: &PatchDictionary,
) -> Result<()> {
    image.validate()?;
    let stride = validate_add(residual, reference, dictionary)?;
    if image.width != residual.width
        || image.height != residual.height
        || image.bit_depth != residual.bit_depth
        || image.color_channels != residual.color_channels
        || image.planes.len() != residual.planes.len()
    {
        return Err(invalid("integer patch replay metadata differs"));
    }
    let mut events = Vec::new();
    events
        .try_reserve(
            dictionary
                .positions
                .len()
                .checked_mul(2)
                .ok_or(Error::Limit("patch replay events"))?,
        )
        .map_err(|_| Error::Limit("patch replay events"))?;
    for (index, position) in dictionary.positions.iter().enumerate() {
        if dictionary.blendings[index * stride].mode == PatchBlendMode::None {
            continue;
        }
        let crop = dictionary.references[position.ref_pos_idx];
        events.push((position.y as usize, index, true));
        events.push((position.y as usize + crop.height as usize, index, false));
    }
    if events.is_empty() {
        return Ok(());
    }
    events.sort_unstable();
    let maximum = ((1u32 << image.bit_depth) - 1) as f32;
    let scale = 1.0f32 / maximum;
    let mut row = zeros::<f32>(image.width)?;
    let mut active = BTreeSet::new();
    let mut ranges: Vec<(usize, usize)> = Vec::new();
    let mut next = 0;
    while next < events.len() {
        let first_y = events[next].0;
        while next < events.len() && events[next].0 == first_y {
            let (_, index, starts) = events[next];
            if starts {
                active.insert(index);
            } else {
                active.remove(&index);
            }
            next += 1;
        }
        if next == events.len() {
            break;
        }
        if active.is_empty() {
            continue;
        }
        ranges.clear();
        for &index in &active {
            let position = dictionary.positions[index];
            let crop = dictionary.references[position.ref_pos_idx];
            ranges.push((
                position.x as usize,
                position.x as usize + crop.width as usize,
            ));
        }
        ranges.sort_unstable();
        let mut merged = 0;
        for index in 0..ranges.len() {
            let (start, end) = ranges[index];
            if merged != 0 && start <= ranges[merged - 1].1 {
                ranges[merged - 1].1 = ranges[merged - 1].1.max(end);
            } else {
                ranges[merged] = (start, end);
                merged += 1;
            }
        }
        ranges.truncate(merged);
        for y in first_y..events[next].0 {
            for channel in 0..image.color_channels {
                let offset = y * image.width;
                for &(start, end) in &ranges {
                    for (out, &word) in row[start..end]
                        .iter_mut()
                        .zip(&residual.planes[channel][offset + start..offset + end])
                    {
                        *out = word as f32 * scale;
                    }
                }
                // GetPatchesForRow sorts by dictionary position index. Do not
                // reassociate sums or multiply an integer sum by scale once.
                for &index in &active {
                    let position = dictionary.positions[index];
                    let crop = dictionary.references[position.ref_pos_idx];
                    let start = position.x as usize;
                    let end = start + crop.width as usize;
                    let src = (crop.y0 as usize + y - position.y as usize) * reference.width
                        + crop.x0 as usize;
                    for (out, &word) in row[start..end]
                        .iter_mut()
                        .zip(&reference.planes[channel][src..src + crop.width as usize])
                    {
                        *out += word as f32 * scale;
                    }
                }
                for &(start, end) in &ranges {
                    for (x, &value) in row.iter().enumerate().take(end).skip(start) {
                        let expected = image.planes[channel][offset + x];
                        if !recovers(value, maximum, expected, image.bit_depth <= 8) {
                            return Err(Error::Encoding(format!(
                                "source integer patch Add decoder replay changes channel {channel}, sample {} (expected {expected})",
                                offset + x,
                            )));
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::{
        PatchBlending, PatchPosition, PatchReferencePosition, REFERENCE_ID, compose_add,
        subtract_add,
    };
    use super::*;

    fn dictionary(count: usize) -> PatchDictionary {
        PatchDictionary {
            references: vec![PatchReferencePosition {
                reference: REFERENCE_ID,
                x0: 0,
                y0: 0,
                width: 1,
                height: 1,
            }],
            positions: vec![
                PatchPosition {
                    x: 0,
                    y: 0,
                    ref_pos_idx: 0
                };
                count
            ],
            blendings: vec![
                PatchBlending {
                    mode: PatchBlendMode::Add,
                    alpha_channel: 0,
                    clamp: false
                };
                count
            ],
            num_extra_channels: 0,
        }
    }

    #[test]
    fn exact_integer_replay_can_fail_decoder_recovery_after_many_overlaps() {
        for (bits, count, original, delta) in [(8, 1024, 0, 127), (16, 256, 1, -65535)] {
            let planes = [vec![original]];
            let image = ImageView {
                width: 1,
                height: 1,
                bit_depth: bits,
                color_channels: 1,
                planes: &planes,
            };
            let reference = SignedPatchImage {
                width: 1,
                height: 1,
                bit_depth: bits,
                color_channels: 1,
                planes: vec![vec![delta]],
            };
            let dictionary = dictionary(count);
            let residual = subtract_add(image, &reference, &dictionary).unwrap();
            assert_eq!(
                compose_add(&residual, &reference, &dictionary).unwrap(),
                planes
            );
            let error = verify_add_replay(image, &residual, &reference, &dictionary).unwrap_err();
            assert!(error.to_string().contains("decoder replay changes"));
        }
    }

    #[test]
    fn different_normalized_float_bits_are_allowed_only_when_original_word_recovers() {
        for bits in [8, 16] {
            let maximum = (1 << bits) - 1;
            let scale = 1.0f32 / maximum as f32;
            let original = 63;
            let delta = original - maximum;
            let reconstructed = maximum as f32 * scale + delta as f32 * scale;
            assert_ne!(reconstructed.to_bits(), (original as f32 * scale).to_bits());
            assert!(recovers(reconstructed, maximum as f32, original, bits <= 8));
            let planes = [vec![original]];
            let image = ImageView {
                width: 1,
                height: 1,
                bit_depth: bits,
                color_channels: 1,
                planes: &planes,
            };
            let reference = SignedPatchImage {
                width: 1,
                height: 1,
                bit_depth: bits,
                color_channels: 1,
                planes: vec![vec![delta]],
            };
            let dictionary = dictionary(1);
            let residual = subtract_add(image, &reference, &dictionary).unwrap();
            verify_add_replay(image, &residual, &reference, &dictionary).unwrap();
        }
    }

    #[test]
    fn source_components_can_have_overlapping_bounding_rectangles() {
        let (width, height) = (128, 96);
        let mut planes = [vec![255; width * height]];
        for (x0, y0) in [(17, 17), (81, 49)] {
            for y in 0..25 {
                for x in 0..25 {
                    if x == 0 || x == 24 || y == 0 || y == 24 {
                        planes[0][(y0 + y) * width + x0 + x] = 0;
                    } else if (9..15).contains(&x) && (9..14).contains(&y) {
                        planes[0][(y0 + y) * width + x0 + x] = 64;
                    }
                }
            }
        }
        let image = ImageView {
            width,
            height,
            bit_depth: 8,
            color_channels: 1,
            planes: &planes,
        };
        for fast in [false, true] {
            let config = super::super::SourcePatchConfig {
                fast,
                ..Default::default()
            };
            let plan = super::super::discover_source(image, &config)
                .unwrap()
                .expect("active nested components");
            let mut coverage = vec![0; width * height];
            for position in &plan.dictionary.positions {
                let crop = plan.dictionary.references[position.ref_pos_idx];
                for y in position.y..position.y + crop.height {
                    for x in position.x..position.x + crop.width {
                        coverage[y as usize * width + x as usize] += 1;
                    }
                }
            }
            assert!(
                coverage.iter().any(|&count| count > 1),
                "component disjointness does not imply rectangle disjointness"
            );
            assert_eq!(
                compose_add(&plan.residual, &plan.reference, &plan.dictionary).unwrap(),
                planes
            );
            verify_add_replay(image, &plan.residual, &plan.reference, &plan.dictionary).unwrap();
        }
    }

    #[test]
    fn byte_output_dither_and_ties_are_part_of_exact_recovery_not_a_tolerance() {
        assert!(recovers(63.0 / 255.0, 255.0, 63, true));
        let value = 63.25 / 255.0;
        assert!(recovers(value, 255.0, 63, false));
        assert!(!recovers(value, 255.0, 63, true));
        assert!(recovers(62.5 / 255.0, 255.0, 62, false));
        assert!(!recovers(62.5 / 255.0, 255.0, 63, false));
        assert!(!recovers(f32::NAN, 255.0, 0, true));
    }

    #[test]
    fn row_events_preserve_overlaps_gaps_and_none_blends() {
        let planes = [vec![63; 7 * 9]];
        let image = ImageView {
            width: 7,
            height: 9,
            bit_depth: 8,
            color_channels: 1,
            planes: &planes,
        };
        let reference = SignedPatchImage {
            width: 2,
            height: 3,
            bit_depth: 8,
            color_channels: 1,
            planes: vec![vec![-128; 6]],
        };
        let mut dictionary = dictionary(4);
        dictionary.references[0].width = 2;
        dictionary.references[0].height = 3;
        for (position, (x, y)) in
            dictionary
                .positions
                .iter_mut()
                .zip([(4, 6), (2, 1), (1, 2), (0, 0)])
        {
            position.x = x;
            position.y = y;
        }
        dictionary.blendings[3].mode = PatchBlendMode::None;
        let residual = subtract_add(image, &reference, &dictionary).unwrap();
        assert_eq!(
            compose_add(&residual, &reference, &dictionary).unwrap(),
            planes
        );
        verify_add_replay(image, &residual, &reference, &dictionary).unwrap();
        let mut bad = residual;
        bad.planes[0][7 + 2] += 1;
        assert!(verify_add_replay(image, &bad, &reference, &dictionary).is_err());
    }
}
