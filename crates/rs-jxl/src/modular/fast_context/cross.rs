//! Standard property 18 over the nearest compatible previous channel.

use crate::modular::{group::Channel, ma_train::ChannelView, predictor::clamped_gradient};

pub(super) fn previous<'a>(planes: &'a [Channel], c: usize, group: u32) -> Option<ChannelView<'a>> {
    let current = &planes[c];
    planes[..c]
        .iter()
        .enumerate()
        .rev()
        .find(|(_, reference)| {
            reference.width == current.width
                && reference.height == current.height
                && reference.hshift == current.hshift
                && reference.vshift == current.vshift
        })
        .map(|(channel, reference)| ChannelView {
            channel: channel as u32,
            group,
            width: reference.width,
            height: reference.height,
            stride: reference.width,
            pixels: &reference.samples,
            is_meta: reference.hshift < 0 || reference.vshift < 0,
        })
}

/// The reference, when present, and coordinates must already be validated.
#[inline]
pub(super) fn value(reference: Option<&ChannelView<'_>>, x: usize, y: usize) -> i32 {
    let Some(reference) = reference else {
        return 0;
    };
    let pos = y * reference.stride + x;
    let left = if x > 0 { reference.pixels[pos - 1] } else { 0 };
    let top = if y > 0 {
        reference.pixels[pos - reference.stride]
    } else {
        left
    };
    let nw = if x > 0 && y > 0 {
        reference.pixels[pos - reference.stride - 1]
    } else {
        left
    };
    reference.pixels[pos]
        .wrapping_sub(clamped_gradient(left, top, nw))
        .wrapping_abs()
}

/// Output covers the current view's width; a present reference matches it.
pub(super) fn fill(reference: Option<&ChannelView<'_>>, y: usize, output: &mut [i32]) {
    let Some(reference) = reference else {
        output.fill(0);
        return;
    };
    let Some((first, rest)) = output.split_first_mut() else {
        return;
    };
    *first = value(Some(reference), 0, y);
    let begin = y * reference.stride;
    let current = &reference.pixels[begin..begin + rest.len() + 1];
    if y == 0 {
        for (out, pair) in rest.iter_mut().zip(current.windows(2)) {
            *out = pair[1].wrapping_sub(pair[0]).wrapping_abs();
        }
    } else {
        let top =
            &reference.pixels[begin - reference.stride..begin - reference.stride + current.len()];
        for ((out, pair), north) in rest.iter_mut().zip(current.windows(2)).zip(top.windows(2)) {
            *out = pair[1]
                .wrapping_sub(clamped_gradient(pair[0], north[1], north[0]))
                .wrapping_abs();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modular::{encode, weighted};

    fn plane(width: usize, height: usize, hshift: i32, vshift: i32, seed: usize) -> Channel {
        let values = [i32::MIN, i32::MAX, -1, 0, 1, -17, 29];
        Channel {
            width,
            height,
            hshift,
            vshift,
            samples: (0..width * height)
                .map(|i| values[(i + seed) % values.len()])
                .collect(),
        }
    }

    fn differential(planes: &[Channel], group: u32) {
        for c in 0..planes.len() {
            let reference = previous(planes, c, group);
            let current = &planes[c];
            let mut row = vec![123; current.width];
            let mut visited = 0;
            encode::visit_channel(
                planes,
                c,
                &weighted::Header::default(),
                false,
                1,
                group as usize,
                |channel, pos, _, _, properties, _| {
                    assert_eq!(channel, c);
                    let x = pos % current.width;
                    let y = pos / current.width;
                    if x == 0 {
                        fill(reference.as_ref(), y, &mut row);
                    }
                    assert_eq!(value(reference.as_ref(), x, y), properties[18]);
                    assert_eq!(row[x], properties[18]);
                    visited += 1;
                    Ok(())
                },
            )
            .unwrap();
            assert_eq!(visited, current.width * current.height);
        }
    }

    #[test]
    fn mapping_skips_incompatible_dimensions_and_shifts() {
        let planes = [
            plane(5, 3, 0, 0, 0),
            plane(4, 3, 0, 0, 1),
            plane(5, 2, 0, 0, 2),
            plane(5, 3, 1, 0, 3),
            plane(5, 3, 0, 1, 4),
            plane(5, 3, 0, 0, 5),
            plane(5, 3, 0, 0, 6),
        ];
        assert!(previous(&planes, 0, 7).is_none());
        assert_eq!(previous(&planes, 5, 7).unwrap().channel, 0);
        assert_eq!(previous(&planes, 6, 7).unwrap().channel, 5);
        differential(&planes, 7);
    }

    #[test]
    fn metadata_and_sparse_groups_keep_stream_local_references() {
        let first = [plane(3, 2, -1, -1, 0), plane(3, 2, -1, -1, 1)];
        let second = [plane(3, 2, 0, 0, 2), plane(3, 2, 1, 0, 3)];
        let reference = previous(&first, 1, 91).unwrap();
        assert!(reference.is_meta);
        assert_eq!(reference.group, 91);
        assert_eq!(reference.channel, 0);
        assert!(previous(&second, 1, 4099).is_none());
        differential(&first, 91);
        differential(&second, 4099);
    }

    #[test]
    fn signed_extremes_and_narrow_rows_match_coding() {
        for width in [1, 2, 7, 16, 17, 33] {
            let planes = [plane(width, 5, 0, 0, 0), plane(width, 5, 0, 0, 3)];
            differential(&planes, 0);
        }
        let planes = [plane(1, 1, 0, 0, 0), plane(1, 1, 0, 0, 1)];
        assert_eq!(value(previous(&planes, 1, 0).as_ref(), 0, 0), i32::MIN);
        differential(&planes, 0);
    }

    #[test]
    fn missing_and_empty_references() {
        let mut row = [123; 4];
        fill(None, 99, &mut row);
        assert_eq!(row, [0; 4]);
        assert_eq!(value(None, 99, 99), 0);
        for (width, height) in [(0, 0), (0, 3), (3, 0)] {
            let planes = [plane(width, height, 0, 0, 0), plane(width, height, 0, 0, 1)];
            assert!(previous(&planes, 1, 17).is_some());
            fill(previous(&planes, 1, 17).as_ref(), 0, &mut []);
            differential(&planes, 17);
        }
    }

    #[test]
    fn padded_reference_uses_its_own_stride() {
        let packed = plane(3, 3, 0, 0, 0);
        let pixels: Vec<_> = packed
            .samples
            .chunks(3)
            .flat_map(|row| row.iter().copied().chain([123, 456]))
            .collect();
        let reference = ChannelView {
            channel: 0,
            group: 31,
            width: 3,
            height: 3,
            stride: 5,
            pixels: &pixels,
            is_meta: false,
        };
        let planes = [packed.clone(), packed];
        let expected = previous(&planes, 1, 31).unwrap();
        for y in 0..3 {
            let mut row = [0; 3];
            fill(Some(&reference), y, &mut row);
            for (x, actual) in row.into_iter().enumerate() {
                assert_eq!(actual, value(Some(&reference), x, y));
                assert_eq!(actual, value(Some(&expected), x, y));
            }
        }
    }
}
