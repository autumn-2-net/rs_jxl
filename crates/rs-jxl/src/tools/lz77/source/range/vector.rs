// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Derived from BSD-3-Clause libjxl; see docs/PORTING_KERNELS.md.

//! enc_lz77_bucket.cc::RelaxCosts, b866b3d: strict f32 comparison and masked
//! updates of all four SoA fields. Native x86 SSE2/AVX2, scalar elsewhere.

use super::super::matchers::filled;
use super::Step;
use crate::Result;

#[derive(Clone, Copy)]
pub(super) struct Update {
    pub start: usize,
    pub end: usize,
    pub first_length: u32,
    pub distance: u32,
    pub context: u32,
    pub cost: f32,
}

pub(super) struct VectorSteps {
    costs: Vec<f32>,
    lengths: Vec<u32>,
    distances: Vec<u32>,
    contexts: Vec<u32>,
    relax: fn(&mut Self, Update),
}

impl VectorSteps {
    pub(super) fn new(count: usize) -> Result<Self> {
        #[allow(unused_mut)]
        let mut relax: fn(&mut Self, Update) = scalar;
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                // SAFETY: selected once after runtime AVX2 detection. The only
                // caller validates equal-length arrays and the update bounds.
                relax = |steps, update| unsafe { x86::avx2(steps, update) };
            } else if std::is_x86_feature_detected!("sse2") {
                // SAFETY: same bounds contract, with runtime SSE2 detection.
                relax = |steps, update| unsafe { x86::sse2(steps, update) };
            }
        }
        Ok(Self {
            costs: filled(count, f32::INFINITY)?,
            lengths: filled(count, 0)?,
            distances: filled(count, 0)?,
            contexts: filled(count, 0)?,
            relax,
        })
    }

    pub(super) fn at(&self, pos: usize) -> Step {
        Step {
            cost: self.costs[pos],
            length: self.lengths[pos],
            distance: self.distances[pos],
            context: self.contexts[pos],
        }
    }

    pub(super) fn set(&mut self, pos: usize, step: Step) {
        self.costs[pos] = step.cost;
        self.lengths[pos] = step.length;
        self.distances[pos] = step.distance;
        self.contexts[pos] = step.context;
    }

    pub(super) fn update(&mut self, update: Update) {
        assert!(update.start <= update.end && update.end <= self.costs.len());
        assert!(
            u32::try_from(update.end - update.start)
                .ok()
                .and_then(|count| update.first_length.checked_add(count))
                .is_some()
        );
        (self.relax)(self, update);
    }
}

fn scalar(steps: &mut VectorSteps, update: Update) {
    for pos in update.start..update.end {
        if update.cost < steps.costs[pos] {
            steps.set(
                pos,
                Step {
                    cost: update.cost,
                    length: update.first_length + (pos - update.start) as u32,
                    distance: update.distance,
                    context: update.context,
                },
            );
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86 {
    use super::*;
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    #[target_feature(enable = "sse2")]
    pub(super) unsafe fn sse2(steps: &mut VectorSteps, update: Update) {
        let mut pos = update.start;
        // SAFETY: CPU feature was checked by new(). Four unaligned lanes are
        // read/written only while all four equal-length arrays contain them.
        unsafe {
            let offered = _mm_set1_ps(update.cost);
            while update.end - pos >= 4 {
                let old = _mm_loadu_ps(steps.costs.as_ptr().add(pos));
                let better = _mm_cmplt_ps(offered, old);
                if _mm_movemask_ps(better) != 0 {
                    let mask = _mm_castps_si128(better);
                    let select = |new, old| {
                        _mm_or_si128(_mm_and_si128(mask, new), _mm_andnot_si128(mask, old))
                    };
                    let first = update.first_length + (pos - update.start) as u32;
                    let length = _mm_setr_epi32(
                        first as i32,
                        (first + 1) as i32,
                        (first + 2) as i32,
                        (first + 3) as i32,
                    );
                    _mm_storeu_ps(
                        steps.costs.as_mut_ptr().add(pos),
                        _mm_or_ps(_mm_and_ps(better, offered), _mm_andnot_ps(better, old)),
                    );
                    for (array, new) in [
                        (&mut steps.lengths, length),
                        (&mut steps.distances, _mm_set1_epi32(update.distance as i32)),
                        (&mut steps.contexts, _mm_set1_epi32(update.context as i32)),
                    ] {
                        let ptr = array.as_mut_ptr().add(pos).cast::<__m128i>();
                        _mm_storeu_si128(ptr, select(new, _mm_loadu_si128(ptr)));
                    }
                }
                pos += 4;
            }
        }
        scalar(
            steps,
            Update {
                start: pos,
                first_length: update.first_length + (pos - update.start) as u32,
                ..update
            },
        );
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn avx2(steps: &mut VectorSteps, update: Update) {
        let mut pos = update.start;
        // SAFETY: runtime AVX2 detection and the same bounds contract as SSE2;
        // the loop checks eight complete lanes before any load or store.
        unsafe {
            let offered = _mm256_set1_ps(update.cost);
            while update.end - pos >= 8 {
                let old = _mm256_loadu_ps(steps.costs.as_ptr().add(pos));
                let better = _mm256_cmp_ps::<_CMP_LT_OQ>(offered, old);
                if _mm256_movemask_ps(better) != 0 {
                    let mask = _mm256_castps_si256(better);
                    let first = update.first_length + (pos - update.start) as u32;
                    let length = _mm256_add_epi32(
                        _mm256_set1_epi32(first as i32),
                        _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7),
                    );
                    _mm256_storeu_ps(
                        steps.costs.as_mut_ptr().add(pos),
                        _mm256_blendv_ps(old, offered, better),
                    );
                    for (array, new) in [
                        (&mut steps.lengths, length),
                        (
                            &mut steps.distances,
                            _mm256_set1_epi32(update.distance as i32),
                        ),
                        (
                            &mut steps.contexts,
                            _mm256_set1_epi32(update.context as i32),
                        ),
                    ] {
                        let ptr = array.as_mut_ptr().add(pos).cast::<__m256i>();
                        _mm256_storeu_si256(
                            ptr,
                            _mm256_blendv_epi8(_mm256_loadu_si256(ptr), new, mask),
                        );
                    }
                }
                pos += 8;
            }
        }
        scalar(
            steps,
            Update {
                start: pos,
                first_length: update.first_length + (pos - update.start) as u32,
                ..update
            },
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_and_scalar_update_every_field_without_touching_ties_or_tails() {
        #[allow(unused_mut)]
        let mut variants: Vec<fn(&mut VectorSteps, Update)> = vec![scalar];
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("sse2") {
                // SAFETY: feature checked here; update() checks slice bounds.
                variants.push(|s, u| unsafe { x86::sse2(s, u) });
            }
            if std::is_x86_feature_detected!("avx2") {
                // SAFETY: feature checked here; update() checks slice bounds.
                variants.push(|s, u| unsafe { x86::avx2(s, u) });
            }
        }
        for variant in variants {
            for start in 0..9 {
                for count in 0..35 {
                    let mut expected = VectorSteps::new(64).unwrap();
                    let mut actual = VectorSteps::new(64).unwrap();
                    for i in 0..64 {
                        let cost = [f32::INFINITY, 0.0, -0.0, 7.0, 8.0, f32::NAN][i % 6];
                        let step = Step {
                            cost,
                            length: 99,
                            distance: 123,
                            context: u32::MAX,
                        };
                        expected.set(i, step);
                        actual.set(i, step);
                    }
                    actual.relax = variant;
                    let update = Update {
                        start,
                        end: start + count,
                        first_length: 3,
                        distance: 1 << 20,
                        context: 0x8765_4321,
                        cost: 7.0,
                    };
                    scalar(&mut expected, update);
                    actual.update(update);
                    for i in 0..64 {
                        let a = actual.at(i);
                        let b = expected.at(i);
                        assert_eq!(
                            (a.cost.to_bits(), a.length, a.distance, a.context),
                            (b.cost.to_bits(), b.length, b.distance, b.context)
                        );
                    }
                }
            }
        }
    }
}
