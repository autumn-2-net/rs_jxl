// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Derived from BSD-3-Clause libjxl; see docs/PORTING_KERNELS.md.

//! enc_lz77_index.h::LZ77StaticIndex<Value> at b866b3d. Narrowing is lossless
//! only when EVERY input value fits u16; there is no reserved sentinel value.

use super::super::search::{match_length, word_length, word_length_u16};
use super::SourceRunKeys;
use crate::Result;

pub(super) enum Values<'a> {
    Wide(&'a [u32]),
    Narrow(Vec<u16>),
}

impl<'a> Values<'a> {
    pub(super) fn new(values: &'a [u32], compact: bool) -> Result<Self> {
        if compact && values.iter().all(|&value| value <= u32::from(u16::MAX)) {
            let mut narrow = super::super::reserved(values.len())?;
            narrow.extend(values.iter().map(|&value| value as u16));
            Ok(Self::Narrow(narrow))
        } else {
            Ok(Self::Wide(values))
        }
    }

    pub(super) fn len(&self) -> usize {
        match self {
            Self::Wide(values) => values.len(),
            Self::Narrow(values) => values.len(),
        }
    }

    pub(super) fn get(&self, position: usize) -> u32 {
        match self {
            Self::Wide(values) => values[position],
            Self::Narrow(values) => u32::from(values[position]),
        }
    }

    pub(super) fn hash(&self, position: usize) -> u32 {
        match self {
            Self::Wide(values) => hash(values, position),
            Self::Narrow(values) => hash(values, position),
        }
    }

    pub(super) fn long_hash(
        &self,
        zeros: u32,
        position: usize,
        window: usize,
        full: bool,
        mode: SourceRunKeys,
    ) -> u32 {
        match self {
            Self::Wide(values) => long_hash(values, zeros, position, window, full, mode),
            Self::Narrow(values) => long_hash(values, zeros, position, window, full, mode),
        }
    }

    pub(super) fn length(
        &self,
        position: usize,
        previous: usize,
        prefix: usize,
        limit: usize,
        word: bool,
    ) -> usize {
        prefix
            + match self {
                Self::Wide(values) => {
                    let a = &values[position + prefix..position + limit];
                    let b = &values[previous + prefix..previous + limit];
                    if word {
                        word_length(a, b)
                    } else {
                        match_length(a, b)
                    }
                }
                Self::Narrow(values) => {
                    let a = &values[position + prefix..position + limit];
                    let b = &values[previous + prefix..previous + limit];
                    if word {
                        word_length_u16(a, b)
                    } else {
                        match_length(a, b)
                    }
                }
            }
    }
}

fn hash<Value: Copy + Into<u32>>(values: &[Value], position: usize) -> u32 {
    if values.len().saturating_sub(position) < 3 {
        return 0;
    }
    let mut hash = 0u32;
    for &value in &values[position..position + 3] {
        hash ^= value
            .into()
            .wrapping_add(0x9e37_79b9)
            .wrapping_add(hash << 6)
            .wrapping_add(hash >> 2);
    }
    hash ^= hash >> 16;
    hash = hash.wrapping_mul(0x85eb_ca6b);
    hash ^= hash >> 13;
    hash = hash.wrapping_mul(0xc2b2_ae35);
    hash ^ (hash >> 16)
}

pub(super) fn long_hash<Value: Copy + Into<u32>>(
    values: &[Value],
    zeros: u32,
    position: usize,
    window: usize,
    full: bool,
    mode: SourceRunKeys,
) -> u32 {
    let mut key = hash(values, position);
    if (mode == SourceRunKeys::LengthAndSuffix && zeros >= 8)
        || (mode == SourceRunKeys::SharedSuffix
            && zeros >= 8
            && (zeros as usize) < window
            && position + zeros as usize + 3 <= values.len())
    {
        if mode == SourceRunKeys::LengthAndSuffix {
            key ^= zeros.wrapping_mul(0x9e37_79b9);
        }
        let next = position + zeros as usize;
        for &value in &values[next..values.len().min(next + 3)] {
            key ^= value
                .into()
                .wrapping_add(0x9e37_79b9)
                .wrapping_add(key << 6)
                .wrapping_add(key >> 2);
        }
        key ^= key >> 16;
        key = key.wrapping_mul(0x85eb_ca6b);
        return key ^ (key >> 13);
    }
    key ^= values[position + 3].into().wrapping_mul(0x85eb_ca6b);
    key ^= values[position + 7].into().wrapping_mul(0xc2b2_ae35);
    if full {
        key ^= values[position + 4].into().wrapping_mul(0x9e37_79b9);
        key ^= values[position + 5].into().wrapping_mul(0x27d4_eb2d);
        key ^= values[position + 6].into().wrapping_mul(0x1656_67b1);
    }
    key ^ (key >> 16)
}
