// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// BSD-3-Clause; see docs/PORTING_KERNELS.md for the retained notice.
//! Search-local reuse of enc_coeff_order.cc / EncodeGlobalACInfo / AC groups.
//! The immutable JPEG borrow is part of the key, not an image hash. Only one
//! policy group survives; coefficient planes, tokens and entropy models do not.

use super::{
    CoefficientEntropy, Dct8Orders, FrameGeometry, JpegData, Options, OrderPolicy, SpeedTier,
    entropy, invalid,
};
use crate::{
    Error, Result,
    bit_writer::BitWriter,
    entropy::options::EntropyOptions,
    jpeg::{
        cfl::CflMode, coefficient_order::source_used_orders, transcode_layout::ChromaSubsampling,
    },
};

#[derive(Clone, Copy)]
struct FrameIdentity<'a>(&'a JpegData);

impl PartialEq for FrameIdentity<'_> {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self.0, other.0)
    }
}
impl Eq for FrameIdentity<'_> {}

impl std::fmt::Debug for FrameIdentity<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("FrameIdentity")
            .field(&(self.0 as *const JpegData))
            .finish()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OrderKey {
    Default,
    Sampled,
    Full,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct AcKey<'a> {
    frame: FrameIdentity<'a>,
    dimensions: (u32, u32),
    subsampling: ChromaSubsampling,
    cfl: CflMode,
    order: OrderKey,
    entropy: EntropyOptions,
    rle: bool,
    max_coefficients: usize,
    max_tokens: usize,
    max_output_bytes: usize,
}

impl<'a> AcKey<'a> {
    fn new(jpeg: &'a JpegData, geometry: FrameGeometry, options: &Options) -> Result<Self> {
        let (width, height) = geometry.dimensions_in_blocks();
        let customize = match options.order {
            OrderPolicy::Source => {
                source_used_orders(options.tier, width, height)
                    .map_err(invalid)?
                    .1
                    != 0
            }
            OrderPolicy::Default => false,
            OrderPolicy::Adaptive => true,
        };
        let order = if !customize {
            OrderKey::Default
        } else if options.tier >= SpeedTier::Squirrel {
            OrderKey::Sampled
        } else {
            OrderKey::Full
        };
        // CoefficientEntropy::source's complete resolved entropy/LZ policy.
        // Threading, sampling and reconstruction quality do not affect AC.
        let mut entropy = EntropyOptions::for_speed_tier(options.tier);
        if options.decoding_speed >= 1 {
            entropy.max_histograms = 6;
        }
        Ok(Self {
            frame: FrameIdentity(jpeg),
            dimensions: geometry.dimensions(),
            subsampling: geometry.subsampling(),
            cfl: options.cfl,
            order,
            entropy,
            rle: options.tier <= SpeedTier::Tortoise,
            max_coefficients: options.max_coefficients,
            max_tokens: options.max_tokens,
            max_output_bytes: options.max_output_bytes,
        })
    }
}

struct Fragment {
    bytes: Box<[u8]>,
    bits: usize,
}

impl Fragment {
    fn finish(mut writer: BitWriter) -> Result<Self> {
        let bits = writer.bits_written();
        writer.zero_pad_to_byte()?;
        Ok(Self {
            bytes: writer.into_bytes()?.into_boxed_slice(),
            bits,
        })
    }

    fn append(&self, writer: &mut BitWriter) -> Result<()> {
        let full_bytes = self.bits / 8;
        if writer.bits_written().is_multiple_of(8) {
            writer.append_byte_aligned(&self.bytes[..full_bytes])?;
        } else {
            for chunk in self.bytes[..full_bytes].chunks(7) {
                let mut word = [0; 8];
                word[..chunk.len()].copy_from_slice(chunk);
                writer.write((chunk.len() * 8) as u8, u64::from_le_bytes(word))?;
            }
        }
        let tail = (self.bits % 8) as u8;
        if tail != 0 {
            writer.write(tail, u64::from(self.bytes[full_bytes]) & ((1 << tail) - 1))?;
        }
        Ok(())
    }
}

pub(super) struct AcPayload {
    global: Fragment,
    groups: Vec<Fragment>,
}

impl AcPayload {
    fn build(
        orders: &Dct8Orders,
        entropy: &CoefficientEntropy,
        groups: usize,
        max_bytes: usize,
    ) -> Result<Self> {
        if entropy.group_count() != groups {
            return Err(Error::Encoding("JPEG AC cache group count mismatch".into()));
        }
        // This fragment starts AFTER the candidate-specific quant-table stream.
        let mut global = BitWriter::new(max_bytes);
        entropy::write_single_histogram_group(&mut global, groups)?;
        entropy::write_orders(orders, &mut global)?;
        entropy.write_model(&mut global)?;
        let mut total_bits = global.bits_written();
        let global = Fragment::finish(global)?;
        let mut streams = Vec::new();
        streams
            .try_reserve_exact(groups)
            .map_err(|_| Error::Limit("JPEG AC cache groups"))?;
        for index in 0..groups {
            let mut writer = BitWriter::new(max_bytes - total_bits / 8);
            entropy.write_group(index, &mut writer)?;
            total_bits = total_bits
                .checked_add(writer.bits_written())
                .filter(|&bits| bits.div_ceil(8) <= max_bytes)
                .ok_or(Error::Limit("JPEG AC cache output"))?;
            streams.push(Fragment::finish(writer)?);
        }
        // Boxed fragments retain no spare capacity. Total retained payload is
        // at most max_bytes + groups bytes (independent fragment tail padding),
        // plus O(groups) descriptors. No cumulative per-candidate retention.
        Ok(Self {
            global,
            groups: streams,
        })
    }

    pub(super) fn write_global(&self, writer: &mut BitWriter) -> Result<()> {
        self.global.append(writer)
    }

    pub(super) fn write_group(&self, index: usize, writer: &mut BitWriter) -> Result<()> {
        self.groups
            .get(index)
            .ok_or_else(|| Error::InvalidInput("JPEG AC group index out of range".into()))?
            .append(writer)
    }
}

pub(crate) struct SearchCache<'a> {
    jpeg: &'a JpegData,
    entry: Option<(AcKey<'a>, AcPayload)>,
    #[cfg(test)]
    hits: usize,
    #[cfg(test)]
    builds: usize,
}

impl<'a> SearchCache<'a> {
    pub(crate) fn new(jpeg: &'a JpegData) -> Self {
        Self {
            jpeg,
            entry: None,
            #[cfg(test)]
            hits: 0,
            #[cfg(test)]
            builds: 0,
        }
    }

    pub(super) fn get_or_build(
        &mut self,
        jpeg: &JpegData,
        geometry: FrameGeometry,
        options: &Options,
        build: impl FnOnce() -> Result<(Dct8Orders, CoefficientEntropy)>,
    ) -> Result<&AcPayload> {
        if !std::ptr::eq(self.jpeg, jpeg) {
            self.entry = None;
            return Err(Error::InvalidInput(
                "JPEG AC cache belongs to another frame".into(),
            ));
        }
        let key = AcKey::new(self.jpeg, geometry, options)?;
        if self.entry.as_ref().is_some_and(|(old, _)| old == &key) {
            #[cfg(test)]
            {
                self.hits += 1;
            }
        } else {
            // Evict before allocating a new policy group; failures publish none.
            self.entry = None;
            let (orders, entropy) = build()?;
            let payload = AcPayload::build(
                &orders,
                &entropy,
                geometry.group_count(),
                options.max_output_bytes,
            )?;
            self.entry = Some((key, payload));
            #[cfg(test)]
            {
                self.builds += 1;
            }
        }
        Ok(&self.entry.as_ref().expect("completed AC cache entry").1)
    }
}

#[cfg(test)]
#[path = "search_cache_tests.rs"]
mod tests;
