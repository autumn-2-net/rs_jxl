// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Derived from the BSD-3-Clause libjxl sources listed below. The complete
// upstream notice and license are retained in docs/PORTING_KERNELS.md.

//! Direct port of `HybridUintConfig::Encode` in `lib/jxl/dec_ans.h` at
//! libjxl commit `b866b3defec13494bfbaf705db93581cda118e37`.
//!
//! Configuration bounds follow `EncodeUintConfig` in `lib/jxl/enc_ans.cc`
//! and `PREFIX_MAX_BITS` in `lib/jxl/ans_params.h` at the same commit.
//! Only integer tokenization is implemented, not config serialization,
//! histogram/alphabet selection, entropy coding, or bitstream decoding.

use std::fmt;

/// Invalid parameters for a JXL hybrid unsigned-integer configuration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HybridUintConfigError {
    SplitExponentTooLarge,
    MsbExceedsSplitExponent,
    LsbExceedsRemainingBits,
}

impl fmt::Display for HybridUintConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::SplitExponentTooLarge => "split exponent must be in 0..=15",
            Self::MsbExceedsSplitExponent => "MSB count must not exceed the split exponent",
            Self::LsbExceedsRemainingBits => {
                "MSB and LSB counts must sum to at most the split exponent"
            }
        })
    }
}

impl std::error::Error for HybridUintConfigError {}

/// The entropy token and its separate, low-aligned extra-bit payload.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HybridUintEncoding {
    pub token: u32,
    pub nbits: u32,
    pub bits: u32,
}

/// Validated parameters for libjxl's hybrid unsigned-integer mapping.
///
/// The split exponent is in `0..=15`, and `msb_in_token + lsb_in_token`
/// must not exceed it. Private fields preserve these invariants after creation.
/// This is the tokenization domain, not a guarantee that the resulting tokens
/// fit a particular ANS or prefix-code alphabet. In particular, the config
/// writer omits MSB/LSB counts when the split equals the alphabet exponent;
/// only direct tokens are usable in that case, so those counts do not matter.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HybridUintConfig {
    split_exponent: u32,
    split_token: u32,
    msb_in_token: u32,
    lsb_in_token: u32,
}

impl HybridUintConfig {
    /// `PREFIX_MAX_BITS` from the pinned `lib/jxl/ans_params.h`.
    pub const MAX_SPLIT_EXPONENT: u32 = 15;

    pub const fn new(
        split_exponent: u32,
        msb_in_token: u32,
        lsb_in_token: u32,
    ) -> Result<Self, HybridUintConfigError> {
        if split_exponent > Self::MAX_SPLIT_EXPONENT {
            return Err(HybridUintConfigError::SplitExponentTooLarge);
        }
        if msb_in_token > split_exponent {
            return Err(HybridUintConfigError::MsbExceedsSplitExponent);
        }
        // Subtraction avoids overflow even for adversarial u32 arguments.
        if lsb_in_token > split_exponent - msb_in_token {
            return Err(HybridUintConfigError::LsbExceedsRemainingBits);
        }
        Ok(Self {
            split_exponent,
            split_token: 1u32 << split_exponent,
            msb_in_token,
            lsb_in_token,
        })
    }

    pub const fn split_exponent(&self) -> u32 {
        self.split_exponent
    }

    pub const fn split_token(&self) -> u32 {
        self.split_token
    }

    pub const fn msb_in_token(&self) -> u32 {
        self.msb_in_token
    }

    pub const fn lsb_in_token(&self) -> u32 {
        self.lsb_in_token
    }

    /// Direct port of `HybridUintConfig::LsbMask`.
    pub const fn lsb_mask(&self) -> u32 {
        (1u32 << self.lsb_in_token) - 1
    }

    /// Tokenize any `u32`, including zero and `u32::MAX`.
    ///
    /// Validation makes this infallible: in the non-direct branch,
    /// `msb + lsb <= split <= n <= 31`. Thus all shifts are below 32,
    /// subtractions cannot underflow, and the token sum fits in `u32`.
    /// Explicit unsigned shift operands also handle inputs with bit 31 set.
    #[inline]
    pub fn encode(&self, value: u32) -> HybridUintEncoding {
        if value < self.split_token {
            HybridUintEncoding {
                token: value,
                nbits: 0,
                bits: 0,
            }
        } else {
            let n = 31 - value.leading_zeros();
            let m = value - (1u32 << n);
            let token = self.split_token
                + ((n - self.split_exponent) << (self.msb_in_token + self.lsb_in_token))
                + ((m >> (n - self.msb_in_token)) << self.lsb_in_token)
                + (m & ((1u32 << self.lsb_in_token) - 1));
            let nbits = n - self.msb_in_token - self.lsb_in_token;
            let bits = (value >> self.lsb_in_token) & ((1u32 << nbits) - 1);
            HybridUintEncoding { token, nbits, bits }
        }
    }
}

impl Default for HybridUintConfig {
    fn default() -> Self {
        Self {
            split_exponent: 4,
            split_token: 16,
            msb_in_token: 2,
            lsb_in_token: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Independent arithmetic inverse: partition the token using division and
    // remainder, then concatenate the digit groups by multiplication in u64.
    // This is deliberately test-only, with no entropy reader or decoder API.
    fn inverse(config: HybridUintConfig, encoded: HybridUintEncoding) -> u32 {
        let split = 2u64.pow(config.split_exponent());
        let token = u64::from(encoded.token);
        if token < split {
            assert_eq!(encoded.nbits, 0);
            assert_eq!(encoded.bits, 0);
            return encoded.token;
        }

        let high_radix = 2u64.pow(config.msb_in_token());
        let low_radix = 2u64.pow(config.lsb_in_token());
        let group_size = high_radix * low_radix;
        let offset = token - split;
        let exponent = u64::from(config.split_exponent()) + offset / group_size;
        assert!(exponent <= 31);
        let extra = exponent - u64::from(config.msb_in_token() + config.lsb_in_token());
        assert_eq!(u64::from(encoded.nbits), extra);
        let extra_radix = 2u64.pow(u32::try_from(extra).unwrap());
        assert!(u64::from(encoded.bits) < extra_radix);
        let high = (offset % group_size) / low_radix;
        let low = offset % low_radix;
        let value = ((high_radix + high) * extra_radix + u64::from(encoded.bits)) * low_radix + low;
        u32::try_from(value).expect("encoded integer must fit u32")
    }

    fn check_roundtrip(config: HybridUintConfig, value: u32) {
        let encoded = config.encode(value);
        assert_eq!(inverse(config, encoded), value, "{config:?}: {encoded:?}");
    }

    fn check_vectors(config: HybridUintConfig, vectors: &[(u32, u32, u32, u32)]) {
        for &(value, token, nbits, bits) in vectors {
            assert_eq!(
                config.encode(value),
                HybridUintEncoding { token, nbits, bits },
                "{config:?}, value={value}"
            );
            check_roundtrip(config, value);
        }
    }

    #[test]
    fn default_and_accessors() {
        let config = HybridUintConfig::default();
        assert_eq!(config, HybridUintConfig::new(4, 2, 0).unwrap());
        assert_eq!(config.split_exponent(), 4);
        assert_eq!(config.split_token(), 16);
        assert_eq!(config.msb_in_token(), 2);
        assert_eq!(config.lsb_in_token(), 0);
        assert_eq!(config.lsb_mask(), 0);
        let config = HybridUintConfig::new(6, 1, 5).unwrap();
        assert_eq!(config.lsb_mask(), 31);
    }

    #[test]
    fn exact_default_vectors_from_dec_ans_header() {
        let config = HybridUintConfig::default();
        for value in 0..16 {
            check_vectors(config, &[(value, value, 0, 0)]);
        }
        // First seven vectors are the examples in dec_ans.h.
        check_vectors(
            config,
            &[
                (16, 16, 2, 0),
                (17, 16, 2, 1),
                (20, 17, 2, 0),
                (24, 18, 2, 0),
                (28, 19, 2, 0),
                (32, 20, 3, 0),
                (65_535, 63, 13, 8_191),
                (19, 16, 2, 3),
                (31, 19, 2, 3),
                (63, 23, 3, 7),
                (64, 24, 4, 0),
                (65_536, 64, 14, 0),
                (0x7fff_ffff, 123, 28, 0x0fff_ffff),
                (0x8000_0000, 124, 29, 0),
                (u32::MAX, 127, 29, 0x1fff_ffff),
            ],
        );
    }

    #[test]
    fn exact_zero_split_and_context_map_vectors() {
        check_vectors(
            HybridUintConfig::new(0, 0, 0).unwrap(),
            &[
                (0, 0, 0, 0),
                (1, 1, 0, 0),
                (2, 2, 1, 0),
                (3, 2, 1, 1),
                (4, 3, 2, 0),
                (7, 3, 2, 3),
                (0x7fff_ffff, 31, 30, 0x3fff_ffff),
                (0x8000_0000, 32, 31, 0),
                (u32::MAX, 32, 31, 0x7fff_ffff),
            ],
        );
        check_vectors(
            HybridUintConfig::new(2, 0, 1).unwrap(),
            &[
                (3, 3, 0, 0),
                (4, 4, 1, 0),
                (5, 5, 1, 0),
                (6, 4, 1, 1),
                (7, 5, 1, 1),
                (8, 6, 2, 0),
                (15, 7, 2, 3),
                (16, 8, 3, 0),
                (u32::MAX, 63, 30, 0x3fff_ffff),
            ],
        );
    }

    #[test]
    fn exact_msb_lsb_and_zero_extra_bit_vectors() {
        check_vectors(
            HybridUintConfig::new(4, 1, 2).unwrap(),
            &[
                (15, 15, 0, 0),
                (16, 16, 1, 0),
                (19, 19, 1, 0),
                (20, 16, 1, 1),
                (23, 19, 1, 1),
                (24, 20, 1, 0),
                (31, 23, 1, 1),
                (32, 24, 2, 0),
                (63, 31, 2, 3),
                (u32::MAX, 239, 28, 0x0fff_ffff),
            ],
        );
        check_vectors(
            HybridUintConfig::new(4, 2, 2).unwrap(),
            &[
                (16, 16, 0, 0),
                (17, 17, 0, 0),
                (31, 31, 0, 0),
                (32, 32, 1, 0),
                (36, 32, 1, 1),
                (63, 47, 1, 1),
                (64, 48, 2, 0),
                (u32::MAX, 463, 27, 0x07ff_ffff),
            ],
        );
    }

    #[test]
    fn invalid_configs_including_overflowing_arguments() {
        use HybridUintConfigError::*;
        for split in [16, 31, 32, u32::MAX] {
            assert_eq!(
                HybridUintConfig::new(split, 0, 0),
                Err(SplitExponentTooLarge)
            );
        }
        for split in 0..=15 {
            for msb in [split + 1, 32, u32::MAX] {
                assert_eq!(
                    HybridUintConfig::new(split, msb, 0),
                    Err(MsbExceedsSplitExponent)
                );
                assert!(HybridUintConfig::new(split, msb, u32::MAX).is_err());
            }
            for msb in 0..=split {
                for lsb in [split - msb + 1, 32, u32::MAX] {
                    assert_eq!(
                        HybridUintConfig::new(split, msb, lsb),
                        Err(LsbExceedsRemainingBits)
                    );
                }
            }
        }
        assert!(HybridUintConfig::new(u32::MAX, u32::MAX, u32::MAX).is_err());
    }

    #[test]
    fn full_config_domain_and_all_u32_power_boundaries() {
        // Include 13..=15 and unrestricted MSB/LSB counts, not just the
        // encoder's current search presets or the smaller ANS alphabet.
        for split in 0..=15 {
            for msb in 0..=split {
                for lsb in 0..=split - msb {
                    let config = HybridUintConfig::new(split, msb, lsb).unwrap();
                    assert_eq!(config.split_token(), 1u32 << split);
                    assert_eq!(config.lsb_mask(), (1u32 << lsb) - 1);
                    check_roundtrip(config, 0);
                    check_roundtrip(config, u32::MAX - 1);
                    check_roundtrip(config, u32::MAX);
                    for exponent in 0..32 {
                        let power = 1u32 << exponent;
                        for value in [power - 1, power, power + 1] {
                            check_roundtrip(config, value);
                        }
                    }
                    for value in [0x5555_5555, 0xaaaa_aaaa, 0x0123_4567, 0xfedc_ba98] {
                        check_roundtrip(config, value);
                    }
                }
            }
        }
    }

    #[test]
    fn exact_large_split_vectors_do_not_truncate_tokens() {
        check_vectors(
            HybridUintConfig::new(15, 0, 0).unwrap(),
            &[
                (32_767, 32_767, 0, 0),
                (32_768, 32_768, 15, 0),
                (65_535, 32_768, 15, 32_767),
                (65_536, 32_769, 16, 0),
                (u32::MAX, 32_784, 31, 0x7fff_ffff),
            ],
        );
        for (msb, lsb) in [(15, 0), (0, 15), (7, 8)] {
            check_vectors(
                HybridUintConfig::new(15, msb, lsb).unwrap(),
                &[
                    (32_767, 32_767, 0, 0),
                    (32_768, 32_768, 0, 0),
                    (65_535, 65_535, 0, 0),
                    (65_536, 65_536, 1, 0),
                    (u32::MAX, 589_823, 16, 65_535),
                ],
            );
        }
    }

    #[test]
    fn encoder_configurations_exhaustive_u16_and_full_width_samples() {
        // ChooseUintConfigs(kBest/kFast), enc_ans_params.h::UintConfig,
        // enc_lz77.cc::LenCost/DistCost, and entropy_coder_test.cc.
        let configs = [
            (4, 2, 0),
            (4, 1, 0),
            (4, 2, 1),
            (4, 2, 2),
            (4, 1, 2),
            (5, 2, 0),
            (5, 1, 0),
            (5, 2, 1),
            (5, 2, 2),
            (5, 1, 2),
            (3, 2, 0),
            (3, 1, 0),
            (3, 2, 1),
            (3, 1, 2),
            (4, 1, 3),
            (5, 1, 4),
            (5, 2, 3),
            (6, 1, 5),
            (6, 2, 4),
            (6, 0, 0),
            (0, 0, 0),
            (2, 0, 1),
            (7, 0, 0),
            (8, 0, 0),
            (9, 0, 0),
            (10, 0, 0),
            (11, 0, 0),
            (12, 0, 0),
            (1, 0, 0),
            (4, 1, 1),
        ];
        for (split, msb, lsb) in configs {
            let config = HybridUintConfig::new(split, msb, lsb).unwrap();
            for value in 0..=u32::from(u16::MAX) {
                check_roundtrip(config, value);
            }
            let mut value = 0x1234_5678u32;
            for _ in 0..4096 {
                value = value.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                check_roundtrip(config, value);
            }
        }
    }
}
