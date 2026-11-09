//! Exact token LCP kernels, implemented originally in Rust.
//!
//! Inspired by the equality-mask / first-difference idiom in misa77
//! `src/isa/lib_avx2.h` (research snapshot misa77-20261109, commit
//! 882cfdad879f473afebe828ff19832bf9e4b1754).
//! That reference is MIT, Copyright (c) 2026 Shreyas Ghildiyal. This is not
//! a translation of its byte codec, copy routines, or wire format.

use crate::{Error, Result};

#[cfg(all(
    any(feature = "experimental-lz77", test),
    any(target_arch = "x86", target_arch = "x86_64")
))]
mod x86;

#[cfg(test)]
mod tests;

/// Exact match-extension implementation. Legacy leaves the caller unchanged.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum MatchExtension {
    #[default]
    Legacy,
    Scalar,
    Avx2,
    /// Requires AVX-512F and AVX-512BW for both token widths.
    Avx512,
}

impl MatchExtension {
    /// Reject unavailable experiments before starting an encode.
    pub fn validate(self) -> Result<()> {
        if self == Self::Legacy {
            return Ok(());
        }
        #[cfg(feature = "experimental-lz77")]
        {
            Extension::new(self).map(|_| ())
        }
        #[cfg(not(feature = "experimental-lz77"))]
        {
            Err(Error::InvalidInput(
                "match extension requires the experimental-lz77 Cargo feature".into(),
            ))
        }
    }
}

#[cfg(any(feature = "experimental-lz77", test))]
#[derive(Clone, Copy, Debug)]
enum Backend {
    Scalar,
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    Avx2,
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    Avx512,
}

/// Private backend state is the capability proof for all subsequent calls.
#[cfg(any(feature = "experimental-lz77", test))]
#[derive(Clone, Copy, Debug)]
pub(crate) struct Extension(Backend);

#[cfg(any(feature = "experimental-lz77", test))]
impl Extension {
    pub(crate) fn new(mode: MatchExtension) -> Result<Option<Self>> {
        match mode {
            MatchExtension::Legacy => Ok(None),
            MatchExtension::Scalar => Ok(Some(Self(Backend::Scalar))),
            MatchExtension::Avx2 => {
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                if std::is_x86_feature_detected!("avx2") {
                    return Ok(Some(Self(Backend::Avx2)));
                }
                Err(Error::InvalidInput(
                    "AVX2 match extension is not supported on this CPU".into(),
                ))
            }
            MatchExtension::Avx512 => {
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                if std::is_x86_feature_detected!("avx512f")
                    && std::is_x86_feature_detected!("avx512bw")
                {
                    return Ok(Some(Self(Backend::Avx512)));
                }
                Err(Error::InvalidInput(
                    "AVX-512F/BW match extension is not supported on this CPU".into(),
                ))
            }
        }
    }

    #[inline]
    pub(crate) fn length_u32(self, a: &[u32], b: &[u32]) -> usize {
        assert_eq!(a.len(), b.len(), "match-extension lengths must be equal");
        match self.0 {
            Backend::Scalar => scalar_u32(a, b),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Backend::Avx2 => {
                // SAFETY: construction proved AVX2; lengths were checked above.
                unsafe { x86::length_u32_avx2(a, b) }
            }
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Backend::Avx512 => {
                // SAFETY: construction proved AVX-512F/BW; lengths are equal.
                unsafe { x86::length_u32_avx512(a, b) }
            }
        }
    }

    #[cfg(test)]
    #[inline]
    pub(crate) fn length_u16(self, a: &[u16], b: &[u16]) -> usize {
        assert_eq!(a.len(), b.len(), "match-extension lengths must be equal");
        match self.0 {
            Backend::Scalar => scalar_u16(a, b),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Backend::Avx2 => {
                // SAFETY: construction proved AVX2; lengths were checked above.
                unsafe { x86::length_u16_avx2(a, b) }
            }
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Backend::Avx512 => {
                // SAFETY: construction proved AVX-512F/BW; lengths are equal.
                unsafe { x86::length_u16_avx512(a, b) }
            }
        }
    }

    #[inline]
    pub(crate) fn lengths4_u32(self, a: &[u32], b: [&[u32]; 4]) -> [usize; 4] {
        assert!(
            b.iter().all(|candidate| candidate.len() == a.len()),
            "match-extension lengths must be equal"
        );
        match self.0 {
            Backend::Scalar => b.map(|candidate| scalar_u32(a, candidate)),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Backend::Avx2 => {
                // SAFETY: construction proved AVX2; all lengths are equal.
                unsafe { x86::lengths4_u32_avx2(a, b) }
            }
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Backend::Avx512 => {
                // SAFETY: construction proved AVX-512F/BW; all lengths are equal.
                unsafe { x86::lengths4_u32_avx512(a, b) }
            }
        }
    }
}

#[cfg(any(feature = "experimental-lz77", test))]
#[inline]
fn scalar_u32(a: &[u32], b: &[u32]) -> usize {
    let mut length = 0;
    while length < a.len() && a[length] == b[length] {
        length += 1;
    }
    length
}

#[cfg(test)]
#[inline]
fn scalar_u16(a: &[u16], b: &[u16]) -> usize {
    let mut length = 0;
    while length < a.len() && a[length] == b[length] {
        length += 1;
    }
    length
}
