//! Scalar skcms transfer math used by the direct djxl compatibility paths.
//! Sources: local libjxl third_party/skcms/src/Transform_inl.h (approx_pow,
//! apply_tf), skcms.cc (skcms_TransferFunction_invert), and
//! lib/jxl/cms/jxl_cms_internal.h (sRGB ICC parameters).
//!
//! This module does not parse ICC, clamp device colors, or approximate arbitrary
//! destination curves. Its F32 boundaries deliberately match skcms, not powf.

// Copyright (c) 2018 Google Inc. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the name of Google Inc. nor the names of its contributors may be
//   used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

pub(super) struct Transfer {
    g: f32,
    a: f32,
    b: f32,
    c: f32,
    d: f32,
    e: f32,
}

impl Transfer {
    /// JXL gamma is the encoding exponent; ICC type 0 stores its reciprocal.
    pub(super) fn gamma(gamma: f32) -> Self {
        Self {
            g: ((1.0 / gamma) * 65536.0).round() / 65536.0,
            a: 1.0,
            b: 0.0,
            c: 0.0,
            d: 0.0,
            e: 0.0,
        }
    }

    pub(super) fn srgb() -> Self {
        // S15Fixed16 values written by libjxl, not unquantized sRGB constants.
        Self {
            g: 157286.0 / 65536.0,
            a: 62119.0 / 65536.0,
            b: 3417.0 / 65536.0,
            c: 5072.0 / 65536.0,
            d: 2651.0 / 65536.0,
            e: 0.0,
        }
    }

    pub(super) fn inverse(self) -> Self {
        let a = approx_pow(self.a, -self.g);
        let g = 1.0 / self.g;
        // Here a+b=1 and e=f=0 exactly, so source(1)=1. skcms adjusts the
        // inverse's offset to preserve inverse(source(1))=1. Its scalar powf_
        // is the same approximation as the pixel stages, NOT system powf.
        Self {
            g,
            a,
            b: 0.0,
            c: 1.0 / self.c,
            d: self.c * self.d,
            e: 1.0 - approx_pow(a, g),
        }
    }

    pub(super) fn apply(&self, value: f32) -> f32 {
        if !value.is_finite() {
            return value;
        }
        let x = value.abs();
        let magnitude = if x < self.d {
            self.c * x
        } else {
            approx_pow(self.a * x + self.b, self.g) + self.e
        };
        magnitude.copysign(value)
    }
}

#[allow(clippy::excessive_precision)] // Preserve the upstream F32 literals.
fn approx_pow(x: f32, exponent: f32) -> f32 {
    if x == 0.0 || x == 1.0 || !x.is_finite() {
        return x;
    }
    let bits = x.to_bits();
    let e = bits as f32 * (1.0 / (1u32 << 23) as f32);
    let m = f32::from_bits((bits & 0x007f_ffff) | 0x3f00_0000);
    let log = e - 124.225514990 - 1.498030302 * m - 1.725879990 / (0.3520887068 + m);
    let power = log * exponent;
    let fract = power - power.floor();
    let fbits = (1u32 << 23) as f32
        * (power + 121.274057500 - 1.490129070 * fract + 27.728023300 / (4.84252568 - fract));
    // This bounds the exponent's bit representation, not RGB/PCS coordinates.
    f32::from_bits(fbits.clamp(0.0, 0x7f80_0000u32 as f32) as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn srgb_inverse_coefficients_use_skcms_scalar_pow() {
        let inverse = Transfer::srgb().inverse();
        assert_eq!(inverse.a, 1.137_306_2);
        assert_eq!(inverse.e, -0.054_985_046);
        // Source-derived scalar reference words. Using system powf for the
        // coefficients produces a different curve even with approx_pow pixels.
        for (input, expected_bits) in [
            (0.01, 0x3dcc_8b00),
            (0.1, 0x3eb2_cb40),
            (0.5, 0x3f3c_4200),
            (1.0, 0x3f80_0000),
            (2.0, 0x3fad_3980),
        ] {
            assert_eq!(inverse.apply(input).to_bits(), expected_bits);
            assert_eq!(inverse.apply(-input).to_bits(), expected_bits | 0x8000_0000);
        }
    }

    #[test]
    fn jxl_gamma_is_inverted_and_quantized_before_approximation() {
        let gamma = Transfer::gamma(1.0 / 2.2);
        assert_eq!(gamma.g, 144179.0 / 65536.0);
        assert_eq!(gamma.apply(128.0 / 255.0), 0.219_507_22);
        assert_eq!(gamma.apply(1.0), 1.0);
        assert_eq!(gamma.apply(-0.0).to_bits(), (-0.0f32).to_bits());
        assert!(gamma.apply(2.0) > 4.0);
        let dci = Transfer::gamma(1.0 / 2.6);
        assert_eq!(dci.g, 170394.0 / 65536.0);
    }
}
