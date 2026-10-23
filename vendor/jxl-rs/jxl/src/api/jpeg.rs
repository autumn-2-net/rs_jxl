// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::frame::jpeg::JpegCapture;
use crate::util::sync::Arc;

/// Quantized JPEG data recovered from a JPEG XL coefficient-transcode frame.
/// Marker/scan metadata from the `jbrd` box is recovered separately.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JxlJpegCoefficients {
    pub width: u32,
    pub height: u32,
    pub is_ycbcr: bool,
    /// JPEG component order: Y/Cb/Cr, R/G/B, or the sole grayscale component.
    pub components: Vec<JxlJpegComponent>,
}

/// One JPEG component, including MCU padding blocks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JxlJpegComponent {
    /// Source JXL channel (0 = X/Cb/R, 1 = Y/G/gray, 2 = B/Cr).
    pub jxl_channel: usize,
    /// Relative sampling factors; original marker factors belong to `jbrd`.
    pub horizontal_sampling_factor: u8,
    pub vertical_sampling_factor: u8,
    pub width_in_blocks: usize,
    pub height_in_blocks: usize,
    /// Natural JPEG order, not zigzag or JXL's transposed order.
    /// Table identifiers and duplicate tables belong to `jbrd`.
    pub quantization: [u16; 64],
    /// Row-major blocks of 64 natural-order quantized coefficients, DC first.
    pub coefficients: Vec<i16>,
}

/// Opt-in capture that survives the decoder releasing its last frame.
/// Obtain immediately after `Event::FrameHeader`, then continue decoding.
#[derive(Clone)]
pub struct JxlJpegCoefficientCapture {
    pub(crate) inner: Arc<JpegCapture>,
}

impl JxlJpegCoefficientCapture {
    /// Takes coefficients once, only after a successful `FrameComplete` event.
    /// Returns `None` while incomplete, after an error, or after taking them.
    pub fn take(&self) -> Option<JxlJpegCoefficients> {
        self.inner.take()
    }
}
