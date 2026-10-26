// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::frame::native_integer::NativeIntegerCapture;
use crate::util::sync::Arc;

/// Original integer samples, with no normalization or output color conversion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JxlNativeIntegerPlane {
    pub bits_per_sample: u32,
    /// Row-major samples, all in `0..(1 << bits_per_sample)`.
    pub samples: Vec<u32>,
}

/// Exact full-resolution original-color Modular output. Color planes precede
/// the extra planes, which remain in codestream order (including alpha).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JxlNativeIntegerFrame {
    pub width: usize,
    pub height: usize,
    pub num_color_channels: usize,
    pub planes: Vec<JxlNativeIntegerPlane>,
}

/// Exact integer decoder output, independent of floating-point pixel rendering.
/// Enable before input or at `BasicInfo`. Read after each `FrameComplete` event.
/// The next frame replaces unread output; hidden references are retained
/// separately. `coalescing` and `adjust_orientation` are respected.
///
/// Supports original-color integer Modular (1..31 bits), full-resolution extra
/// planes, reversible transforms, integer patches and crop/reference composition.
/// Unsupported fractional rendering returns `NativeIntegerUnsupported`, never
/// an approximation. This mode does not populate ordinary pixel buffers.
#[derive(Clone)]
pub struct JxlNativeIntegerCapture {
    pub(crate) inner: Arc<NativeIntegerCapture>,
}

impl JxlNativeIntegerCapture {
    /// Takes the latest completed frame. No partial frame is ever returned.
    /// Await `Complete` as well if a valid whole container is required.
    pub fn take(&self) -> Option<JxlNativeIntegerFrame> {
        self.inner.take()
    }
}
