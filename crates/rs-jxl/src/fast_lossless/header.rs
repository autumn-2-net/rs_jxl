// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// BSD-3-Clause; retained notice in docs/PORTING_KERNELS.md.
//! b866b3d enc_fast_lossless.cc: image, frame, TOC and fixed Modular headers.

use super::{FrameInfo, entropy::PrefixCode};
use crate::bit_writer::BitWriter;
use crate::{Error, Result};

fn fields(writer: &mut BitWriter, values: &[(u8, u64)]) -> Result<()> {
    for &(n, bits) in values {
        writer.write(n, bits)?;
    }
    Ok(())
}

fn size(writer: &mut BitWriter, value: usize) -> Result<()> {
    let value = value - 1;
    let bucket = if value < 1 << 9 {
        0
    } else if value < 1 << 13 {
        1
    } else if value < 1 << 18 {
        2
    } else {
        3
    };
    writer.write(2, bucket)?;
    writer.write([9, 13, 18, 30][bucket as usize], value as u64)?;
    Ok(())
}

fn depth(writer: &mut BitWriter, bits: u8) -> Result<()> {
    writer.write(1, 0)?;
    match bits {
        8 => writer.write(2, 0)?,
        10 => writer.write(2, 1)?,
        12 => writer.write(2, 2)?,
        _ => {
            writer.write(2, 3)?;
            writer.write(6, u64::from(bits - 1))?;
        }
    }
    Ok(())
}

pub(super) fn image(writer: &mut BitWriter, image: impl Into<FrameInfo>) -> Result<()> {
    let image = image.into();
    fields(writer, &[(16, 0x0aff), (1, 0)])?;
    size(writer, image.height)?;
    writer.write(3, 0)?;
    size(writer, image.width)?;
    fields(writer, &[(1, 0), (1, 0)])?;
    depth(writer, image.bit_depth)?;
    writer.write(1, u64::from(image.bit_depth <= 14))?;
    let alpha = image.channels == 2 || image.channels == 4;
    writer.write(2, u64::from(alpha))?;
    if alpha {
        writer.write(1, u64::from(image.bit_depth == 8))?;
        if image.bit_depth != 8 {
            writer.write(2, 0)?;
            depth(writer, image.bit_depth)?;
            fields(writer, &[(2, 0), (2, 0), (1, 0)])?;
        }
    }
    writer.write(1, 0)?;
    if image.channels >= 3 {
        writer.write(1, 1)?;
    } else {
        fields(
            writer,
            &[
                (1, 0),
                (1, 0),
                (2, 1),
                (2, 1),
                (1, 0),
                (2, 2),
                (4, 11),
                (2, 1),
            ],
        )?;
    }
    fields(writer, &[(2, 0), (1, 1)])?;
    writer.zero_pad_to_byte()?;
    Ok(())
}

pub(super) fn frame(
    writer: &mut BitWriter,
    channels: u8,
    is_last: bool,
    sizes: &[usize],
) -> Result<()> {
    fields(writer, &[(1, 0), (2, 0), (1, 1), (2, 0), (1, 0), (2, 0)])?;
    let alpha = channels == 2 || channels == 4;
    if alpha {
        writer.write(2, 0)?;
    }
    fields(writer, &[(2, 1), (2, 0), (1, 0), (2, 0)])?;
    if alpha {
        writer.write(2, 0)?;
    }
    writer.write(1, u64::from(is_last))?;
    if !is_last {
        writer.write(2, 0)?;
    }
    fields(
        writer,
        &[(2, 0), (1, 0), (1, 0), (2, 0), (2, 0), (2, 0), (1, 0)],
    )?;
    writer.zero_pad_to_byte()?;
    for &size in sizes {
        let bucket = if size < 1024 {
            0
        } else if size < 17408 {
            1
        } else if size < 4211712 {
            2
        } else {
            3
        };
        let value = size - [0, 1024, 17408, 4211712][bucket];
        let bits = [10, 14, 22, 30][bucket];
        if value as u64 >= 1u64 << bits {
            return Err(Error::Limit("FJXL group size"));
        }
        writer.write(2, bucket as u64)?;
        writer.write(bits, value as u64)?;
    }
    writer.zero_pad_to_byte()?;
    Ok(())
}

pub(super) fn global_common(writer: &mut BitWriter, codes: &[PrefixCode; 4]) -> Result<()> {
    fields(
        writer,
        &[
            (1, 1),
            (1, 1),
            (1, 0),
            (1, 1),
            (2, 0),
            (1, 1),
            (4, 0),
            (6, 0b100011),
            (2, 1),
            (2, 3),
            (2, 0),
            (2, 1),
            (2, 2),
            (2, 3),
            (1, 0),
        ],
    )?;
    let tree = [
        1, 2, 1, 4, 1, 0, 0, 5, 0, 0, 0, 0, 5, 0, 0, 0, 0, 5, 0, 0, 0, 0, 5, 0, 0, 0,
    ];
    for symbol in tree {
        writer.write([2, 2, 3, 3, 4, 4][symbol], [0, 2, 1, 5, 3, 7][symbol])?;
    }
    fields(
        writer,
        &[
            (1, 1),
            (2, 0),
            (4, 10),
            (4, 4),
            (3, 0),
            (3, 0),
            (1, 1),
            (2, 3),
            (3, 4),
            (3, 3),
            (3, 2),
            (3, 1),
            (3, 0),
            (1, 1),
        ],
    )?;
    for _ in 0..5 {
        writer.write(4, 0)?;
    }
    writer.write(5, 1)?;
    for _ in 0..4 {
        // C++ Write(8, 256) spills its high bit onto the next mandatory one.
        // The bounded writer emits the equivalent low eight bits explicitly.
        fields(writer, &[(1, 1), (4, 8), (8, 0)])?;
    }
    fields(writer, &[(2, 1), (2, 0), (1, 1)])?;
    for code in codes {
        code.write(writer)?;
    }
    fields(writer, &[(1, 1), (1, 1)])?;
    Ok(())
}

pub(super) fn transforms(
    writer: &mut BitWriter,
    channels: u8,
    palette_size: Option<usize>,
) -> Result<()> {
    if let Some(colors) = palette_size {
        fields(writer, &[(2, 1), (2, 1), (5, 0)])?;
        match channels {
            1 => writer.write(2, 0)?,
            3 => writer.write(2, 1)?,
            4 => writer.write(2, 2)?,
            _ => {
                writer.write(2, 3)?;
                writer.write(13, u64::from(channels - 1))?;
            }
        }
        if colors < 256 {
            fields(writer, &[(2, 0), (8, colors as u64)])?;
        } else {
            fields(writer, &[(2, 1), (10, (colors - 256) as u64)])?;
        }
        fields(writer, &[(2, 0), (4, 0)])?;
    } else if channels >= 3 {
        fields(writer, &[(2, 1), (2, 0), (5, 0), (2, 0)])?;
    } else {
        writer.write(2, 0)?;
    }
    Ok(())
}
