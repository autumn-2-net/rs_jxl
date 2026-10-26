//! PNG and JPEG pixel input shared by the CLI and bindings.

mod jpeg;
mod png;

use rs_jxl::Image;

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

pub struct Decoded {
    pub image: Image,
    pub jpeg: bool,
}

pub fn decode(bytes: &[u8], strip_metadata: bool) -> Result<Decoded> {
    if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        Ok(Decoded {
            image: png::decode(bytes, strip_metadata)?,
            jpeg: false,
        })
    } else if bytes.starts_with(&[0xff, 0xd8]) {
        Ok(Decoded {
            image: jpeg::decode(bytes, strip_metadata)?,
            jpeg: true,
        })
    } else {
        Err("unsupported input: expected a PNG or JPEG signature".into())
    }
}

fn checked_size(
    width: u32,
    height: u32,
    channels: usize,
    bytes_per_sample: usize,
) -> Result<usize> {
    if width == 0 || height == 0 {
        return Err("image dimensions must be nonzero".into());
    }
    let pixels = u64::from(width) * u64::from(height);
    let size = usize::try_from(pixels)?
        .checked_mul(channels)
        .and_then(|n| n.checked_mul(bytes_per_sample))
        .filter(|&n| n <= isize::MAX as usize)
        .ok_or("decoded image size exceeds addressable allocation size")?;
    Ok(size)
}

fn zeroed_bytes(size: usize) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(size)?;
    bytes.resize(size, 0);
    Ok(bytes)
}

fn metadata(kind: &str, strip: bool) -> Result<()> {
    if strip {
        Ok(())
    } else {
        Err(format!(
            "{kind} metadata cannot be preserved; use --strip-metadata to explicitly \
             discard it without color conversion or EXIF orientation transforms"
        )
        .into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_unknown_or_empty_input() {
        for bytes in [&b""[..], &b"not a PNG"[..], &b"\xff\x0a"[..]] {
            assert!(decode(bytes, false).is_err());
        }
    }

    #[test]
    fn checks_dimensions_and_products_before_allocation() {
        assert_eq!(checked_size(2, 3, 4, 2).unwrap(), 48);
        assert!(checked_size(0, 3, 1, 1).is_err());
        assert!(checked_size(u32::MAX, u32::MAX, 4, 2).is_err());
        assert_eq!(checked_size(16_385, 1, 1, 1).unwrap(), 16_385);
        assert_eq!(checked_size(16_384, 16_384, 1, 1).unwrap(), 268_435_456);
        assert_eq!(checked_size(16_384, 16_384, 4, 1).unwrap(), 1_073_741_824);
        assert!(checked_size(8192, 8192, usize::MAX, 2).is_err());
        assert!(zeroed_bytes(usize::MAX).is_err());
    }
}
