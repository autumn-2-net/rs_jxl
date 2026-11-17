use super::{
    pixel_model::{self, Plane},
    transform,
};
use crate::arith::{Decoder, Encoder};
fn u32s(v: &mut Vec<u8>, x: impl IntoIterator<Item = u32>) {
    for n in x {
        v.extend_from_slice(&n.to_le_bytes());
    }
}
#[test]
#[ignore = "requires RS_JXL_PLUS_PIXEL_ORACLE compiled original C"]
fn original_c_pixels_and_transforms() {
    let exe = std::env::var_os("RS_JXL_PLUS_PIXEL_ORACLE").expect("pixel oracle path");
    let temp = std::env::temp_dir().canonicalize().unwrap();
    let dir = temp.join(format!(
        "pja-pixel-oracle-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir(&dir).unwrap();
    let input = dir.join("input");
    let output = dir.join("output");
    let run = |mode: &str, bytes: &[u8]| {
        std::fs::write(&input, bytes).unwrap();
        let status = std::process::Command::new(&exe)
            .args([
                std::ffi::OsStr::new(mode),
                input.as_os_str(),
                output.as_os_str(),
            ])
            .status()
            .unwrap();
        assert!(status.success(), "{mode}");
        std::fs::read(&output).unwrap()
    };
    for hb in [12, 16, 22] {
        for depths in [
            vec![1, 1],
            vec![2],
            vec![8, 8, 8],
            vec![12, 16],
            vec![1, 8, 16],
        ] {
            let original: Vec<_> = depths
                .iter()
                .enumerate()
                .map(|(c, &depth)| Plane {
                    width: 41,
                    height: 23,
                    depth,
                    samples: (0..41 * 23)
                        .map(|i| ((i * 31 + (i / 17) * 19 + c * 37) % (1usize << depth)) as u16)
                        .collect(),
                })
                .collect();
            let mut desc = Vec::new();
            u32s(&mut desc, [hb, original.len() as u32]);
            for p in &original {
                u32s(&mut desc, [p.width as u32, p.height as u32, p.depth as u32]);
                for v in &p.samples {
                    desc.extend_from_slice(&v.to_le_bytes());
                }
            }
            let expected = run("model", &desc);
            let mut planes = original.clone();
            let mut enc = Encoder::new(1 << 20);
            pixel_model::code(&mut enc, &mut planes, hb, 1 << 20).unwrap();
            assert_eq!(enc.finish().unwrap(), expected, "hb{hb} depths{depths:?}");
            for p in &mut planes {
                p.samples.fill(0);
            }
            pixel_model::code(&mut Decoder::new(&expected), &mut planes, hb, 1 << 20).unwrap();
            for (a, b) in planes.iter().zip(original) {
                assert_eq!(a.samples, b.samples);
            }
        }
    }
    for (w, h, x, y) in [
        (1, 1, 1, 1),
        (1, 9, 0, 3),
        (8, 1, 3, 0),
        (9, 7, 3, 1),
        (32, 24, 0, 0),
    ] {
        let original: Vec<i32> = (0..w * h).map(|i| (i * 31 % 91) as i32 - 45).collect();
        let mut actual = original.clone();
        for fwd in [true, false] {
            let mut desc = Vec::new();
            u32s(&mut desc, [w, h, x, y, 3, u32::from(fwd)]);
            for v in &actual {
                desc.extend_from_slice(&v.to_le_bytes());
            }
            let expected = run("dwt", &desc);
            transform::dwt(
                &mut actual,
                w as usize,
                w as usize,
                h as usize,
                x as i64,
                y as i64,
                3,
                fwd,
            )
            .unwrap();
            let bytes: Vec<_> = actual.iter().flat_map(|v| v.to_le_bytes()).collect();
            assert_eq!(bytes, expected);
        }
        assert_eq!(actual, original);
    }
    let resolved = dir.canonicalize().unwrap();
    assert!(resolved.starts_with(&temp) && resolved != temp);
    std::fs::remove_dir_all(resolved).unwrap();
}
