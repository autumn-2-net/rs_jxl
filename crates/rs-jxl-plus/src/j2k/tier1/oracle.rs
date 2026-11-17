//! Independent original-C development oracle. PJA source SHA and licenses
//! are recorded in ../tier1.rs and tools/pja_j2k_reference.c.
use std::{path::PathBuf, process::Command};
pub(crate) fn run(mode: &str, input: &[u8]) -> Vec<u8> {
    run_env("RS_JXL_PLUS_J2K_ORACLE", mode, input)
}
pub(crate) fn run_env(env: &str, mode: &str, input: &[u8]) -> Vec<u8> {
    let exe = std::env::var_os(env).unwrap_or_else(|| panic!("set {env}"));
    let dir = std::env::temp_dir().join(format!(
        "pja-j2k-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir(&dir).unwrap();
    let desc = dir.join("descriptor");
    let output = dir.join("output");
    std::fs::write(&desc, input).unwrap();
    let result = Command::new(PathBuf::from(exe))
        .arg(mode)
        .arg(&desc)
        .arg(&output)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "C oracle {mode}: {}; artifacts {}",
        String::from_utf8_lossy(&result.stderr),
        dir.display()
    );
    let out = std::fs::read(&output).unwrap();
    std::fs::remove_dir_all(dir).unwrap();
    out
}
pub(crate) fn word(out: &mut Vec<u8>, n: usize) {
    out.extend_from_slice(&(n as u32).to_le_bytes());
}
#[test]
#[ignore = "requires RS_JXL_PLUS_J2K_ORACLE original-C helper"]
fn original_c_tier1_and_mq() {
    use super::*;
    let bits: Vec<u8> = (0..4096).map(|i| (i * 37 + i / 3) as u8).collect();
    for style in [0, 1, 2, 3, 8, 9, 32, 35] {
        for ori in 0..4 {
            let b = Block {
                width: 5,
                height: 7,
                orientation: ori,
                style,
                passes: 15,
            };
            let r = replay(b, &bits, 3, bits.len() * 8, 8192).unwrap();
            let mut desc = Vec::new();
            for n in [
                5,
                7,
                ori,
                style as usize,
                15,
                3,
                bits.len() * 8,
                8192,
                bits.len(),
            ] {
                word(&mut desc, n);
            }
            desc.extend(&bits);
            let mut expected = Vec::new();
            word(&mut expected, r.symbols.len());
            word(&mut expected, r.used_bits);
            expected.extend(&r.symbols);
            for &n in &r.pass_ends {
                word(&mut expected, n);
            }
            assert_eq!(run("tier1", &desc), expected, "style{style} ori{ori}");
        }
    }
    for reset in [false, true] {
        let symbols: Vec<u8> = (0..500)
            .map(|i| (((i * 7 % 19) << 1) | ((i * 17 + i / 5) % 2)) as u8)
            .collect();
        let pe = [0, 17, 100, 499];
        let ctx = reset_contexts();
        let e = mq_encode(&symbols, &pe, reset, ctx).unwrap();
        let mut desc = Vec::new();
        for n in [symbols.len(), pe.len(), reset as usize] {
            word(&mut desc, n)
        }
        desc.extend(ctx);
        desc.extend(&symbols);
        for n in pe {
            word(&mut desc, n)
        }
        let mut expected = Vec::new();
        for n in [e.flush.len(), e.erterm.len(), e.committed] {
            word(&mut expected, n)
        }
        expected.extend(e.flush);
        expected.extend(e.erterm);
        for row in e.pass_lengths {
            for n in row {
                word(&mut expected, n)
            }
        }
        expected.extend(e.contexts);
        assert_eq!(run("mq", &desc), expected);
    }
}
pub(crate) fn bands_descriptor(
    bands: &[super::super::model::Band],
    maps: usize,
    blocks: usize,
) -> Vec<u8> {
    let mut desc = Vec::new();
    for n in [10, bands.len(), maps, blocks] {
        word(&mut desc, n)
    }
    for g in bands {
        for n in [
            g.comp,
            g.res,
            g.orient,
            g.nres,
            g.width as i32,
            g.height as i32,
            g.mb,
            g.parent.map_or(-1, |v| v as i32),
            g.cross0.map_or(-1, |v| v as i32),
            g.cross1.map_or(-1, |v| v as i32),
        ] {
            desc.extend(n.to_le_bytes())
        }
        desc.extend(g.map.map_or(-1, |v| v as i64).to_le_bytes());
    }
    desc
}
