//! Development-only equivalence against the original PJA C v7 model, MIT,
//! revision 1058f63fbfb9c60cc6640b7bb130d6f7c7ea7dbd. See LICENSE-PJA
//! and NOTICE-PJA. The PJA7REF1 descriptor is specified in
//! tools/pja_v7_reference.c. Prediction tables are shared from Rust;
//! probability tables, arithmetic and coefficient decisions run independently.
use crate::{
    model::{self, Component},
    predict::tables,
};
use std::{
    path::{Path, PathBuf},
    process::Command,
};

fn u32le(out: &mut Vec<u8>, v: usize) {
    out.extend_from_slice(&u32::try_from(v).unwrap().to_le_bytes());
}
fn bytes(out: &mut Vec<u8>, data: &[u8]) {
    u32le(out, data.len());
    out.extend_from_slice(data);
}
fn longs(out: &mut Vec<u8>, data: &[i64]) {
    u32le(out, data.len());
    for v in data {
        out.extend_from_slice(&v.to_le_bytes());
    }
}
fn zigzag_bytes(components: &[Component]) -> Vec<u8> {
    let t = tables();
    let mut out = Vec::new();
    for c in components {
        for block in c.coefficients.chunks_exact(64) {
            for &p in &t.zz_table {
                out.extend_from_slice(&block[p as usize].to_le_bytes());
            }
        }
    }
    out
}
fn descriptor(components: &[Component]) -> Vec<u8> {
    let t = tables();
    let mut out = b"PJA7REF1".to_vec();
    for a in [
        &t.interior[..],
        &t.interior_left,
        &t.ib_na,
        &t.ib_nb,
        &t.ib_wa,
        &t.ib_wb,
        &t.ib_none,
    ] {
        bytes(&mut out, a);
    }
    // Upstream RATIO_OF[1..49,0..48] omits Rust's unused denominator-zero row.
    let ratios: Vec<u8> = t.ratio_of[1..].iter().flatten().copied().collect();
    bytes(&mut out, &ratios);
    for a in [
        &t.unzig[..],
        &t.zz_table,
        &t.nz_table,
        &t.rem_table,
        &t.k_bucket,
    ] {
        bytes(&mut out, a);
    }
    longs(&mut out, &t.w3);
    longs(&mut out, &t.e3);
    for offset in [0, 8, 16, 24] {
        let values: Vec<i64> =
            t.k4.iter()
                .flat_map(|k| k[offset..offset + 8].iter().map(|&v| v as i32 as i64))
                .collect();
        longs(&mut out, &values);
    }
    out.extend_from_slice(&t.interior_zz.to_le_bytes());
    out.extend_from_slice(&t.edge_zz.to_le_bytes());
    u32le(&mut out, components.len());
    for c in components {
        for v in [
            c.width,
            c.height,
            components[0].h_samp / c.h_samp,
            components[0].v_samp / c.v_samp,
        ] {
            u32le(&mut out, v);
        }
        for &p in &t.zz_table {
            out.extend_from_slice(&c.quant[p as usize].to_le_bytes());
        }
        out.extend_from_slice(&zigzag_bytes(std::slice::from_ref(c)));
    }
    out
}
fn run(exe: &Path, arguments: &[&Path], mode: &str) {
    let result = Command::new(exe)
        .arg(mode)
        .args(arguments)
        .output()
        .expect("launch original PJA oracle");
    assert!(
        result.status.success(),
        "original C {mode} failed ({}): {}",
        result.status,
        String::from_utf8_lossy(&result.stderr)
    );
}
fn fixtures() -> Vec<(String, Vec<Component>)> {
    let mut cases = Vec::new();
    for (sampling, mcu_w, mcu_h) in [
        (vec![(1, 1)], 1, 1),
        (vec![(1, 1)], 3, 3),
        (vec![(1, 1), (1, 1), (1, 1)], 3, 2),
        (vec![(2, 2), (1, 1), (1, 1)], 2, 2),
        (vec![(2, 1), (1, 1), (1, 1)], 2, 3),
    ] {
        for pattern in 0..4 {
            let mut components = Vec::new();
            for (ci, &(h, v)) in sampling.iter().enumerate() {
                let width = mcu_w * h;
                let height = mcu_h * v;
                let mut coefficients = vec![0i16; width * height * 64];
                let mut state = 0x327a88u32 + ci as u32 * 71 + pattern * 8191;
                for (bi, block) in coefficients.chunks_exact_mut(64).enumerate() {
                    for (p, x) in block.iter_mut().enumerate() {
                        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                        *x = match pattern {
                            0 => 0,
                            1 => {
                                if p == 0 || p % 8 == 0 || p < 8 {
                                    ((state >> 16) % 127) as i16 - 63
                                } else {
                                    0
                                }
                            }
                            2 => {
                                if state.is_multiple_of(13) {
                                    ((state >> 16) % 511) as i16 - 255
                                } else {
                                    0
                                }
                            }
                            _ => ((state >> 16) % 2047) as i16 - 1023,
                        };
                    }
                    if pattern == 3 {
                        block[(bi * 7 + ci) % 64] = if bi % 2 == 0 { 32767 } else { -32767 };
                    }
                }
                let mut quant = [1u16; 64];
                for (p, q) in quant.iter_mut().enumerate() {
                    *q = match pattern {
                        0 => 1,
                        1 => (p % 255 + 1) as u16,
                        2 => {
                            if p % 7 == 0 {
                                65535
                            } else {
                                257
                            }
                        }
                        _ => {
                            if (p + ci) % 3 == 0 {
                                65535
                            } else {
                                1
                            }
                        }
                    };
                }
                components.push(Component {
                    width,
                    height,
                    h_samp: h,
                    v_samp: v,
                    quant,
                    coefficients,
                });
            }
            cases.push((
                format!("{}c-{}x{}-p{}", sampling.len(), mcu_w, mcu_h, pattern),
                components,
            ));
        }
    }
    cases
}

#[test]
#[ignore = "requires independently built original-C executable in RS_JXL_PLUS_PJA_ORACLE"]
fn original_c_v7_equivalence() {
    let exe = PathBuf::from(
        std::env::var_os("RS_JXL_PLUS_PJA_ORACLE")
            .expect("set RS_JXL_PLUS_PJA_ORACLE to tools/pja_v7_reference.c executable"),
    );
    let exe = exe.canonicalize().expect("original-C executable path");
    let unique = format!(
        "rs-jxl-plus-pja-oracle-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );
    let dir = std::env::temp_dir().join(unique);
    std::fs::create_dir_all(&dir).unwrap();
    // Keep failing artifacts for diagnosis; remove only this test's unique directory on success.
    for (name, original) in fixtures() {
        let desc = dir.join(format!("{name}.descriptor"));
        let c_encoded = dir.join(format!("{name}.c-stream"));
        let rust_encoded = dir.join(format!("{name}.rust-stream"));
        let c_decoded = dir.join(format!("{name}.c-coefficients"));
        std::fs::write(&desc, descriptor(&original)).unwrap();
        let rust = model::encode(&original, 16 << 20).unwrap();
        assert_eq!(rust.len(), 1, "fixture must fit one band");
        std::fs::write(&rust_encoded, &rust[0]).unwrap();
        run(&exe, &[&desc, &c_encoded], "encode");
        let c = std::fs::read(&c_encoded).unwrap();
        assert_eq!(
            c,
            rust[0],
            "original C vs Rust arithmetic bytes: {name}; artifacts {}",
            dir.display()
        );
        run(&exe, &[&desc, &rust_encoded, &c_decoded], "decode");
        assert_eq!(
            std::fs::read(&c_decoded).unwrap(),
            zigzag_bytes(&original),
            "original C decoding Rust bytes: {name}"
        );
        let mut decoded = original.clone();
        for c in &mut decoded {
            c.coefficients.fill(0);
        }
        model::decode(&mut decoded, &[c.as_slice()]).unwrap();
        assert_eq!(decoded, original, "Rust decoding original C bytes: {name}");
    }
    std::fs::remove_dir_all(&dir).unwrap();
}
