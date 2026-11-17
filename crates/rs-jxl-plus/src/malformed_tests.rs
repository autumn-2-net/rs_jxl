//! Small deterministic adversarial checks for the native coefficient model.
//! These exercise bounded decisions on arbitrary arithmetic bytes; successful
//! decoding alone does not establish container integrity or JPEG validity.
use crate::model::{self, Component};

fn components(chroma: bool, quantum: u16) -> Vec<Component> {
    let sampling: &[(usize, usize)] = if chroma {
        &[(2, 2), (1, 1), (1, 1)]
    } else {
        &[(1, 1)]
    };
    sampling
        .iter()
        .map(|&(h, v)| Component {
            // Two MCU columns exercise both missing and present neighbours.
            width: 2 * h,
            height: v,
            h_samp: h,
            v_samp: v,
            quant: [quantum; 64],
            coefficients: vec![0; 2 * h * v * 64],
        })
        .collect()
}

#[test]
fn arbitrary_short_arithmetic_streams_do_not_panic() {
    // 2 shapes * 2 quantum extremes * 24 streams = 96 bounded cases.
    // Empty and 1..3-byte streams deliberately enter the original decoder's
    // end-zero-padding behavior, beyond the container's four-byte minimum.
    for chroma in [false, true] {
        for quantum in [1, 65535] {
            for case in 0..24usize {
                let mut bytes = vec![0; case % 17];
                let mut state = 0x93ac_715du32 ^ case as u32;
                for (i, byte) in bytes.iter_mut().enumerate() {
                    state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                    *byte = match case / 6 {
                        0 => 0,
                        1 => 255,
                        2 => {
                            if i % 2 == 0 {
                                0x55
                            } else {
                                0xaa
                            }
                        }
                        _ => (state >> 24) as u8,
                    };
                }
                let mut frame = components(chroma, quantum);
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    model::decode(&mut frame, &[bytes.as_slice()])
                }));
                assert!(
                    result.is_ok(),
                    "model panic: 420={chroma}, q={quantum}, case={case}, bytes={bytes:02x?}"
                );
                // Rejection is expected for some malformed decisions. A
                // successful result need only preserve the validated storage.
                if result.unwrap().is_ok() {
                    for component in frame {
                        assert_eq!(
                            component.coefficients.len(),
                            component.width * component.height * 64
                        );
                    }
                }
            }
        }
    }
}
