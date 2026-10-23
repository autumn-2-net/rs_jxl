// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// BSD-3-Clause; retained notice in docs/PORTING_KERNELS.md.

use super::*;
use schema::*;
use std::cell::Cell;
use std::io;
use std::panic::{AssertUnwindSafe, catch_unwind};

fn parameters_fixture() -> FrameParameters {
    FrameParameters::from_values(&[2, 1, 6, 0, -1, 1024, 6, 3, 1, -1, 0, -1], -1).unwrap()
}

fn shape_fixture() -> InputShape {
    InputShape {
        kind: InputKind::Pixels,
        width: 32,
        height: 17,
        channels: 4,
    }
}

const GOLDEN: &[u8] = b"JXL_FINAL_PLAN_V1 4\nframe 12 2 1 6 0 -1 1024 6 3 1 -1 0 -1\ninput 4 1 32 17 4\nopaque 3 -9223372036854775808 0 9223372036854775807\nempty 0\n";

#[test]
fn v1_writer_matches_source_golden_and_replay_exchanges_in_order() {
    let mut capture = FinalPlan::capture(Limits::default()).unwrap();
    capture.parameters(&mut parameters_fixture()).unwrap();
    capture.input(shape_fixture()).unwrap();
    capture
        .exchange("opaque", &mut vec![i64::MIN, 0, i64::MAX])
        .unwrap();
    capture.exchange("empty", &mut vec![]).unwrap();
    assert_eq!(capture.cursor(), 0);
    assert_eq!(capture.tree_limit(), 1160);
    capture.finished().unwrap();
    let mut bytes = Vec::new();
    capture.write(&mut bytes).unwrap();
    assert_eq!(bytes, GOLDEN);

    let mut replay = FinalPlan::read(bytes.as_slice(), Limits::default()).unwrap();
    assert_eq!(replay.mode(), Mode::Replay);
    assert_eq!(replay.tree_limit(), 0);
    assert!(replay.finished().is_err());
    let mut untouched = vec![123];
    assert!(replay.exchange("input", &mut untouched).is_err());
    assert_eq!(replay.cursor(), 0);
    assert_eq!(untouched, [123]);
    let mut parameters = parameters_fixture();
    parameters.speed_tier = 9;
    parameters.palette_colors = 0;
    replay.parameters(&mut parameters).unwrap();
    assert_eq!(parameters, parameters_fixture());
    replay.input(shape_fixture()).unwrap();
    replay.exchange("opaque", &mut untouched).unwrap();
    assert_eq!(untouched, [i64::MIN, 0, i64::MAX]);
    replay.exchange("empty", &mut untouched).unwrap();
    assert!(untouched.is_empty());
    replay.finished().unwrap();
    assert!(replay.exchange("empty", &mut untouched).is_err());
    assert_eq!(replay.cursor(), 4);
    let mut again = Vec::new();
    replay.write(&mut again).unwrap();
    assert_eq!(again, GOLDEN);
}

#[test]
fn reader_accepts_classic_whitespace_signs_zeros_and_byte_tags() {
    let input =
        b"\x0bJXL_FINAL_PLAN_V1\t+0002\r\n\xff\0\x0c3 +0001 -0 -9223372036854775808\rempty 0\t\x0b";
    let plan = FinalPlan::read(&input[..], Limits::default()).unwrap();
    assert_eq!(plan.records()[0].name(), b"\xff\0");
    assert_eq!(plan.records()[0].values(), [1, 0, i64::MIN]);
    let mut written = Vec::new();
    plan.write(&mut written).unwrap();
    assert_eq!(
        written,
        b"JXL_FINAL_PLAN_V1 2\n\xff\0 3 1 0 -9223372036854775808\nempty 0\n"
    );
    assert_eq!(
        FinalPlan::read(written.as_slice(), Limits::default())
            .unwrap()
            .records(),
        plan.records()
    );
    assert!(Record::new("", vec![]).is_err());
    assert!(Record::new("two words", vec![]).is_err());
    assert!(Record::new([b'a'; 65], vec![]).is_err());
    assert!(Record::new([b'a'; 64], vec![]).is_ok());
}

#[test]
fn malformed_truncated_and_trailing_data_are_rejected() {
    for input in [
        "",
        "JXL_FINAL_PLAN_V2 1 x 0",
        "JXL_FINAL_PLAN_V1 0",
        "JXL_FINAL_PLAN_V1 -1",
        "JXL_FINAL_PLAN_V1 1000001",
        "JXL_FINAL_PLAN_V1 1",
        "JXL_FINAL_PLAN_V1 1 x",
        "JXL_FINAL_PLAN_V1 1 x -0",
        "JXL_FINAL_PLAN_V1 1 x 16000001",
        "JXL_FINAL_PLAN_V1 1 x 1",
        "JXL_FINAL_PLAN_V1 2 x 0",
        "JXL_FINAL_PLAN_V1 1 x 1 +",
        "JXL_FINAL_PLAN_V1 1 x 1 -",
        "JXL_FINAL_PLAN_V1 1 x 1 9223372036854775808",
        "JXL_FINAL_PLAN_V1 1 x 1 -9223372036854775809",
        "JXL_FINAL_PLAN_V1 1 x 1 1.0",
        "JXL_FINAL_PLAN_V1 1 x 1 0x1",
        "JXL_FINAL_PLAN_V1 1 x 1 1junk",
        "JXL_FINAL_PLAN_V1 1 x 0 junk",
    ] {
        assert!(
            FinalPlan::read(input.as_bytes(), Limits::default()).is_err(),
            "{input}"
        );
    }
    let long_name = format!("JXL_FINAL_PLAN_V1 1 {} 0", "a".repeat(65));
    assert!(FinalPlan::read(long_name.as_bytes(), Limits::default()).is_err());
}

#[test]
fn budgets_are_checked_before_capture_mutation_or_output() {
    for limits in [
        Limits {
            records: 0,
            ..Limits::default()
        },
        Limits {
            records: MAX_RECORDS + 1,
            ..Limits::default()
        },
        Limits {
            values: MAX_VALUES + 1,
            ..Limits::default()
        },
        Limits {
            bytes: 0,
            ..Limits::default()
        },
        Limits {
            bytes: u64::MAX,
            ..Limits::default()
        },
    ] {
        assert!(FinalPlan::capture(limits).is_err());
    }
    let size = wire::header_size(1) + wire::record_size(b"x", &[1, 2]).unwrap();
    let limits = Limits {
        records: 1,
        values: 2,
        bytes: size,
    };
    let mut plan = FinalPlan::capture(limits).unwrap();
    let mut values = vec![1, 2];
    plan.exchange("x", &mut values).unwrap();
    assert!(plan.exchange("x", &mut values).is_err());
    assert_eq!(plan.records().len(), 1);
    assert_eq!(values, [1, 2]);
    let mut bytes = Vec::new();
    plan.write(&mut bytes).unwrap();
    assert_eq!(bytes.len() as u64, size);
    assert!(FinalPlan::read(bytes.as_slice(), limits).is_ok());
    assert!(
        FinalPlan::read(
            bytes.as_slice(),
            Limits {
                values: 1,
                ..limits
            }
        )
        .is_err()
    );
    assert!(
        FinalPlan::read(
            bytes.as_slice(),
            Limits {
                bytes: size - 1,
                ..limits
            }
        )
        .is_err()
    );
    bytes.push(b' ');
    assert!(FinalPlan::read(bytes.as_slice(), limits).is_err());
    let mut short = FinalPlan::capture(Limits {
        bytes: size - 1,
        ..limits
    })
    .unwrap();
    assert!(short.exchange("x", &mut values).is_err());
    assert!(short.records().is_empty());
    let mut empty_sink = Vec::new();
    assert!(short.write(&mut empty_sink).is_err());
    assert!(empty_sink.is_empty());
    let mut cumulative = FinalPlan::capture(Limits {
        records: 3,
        values: 2,
        bytes: 1024,
    })
    .unwrap();
    cumulative.exchange("a", &mut vec![1]).unwrap();
    cumulative.exchange("b", &mut vec![2]).unwrap();
    assert!(cumulative.exchange("c", &mut vec![3]).is_err());
    assert_eq!(cumulative.records().len(), 2);
}

#[test]
fn input_budget_bounds_reader_even_for_unending_whitespace() {
    struct Spaces(Rc<Cell<usize>>);
    impl Read for Spaces {
        fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
            output.fill(b' ');
            self.0.set(self.0.get() + output.len());
            Ok(output.len())
        }
    }
    let count = Rc::new(Cell::new(0));
    assert!(
        FinalPlan::read(
            Spaces(count.clone()),
            Limits {
                bytes: 80,
                ..Limits::default()
            }
        )
        .is_err()
    );
    assert_eq!(count.get(), 81);
}

#[test]
fn read_write_and_flush_errors_are_returned() {
    struct Failing;
    impl Read for Failing {
        fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
            Err(io::Error::other("read"))
        }
    }
    struct Sink {
        remaining: usize,
        fail_flush: bool,
    }
    impl Write for Sink {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            if self.remaining == 0 {
                return Err(io::Error::other("write"));
            }
            let n = bytes.len().min(self.remaining);
            self.remaining -= n;
            Ok(n)
        }
        fn flush(&mut self) -> io::Result<()> {
            if self.fail_flush {
                Err(io::Error::other("flush"))
            } else {
                Ok(())
            }
        }
    }
    assert!(matches!(
        FinalPlan::read(Failing, Limits::default()),
        Err(Error::Io(_))
    ));
    let plan = FinalPlan::read(GOLDEN, Limits::default()).unwrap();
    assert!(matches!(
        plan.write(Sink {
            remaining: 3,
            fail_flush: false
        }),
        Err(Error::Io(_))
    ));
    assert!(matches!(
        plan.write(Sink {
            remaining: usize::MAX,
            fail_flush: true
        }),
        Err(Error::Io(_))
    ));
}

#[test]
fn serial_tls_scopes_restore_nested_suspension_errors_and_panics() {
    assert!(current().is_none());
    let capture = Handle::from(FinalPlan::capture(Limits::default()).unwrap());
    let replay = Handle::from(
        FinalPlan::read(&b"JXL_FINAL_PLAN_V1 1 x 1 7\n"[..], Limits::default()).unwrap(),
    );
    let serial_before = crate::execution::is_serial();
    scope(Some(&capture), || {
        assert!(crate::execution::is_serial());
        assert_eq!(crate::execution::worker_count(), 1);
        assert!(!replaying());
        exchange("first", &mut vec![1]).unwrap();
        scope(None, || {
            assert!(current().is_none());
            exchange("not recorded", &mut vec![2]).unwrap();
            assert!(crate::execution::is_serial());
        });
        let error: Result<()> = scope(Some(&replay), || {
            assert!(replaying());
            let mut values = vec![];
            exchange("x", &mut values)?;
            assert_eq!(values, [7]);
            replay.finished()?;
            Err(Error::Encoding("test".into()))
        });
        assert!(error.is_err());
        assert!(!replaying());
        assert!(
            catch_unwind(AssertUnwindSafe(|| scope(Some(&replay), || panic!("test")))).is_err()
        );
        assert!(Rc::ptr_eq(&current().unwrap().plan, &capture.plan));
        assert!(
            std::thread::spawn(|| current().is_none() && !crate::execution::is_serial())
                .join()
                .unwrap()
        );
        exchange("last", &mut vec![3]).unwrap();
    });
    assert!(current().is_none());
    assert_eq!(crate::execution::is_serial(), serial_before);
    assert!(
        catch_unwind(AssertUnwindSafe(|| scope(Some(&capture), || panic!(
            "outer"
        ))))
        .is_err()
    );
    assert!(current().is_none());
    assert_eq!(crate::execution::is_serial(), serial_before);
    let mut bytes = Vec::new();
    capture.write(&mut bytes).unwrap();
    assert_eq!(bytes, b"JXL_FINAL_PLAN_V1 2\nfirst 1 1\nlast 1 3\n");
}

#[test]
fn reentrant_writer_gets_borrow_error_instead_of_panicking() {
    struct Callback(Handle);
    impl Write for Callback {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            assert!(matches!(
                self.0.exchange("nested", &mut vec![]),
                Err(Error::Encoding(_))
            ));
            Ok(bytes.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    let handle = Handle::from(FinalPlan::read(GOLDEN, Limits::default()).unwrap());
    handle.write(Callback(handle.clone())).unwrap();
    assert_eq!(handle.cursor().unwrap(), 0);
}

#[test]
fn unsupported_marker_is_separate_from_finished_and_not_serialized() {
    let mut plan = FinalPlan::capture(Limits::default()).unwrap();
    plan.exchange("patches", &mut vec![0]).unwrap();
    plan.mark_unsupported(Unsupported::SqueezeOrLossyPalette);
    plan.mark_unsupported(Unsupported::PatchReferenceCapture);
    assert_eq!(plan.unsupported(), Some(Unsupported::PatchReferenceCapture));
    assert!(plan.ensure_supported().is_err());
    plan.finished().unwrap();
    let mut bytes = Vec::new();
    plan.write(&mut bytes).unwrap();
    let read = FinalPlan::read(bytes.as_slice(), Limits::default()).unwrap();
    assert_eq!(read.unsupported(), None);
    assert_eq!(read.tree_limit(), 0);
}

#[test]
fn frame_parameters_validate_each_source_field_and_invisible_equality() {
    let fixture = parameters_fixture();
    let bounds = [
        (0, 9),
        (-1, 3),
        (-1, 41),
        (-1, 1),
        (-1, 1),
        (-65536, 70000),
        (-1, 15),
        (-1, 7),
        (0, 1),
        (-1, 0),
        (0, 1),
    ];
    for (i, (lo, hi)) in bounds.into_iter().enumerate() {
        for value in [lo, hi] {
            let mut values = fixture.to_values();
            values[i] = value;
            assert!(FrameParameters::from_values(&values, -1).is_ok());
        }
        for value in [lo - 1, hi + 1] {
            let mut values = fixture.to_values();
            values[i] = value;
            assert!(
                FrameParameters::from_values(&values, -1).is_err(),
                "field {i}"
            );
        }
    }
    assert!(FrameParameters::from_values(&fixture.to_values()[..11], -1).is_err());
    assert!(FrameParameters::from_values(&fixture.to_values(), 0).is_err());
    let mut plan = FinalPlan::read(GOLDEN, Limits::default()).unwrap();
    let mut parameters = fixture;
    parameters.keep_invisible = 0;
    let before = parameters;
    assert!(plan.parameters(&mut parameters).is_err());
    assert_eq!(parameters, before);
    // Exchange succeeded first, just as in the source hook.
    assert_eq!(plan.cursor(), 1);
}

#[test]
fn input_shape_equality_and_capped_tree_budget() {
    let shape = shape_fixture();
    assert_eq!(shape.to_values().unwrap(), [1, 32, 17, 4]);
    assert_eq!(shape.tree_limit(), 1160);
    for index in 0..4 {
        let mut values = shape.to_values().unwrap();
        values[index] += 1;
        assert!(shape.validate_record(&values).is_err());
    }
    let huge = InputShape {
        width: u64::MAX,
        height: u64::MAX,
        ..shape
    };
    assert_eq!(huge.tree_limit(), MAX_TREE_NODES);
    assert!(huge.to_values().is_err());
    assert_eq!(
        InputShape {
            channels: 0,
            ..huge
        }
        .tree_limit(),
        1024
    );
    let mut plan = FinalPlan::capture(Limits::default()).unwrap();
    assert!(plan.set_tree_limit(MAX_TREE_NODES + 1).is_err());
    assert_eq!(plan.tree_limit(), 0);
}

#[test]
fn scalar_record_schemas_reject_wrong_sizes_and_values() {
    for tag in Tag::ALL {
        assert_eq!(Tag::from_name(tag.name().as_bytes()), Some(tag));
    }
    assert_eq!(Tag::from_name(b"unknown"), None);
    assert_eq!(
        global_state(&[32, i32::MIN.into(), 41]).unwrap(),
        [32, i32::MIN, 41]
    );
    assert!(global_state(&[33, 0, 0]).is_err());
    assert!(global_state(&[0, i64::MAX, 0]).is_err());
    assert_eq!(wp(&[7, 4], 7).unwrap(), 4);
    assert!(wp(&[8, 4], 7).is_err());
    assert!(wp(&[7, 5], 7).is_err());
    assert!(prefix(&[1]).unwrap());
    assert!(!prefix(&[0]).unwrap());
    assert!(prefix(&[]).is_err());
    assert!(prefix(&[2]).is_err());
    assert_eq!(ans(&[12]).unwrap(), 12);
    assert!(ans(&[13]).is_err());
    assert!(patches(&[0]).is_ok());
    assert!(patches(&[1]).is_err());
    assert!(MapMode::from_values(&[1, 1], 3).unwrap().simple);
    assert!(MapMode::from_values(&[1, 0], 4).is_err());
    assert!(MapMode::from_values(&[0, 2], 4).is_err());
    assert!(cfl(&[-128, 127], 2, 1).is_ok());
    assert!(cfl(&[-129, 127], 2, 1).is_err());
    assert!(cfl(&[], usize::MAX, 2).is_err());
    assert!(dc_thresholds(&[-1025, 0, 1023]).is_ok());
    for values in [&[-1026][..], &[1024], &[1, 1], &[1, 0], &[0; 8]] {
        assert!(dc_thresholds(values).is_err());
    }
}

#[test]
fn clusters_and_uint_preserve_noncanonical_but_source_valid_decisions() {
    assert_eq!(clusters(&[0, 3, 0], 3, 0, 0).unwrap(), 4);
    for (map, contexts, previous, offset) in [
        (&[][..], 0, 0, 0),
        (&[0][..], 2, 0, 0),
        (&[-1][..], 1, 0, 0),
        (&[256][..], 1, 0, 0),
        (&[0][..], 1, 1, 0),
        (&[0][..], 1, 0, 1),
    ] {
        assert!(clusters(map, contexts, previous, offset).is_err());
    }
    let values = [8, 4, 2, 1, 8, 0, 0];
    let configs = UintConfigs::from_values(&values, 2, false).unwrap();
    assert_eq!(configs.log_alpha_size, 8);
    assert_eq!(
        configs.configs().collect::<Vec<_>>(),
        [
            UintConfig {
                split: 4,
                msb: 2,
                lsb: 1
            },
            UintConfig {
                split: 8,
                msb: 0,
                lsb: 0
            }
        ]
    );
    assert!(UintConfigs::from_values(&[15, 15, 0, 0], 1, true).is_ok());
    for values in [
        [4, 0, 0, 0],
        [9, 4, 0, 0],
        [8, 8, 1, 0],
        [8, 4, 3, 2],
        [8, 4, i64::MAX, i64::MAX],
        [8, -1, 0, 0],
    ] {
        assert!(UintConfigs::from_values(&values, 1, false).is_err());
    }
    assert!(UintConfigs::from_values(&[], usize::MAX, false).is_err());
}

#[test]
fn transform_rows_require_validation_against_evolving_channels() {
    let values = [0, 0, 6, 0, 0, 0, 0, 1, 1, 0, 2, 8, 6, 1];
    let rows: Vec<_> = Transforms::from_values(&values)
        .unwrap()
        .decisions()
        .collect();
    rows[0].validate_channels(3).unwrap();
    rows[1].validate_channels(3).unwrap();
    assert!(rows[0].validate_channels(2).is_err());
    assert!(rows[1].validate_channels(2).is_err());
    assert!(Transforms::from_values(&[]).is_ok());
    assert!(Transforms::from_values(&[0; 8]).is_err());
    assert!(Transforms::from_values(&[2, 0, 0, 0, 0, 0, 0]).is_err());
    assert!(Transforms::from_values(&[0, 0, 42, 0, 0, 0, 0]).is_err());
    let empty_palette = [1, 0, 0, 0, 0, 0, 0];
    assert!(
        Transforms::from_values(&empty_palette)
            .unwrap()
            .decisions()
            .next()
            .unwrap()
            .validate_channels(3)
            .is_err()
    );
}

#[test]
fn tree_schema_validates_breadth_first_topology_and_branch_intervals() {
    let leaf = [-1, 0, 6, 1];
    assert_eq!(
        tree(&leaf, 1).unwrap(),
        [TreeNode::Leaf {
            offset: 0,
            predictor: 6,
            multiplier: 1
        }]
    );
    let mut values = vec![0, 5, 0, 1, 0, 7, 0, 1, 0, 3, 0, 1];
    for _ in 0..4 {
        values.extend(leaf);
    }
    let nodes = tree(&values, 7).unwrap();
    assert_eq!(
        nodes[0],
        TreeNode::Split {
            property: 0,
            threshold: 5,
            left: 1,
            right: 2
        }
    );
    assert!(tree(&values, 6).is_err());
    values[5] = 5;
    assert!(tree(&values, 7).is_err());
    values[5] = 7;
    values[9] = 5;
    assert!(tree(&values, 7).is_err());
    for invalid in [
        vec![],
        vec![0, 0, 0, 1],
        vec![-1, 0, 14, 1],
        vec![-1, 0, 0, 0],
        vec![-1, i64::MAX, 0, 1],
        vec![0, i32::MAX.into(), 0, 1, -1, 0, 0, 1, -1, 0, 0, 1],
        vec![-1, 0, 0, 1, -1, 0, 0, 1, -1, 0, 0, 1],
    ] {
        assert!(tree(&invalid, 100).is_err());
    }
}

#[test]
fn tree_schema_preserves_strict_source_stack_bound() {
    fn chain(splits: i64) -> Vec<i64> {
        let mut values = vec![0, 0, 0, 1];
        for i in 1..splits {
            values.extend([0, i, 0, 1, -1, 0, 0, 1]);
        }
        values.extend([-1, 0, 0, 1, -1, 0, 0, 1]);
        values
    }
    assert!(tree(&chain(2046), MAX_TREE_NODES).is_ok());
    assert!(tree(&chain(2047), MAX_TREE_NODES).is_err());
}

#[test]
fn lz_schema_reuses_header_and_match_spans_without_searching() {
    let header = LzHeader {
        enabled: false,
        contexts: 3,
        streams: 2,
        min_symbol: 224,
        min_length: 3,
        length_config: UintConfig {
            split: 0,
            msb: 0,
            lsb: 0,
        },
    };
    let mut values = header.to_values().unwrap();
    values[0] = 1;
    assert!(LzHeader::from_values(&values, header).unwrap().enabled);
    for field in 1..8 {
        let mut bad = values;
        bad[field] += 1;
        assert!(LzHeader::from_values(&bad, header).is_err());
    }
    let values = [3, 4, 2, 8, 3, 0];
    assert_eq!(
        Matches::from_values(&values, 11, 3)
            .unwrap()
            .spans()
            .count(),
        2
    );
    for bad in [
        &[3, 4][..],
        &[3, 2, 0],
        &[3, 20, 0],
        &[3, 4, -1],
        &[3, 4, i64::MAX],
        &[3, 4, 0, 6, 3, 0],
        &[11, 3, 0],
    ] {
        assert!(Matches::from_values(bad, 11, 3).is_err());
    }
    // Structure alone cannot establish a valid backward distance or equality.
    assert!(Matches::from_values(&[0, 3, 0], 3, 3).is_ok());
}

#[test]
fn coefficient_order_checks_pass_count_permutations_and_supplied_natural_order() {
    let natural = std::array::from_fn(|i| i as u32);
    let mut values = vec![0];
    for _ in 0..3 {
        values.extend(natural.map(i64::from));
    }
    assert!(
        !CoeffOrder::from_values(&values, 1, &natural)
            .unwrap()
            .used_order
    );
    assert!(CoeffOrder::from_values(&values, 2, &natural).is_err());
    values.swap(2, 3);
    assert!(CoeffOrder::from_values(&values, 1, &natural).is_err());
    values[0] = 1;
    assert!(
        CoeffOrder::from_values(&values, 1, &natural)
            .unwrap()
            .used_order
    );
    values[2] = values[3];
    assert!(CoeffOrder::from_values(&values, 1, &natural).is_err());
}
