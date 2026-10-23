use super::*;

#[test]
fn invalid_input_rejected_before_frame_work() {
    assert!(encode_coefficients(&JpegData::default(), &Options::default()).is_err());
}

#[test]
fn jpeg_execution_lends_one_pool_and_null_scopes_never_enter_it() {
    use crate::execution;
    let caller = std::thread::current().id();
    for threads in [1, 3] {
        for cooperative_ma in [None, Some(WorkerLimits::default())] {
            let options = Options {
                threads,
                cooperative_ma,
                ..Default::default()
            };
            with_threads(&options, |pool| {
                assert_eq!(pool.is_some(), threads > 1);
                assert_eq!(execution::is_serial(), threads == 1);
                if let Some(pool) = pool {
                    assert!(pool.current_thread_index().is_some());
                    let here = std::thread::current().id();
                    execution::with_pool(pool, || assert_eq!(std::thread::current().id(), here));
                    let nested = execution::map(&[0; 32], |_| {
                        assert!(pool.current_thread_index().is_some());
                        execution::worker_count()
                    });
                    assert_eq!(nested, vec![threads; 32]);
                } else {
                    assert_eq!(std::thread::current().id(), caller);
                }
                Ok(())
            })
            .unwrap();
            execution::serial(|| {
                with_threads(&options, |pool| {
                    assert!(pool.is_none());
                    assert!(execution::is_serial());
                    assert_eq!(std::thread::current().id(), caller);
                    Ok(())
                })
            })
            .unwrap();
        }
    }
    assert!(!execution::is_serial());
}

#[test]
fn cooperative_limits_reject_before_pool_or_frame_callbacks() {
    assert!(Options::default().cooperative_ma.is_none());
    for limits in [
        WorkerLimits {
            tasks: 0,
            scratch_bytes: 1,
        },
        WorkerLimits {
            tasks: 65,
            scratch_bytes: 1,
        },
        WorkerLimits {
            tasks: 1,
            scratch_bytes: 0,
        },
    ] {
        let options = Options {
            threads: 3,
            cooperative_ma: Some(limits),
            ..Default::default()
        };
        assert!(matches!(
            with_threads(&options, |_| -> Result<()> {
                panic!("invalid limits reached the frame callback")
            }),
            Err(Error::InvalidInput(_))
        ));
    }
}

#[test]
#[ignore = "requires RS_JXL_JPEG_INPUT and RS_JXL_JPEG_OUTPUT for independent oracle bring-up"]
fn write_bringup_frame() {
    let input = std::env::var_os("RS_JXL_JPEG_INPUT").expect("input path");
    let output = std::env::var_os("RS_JXL_JPEG_OUTPUT").expect("output path");
    let bytes = std::fs::read(input).unwrap();
    let mut jpeg = super::super::coefficient_reader::read_coefficients(&bytes).unwrap();
    for record in &mut jpeg.records {
        if let super::super::data::JpegRecord::Entropy(data) = record {
            data.bytes.clear();
        }
    }
    let encoded = encode_coefficients(&jpeg, &Options::default()).unwrap();
    std::fs::write(output, encoded).unwrap();
}
