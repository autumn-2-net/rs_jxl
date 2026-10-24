use super::{Error, backend_error};

#[test]
fn backend_resource_io_and_malformed_errors_remain_distinct() {
    use jxl_decoder::error::Error as BackendError;
    for error in [
        BackendError::ImageSizeTooLarge(8, 8),
        BackendError::ImageDimensionTooLarge(u64::MAX),
        BackendError::ImageOutOfMemory(8, 8),
        BackendError::OutOfMemory(Vec::<u8>::new().try_reserve(usize::MAX).unwrap_err()),
    ] {
        assert!(matches!(backend_error(error), Error::Limit(_)));
    }
    let Error::Io(error) = backend_error(BackendError::IOError(std::io::Error::from(
        std::io::ErrorKind::UnexpectedEof,
    ))) else {
        panic!("backend I/O error lost its type");
    };
    assert_eq!(error.kind(), std::io::ErrorKind::UnexpectedEof);
    assert!(matches!(
        backend_error(BackendError::InvalidBox),
        Error::Decoding(_)
    ));
}
