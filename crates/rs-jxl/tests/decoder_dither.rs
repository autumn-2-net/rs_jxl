//! Exercise the maintained output table without unrelated upstream fixtures.
mod api {
    pub use jxl_decoder::api::DitherMode;
}

mod util {
    pub fn box_array<T: Clone, const N: usize>(elem: T) -> Box<[T; N]> {
        vec![elem; N].into_boxed_slice().try_into().ok().unwrap()
    }
}

#[path = "../../../vendor/jxl-rs/jxl/src/util/dither.rs"]
mod dither;
