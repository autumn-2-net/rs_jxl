//! Focused internal test/benchmark target; avoids linking unrelated JPEG/J2K
//! test fixtures during pixel experiment iteration. Uses the same source files.
//! Run: cargo test --release -p rs-jxl-plus --example pixel_lab
pub use rs_jxl_plus::{Error, Result};

#[allow(dead_code)]
#[path = "../src/arith.rs"]
mod arith;
#[cfg(test)]
#[path = "../src/pixel_compose_tests.rs"]
mod pixel_compose_tests;
#[cfg(test)]
#[path = "../src/pixel_model.rs"]
mod pixel_model;
#[cfg(test)]
#[path = "../src/pixel_tests.rs"]
mod pixel_tests;

fn main() {
    eprintln!(
        "Internal pixel laboratory: use cargo test --release -p rs-jxl-plus --example pixel_lab"
    );
}
