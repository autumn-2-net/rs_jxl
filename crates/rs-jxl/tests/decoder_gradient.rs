//! Exercise the actual vendor helper without the upstream test crate's absent
//! image fixtures. Stand-ins expose only its private data types; end-to-end
//! tests in gradient_interop use the complete production decoder.
mod error {
    pub use jxl_decoder::error::Result;
}

mod frame {
    pub mod modular {
        pub use rs_jxl::modular::predictor::Predictor;

        pub mod predict {
            pub use rs_jxl::modular::predictor::Neighbors as PredictionData;
        }

        pub mod tree {
            use super::Predictor;

            #[derive(Clone, Copy)]
            pub enum TreeNode {
                Split {
                    property: u8,
                    val: i32,
                    left: u32,
                    right: u32,
                },
                Leaf {
                    predictor: Predictor,
                    offset: i32,
                    multiplier: u32,
                    id: u32,
                },
            }
        }
    }
}

#[path = "../../../vendor/jxl-rs/jxl/src/frame/modular/decode/gradient.rs"]
mod gradient;
