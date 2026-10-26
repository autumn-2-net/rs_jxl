//! Compile the actual maintained weight helper without the upstream test crate,
//! whose unrelated include_bytes! fixtures are not bundled with this vendor.
//! The reciprocal table is generated independently from its specified formula;
//! end-to-end decoder tests separately exercise the production parent table.

const DIVLOOKUP: [u32; 64] = {
    let mut table = [0; 64];
    let mut i = 0;
    while i < table.len() {
        table[i] = (1 << 24) / (i as u32 + 1);
        i += 1;
    }
    table
};

#[path = "../../../vendor/jxl-rs/jxl/src/frame/modular/weighted_weights.rs"]
mod weighted_weights;
