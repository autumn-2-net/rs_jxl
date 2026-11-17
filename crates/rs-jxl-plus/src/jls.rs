//! Native JPEG-LS decomposition and byte-exact reconstruction.
//! PJA pas/pjajls.pas, MIT, revision 1058f63fbfb9c60cc6640b7bb130d6f7c7ea7dbd.
//! Source behavior follows CharLS; retain LICENSE-PJA/NOTICE-PJA (CharLS BSD).
use crate::{Error, Result};
mod entropy;
mod headers;
mod state;
#[cfg(test)]
pub(crate) mod tests;
#[cfg(test)]
pub(crate) use headers::decompose;
pub(crate) use headers::{is_jls, parse_headers, recompose, validate, verify};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Component {
    pub id: u8,
    pub h: usize,
    pub v: usize,
    pub width: usize,
    pub height: usize,
    pub samples: Vec<u16>,
}
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Scan {
    pub components: Vec<usize>,
    pub near: i32,
    pub ilv: i32,
    pub maxval: i32,
    pub t1: i32,
    pub t2: i32,
    pub t3: i32,
    pub reset: i32,
    pub ri: i32,
    pub cut: usize,
    pub runs: Vec<i32>,
}
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Image {
    pub width: usize,
    pub height: usize,
    pub bits: usize,
    pub components: Vec<Component>,
    pub scans: Vec<Scan>,
    pub raw: Vec<u8>,
}
#[derive(Clone, Debug)]
pub(crate) struct Limits {
    pub max_input_bytes: usize,
    pub max_output_bytes: usize,
    pub max_samples: usize,
    pub max_scans: usize,
    pub max_runs: usize,
}
impl Default for Limits {
    fn default() -> Self {
        Self {
            max_input_bytes: 512 << 20,
            max_output_bytes: 512 << 20,
            max_samples: 1 << 27,
            max_scans: 1024,
            max_runs: 1 << 27,
        }
    }
}
pub(crate) trait Symbols {
    fn begin_scan(&mut self, s: &Scan, j: &Image) -> Result<()>;
    #[allow(clippy::too_many_arguments)]
    fn mapped(
        &mut self,
        m: i32,
        c: usize,
        i: usize,
        k: i32,
        ctx: i32,
        px: i32,
        sgn: i32,
        inv: bool,
    ) -> Result<i32>;
    fn sample(&mut self, c: usize, i: usize, x: i32) -> Result<()>;
    fn run_bit(&mut self, bit: i32, c: usize, i: usize, index: usize) -> Result<i32>;
    fn run_rest(&mut self, v: i32, bits: i32, c: usize, index: usize) -> Result<i32>;
    fn run_end(&mut self, c: usize, i: usize, len: usize) -> Result<()>;
}
pub(crate) fn default_thresholds(maxval: i32, near: i32) -> (i32, i32, i32) {
    let clamp = |i: i32, lo: i32| if i > maxval || i < lo { lo } else { i };
    if maxval >= 128 {
        let f = (maxval.min(4095) + 128) / 256;
        let a = clamp(f + 2 + 3 * near, near + 1);
        let b = clamp(f * 4 + 3 + 5 * near, a);
        (a, b, clamp(f * 17 + 4 + 7 * near, b))
    } else {
        let f = 256 / (maxval + 1);
        let a = clamp(2.max(3 / f + 3 * near), near + 1);
        let b = clamp(3.max(7 / f + 5 * near), a);
        (a, b, clamp(4.max(21 / f + 7 * near), b))
    }
}
pub(crate) fn med(a: i32, b: i32, c: i32) -> i32 {
    if c >= a.max(b) {
        a.min(b)
    } else if c <= a.min(b) {
        a.max(b)
    } else {
        a + b - c
    }
}
pub(crate) fn grouped(j: &Image, s: &Scan) -> bool {
    s.components
        .iter()
        .any(|&c| j.components[c].h != 1 || j.components[c].v != 1)
}
pub(crate) fn scan_rows(j: &Image, s: &Scan) -> usize {
    if s.ilv == 0 {
        j.components[s.components[0]].height
    } else if !grouped(j, s) {
        j.height
    } else {
        s.components
            .iter()
            .map(|&c| j.components[c].height.div_ceil(j.components[c].v))
            .max()
            .unwrap_or(0)
    }
}
pub(crate) fn code_symbols(
    j: &mut Image,
    symbols: &mut dyn Symbols,
    encoding: bool,
    limits: &Limits,
) -> Result<()> {
    validate(j, limits)?;
    for si in 0..j.scans.len() {
        let s = j.scans[si].clone();
        symbols.begin_scan(&s, j)?;
        let mut st = state::State::new(j, &s);
        if !encoding {
            st.runs.clear();
        }
        let rows = scan_rows(j, &s);
        let mut y = 0;
        while y < rows {
            let end = if s.ri > 0 {
                rows.min(y + s.ri as usize)
            } else {
                rows
            };
            if y > 0 {
                st.reset();
            }
            let mut io = entropy::Io::symbols(symbols, encoding);
            st.lines(j, &s, &mut io, y, end, limits)?;
            y = end;
        }
        if encoding && s.near > 0 && st.run_pos != s.runs.len() {
            return Err(Error::Invalid("JPEG-LS unused recorded runs"));
        }
        if !encoding {
            j.scans[si].runs = st.runs;
        }
    }
    Ok(())
}
