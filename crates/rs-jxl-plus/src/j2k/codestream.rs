// PJA MIT translation; see ../../LICENSE-PJA and ../../NOTICE-PJA.
//! Native pjaj2kcs.pas, pinned 1058f63fbfb9c60cc6640b7bb130d6f7c7ea7dbd.
#![allow(non_snake_case)] // Source field names retained to audit the packet state machine.
use crate::{Error, Result};
mod assembly;
mod bytes;
mod encode_packets;
mod geometry;
mod headers;
mod packets;
mod segments;
#[cfg(test)]
mod tests;
use bytes::{BR, BW};
const MAXNL: usize = 32;
const MAXCOMPS: usize = 16384;
const MAXTILES: usize = 65535;
const MAXCBS: usize = 1 << 21;
const MAXPRECINCTS: usize = 1 << 22;
const MAXITER: usize = 1 << 26;
const MAXENTRIES: usize = 1 << 25;
const MAXPASSES: usize = 120;
const ZBPLIMIT: i32 = 128;
const MINWINDOW: usize = 32;
const TAILMAX: usize = 1 << 24;
const INF: i32 = i32::MAX;
const SH_HDR: usize = 0;
const SH_LAY: usize = 1;
const SH_T2: usize = 2;
const SH_ZBP: usize = 3;
const SH_TREE: usize = 4;
const SH_LB: usize = 5;
const SH_T1R: usize = 6;
const SH_T1X: usize = 7;
const SH_HDRX: usize = 8;
const SH_DEC: usize = 9;
const NSTREAMS: usize = 10;
const R_FLUSH: u8 = 0;
const R_ERTERM: u8 = 1;
const R_MINF: u8 = 2;
const R_MINE: u8 = 3;
const R_TAILE: u8 = 4;
const R_TAILF: u8 = 5;
const R_RAW: u8 = 6;
const R_EMPTY: u8 = 7;
const R_PAD0: u8 = 10;
const R_PAD1: u8 = 11;
const R_PAD2: u8 = 12;
const R_RAWB: u8 = 13;
fn refuse(s: &'static str) -> Error {
    Error::Invalid(s)
}
fn be16(d: &[u8], p: usize) -> Result<u16> {
    let b = d.get(p..p + 2).ok_or(refuse("J2K marker cut short"))?;
    Ok(u16::from_be_bytes([b[0], b[1]]))
}
fn be32(d: &[u8], p: usize) -> Result<u32> {
    let b = d.get(p..p + 4).ok_or(refuse("J2K marker cut short"))?;
    Ok(u32::from_be_bytes([b[0], b[1], b[2], b[3]]))
}
fn ceil(a: i64, b: i64) -> i64 {
    -(-a).div_euclid(b)
}
fn cp2(a: i64, n: usize) -> i64 {
    ceil(a, 1i64 << n)
}
fn fp2(a: i64, n: usize) -> i64 {
    a >> n
}
fn flog(x: usize) -> usize {
    if x == 0 {
        0
    } else {
        (usize::BITS - 1 - x.leading_zeros()) as usize
    }
}
#[derive(Clone, Debug, Default)]
pub(crate) struct Stats {
    pub codestreams: u64,
    pub tiles: u64,
    pub packets: u64,
    pub hdr_exceptions: u64,
    pub tree_overrides: u64,
    pub lb_non_minimal: u64,
    pub cbs: u64,
    pub cb_raw: u64,
    pub segs: u64,
    pub raw_segs: u64,
    pub bounds: u64,
    pub tails: u64,
    pub tail_bits: u64,
    pub decisions: u64,
    pub rule: [u64; 16],
    pub brule: [u64; 4],
    pub side: [u64; 10],
    pub ht_cup: [u64; 16],
    pub ht_ref: [u64; 8],
    pub ht_raw: [u64; 8],
    pub ht_cbs: u64,
    pub ht_samples: u64,
}
#[derive(Clone, Debug)]
pub(crate) struct Limits {
    pub max_input_bytes: usize,
    pub max_output_bytes: usize,
    pub max_samples: u64,
    pub max_side_bytes: usize,
}
impl Default for Limits {
    fn default() -> Self {
        Self {
            max_input_bytes: 1 << 30,
            max_output_bytes: 1 << 30,
            max_samples: 1 << 30,
            max_side_bytes: 1 << 30,
        }
    }
}
#[derive(Clone, Debug, Default)]
struct CodC {
    nl: usize,
    xcb: usize,
    ycb: usize,
    sty: u32,
    tr: u8,
    prec: bool,
    pp: Vec<u8>,
}
#[derive(Clone, Debug, Default)]
struct QntC {
    style: u8,
    guard: i32,
    ex: Vec<i32>,
}
#[derive(Clone, Debug, Default)]
struct Poc {
    rs: usize,
    cs: usize,
    lye: usize,
    re: usize,
    ce: usize,
    prg: u8,
}
#[derive(Clone, Debug, Default)]
struct Item {
    code: u16,
    raw: Vec<u8>,
    kind: u8,
    z: u8,
    st: u8,
    cnt: usize,
}
#[derive(Clone, Debug, Default)]
struct Tp {
    isot: usize,
    tpsot: u8,
    tnsot: u8,
    psotMode: u8,
    psot: usize,
    hstart: usize,
    dstart: usize,
    dend: usize,
    items: Vec<Item>,
    npk: usize,
    pk0: usize,
    gap: Vec<u8>,
    ppmOff: usize,
    ppmLen: usize,
    plens: Vec<usize>,
    bytes: Vec<u8>,
}
#[derive(Clone, Debug, Default)]
struct CodG {
    have: bool,
    scod: u8,
    prog: u8,
    layers: usize,
    mct: u8,
    cod: CodC,
}
#[derive(Clone, Debug, Default)]
struct Params {
    g: CodG,
    cod: Vec<CodC>,
    qnt: Vec<QntC>,
    pocs: Vec<Poc>,
}
#[derive(Clone, Debug, Default)]
struct BandG {
    orient: usize,
    x0: i64,
    y0: i64,
    x1: i64,
    y1: i64,
    cbw: usize,
    cbh: usize,
    gx0: i64,
    gy0: i64,
    ncbx: usize,
    ncby: usize,
    base: usize,
    mb: i32,
    comp: usize,
    res: usize,
}
#[derive(Clone, Debug, Default)]
struct ResG {
    x0: i64,
    y0: i64,
    x1: i64,
    y1: i64,
    ppx: usize,
    ppy: usize,
    pw: usize,
    ph: usize,
    prBase: usize,
    nb: usize,
    band: [usize; 3],
}
#[derive(Clone, Debug, Default)]
struct CompG {
    dx: usize,
    dy: usize,
    x0: i64,
    y0: i64,
    x1: i64,
    y1: i64,
    nres: usize,
    res0: usize,
    sty: u32,
}
#[derive(Default)]
struct State<'a> {
    link: Option<&'a mut dyn Link>,
    kind: Kind,
    an: bool,
    cs: Vec<u8>,
    sw: [BW; 10],
    sr: [BR; 10],
    dbits: usize,
    st: Stats,
    limits: Limits,
    xsiz: i64,
    ysiz: i64,
    xosiz: i64,
    yosiz: i64,
    xtsiz: i64,
    ytsiz: i64,
    xtosiz: i64,
    ytosiz: i64,
    ncomp: usize,
    nxt: usize,
    nyt: usize,
    cdx: Vec<usize>,
    cdy: Vec<usize>,
    csiz: Vec<u8>,
    main: Vec<Item>,
    tps: Vec<Tp>,
    tail: Vec<u8>,
    mainG: CodG,
    mainCoc: Vec<Option<CodC>>,
    mainQcd: Option<QntC>,
    mainQcc: Vec<Option<QntC>>,
    mainPoc: Vec<Poc>,
    ppm: Vec<u8>,
    hasPpm: bool,
    hasRgn: bool,
    tileRgn: bool,
    iterations: usize,
    tileNo: usize,
    tx0: i64,
    ty0: i64,
    tx1: i64,
    ty1: i64,
    par: Params,
    comps: Vec<CompG>,
    ress: Vec<ResG>,
    bands: Vec<BandG>,
    ncb: usize,
    nprec: usize,
    hdrPacked: bool,
    pkd: Vec<u8>,
    pkdPos: usize,
    tileTps: Vec<usize>,
    ieTotal: usize,
    ieCap: usize,
    cbBand: Vec<i32>,
    cbFirst: Vec<i32>,
    cbZbp: Vec<i32>,
    cbPasses: Vec<i32>,
    cbNseg: Vec<i32>,
    cbSegHave: Vec<i32>,
    cbSegMax: Vec<i32>,
    cbLb: Vec<i32>,
    cbHead: Vec<i32>,
    cbTail: Vec<i32>,
    cbPb: Vec<i32>,
    cbLeaf: Vec<i32>,
    cbELb: Vec<i32>,
    bandZ: Vec<i32>,
    prNext: Vec<i32>,
    prPb: Vec<i32>,
    pbCw: Vec<i32>,
    pbCh: Vec<i32>,
    pbLx: Vec<i32>,
    pbLy: Vec<i32>,
    pbBand: Vec<i32>,
    pbNode: Vec<i32>,
    pbNn: Vec<i32>,
    ndParent: Vec<i32>,
    ndLow: Vec<i32>,
    ndVal: Vec<i32>,
    eLow: Vec<i32>,
    eVal: Vec<i32>,
    lgNode: Vec<i32>,
    lgLow: Vec<i32>,
    lgVal: Vec<i32>,
    pkC: Vec<i32>,
    pkR: Vec<i32>,
    pkP: Vec<i32>,
    pkL: Vec<i32>,
    pkFlags: Vec<i32>,
    pkNsop: Vec<i32>,
    pkIe0: Vec<i32>,
    pkIeN: Vec<i32>,
    ieCb: Vec<i32>,
    ieFlags: Vec<i32>,
    ieNp: Vec<i32>,
    ieInc: Vec<i32>,
    iePc0: Vec<i32>,
    iePcN: Vec<i32>,
    pcSeg: Vec<i32>,
    pcNp: Vec<i32>,
    pcNext: Vec<i32>,
    tCb: Vec<i32>,
    tIncl: Vec<i32>,
    tFirst: Vec<i32>,
    tZbp: Vec<i32>,
    tNp: Vec<i32>,
    tInc: Vec<i32>,
    tL0: Vec<i32>,
    tLn: Vec<i32>,
    tNps: Vec<i32>,
    cbHt: Vec<i32>,
    tFound: Vec<i32>,
    cbDOff: Vec<usize>,
    cbCur: Vec<usize>,
    pkHdrPos: Vec<usize>,
    pkHdrLen: Vec<usize>,
    pkStart: Vec<usize>,
    pkEnd: Vec<usize>,
    pcLen: Vec<usize>,
    pcPos: Vec<usize>,
    tLens: Vec<usize>,
    tilePk: Vec<usize>,
    cbData: BW,
    eKnown: Vec<bool>,
    npb: usize,
    nnd: usize,
    nlg: usize,
    npk: usize,
    nsopLast: i32,
    nie: usize,
    npc: usize,
    curTp: usize,
    curPos: usize,
    npkTarget: usize,
    tileOut: BW,
    symb: Vec<u8>,
    fb: Vec<u8>,
    eb: Vec<u8>,
    segdata: Vec<u8>,
    tileMaxCap: usize,
    passend: Vec<usize>,
    stb: Vec<u8>,
    seglen: Vec<usize>,
    segpass: Vec<usize>,
    segp0: Vec<usize>,
    pev: Vec<usize>,
    kpv: Vec<usize>,
    brD: Vec<u8>,
    brP: usize,
    brE: usize,
    brBuf: u32,
    brCt: u32,
    brBad: bool,
    bw: BW,
    bwBuf: u32,
    bwCt: u32,
    ht: bool,
    htq: Vec<i32>,
    htout: Vec<u8>,
    htLastCup: u8,
    htLastRef: u8,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum Kind {
    #[default]
    None,
    Encode,
    Decode,
    Dump,
    PixEncode,
    PixDecode,
}
#[derive(Clone, Debug)]
pub(crate) struct BlockInfo {
    pub band: usize,
    pub x0: usize,
    pub y0: usize,
    pub width: usize,
    pub height: usize,
    pub style: u32,
    pub passes: usize,
    pub zbp: i32,
    pub orientation: usize,
}
#[derive(Clone, Debug)]
pub(crate) struct PixComp {
    pub dx: usize,
    pub dy: usize,
    pub nres: usize,
    pub depth: usize,
    pub signed: bool,
    pub ok: bool,
    pub x0: i64,
    pub y0: i64,
    pub x1: i64,
    pub y1: i64,
}
#[derive(Clone, Debug)]
pub(crate) struct PixBand {
    pub comp: usize,
    pub res: usize,
    pub orient: usize,
    pub mb: i32,
    pub x0: i64,
    pub y0: i64,
    pub x1: i64,
    pub y1: i64,
}
#[derive(Clone, Debug)]
pub(crate) struct PixTile {
    pub tile: usize,
    pub mct: bool,
    pub rgn: bool,
    pub ix0: i64,
    pub iy0: i64,
    pub ix1: i64,
    pub iy1: i64,
    pub comps: Vec<PixComp>,
    pub bands: Vec<PixBand>,
}
pub(crate) trait Link {
    fn kind(&self) -> Kind;
    fn start_slice(&mut self, k: usize, hb: usize) -> Result<()>;
    fn end_slice(&mut self, k: usize) -> Result<()>;
    fn tile(&mut self, pixels: &PixTile, bands: Vec<super::model::Band>, maps: usize)
    -> Result<()>;
    fn block(
        &mut self,
        info: &BlockInfo,
        symbols: &mut Vec<u8>,
        pass_ends: &mut Vec<usize>,
        cap: usize,
    ) -> Result<usize>;
    fn ht_block(
        &mut self,
        info: &BlockInfo,
        np: usize,
        plane: i32,
        values: &mut Vec<i32>,
    ) -> Result<()>;
    fn tile_end(&mut self) -> Result<()>;
}
impl State<'_> {
    fn classic_values(
        &mut self,
        cb: usize,
        x0: usize,
        y0: usize,
        width: usize,
        height: usize,
        style: u32,
        passes: usize,
        cap: usize,
        input: Option<&super::tier1::Passes>,
    ) -> Result<super::tier1::Passes> {
        let info = BlockInfo {
            band: self.cbBand[cb] as usize,
            x0,
            y0,
            width,
            height,
            style,
            passes,
            zbp: self.cbZbp[cb],
            orientation: self.bands[self.cbBand[cb] as usize].orient,
        };
        let mut symbols = input.map_or_else(Vec::new, |p| p.symbols.clone());
        let mut pass_ends = input.map_or_else(Vec::new, |p| p.pass_ends.clone());
        let n = self
            .link
            .as_deref_mut()
            .ok_or(refuse("J2K missing decision model"))?
            .block(&info, &mut symbols, &mut pass_ends, cap)?;
        if n > cap
            || n != symbols.len()
            || pass_ends.len() != passes
            || pass_ends.last().is_some_and(|&v| v != n)
            || pass_ends.windows(2).any(|w| w[0] > w[1])
        {
            return Err(refuse("J2K model decision count"));
        }
        Ok(super::tier1::Passes {
            symbols,
            pass_ends,
            used_bits: input.map_or(0, |p| p.used_bits),
        })
    }
    fn ht_values(
        &mut self,
        cb: usize,
        x0: usize,
        y0: usize,
        width: usize,
        height: usize,
        causal: bool,
        np: usize,
        plane: i32,
        q: &mut [i32],
    ) -> Result<()> {
        let info = BlockInfo {
            band: self.cbBand[cb] as usize,
            x0,
            y0,
            width,
            height,
            style: if causal { 8 } else { 0 },
            passes: np,
            zbp: self.cbZbp[cb],
            orientation: self.bands[self.cbBand[cb] as usize].orient,
        };
        let mut values = q.to_vec();
        self.link
            .as_deref_mut()
            .ok_or(refuse("J2K missing HT model"))?
            .ht_block(&info, np, plane, &mut values)?;
        if values.len() != q.len() {
            return Err(refuse("J2K HT model value count"));
        }
        q.copy_from_slice(&values);
        Ok(())
    }
}
pub(crate) fn side_bound(n: usize) -> usize {
    n.saturating_add(n / 4)
        .saturating_add(65536)
        .min(i32::MAX as usize)
}

#[derive(Clone, Debug)]
struct Slice {
    first: usize,
    hb: usize,
    pos: [usize; 8],
}
fn tiles(j: &State) -> Vec<usize> {
    let mut used = vec![false; j.nxt * j.nyt];
    for tp in &j.tps {
        used[tp.isot] = true;
    }
    used.into_iter()
        .enumerate()
        .filter_map(|(i, b)| b.then_some(i))
        .collect()
}
fn check_samples(j: &State) -> Result<()> {
    let area = (j.xsiz - j.xosiz) as u128 * (j.ysiz - j.yosiz) as u128 * j.ncomp as u128;
    if j.limits.max_samples > 0 && area > j.limits.max_samples as u128 {
        return Err(Error::Limit("J2K samples"));
    }
    Ok(())
}
fn version(ht: bool, kind: Kind) -> u8 {
    1 + u8::from(matches!(
        kind,
        Kind::Encode | Kind::Decode | Kind::PixEncode | Kind::PixDecode
    )) + 2 * u8::from(ht)
}
pub(crate) fn analyze(data: &[u8], ht: bool, limits: &Limits) -> Result<(Vec<u8>, Stats)> {
    analyze_inner(data, ht, limits, None)
}
pub(crate) fn analyze_with(
    data: &[u8],
    ht: bool,
    limits: &Limits,
    link: &mut dyn Link,
) -> Result<(Vec<u8>, Stats)> {
    analyze_inner(data, ht, limits, Some(link))
}
fn analyze_inner(
    data: &[u8],
    ht: bool,
    limits: &Limits,
    link: Option<&mut dyn Link>,
) -> Result<(Vec<u8>, Stats)> {
    if data.len() > limits.max_input_bytes {
        return Err(Error::Limit("J2K input bytes"));
    }
    let kind = link.as_ref().map_or(Kind::None, |l| l.kind());
    if matches!(kind, Kind::Decode | Kind::PixDecode) {
        return Err(refuse("J2K analyze link direction"));
    }
    let mut j = State {
        an: true,
        cs: data.to_vec(),
        ht,
        limits: limits.clone(),
        link,
        kind,
        ..State::default()
    };
    j.structure()?;
    check_samples(&j)?;
    let tile_list = tiles(&j);
    let nslice = if kind == Kind::None {
        0
    } else {
        tile_list.len().min(data.len() / (1 << 20)).min(64).max(1)
    };
    let mut slices = Vec::<Slice>::new();
    let mut current = None;
    for (t, &tile) in tile_list.iter().enumerate() {
        if nslice > 0 {
            let k = if let Some(k) = current {
                if k + 1 < nslice && t == (k + 1) * tile_list.len() / nslice {
                    k + 1
                } else {
                    k
                }
            } else {
                0
            };
            if current != Some(k) {
                if let Some(previous) = current {
                    j.link
                        .as_deref_mut()
                        .ok_or(refuse("J2K missing model"))?
                        .end_slice(previous)?;
                }
                let mut hb = 10;
                while hb < 20 && (1usize << (hb + 1)) < data.len() / nslice {
                    hb += 1;
                }
                let mut pos = [0; 8];
                for i in SH_LAY..=SH_HDRX {
                    pos[i - SH_LAY] = j.sw[i].d.len();
                }
                slices.push(Slice { first: t, hb, pos });
                j.link
                    .as_deref_mut()
                    .ok_or(refuse("J2K missing model"))?
                    .start_slice(k, hb)?;
                current = Some(k);
            }
        }
        j.tile(tile)?;
    }
    if let Some(k) = current {
        j.link
            .as_deref_mut()
            .ok_or(refuse("J2K missing model"))?
            .end_slice(k)?;
    }
    j.write_headers()?;
    let mut blob = BW::default();
    blob.put(version(ht, kind))?;
    for w in &j.sw {
        blob.v(w.d.len() as u64)?;
    }
    for w in &j.sw {
        blob.put_n(&w.d, 0, w.d.len())?;
    }
    if matches!(kind, Kind::Encode | Kind::PixEncode) {
        blob.v(slices.len() as u64)?;
        for (k, s) in slices.iter().enumerate() {
            if k > 0 {
                blob.v(s.first as u64)?;
            }
            blob.put(s.hb as u8)?;
            if k > 0 {
                for p in s.pos {
                    blob.v(p as u64)?;
                }
            }
        }
    }
    if blob.d.len() > limits.max_side_bytes {
        return Err(Error::Limit("J2K analyzed bytes"));
    }
    if kind == Kind::None {
        let back = rebuild(&blob.d, ht, limits)?;
        if back != data {
            return Err(refuse("J2K analysis reconstruction mismatch"));
        }
    }
    j.st.codestreams += 1;
    for i in 0..NSTREAMS {
        j.st.side[i] += j.sw[i].d.len() as u64;
    }
    Ok((blob.d, j.st))
}
pub(crate) fn rebuild(blob: &[u8], ht: bool, limits: &Limits) -> Result<Vec<u8>> {
    rebuild_inner(blob, ht, limits, None)
}
pub(crate) fn rebuild_with(
    blob: &[u8],
    ht: bool,
    limits: &Limits,
    link: &mut dyn Link,
) -> Result<Vec<u8>> {
    rebuild_inner(blob, ht, limits, Some(link))
}
fn rebuild_inner(
    blob: &[u8],
    ht: bool,
    limits: &Limits,
    link: Option<&mut dyn Link>,
) -> Result<Vec<u8>> {
    if blob.len() > limits.max_side_bytes {
        return Err(Error::Limit("J2K blob bytes"));
    }
    let kind = link.as_ref().map_or(Kind::None, |l| l.kind());
    if matches!(kind, Kind::Encode | Kind::PixEncode | Kind::Dump) {
        return Err(refuse("J2K rebuild link direction"));
    }
    let mut r = BR {
        d: blob.to_vec(),
        p: 0,
    };
    if r.get()? != version(ht, kind) {
        return Err(refuse("J2K blob version"));
    }
    let mut lengths = [0; 10];
    for n in &mut lengths {
        *n = r.vi(i32::MAX as usize)?;
    }
    let mut j = State {
        an: false,
        ht,
        link,
        kind,
        limits: limits.clone(),
        ..State::default()
    };
    for (i, &n) in lengths.iter().enumerate() {
        j.sr[i] = BR {
            d: r.take(n)?,
            p: 0,
        };
    }
    j.read_headers()?;
    check_samples(&j)?;
    let tile_list = tiles(&j);
    let mut slices: Vec<Slice> = Vec::new();
    if matches!(kind, Kind::Decode | Kind::PixDecode) {
        let n = r.vi(tile_list.len().min(64))?;
        if n == 0 {
            return Err(refuse("J2K slice count"));
        }
        for k in 0..n {
            let first = if k == 0 {
                0
            } else {
                r.vi(tile_list.len() - 1)?
            };
            if k > 0 && first <= slices[k - 1].first {
                return Err(refuse("J2K slice order"));
            }
            let hb = r.get()? as usize;
            if !(10..=20).contains(&hb) {
                return Err(refuse("J2K slice table bits"));
            }
            let mut pos = [0; 8];
            if k > 0 {
                for i in SH_LAY..=SH_HDRX {
                    pos[i - SH_LAY] = r.vi(lengths[i])?;
                }
            }
            slices.push(Slice { first, hb, pos });
        }
    }
    if r.remaining() != 0 {
        return Err(refuse("J2K blob left over"));
    }
    let mut current = None;
    for (t, &tile) in tile_list.iter().enumerate() {
        if !slices.is_empty() {
            let mut k = current.unwrap_or(0);
            while k + 1 < slices.len() && slices[k + 1].first <= t {
                k += 1;
            }
            if current != Some(k) {
                if let Some(previous) = current {
                    j.link
                        .as_deref_mut()
                        .ok_or(refuse("J2K missing model"))?
                        .end_slice(previous)?;
                }
                for i in SH_LAY..=SH_HDRX {
                    if j.sr[i].p != slices[k].pos[i - SH_LAY] {
                        return Err(refuse("J2K slice side position"));
                    }
                }
                j.link
                    .as_deref_mut()
                    .ok_or(refuse("J2K missing model"))?
                    .start_slice(k, slices[k].hb)?;
                current = Some(k);
            }
        }
        j.tile(tile)?;
        if j.tps.iter().map(|tp| tp.bytes.len()).sum::<usize>() > limits.max_output_bytes {
            return Err(Error::Limit("J2K output bytes"));
        }
    }
    if let Some(k) = current {
        j.link
            .as_deref_mut()
            .ok_or(refuse("J2K missing model"))?
            .end_slice(k)?;
    }
    for i in 0..NSTREAMS {
        if i != SH_DEC && j.sr[i].remaining() != 0 {
            return Err(refuse("J2K side information left over"));
        }
    }
    if kind == Kind::None {
        if ht {
            if j.sr[SH_DEC].remaining() != 0 {
                return Err(refuse("J2K values left over"));
            }
        } else if j.dbits.div_ceil(8) != j.sr[SH_DEC].d.len() {
            return Err(refuse("J2K decisions left over"));
        }
    } else if !j.sr[SH_DEC].d.is_empty() {
        return Err(refuse("J2K modeled decision stream must be empty"));
    }
    let cs = j.assemble()?;
    if cs.len() > limits.max_output_bytes {
        return Err(Error::Limit("J2K output bytes"));
    }
    Ok(cs)
}
pub(crate) fn blob_siz(blob: &[u8]) -> Result<Vec<u8>> {
    let mut r = BR {
        d: blob.to_vec(),
        p: 0,
    };
    if !(1..=4).contains(&r.get()?) {
        return Err(refuse("J2K blob version"));
    }
    let mut lens = [0; 10];
    for n in &mut lens {
        *n = r.vi(i32::MAX as usize)?;
    }
    let h = r.take(lens[SH_HDR])?;
    let mut r = BR { d: h, p: 0 };
    if r.vi(1 << 20)? < 1 || r.get()? != 0 {
        return Err(refuse("J2K blob missing SIZ"));
    }
    let n = r.vi(65537)?;
    let siz = r.take(n)?;
    if siz.len() < 4 || be16(&siz, 0)? != 0xff51 {
        return Err(refuse("J2K blob bad SIZ"));
    }
    Ok(siz)
}
fn filled<T: Clone>(n: usize, value: T) -> Result<Vec<T>> {
    let mut v = Vec::new();
    v.try_reserve_exact(n)
        .map_err(|_| Error::Limit("J2K allocation"))?;
    v.resize(n, value);
    Ok(v)
}

impl State<'_> {
    fn model_tile(&mut self) -> Result<()> {
        let mut pixels = PixTile {
            tile: self.tileNo,
            mct: self.par.g.mct != 0,
            rgn: self.tileRgn,
            ix0: self.xosiz,
            iy0: self.yosiz,
            ix1: self.xsiz,
            iy1: self.ysiz,
            comps: Vec::new(),
            bands: Vec::new(),
        };
        for c in 0..self.ncomp {
            let cg = &self.comps[c];
            pixels.comps.push(PixComp {
                dx: cg.dx,
                dy: cg.dy,
                nres: cg.nres,
                depth: (self.csiz[c] & 127) as usize + 1,
                signed: self.csiz[c] & 128 != 0,
                ok: self.par.cod[c].tr == 1
                    && self.par.qnt[c].style == 0
                    && self.par.qnt[c].ex.len() >= 3 * (cg.nres - 1) + 1,
                x0: cg.x0,
                y0: cg.y0,
                x1: cg.x1,
                y1: cg.y1,
            });
        }
        let mut bands = Vec::new();
        let mut area = vec![0usize; self.ncomp];
        let mapcap = 1usize << 27;
        for bg in &self.bands {
            pixels.bands.push(PixBand {
                comp: bg.comp,
                res: bg.res,
                orient: bg.orient,
                mb: bg.mb,
                x0: bg.x0,
                y0: bg.y0,
                x1: bg.x1,
                y1: bg.y1,
            });
            let find = |c: usize, r: usize, o: usize| -> Option<usize> {
                let cg = self.comps.get(c)?;
                if r >= cg.nres {
                    return None;
                }
                if r == 0 {
                    if o == 0 {
                        Some(self.ress[cg.res0].band[0])
                    } else {
                        None
                    }
                } else if o > 0 {
                    Some(self.ress[cg.res0 + r].band[o - 1])
                } else {
                    None
                }
            };
            let width = (bg.x1 - bg.x0) as usize;
            let height = (bg.y1 - bg.y0) as usize;
            let valid = width < i32::MAX as usize && height < i32::MAX as usize;
            let (width, height) = if valid { (width, height) } else { (0, 0) };
            bands.push(super::model::Band {
                comp: bg.comp as i32,
                res: bg.res as i32,
                orient: bg.orient as i32,
                nres: self.comps[bg.comp].nres as i32,
                width,
                height,
                mb: bg.mb,
                parent: if bg.res == 0 {
                    None
                } else if bg.res == 1 {
                    find(bg.comp, 0, 0)
                } else {
                    find(bg.comp, bg.res - 1, bg.orient)
                },
                cross0: if bg.comp > 0 {
                    find(0, bg.res, bg.orient)
                } else {
                    None
                },
                cross1: if bg.comp > 1 {
                    find(1, bg.res, bg.orient)
                } else {
                    None
                },
                map: None,
            });
            if area[bg.comp] <= mapcap {
                area[bg.comp] = area[bg.comp].saturating_add(width.saturating_mul(height));
            }
        }
        let a0 = area[0];
        let a1 = area.get(1).copied().unwrap_or(0);
        let a2 = area.iter().skip(2).copied().max().unwrap_or(0);
        let need = a0.saturating_add(a1).saturating_add(a2);
        let mut base = [0, a0, a0.saturating_add(a1)];
        for b in 0..bands.len() {
            if need <= mapcap {
                let c = (bands[b].comp as usize).min(2);
                bands[b].map = Some(base[c]);
                base[c] += bands[b].width * bands[b].height;
                if bands[b].comp >= 2 && b + 1 < bands.len() && bands[b + 1].comp != bands[b].comp {
                    base[2] = a0 + a1;
                }
            }
        }
        self.link
            .as_deref_mut()
            .ok_or(refuse("J2K no model"))?
            .tile(&pixels, bands, if need > mapcap { 0 } else { need })
    }
    fn tile(&mut self, t: usize) -> Result<()> {
        self.tileNo = t;
        self.tileTps = self
            .tps
            .iter()
            .enumerate()
            .filter_map(|(i, tp)| (tp.isot == t).then_some(i))
            .collect();
        if self.tileTps.is_empty() {
            return Err(refuse("J2K missing tile parts"));
        }
        self.tile_params()?;
        self.geometry()?;
        if self.kind != Kind::None {
            self.model_tile()?;
        }
        self.bandZ = filled(self.bands.len(), 0)?;
        self.pkdPos = 0;
        self.nsopLast = -1;
        self.tileMaxCap = 0;
        self.st.tiles += 1;
        if self.an {
            self.curTp = 0;
            self.curPos = self.tps[self.tileTps[0]].dstart;
            for &ti in &self.tileTps {
                self.tps[ti].npk = 0;
                self.tps[ti].gap.clear();
            }
            self.iterate(Self::emit_analyse)?;
            while self.curTp < self.tileTps.len() {
                let ti = self.tileTps[self.curTp];
                self.tps[ti].gap = self.cs[self.curPos..self.tps[ti].dend].to_vec();
                self.curTp += 1;
                if self.curTp < self.tileTps.len() {
                    self.curPos = self.tps[self.tileTps[self.curTp]].dstart;
                }
            }
            self.sw[SH_LAY].v(self.npk as u64)?;
            let mut k = 0;
            for &ti in &self.tileTps {
                let tp = &mut self.tps[ti];
                self.sw[SH_LAY].v(tp.npk as u64)?;
                self.sw[SH_LAY].v(tp.gap.len() as u64)?;
                self.sw[SH_LAY].put_n(&tp.gap, 0, tp.gap.len())?;
                tp.pk0 = k;
                k += tp.npk;
                tp.plens = (tp.pk0..tp.pk0 + tp.npk)
                    .map(|p| self.pkEnd[p] - self.pkStart[p])
                    .collect();
            }
        } else {
            self.npkTarget = self.sr[SH_LAY].vi(MAXENTRIES)?;
            let mut k = 0usize;
            for &ti in &self.tileTps {
                let tp = &mut self.tps[ti];
                tp.npk = self.sr[SH_LAY].vi(self.npkTarget)?;
                let n = self.sr[SH_LAY].vi(i32::MAX as usize)?;
                tp.gap = self.sr[SH_LAY].take(n)?;
                tp.pk0 = k;
                k = k.checked_add(tp.npk).ok_or(refuse("J2K layout"))?;
            }
            if k != self.npkTarget {
                return Err(refuse("J2K layout"));
            }
            self.iterate(Self::emit_rebuild)?;
            if self.npk != self.npkTarget {
                return Err(refuse("J2K packets missing"));
            }
        }
        self.st.packets += self.npk as u64;
        if !self.hdrPacked {
            self.trees()?;
        }
        self.cbData.d.clear();
        for cb in 0..self.ncb {
            self.cbDOff[cb] = 0;
            if self.cbPasses[cb] > 0 {
                if self.ht {
                    self.do_cb_ht(cb)?;
                } else {
                    self.do_cb(cb)?;
                }
            }
        }
        if matches!(self.kind, Kind::PixEncode | Kind::PixDecode) {
            self.link
                .as_deref_mut()
                .ok_or(refuse("J2K no pixel model"))?
                .tile_end()?;
        }
        self.pass3()?;
        if !self.an {
            for &ti in &self.tileTps {
                let tp = &mut self.tps[ti];
                let mut w = BW::default();
                w.put_n(
                    &[
                        255,
                        144,
                        0,
                        10,
                        (tp.isot >> 8) as u8,
                        tp.isot as u8,
                        0,
                        0,
                        0,
                        0,
                        tp.tpsot,
                        tp.tnsot,
                    ],
                    0,
                    12,
                )?;
                tp.plens = (tp.pk0..tp.pk0 + tp.npk)
                    .map(|p| self.tilePk[p + 1] - self.tilePk[p])
                    .collect();
                let mut pi = 0;
                for it in &tp.items {
                    if it.kind == 2 {
                        if pi + it.cnt > tp.npk {
                            return Err(refuse("J2K PLT"));
                        }
                        let b = assembly::plt_bytes(it.z, &tp.plens, pi, it.cnt)?;
                        w.put_n(&b, 0, b.len())?;
                        pi += it.cnt;
                    } else {
                        w.put_n(&it.raw, 0, it.raw.len())?;
                        if it.code == 0xff58 {
                            pi += assembly::plt_count(&it.raw);
                        }
                    }
                }
                w.put_n(&[255, 147], 0, 2)?;
                let from = self.tilePk[tp.pk0];
                let n = self.tilePk[tp.pk0 + tp.npk] - from;
                w.put_n(&self.tileOut.d, from, n)?;
                w.put_n(&tp.gap, 0, tp.gap.len())?;
                let size = match tp.psotMode {
                    0 => w.d.len(),
                    1 => 0,
                    _ => tp.psot,
                };
                if size > u32::MAX as usize {
                    return Err(Error::Limit("J2K tile part bytes"));
                }
                w.d[6..10].copy_from_slice(&(size as u32).to_be_bytes());
                tp.bytes = w.d;
            }
        }
        if self.sw.iter().map(|w| w.d.len()).sum::<usize>() > self.limits.max_side_bytes {
            return Err(Error::Limit("J2K side bytes"));
        }
        Ok(())
    }
}
