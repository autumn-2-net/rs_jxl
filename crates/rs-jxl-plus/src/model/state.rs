//! PJA InitV7 tables and V7Weights, MIT, source revision
//! 1058f63fbfb9c60cc6640b7bb130d6f7c7ea7dbd, pas/pjamodel.pas.
//! See ../../LICENSE-PJA and ../../NOTICE-PJA.
use super::{AdaptiveBit, BitCoder, probability};

pub(super) const NZ: i32 = 11;
pub(super) const MAG: i32 = 20;
pub(super) const REM: i32 = 9;
pub(super) const KB: i32 = 11;
pub(super) const PRED: i32 = 22;
pub(super) const PQ: i32 = 26;
pub(super) const ROW: i32 = 16;
pub(super) const ACT: i32 = 28;

macro_rules! bit_tables {
    ($( $name:ident => $size:expr ),+ $(,)?) => {
        #[derive(Clone,Copy)]
        pub(super) enum B { $( $name ),+ }
        fn bits() -> Vec<Vec<AdaptiveBit>> { vec![$(vec![AdaptiveBit::default(); $size]),+] }
    }
}
bit_tables! {
    Nz2=>11*11*64, Nz3=>12*11*64, Nz4=>12*12*8,
    Nzm1=>11*8*8,Nzm2=>12*8*8,
    Zero2=>64*20*4,Zero4=>11*20*9,ZLum=>64*28*9,ZCb=>64*28*9,
    Expo3=>11*9*16,XPos=>64*20*16,XRow=>11*16*16,XLum=>11*28*16,XCb=>11*28*16,
    Mant1=>16*11,MFirst=>16*20*16,MRest=>11*256,ISign=>64*9,
    Ec1=>2*11*9*8,Ec2=>2*9*9*8,Ec3=>2*9*22*8,EcX=>2048+2*9*18*8,Ecp=>2*16*11*8,
    Ez6=>14*8*8*4,Ezero4=>14*20,EzLum=>14*26*4,
    Eexpo1=>14*26*16,Eexpo2=>14*20*16,Eexpo3=>26*8*16,Eexpo4=>14*11*16,Eexpo5=>14*12*16,
    ExLum=>14*26*16,ExCb=>14*26*16,Emant1=>16*26,EMant=>14*1280,
    Esr=>2*3*26*9,ESgn=>14*3*3*3*3*3,
    DcZero=>28,Dzf=>20*5,Dz3=>11*8,Dz4=>28*3,DzX=>28*17,
    DcExpo=>28*16,Dx2=>20*16,Dx3=>11*8*16,Dx4=>28*3*16,DxX=>8*17*16,
    DcMant1=>18*28,DcMant=>18*16,Dsf=>14*9*3*3,DsX=>9*9*3,
}
#[derive(Clone, Copy)]
pub(super) enum W {
    Nz,
    Zero,
    Expo,
    EZero,
    EExpo,
    DcZero,
    DcExpo,
    ECnt,
}
pub(super) struct Model {
    #[cfg(test)]
    namespace: u8,
    bits: Vec<Vec<AdaptiveBit>>,
    weights: Vec<Vec<[i32; 8]>>,
}
impl Model {
    pub fn new() -> Self {
        // V7Weights: counter lanes receive 4/3 of an equal share of all
        // inputs (including bias); bias and padding lanes initially zero.
        let mut weights = Vec::new();
        for (n, selectors) in [
            (3, 6),
            (4, 64),
            (5, 15 * 11 * 9),
            (3, 14 * 64),
            (7, 15 * 14 * 4),
            (5, 28),
            (5, 17),
            (5, 2 * 8 * 11),
        ] {
            let mut w = [0i32; 8];
            w[..n].fill(65536 * 4 / (3 * (n as i32 + 1)));
            weights.push(vec![w; selectors]);
        }
        Self {
            #[cfg(test)]
            namespace: 0,
            bits: bits(),
            weights,
        }
    }
    pub fn with_namespace(_namespace: u8) -> Self {
        #[cfg(test)]
        {
            Self {
                namespace: _namespace,
                ..Self::new()
            }
        }
        #[cfg(not(test))]
        {
            Self::new()
        }
    }
    pub fn one<C: BitCoder>(&mut self, cd: &mut C, a: B, i: usize, v: i32) -> i32 {
        #[cfg(test)]
        if cd.static_probability_mode() {
            let key = super::static_model::context_key(
                cd.static_probability_profile(),
                self.namespace,
                0,
                a as usize,
                i,
                &[],
            );
            if let Some(bit) = cd.static_bit(key, v) {
                return bit;
            }
        }
        #[cfg(test)]
        cd.probability_context(self.namespace, 0, a as usize, i, &[]);
        probability::one(cd, &mut self.bits[a as usize][i], v)
    }
    pub fn avg<C: BitCoder>(&mut self, cd: &mut C, a: (B, usize), b: (B, usize), v: i32) -> i32 {
        #[cfg(test)]
        if cd.static_probability_mode() {
            let key = super::static_model::context_key(
                cd.static_probability_profile(),
                self.namespace,
                1,
                a.0 as usize,
                a.1,
                &[(b.0 as usize, b.1)],
            );
            if let Some(bit) = cd.static_bit(key, v) {
                return bit;
            }
        }
        let mut aa = self.bits[a.0 as usize][a.1];
        let mut bb = self.bits[b.0 as usize][b.1];
        #[cfg(test)]
        cd.probability_context(self.namespace, 1, a.0 as usize, a.1, &[(b.0 as usize, b.1)]);
        let bit = probability::avg2(cd, &mut aa, &mut bb, v);
        self.bits[a.0 as usize][a.1] = aa;
        self.bits[b.0 as usize][b.1] = bb;
        bit
    }
    pub fn mix<C: BitCoder, const N: usize>(
        &mut self,
        cd: &mut C,
        w: W,
        selector: usize,
        lr: i32,
        v: i32,
        refs: &[(B, usize); N],
    ) -> i32 {
        #[cfg(test)]
        if cd.static_probability_mode() {
            let mut indices = [(0, 0); 7];
            for (out, &(bank, index)) in indices.iter_mut().zip(refs) {
                *out = (bank as usize, index);
            }
            let key = super::static_model::context_key(
                cd.static_probability_profile(),
                self.namespace,
                2,
                w as usize,
                selector,
                &indices[..refs.len()],
            );
            if let Some(bit) = cd.static_bit(key, v) {
                return bit;
            }
        }
        // Counters belong to separate upstream arrays. Copying this bounded
        // group to the stack allows safe, disjoint mutable references without
        // raw pointers; every learned state is written back immediately.
        let mut counters = [AdaptiveBit::default(); N];
        for (c, &(b, i)) in counters.iter_mut().zip(refs) {
            *c = self.bits[b as usize][i];
        }
        #[cfg(test)]
        {
            let mut indices = [(0, 0); 7];
            for (out, &(bank, index)) in indices.iter_mut().zip(refs) {
                *out = (bank as usize, index);
            }
            cd.probability_context(
                self.namespace,
                2,
                w as usize,
                selector,
                &indices[..refs.len()],
            );
        }
        let bit = probability::mix_array(
            cd,
            &mut self.weights[w as usize][selector],
            &mut counters,
            lr,
            v,
        );
        for (&(b, i), c) in refs.iter().zip(counters) {
            self.bits[b as usize][i] = c;
        }
        bit
    }
}
