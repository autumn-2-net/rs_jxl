//! PJA pas/pjamodel.pas TCoefModel.Init and CodeMantissa, MIT,
//! revision 1058f63fbfb9c60cc6640b7bb130d6f7c7ea7dbd; LICENSE-PJA/NOTICE-PJA.
use super::*;
macro_rules! tables{($($name:ident=>$size:expr),+$(,)?)=>{#[derive(Clone,Copy)]pub(super) enum B{$($name),+}fn sizes()->Vec<usize>{vec![$($size),+]}}}
tables! {Nz1=>11*64,Nz2=>11*11*64,Nz3=>12*11*64,Zero1=>64*9*20,Zero2=>64*20*4,Zero3=>64*17,Zero4=>11*20*9,Zero5=>64*12*9,Expo1=>11*20*15,Expo2=>11*20*15,Expo3=>11*9*15,Expo4=>20*15,Mant1=>16*11,Mant=>16*15,Sign=>11*9,Ezero1=>14*22,Ezero2=>14*11,Ezero3=>14*8*22,Ezero4=>14*20,Ezero5=>14*12*4,Eexpo1=>14*22*15,Eexpo2=>14*20*15,Eexpo3=>22*8*15,Eexpo4=>14*11*15,Eexpo5=>14*12*15,Emant1=>16*22,Esign=>2*3*22,DcZero=>24,DcExpo=>24*17,DcSign=>24,DcMant1=>18*24,DcMant=>18*17,Dz2=>20,Dz3=>11*8,Dz4=>24*3,Dx2=>20*17,Dx3=>11*8*17,Dx4=>24*3*17,Fz=>64*9*20,Fx=>11*9*20*15,Fez=>14*11*8*22,Fex=>14*11*22*8*15,Fdz=>20*24*3,Fdx=>24*3*17}
#[derive(Clone, Copy)]
pub(super) enum W {
    Nz,
    Zero,
    Expo,
    EZero,
    EExpo,
    DcZero,
    DcExpo,
}
pub(super) struct Model {
    pub chroma: bool,
    pub fast: bool,
    bank: Bank,
}
impl Model {
    pub fn new(chroma: bool, fast: bool) -> Self {
        Self {
            chroma,
            fast,
            bank: Bank::new(
                &sizes(),
                &[
                    (4, 6),
                    (6, 64),
                    (5, 15 * 11),
                    (6, 14),
                    (6, 15 * 14),
                    (5, 24),
                    (5, 17),
                ],
            ),
        }
    }
    pub fn one<C: BitCoder>(&mut self, cd: &mut C, b: B, i: usize, v: i32) -> i32 {
        self.bank.one(cd, b as usize, i, v)
    }
    pub fn avg<C: BitCoder>(&mut self, cd: &mut C, a: (B, usize), b: (B, usize), v: i32) -> i32 {
        let mut aa = self.bank.bits[a.0 as usize][a.1];
        let mut bb = self.bank.bits[b.0 as usize][b.1];
        let bit = crate::probability::avg2(cd, &mut aa, &mut bb, v);
        self.bank.bits[a.0 as usize][a.1] = aa;
        self.bank.bits[b.0 as usize][b.1] = bb;
        bit
    }
    pub fn mix<C: BitCoder>(
        &mut self,
        cd: &mut C,
        w: W,
        sel: usize,
        refs: &[(B, usize)],
        v: i32,
    ) -> i32 {
        let mut r = [(0usize, 0usize); 5];
        for (r, &(b, i)) in r.iter_mut().zip(refs) {
            *r = (b as usize, i);
        }
        self.bank.mix(cd, w as usize, sel, &r[..refs.len()], v)
    }
    #[allow(clippy::too_many_arguments)]
    pub fn mantissa<C: BitCoder>(
        &mut self,
        cd: &mut C,
        first: B,
        bucket: usize,
        stride: usize,
        rest: B,
        maxe: usize,
        e: usize,
        v: i32,
    ) -> i32 {
        let mut x = 1i32 << (e - 1);
        for i in (0..e - 1).rev() {
            let bit = if i == e - 2 {
                self.one(cd, first, e * stride + bucket, (v >> i) & 1)
            } else {
                self.one(cd, rest, e * maxe + i, (v >> i) & 1)
            };
            x |= bit << i;
        }
        x
    }
}
