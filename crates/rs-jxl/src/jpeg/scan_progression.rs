// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// BSD-3-Clause; retained notice in docs/PORTING_KERNELS.md.
//! Coefficient coverage checks from jpeg/enc_jpeg_data_reader.cc::ProcessScan,
//! frozen libjxl b866b3defec13494bfbaf705db93581cda118e37.
//!
//! This tracks transmitted bit planes, not decoded pixels or coefficient values.
//! It intentionally retains the source's tolerance of skipped refinement levels:
//! overlapping data and reversed refinement order are rejected, but this is not
//! a claim that every accepted scan sequence is a conforming complete JPEG.

use super::data::{BLOCK_SIZE, MAX_COMPONENTS, ScanInfo};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScanProgression {
    coverage: Vec<[u16; BLOCK_SIZE]>,
}

impl ScanProgression {
    pub fn new(components: usize) -> Result<Self, &'static str> {
        if !(1..=MAX_COMPONENTS).contains(&components) {
            return Err("invalid JPEG progression component count");
        }
        Ok(Self {
            coverage: vec![[0; BLOCK_SIZE]; components],
        })
    }

    /// Mark one scan's bit planes after checking all components. An error leaves
    /// the state unchanged. Sequential coding uses the full band regardless of
    /// the SOS approximation fields, matching ProcessScan's effective values.
    pub fn apply(&mut self, scan: &ScanInfo, progressive: bool) -> Result<(), &'static str> {
        if scan.components.is_empty() || scan.components.len() > self.coverage.len() {
            return Err("invalid JPEG progression scan components");
        }
        let (ss, se, ah, al) = if progressive {
            (scan.ss, scan.se, scan.ah, scan.al)
        } else {
            (0, 63, 0, 0)
        };
        if ss > se || se >= BLOCK_SIZE as u32 || ah > 15 || al > 10 {
            return Err("unsupported JPEG spectral band or approximation level");
        }
        let new_mask = if ah == 0 { u16::MAX << al } else { 1u16 << al };
        let refined_mask = (1u16 << al) - 1;
        let mut seen = [false; MAX_COMPONENTS];
        for component in &scan.components {
            let index = component.component_index;
            if index >= self.coverage.len() || seen[index] {
                return Err("invalid or duplicate JPEG progression component");
            }
            seen[index] = true;
            for &previous in &self.coverage[index][ss as usize..=se as usize] {
                if previous & new_mask != 0 {
                    return Err("overlapping JPEG coefficient scans");
                }
                if previous & refined_mask != 0 {
                    return Err("JPEG scan follows a more refined coefficient scan");
                }
            }
        }
        for component in &scan.components {
            for previous in &mut self.coverage[component.component_index][ss as usize..=se as usize]
            {
                *previous |= new_mask;
            }
        }
        Ok(())
    }

    /// Coverage in natural *scan* order (zigzag indices 0..63), not pixel order.
    pub fn coverage(&self, component: usize) -> Option<&[u16; BLOCK_SIZE]> {
        self.coverage.get(component)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jpeg::data::ComponentScanInfo;

    fn scan(components: &[usize], ss: u32, se: u32, ah: u32, al: u32) -> ScanInfo {
        ScanInfo {
            components: components
                .iter()
                .map(|&component_index| ComponentScanInfo {
                    component_index,
                    ..Default::default()
                })
                .collect(),
            ss,
            se,
            ah,
            al,
            ..Default::default()
        }
    }

    #[test]
    fn progressive_bands_and_component_ids_remain_independent() {
        let mut state = ScanProgression::new(3).unwrap();
        state.apply(&scan(&[2, 0, 1], 0, 0, 0, 2), true).unwrap();
        for component in [2, 0, 1] {
            state.apply(&scan(&[component], 1, 5, 0, 2), true).unwrap();
            state.apply(&scan(&[component], 6, 63, 0, 2), true).unwrap();
            state.apply(&scan(&[component], 0, 63, 2, 1), true).unwrap();
            state.apply(&scan(&[component], 0, 63, 1, 0), true).unwrap();
            assert_eq!(state.coverage(component), Some(&[u16::MAX; BLOCK_SIZE]));
        }
    }

    #[test]
    fn rejection_is_atomic_even_after_a_valid_component() {
        let mut state = ScanProgression::new(3).unwrap();
        state.apply(&scan(&[2], 0, 0, 0, 0), true).unwrap();
        let before = state.clone();
        for bad in [
            scan(&[0, 2], 0, 0, 0, 0),
            scan(&[0, 0], 0, 0, 0, 0),
            scan(&[0, 3], 0, 0, 0, 0),
            scan(&[0], 8, 7, 0, 0),
            scan(&[0], 0, 64, 0, 0),
            scan(&[0], 0, 63, 0, 11),
        ] {
            assert!(state.apply(&bad, true).is_err());
            assert_eq!(state, before);
        }
    }

    #[test]
    fn reversed_refinements_reject_and_sequential_covers_full_band() {
        let mut state = ScanProgression::new(1).unwrap();
        state.apply(&scan(&[0], 0, 63, 0, 2), true).unwrap();
        state.apply(&scan(&[0], 0, 63, 1, 0), true).unwrap();
        assert!(state.apply(&scan(&[0], 0, 63, 2, 1), true).is_err());
        let mut sequential = ScanProgression::new(1).unwrap();
        sequential.apply(&scan(&[0], 0, 0, 0, 2), false).unwrap();
        assert_eq!(sequential.coverage(0), Some(&[u16::MAX; BLOCK_SIZE]));
        assert!(sequential.apply(&scan(&[0], 0, 63, 0, 0), false).is_err());
        assert!(ScanProgression::new(0).is_err());
        assert!(ScanProgression::new(5).is_err());
    }
}
