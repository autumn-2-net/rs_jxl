//! Complete post-transform states retained in standard reference slots.
//! Slot 3 remains private to generated patch atlases; no synthetic copy frame
//! or decoder change is needed to rotate the two ordinary saved canvases.

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct ReferenceSlots {
    pub(super) frames: [Option<usize>; 4],
}

impl ReferenceSlots {
    /// Prefer the immediate predecessor for canonical ties; the other retained
    /// state can be older than both it and the first frame.
    pub(super) fn sources(&self, index: usize) -> Vec<(usize, u8)> {
        let mut sources: Vec<_> = (1u8..=2)
            .filter_map(|slot| self.frames[usize::from(slot)].map(|frame| (frame, slot)))
            .collect();
        sources.sort_by_key(|&(frame, slot)| (frame.checked_add(1) != Some(index), slot));
        sources
    }

    pub(super) fn destination(&self, index: usize, source: Option<u8>, retained: bool) -> u8 {
        if !retained {
            return if index == 0 { 1 } else { 2 };
        }
        let keep = source.or_else(|| {
            (1u8..=2).find(|&slot| {
                self.frames[usize::from(slot)]
                    .is_some_and(|frame| frame.checked_add(1) == Some(index))
            })
        });
        // A delta retains its source. A full replacement retains the previous
        // state, so scene changes do not permanently pin the original frame.
        match keep {
            Some(1) => 2,
            Some(2) => 1,
            _ => 1,
        }
    }

    pub(super) fn saved_layer(
        &mut self,
        index: usize,
        slot: u8,
        duration: u32,
        final_part: bool,
        last: bool,
    ) {
        if !last && (duration == 0 || slot != 0) {
            // An intermediate rectangle is a partial composite, not the full
            // logical input. Do not advertise it as a reusable historical state.
            self.frames[usize::from(slot)] = final_part.then_some(index);
        }
    }
}
