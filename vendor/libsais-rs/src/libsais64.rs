//! Rust translation of upstream [libsais](https://github.com/IlyaGrebnov/libsais)
//! 2.10.4 by Ilya Grebnov.
//!
//! This module exposes the 64-bit suffix array, BWT, unBWT, PLCP and LCP entry
//! points (mirroring `libsais64.h`).

use std::marker::PhantomData;
use std::mem;
#[cfg(feature = "upstream-c")]
use std::mem::MaybeUninit;

use crate::{run_rayon_with_threads, SyncMutPtr};
use rayon::prelude::*;

pub type SaSint = i64;
pub type SaUint = u64;
pub type FastSint = isize;
pub type FastUint = usize;

pub const SAINT_BIT: u32 = 64;
pub const SAINT_MAX: SaSint = i64::MAX;
pub const SAINT_MIN: SaSint = i64::MIN;

pub const ALPHABET_SIZE: usize = 1usize << 8;
pub const UNBWT_FASTBITS: usize = 17;

pub const SUFFIX_GROUP_BIT: u32 = SAINT_BIT - 1;
pub const SUFFIX_GROUP_MARKER: SaSint = 1_i64 << (SUFFIX_GROUP_BIT - 1);

pub const LIBSAIS_LOCAL_BUFFER_SIZE: usize = 1000;
pub const LIBSAIS_PER_THREAD_CACHE_SIZE: usize = 24_576;

pub const LIBSAIS_FLAGS_NONE: SaSint = 0;
pub const LIBSAIS_FLAGS_BWT: SaSint = 1;
pub const LIBSAIS_FLAGS_GSA: SaSint = 2;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ThreadCache {
    pub symbol: SaSint,
    pub index: SaSint,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ThreadState {
    pub position: FastSint,
    pub count: FastSint,
    pub m: FastSint,
    pub last_lms_suffix: FastSint,
    pub buckets: Vec<SaSint>,
    pub cache: Vec<ThreadCache>,
}

impl ThreadState {
    fn new() -> Self {
        Self {
            position: 0,
            count: 0,
            m: 0,
            last_lms_suffix: 0,
            buckets: vec![0; 4 * ALPHABET_SIZE],
            cache: vec![ThreadCache::default(); LIBSAIS_PER_THREAD_CACHE_SIZE],
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Context {
    pub buckets: Vec<SaSint>,
    pub thread_state: Option<Vec<ThreadState>>,
    pub threads: FastSint,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UnbwtContext {
    pub bucket2: Vec<SaUint>,
    pub fastbits: Vec<u16>,
    pub buckets: Option<Vec<SaUint>>,
    pub threads: FastSint,
}

/// Internal helper: buckets index2.
#[doc(hidden)]
pub fn buckets_index2(c: FastUint, s: FastUint) -> FastUint {
    (c << 1) + s
}

/// Internal helper: buckets index4.
#[doc(hidden)]
pub fn buckets_index4(c: FastUint, s: FastUint) -> FastUint {
    (c << 2) + s
}

/// Internal helper: align up.
#[doc(hidden)]
pub fn align_up(value: usize, alignment: usize) -> usize {
    debug_assert!(alignment.is_power_of_two());
    (value + alignment - 1) & !(alignment - 1)
}

/// Internal helper: alloc thread state.
#[doc(hidden)]
pub fn alloc_thread_state(threads: SaSint) -> Option<Vec<ThreadState>> {
    if threads <= 0 {
        return None;
    }

    let len = usize::try_from(threads).ok()?;
    Some((0..len).map(|_| ThreadState::new()).collect())
}

/// Internal helper: create ctx main.
#[doc(hidden)]
pub fn create_ctx_main(threads: SaSint) -> Option<Context> {
    if threads <= 0 {
        return None;
    }

    let thread_state = if threads > 1 {
        Some(alloc_thread_state(threads)?)
    } else {
        None
    };

    Some(Context {
        buckets: vec![0; 8 * ALPHABET_SIZE],
        thread_state,
        threads: threads as FastSint,
    })
}

/// Creates the libsais64 context that allows reusing allocated memory with each libsais64 operation.
///
/// In multi-threaded environments, use one context per thread for parallel executions.
///
/// Returns the context, or `None` on allocation failure.
pub fn create_ctx() -> Option<Context> {
    create_ctx_main(1)
}

/// Destroys the libsais64 context and frees previously allocated memory.
pub fn free_ctx(_ctx: Context) {}

/// Internal helper: unbwt create ctx main.
#[doc(hidden)]
pub fn unbwt_create_ctx_main(threads: SaSint) -> Option<UnbwtContext> {
    if threads <= 0 {
        return None;
    }

    let buckets = if threads > 1 {
        let len = usize::try_from(threads).ok()? * (ALPHABET_SIZE + ALPHABET_SIZE * ALPHABET_SIZE);
        Some(vec![0; len])
    } else {
        None
    };

    Some(UnbwtContext {
        bucket2: vec![0; ALPHABET_SIZE * ALPHABET_SIZE],
        fastbits: vec![0; 1 + (1 << UNBWT_FASTBITS)],
        buckets,
        threads: threads as FastSint,
    })
}

/// Internal helper: unbwt free ctx main.
#[doc(hidden)]
pub fn unbwt_free_ctx_main(_ctx: UnbwtContext) {}

/// Creates the libsais64 reverse-BWT context that allows reusing allocated memory with each `libsais64_unbwt_*` operation.
///
/// In multi-threaded environments, use one context per thread for parallel executions.
///
/// Returns the context, or `None` on allocation failure.
pub fn unbwt_create_ctx() -> Option<UnbwtContext> {
    unbwt_create_ctx_main(1)
}

/// Destroys the libsais64 reverse-BWT context and frees previously allocated memory.
pub fn unbwt_free_ctx(_ctx: UnbwtContext) {}

/// Internal helper: count negative marked suffixes.
#[doc(hidden)]
pub fn count_negative_marked_suffixes(
    sa: &[SaSint],
    block_start: FastSint,
    block_size: FastSint,
) -> SaSint {
    block_slice(sa, block_start, block_size)
        .iter()
        .map(|&value| SaSint::from(value < 0))
        .sum()
}

/// Internal helper: count zero marked suffixes.
#[doc(hidden)]
pub fn count_zero_marked_suffixes(
    sa: &[SaSint],
    block_start: FastSint,
    block_size: FastSint,
) -> SaSint {
    block_slice(sa, block_start, block_size)
        .iter()
        .map(|&value| SaSint::from(value == 0))
        .sum()
}

/// Internal helper: place cached suffixes.
#[doc(hidden)]
pub fn place_cached_suffixes(
    sa: &mut [SaSint],
    cache: &[ThreadCache],
    block_start: FastSint,
    block_size: FastSint,
) {
    let start = usize::try_from(block_start).expect("block_start must be non-negative");
    let len = usize::try_from(block_size).expect("block_size must be non-negative");
    let entries = if cache.len() >= start + len {
        &cache[start..start + len]
    } else {
        &cache[..len]
    };

    for entry in entries {
        let slot = usize::try_from(entry.symbol).expect("cache symbol must be non-negative");
        sa[slot] = entry.index;
    }
}

/// Internal helper: compact and place cached suffixes.
#[doc(hidden)]
pub fn compact_and_place_cached_suffixes(
    sa: &mut [SaSint],
    cache: &mut [ThreadCache],
    block_start: FastSint,
    block_size: FastSint,
) {
    let start = usize::try_from(block_start).expect("block_start must be non-negative");
    let len = usize::try_from(block_size).expect("block_size must be non-negative");
    let read_start = if cache.len() >= start + len { start } else { 0 };
    let read_end = read_start + len;

    let mut write = read_start;
    for read in read_start..read_end {
        let entry = cache[read];
        if entry.symbol >= 0 {
            cache[write] = entry;
            write += 1;
        }
    }

    place_cached_suffixes(sa, cache, block_start, (write - read_start) as FastSint);
}

/// Internal helper: flip suffix markers (OpenMP variant).
#[doc(hidden)]
pub fn flip_suffix_markers_omp(sa: &mut [SaSint], l: SaSint, threads: SaSint) {
    let len = usize::try_from(l).expect("l must be non-negative");
    let omp_num_threads = if threads > 1 && l >= 65_536 {
        usize::try_from(threads).expect("threads must be non-negative")
    } else {
        1
    };
    if omp_num_threads > 1 {
        let chunk_size = ((len / omp_num_threads) & !15usize).max(16);
        run_rayon_with_threads(omp_num_threads, || {
            sa[..len].par_chunks_mut(chunk_size).for_each(|chunk| {
                for value in chunk {
                    *value ^= SAINT_MIN;
                }
            });
        });
        return;
    }

    let omp_block_stride = (len / omp_num_threads) & !15usize;
    for omp_thread_num in 0..omp_num_threads {
        let omp_block_start = omp_thread_num * omp_block_stride;
        let omp_block_size = if omp_thread_num + 1 < omp_num_threads {
            omp_block_stride
        } else {
            len - omp_block_start
        };
        for value in &mut sa[omp_block_start..omp_block_start + omp_block_size] {
            *value ^= SAINT_MIN;
        }
    }
}

/// Internal helper: gather lms suffixes 8u.
#[doc(hidden)]
pub fn gather_lms_suffixes_8u(
    t: &[u8],
    sa: &mut [SaSint],
    n: SaSint,
    mut m: FastSint,
    omp_block_start: FastSint,
    omp_block_size: FastSint,
) {
    if omp_block_size <= 0 {
        return;
    }

    let n = usize::try_from(n).expect("n must be non-negative");
    let block_start =
        usize::try_from(omp_block_start).expect("omp_block_start must be non-negative");
    let block_size = usize::try_from(omp_block_size).expect("omp_block_size must be non-negative");

    let mut j = block_start + block_size;
    let mut c0 = t[block_start + block_size - 1] as FastSint;
    let mut c1 = -1;
    while j < n {
        c1 = t[j] as FastSint;
        if c1 != c0 {
            break;
        }
        j += 1;
    }

    let mut f0 = usize::from(c0 >= c1);
    let mut f1: usize;
    let mut i = block_start + block_size - 2;
    let limit = block_start + 3;

    while i >= limit {
        c1 = t[i] as FastSint;
        f1 = usize::from(c1 > (c0 - f0 as FastSint));
        sa[usize::try_from(m).expect("m must be non-negative")] = (i + 1) as SaSint;
        m -= (f1 & !f0) as FastSint;

        c0 = t[i - 1] as FastSint;
        f0 = usize::from(c0 > (c1 - f1 as FastSint));
        sa[usize::try_from(m).expect("m must be non-negative")] = i as SaSint;
        m -= (f0 & !f1) as FastSint;

        c1 = t[i - 2] as FastSint;
        f1 = usize::from(c1 > (c0 - f0 as FastSint));
        sa[usize::try_from(m).expect("m must be non-negative")] = (i - 1) as SaSint;
        m -= (f1 & !f0) as FastSint;

        c0 = t[i - 3] as FastSint;
        f0 = usize::from(c0 > (c1 - f1 as FastSint));
        sa[usize::try_from(m).expect("m must be non-negative")] = (i - 2) as SaSint;
        m -= (f0 & !f1) as FastSint;

        if i < 4 {
            break;
        }
        i -= 4;
    }

    let tail_limit = limit - 3;
    while i >= tail_limit {
        c1 = c0;
        c0 = t[i] as FastSint;
        f1 = f0;
        f0 = usize::from(c0 > (c1 - f1 as FastSint));
        sa[usize::try_from(m).expect("m must be non-negative")] = (i + 1) as SaSint;
        m -= (f0 & !f1) as FastSint;
        if i == 0 {
            break;
        }
        i -= 1;
    }

    sa[usize::try_from(m).expect("m must be non-negative")] = (i + 1) as SaSint;
}

/// Internal helper: gather lms suffixes 8u (OpenMP variant).
#[doc(hidden)]
pub fn gather_lms_suffixes_8u_omp(
    t: &[u8],
    sa: &mut [SaSint],
    n: SaSint,
    threads: SaSint,
    thread_state: &mut [ThreadState],
) {
    let n_usize = usize::try_from(n).expect("n must be non-negative");
    let omp_num_threads = if threads > 1 && n >= 65_536 {
        usize::try_from(threads)
            .expect("threads must be non-negative")
            .min(thread_state.len())
            .max(1)
    } else {
        1
    };
    if omp_num_threads == 1 {
        gather_lms_suffixes_8u(t, sa, n, n as FastSint - 1, 0, n as FastSint);
        return;
    }

    let omp_block_stride = (n_usize / omp_num_threads) & !15usize;
    let mut suffix_counts_after = vec![0 as FastSint; omp_num_threads];
    let mut m = 0 as FastSint;
    for omp_thread_num in (0..omp_num_threads).rev() {
        suffix_counts_after[omp_thread_num] = m;
        m += thread_state[omp_thread_num].m;
    }

    let sa_ptr = SyncMutPtr::new(sa);
    let suffix_counts_after_slice: &[FastSint] = &suffix_counts_after;
    let last_lms_suffixes: Vec<FastSint> = thread_state[..omp_num_threads]
        .iter()
        .map(|s| {
            if s.m > 0 {
                s.last_lms_suffix
            } else {
                FastSint::MIN
            }
        })
        .collect();

    run_rayon_with_threads(omp_num_threads, || {
        (0..omp_num_threads)
            .into_par_iter()
            .for_each(|omp_thread_num| {
                let omp_block_start = omp_thread_num * omp_block_stride;
                let omp_block_size = if omp_thread_num + 1 < omp_num_threads {
                    omp_block_stride
                } else {
                    n_usize - omp_block_start
                };
                // SAFETY: per-thread disjoint sa block plus a unique tail write.
                let sa = unsafe { sa_ptr.as_slice() };
                gather_lms_suffixes_8u(
                    t,
                    sa,
                    n,
                    n as FastSint - 1 - suffix_counts_after_slice[omp_thread_num],
                    omp_block_start as FastSint,
                    omp_block_size as FastSint,
                );
            });
    });

    for omp_thread_num in 0..omp_num_threads {
        if last_lms_suffixes[omp_thread_num] != FastSint::MIN {
            let dst = usize::try_from(n as FastSint - 1 - suffix_counts_after[omp_thread_num])
                .expect("destination must be non-negative");
            sa[dst] = last_lms_suffixes[omp_thread_num] as SaSint;
        }
    }
}

/// Internal helper: gather lms suffixes 32s.
#[doc(hidden)]
pub fn gather_lms_suffixes_32s(t: &[SaSint], sa: &mut [SaSint], n: SaSint) -> SaSint {
    let n_usize = usize::try_from(n).expect("n must be non-negative");
    let mut i = n as FastSint - 2;
    let mut m = n_usize - 1;
    let mut f0 = 1usize;
    let mut f1: usize;
    let mut c0 = t[n_usize - 1] as FastSint;
    let mut c1: FastSint;

    while i >= 3 {
        c1 = t[i as usize] as FastSint;
        f1 = usize::from(c1 > (c0 - f0 as FastSint));
        sa[m] = (i + 1) as SaSint;
        m -= f1 & !f0;

        c0 = t[(i - 1) as usize] as FastSint;
        f0 = usize::from(c0 > (c1 - f1 as FastSint));
        sa[m] = i as SaSint;
        m -= f0 & !f1;

        c1 = t[(i - 2) as usize] as FastSint;
        f1 = usize::from(c1 > (c0 - f0 as FastSint));
        sa[m] = (i - 1) as SaSint;
        m -= f1 & !f0;

        c0 = t[(i - 3) as usize] as FastSint;
        f0 = usize::from(c0 > (c1 - f1 as FastSint));
        sa[m] = (i - 2) as SaSint;
        m -= f0 & !f1;

        i -= 4;
    }

    while i >= 0 {
        c1 = c0;
        c0 = t[i as usize] as FastSint;
        f1 = f0;
        f0 = usize::from(c0 > (c1 - f1 as FastSint));
        sa[m] = (i + 1) as SaSint;
        m -= f0 & !f1;
        i -= 1;
    }

    (n_usize - 1 - m) as SaSint
}

/// Internal helper: gather compacted lms suffixes 32s.
#[doc(hidden)]
pub fn gather_compacted_lms_suffixes_32s(t: &[SaSint], sa: &mut [SaSint], n: SaSint) -> SaSint {
    let n_usize = usize::try_from(n).expect("n must be non-negative");
    let mut i = n as FastSint - 2;
    let mut m = n_usize - 1;
    let mut f0 = 1usize;
    let mut f1: usize;
    let mut c0 = t[n_usize - 1] as FastSint;
    let mut c1: FastSint;

    while i >= 3 {
        c1 = t[i as usize] as FastSint;
        f1 = usize::from(c1 > (c0 - f0 as FastSint));
        sa[m] = (i + 1) as SaSint;
        m -= f1 & !f0 & usize::from(c0 >= 0);

        c0 = t[(i - 1) as usize] as FastSint;
        f0 = usize::from(c0 > (c1 - f1 as FastSint));
        sa[m] = i as SaSint;
        m -= f0 & !f1 & usize::from(c1 >= 0);

        c1 = t[(i - 2) as usize] as FastSint;
        f1 = usize::from(c1 > (c0 - f0 as FastSint));
        sa[m] = (i - 1) as SaSint;
        m -= f1 & !f0 & usize::from(c0 >= 0);

        c0 = t[(i - 3) as usize] as FastSint;
        f0 = usize::from(c0 > (c1 - f1 as FastSint));
        sa[m] = (i - 2) as SaSint;
        m -= f0 & !f1 & usize::from(c1 >= 0);

        i -= 4;
    }

    while i >= 0 {
        c1 = c0;
        c0 = t[i as usize] as FastSint;
        f1 = f0;
        f0 = usize::from(c0 > (c1 - f1 as FastSint));
        sa[m] = (i + 1) as SaSint;
        m -= f0 & !f1 & usize::from(c1 >= 0);
        i -= 1;
    }

    (n_usize - 1 - m) as SaSint
}

/// Internal helper: count lms suffixes 32s 4k.
#[doc(hidden)]
pub fn count_lms_suffixes_32s_4k(t: &[SaSint], n: SaSint, k: SaSint, buckets: &mut [SaSint]) {
    buckets.fill(0);
    let n_usize = usize::try_from(n).expect("n must be non-negative");
    let _k_usize = usize::try_from(k).expect("k must be non-negative");
    let mut i = n as FastSint - 2;
    let mut f0 = 1usize;
    let mut f1: usize;
    let mut c0 = t[n_usize - 1] as FastSint;
    let mut c1: FastSint;

    while i >= 3 {
        c1 = t[i as usize] as FastSint;
        f1 = usize::from(c1 > (c0 - f0 as FastSint));
        buckets[buckets_index4(c0 as usize, f0 + f0 + f1)] += 1;

        c0 = t[(i - 1) as usize] as FastSint;
        f0 = usize::from(c0 > (c1 - f1 as FastSint));
        buckets[buckets_index4(c1 as usize, f1 + f1 + f0)] += 1;

        c1 = t[(i - 2) as usize] as FastSint;
        f1 = usize::from(c1 > (c0 - f0 as FastSint));
        buckets[buckets_index4(c0 as usize, f0 + f0 + f1)] += 1;

        c0 = t[(i - 3) as usize] as FastSint;
        f0 = usize::from(c0 > (c1 - f1 as FastSint));
        buckets[buckets_index4(c1 as usize, f1 + f1 + f0)] += 1;

        i -= 4;
    }

    while i >= 0 {
        c1 = c0;
        c0 = t[i as usize] as FastSint;
        f1 = f0;
        f0 = usize::from(c0 > (c1 - f1 as FastSint));
        buckets[buckets_index4(c1 as usize, f1 + f1 + f0)] += 1;
        i -= 1;
    }

    buckets[buckets_index4(c0 as usize, f0 + f0)] += 1;
}

/// Internal helper: count lms suffixes 32s 2k.
#[doc(hidden)]
pub fn count_lms_suffixes_32s_2k(t: &[SaSint], n: SaSint, k: SaSint, buckets: &mut [SaSint]) {
    buckets.fill(0);
    let n_usize = usize::try_from(n).expect("n must be non-negative");
    let _k_usize = usize::try_from(k).expect("k must be non-negative");
    let mut i = n as FastSint - 2;
    let mut f0 = 1usize;
    let mut f1: usize;
    let mut c0 = t[n_usize - 1] as FastSint;
    let mut c1: FastSint;

    while i >= 3 {
        c1 = t[i as usize] as FastSint;
        f1 = usize::from(c1 > (c0 - f0 as FastSint));
        buckets[buckets_index2(c0 as usize, f1 & !f0)] += 1;

        c0 = t[(i - 1) as usize] as FastSint;
        f0 = usize::from(c0 > (c1 - f1 as FastSint));
        buckets[buckets_index2(c1 as usize, f0 & !f1)] += 1;

        c1 = t[(i - 2) as usize] as FastSint;
        f1 = usize::from(c1 > (c0 - f0 as FastSint));
        buckets[buckets_index2(c0 as usize, f1 & !f0)] += 1;

        c0 = t[(i - 3) as usize] as FastSint;
        f0 = usize::from(c0 > (c1 - f1 as FastSint));
        buckets[buckets_index2(c1 as usize, f0 & !f1)] += 1;

        i -= 4;
    }

    while i >= 0 {
        c1 = c0;
        c0 = t[i as usize] as FastSint;
        f1 = f0;
        f0 = usize::from(c0 > (c1 - f1 as FastSint));
        buckets[buckets_index2(c1 as usize, f0 & !f1)] += 1;
        i -= 1;
    }

    buckets[buckets_index2(c0 as usize, 0)] += 1;
}

/// Internal helper: count compacted lms suffixes 32s 2k.
#[doc(hidden)]
pub fn count_compacted_lms_suffixes_32s_2k(
    t: &[SaSint],
    n: SaSint,
    k: SaSint,
    buckets: &mut [SaSint],
) {
    buckets.fill(0);
    let n_usize = usize::try_from(n).expect("n must be non-negative");
    let _k_usize = usize::try_from(k).expect("k must be non-negative");
    let mut i = n as FastSint - 2;
    let mut f0 = 1usize;
    let mut f1: usize;
    let mut c0 = t[n_usize - 1] as FastSint;
    let mut c1: FastSint;

    while i >= 3 {
        c1 = t[i as usize] as FastSint;
        f1 = usize::from(c1 > (c0 - f0 as FastSint));
        buckets[buckets_index2((c0 as SaSint & SAINT_MAX) as usize, f1 & !f0)] += 1;

        c0 = t[(i - 1) as usize] as FastSint;
        f0 = usize::from(c0 > (c1 - f1 as FastSint));
        buckets[buckets_index2((c1 as SaSint & SAINT_MAX) as usize, f0 & !f1)] += 1;

        c1 = t[(i - 2) as usize] as FastSint;
        f1 = usize::from(c1 > (c0 - f0 as FastSint));
        buckets[buckets_index2((c0 as SaSint & SAINT_MAX) as usize, f1 & !f0)] += 1;

        c0 = t[(i - 3) as usize] as FastSint;
        f0 = usize::from(c0 > (c1 - f1 as FastSint));
        buckets[buckets_index2((c1 as SaSint & SAINT_MAX) as usize, f0 & !f1)] += 1;

        i -= 4;
    }

    while i >= 0 {
        c1 = c0;
        c0 = t[i as usize] as FastSint;
        f1 = f0;
        f0 = usize::from(c0 > (c1 - f1 as FastSint));
        buckets[buckets_index2((c1 as SaSint & SAINT_MAX) as usize, f0 & !f1)] += 1;
        i -= 1;
    }

    buckets[buckets_index2((c0 as SaSint & SAINT_MAX) as usize, 0)] += 1;
}

/// Internal helper: count and gather lms suffixes 8u.
#[doc(hidden)]
pub fn count_and_gather_lms_suffixes_8u(
    t: &[u8],
    sa: &mut [SaSint],
    n: SaSint,
    buckets: &mut [SaSint],
    omp_block_start: FastSint,
    omp_block_size: FastSint,
) -> SaSint {
    buckets.fill(0);
    let n = n as FastSint;
    let mut m = omp_block_start + omp_block_size - 1;

    if omp_block_size > 0 {
        let prefetch_distance = 256 as FastSint;
        let mut j = m + 1;
        let mut c0 = t[m as usize] as FastSint;
        let mut c1 = -1;
        while j < n {
            c1 = t[j as usize] as FastSint;
            if c1 != c0 {
                break;
            }
            j += 1;
        }

        let mut f0 = usize::from(c0 >= c1);
        let mut f1: usize;
        let mut i = m - 1;
        let limit = omp_block_start + 3;

        while i >= limit {
            let _prefetch_index = i - prefetch_distance;
            c1 = t[i as usize] as FastSint;
            f1 = usize::from(c1 > (c0 - f0 as FastSint));
            sa[m as usize] = (i + 1) as SaSint;
            m -= (f1 & !f0) as FastSint;
            buckets[buckets_index4(c0 as usize, f0 + f0 + f1)] += 1;

            c0 = t[(i - 1) as usize] as FastSint;
            f0 = usize::from(c0 > (c1 - f1 as FastSint));
            sa[m as usize] = i as SaSint;
            m -= (f0 & !f1) as FastSint;
            buckets[buckets_index4(c1 as usize, f1 + f1 + f0)] += 1;

            c1 = t[(i - 2) as usize] as FastSint;
            f1 = usize::from(c1 > (c0 - f0 as FastSint));
            sa[m as usize] = (i - 1) as SaSint;
            m -= (f1 & !f0) as FastSint;
            buckets[buckets_index4(c0 as usize, f0 + f0 + f1)] += 1;

            c0 = t[(i - 3) as usize] as FastSint;
            f0 = usize::from(c0 > (c1 - f1 as FastSint));
            sa[m as usize] = (i - 2) as SaSint;
            m -= (f0 & !f1) as FastSint;
            buckets[buckets_index4(c1 as usize, f1 + f1 + f0)] += 1;

            i -= 4;
        }

        let tail_limit = limit - 3;
        while i >= tail_limit {
            c1 = c0;
            c0 = t[i as usize] as FastSint;
            f1 = f0;
            f0 = usize::from(c0 > (c1 - f1 as FastSint));
            sa[m as usize] = (i + 1) as SaSint;
            m -= (f0 & !f1) as FastSint;
            buckets[buckets_index4(c1 as usize, f1 + f1 + f0)] += 1;
            i -= 1;
        }

        c1 = if i >= 0 {
            t[i as usize] as FastSint
        } else {
            -1
        };
        f1 = usize::from(c1 > (c0 - f0 as FastSint));
        sa[m as usize] = (i + 1) as SaSint;
        m -= (f1 & !f0) as FastSint;
        buckets[buckets_index4(c0 as usize, f0 + f0 + f1)] += 1;
    }

    (omp_block_start + omp_block_size - 1 - m) as SaSint
}

/// Internal helper: count and gather lms suffixes 8u (OpenMP variant).
#[doc(hidden)]
pub fn count_and_gather_lms_suffixes_8u_omp(
    t: &[u8],
    sa: &mut [SaSint],
    n: SaSint,
    buckets: &mut [SaSint],
    threads: SaSint,
    thread_state: &mut [ThreadState],
) -> SaSint {
    let mut m = 0;
    let n_usize = usize::try_from(n).expect("n must be non-negative");
    let omp_num_threads = if threads > 1 && n >= 65_536 {
        usize::try_from(threads)
            .expect("threads must be non-negative")
            .min(thread_state.len())
            .max(1)
    } else {
        1
    };
    let omp_block_stride = (n_usize / omp_num_threads) & !15usize;

    if omp_num_threads == 1 {
        return count_and_gather_lms_suffixes_8u(t, sa, n, buckets, 0, n as FastSint);
    }

    let sa_ptr = SyncMutPtr::new(sa);
    run_rayon_with_threads(omp_num_threads, || {
        thread_state[..omp_num_threads]
            .par_iter_mut()
            .enumerate()
            .for_each(|(omp_thread_num, state)| {
                let omp_block_start = omp_thread_num * omp_block_stride;
                let omp_block_size = if omp_thread_num + 1 < omp_num_threads {
                    omp_block_stride
                } else {
                    n_usize - omp_block_start
                };

                // SAFETY: each thread writes only to sa[omp_block_start..+omp_block_size].
                let sa = unsafe { sa_ptr.as_slice() };

                state.position = FastSint::try_from(omp_block_start + omp_block_size)
                    .expect("position must fit FastSint");
                state.m = FastSint::try_from(count_and_gather_lms_suffixes_8u(
                    t,
                    sa,
                    n,
                    &mut state.buckets,
                    FastSint::try_from(omp_block_start).expect("block start must fit FastSint"),
                    FastSint::try_from(omp_block_size).expect("block size must fit FastSint"),
                ))
                .expect("m must fit FastSint");

                if state.m > 0 {
                    let position =
                        usize::try_from(state.position).expect("position must be non-negative");
                    state.last_lms_suffix =
                        FastSint::try_from(sa[position - 1]).expect("suffix must fit FastSint");
                }
            });
    });

    buckets.fill(0);

    for tnum in (0..omp_num_threads).rev() {
        let state = &mut thread_state[tnum];
        m += SaSint::try_from(state.m).expect("m must fit SaSint");

        if tnum + 1 < omp_num_threads && state.m > 0 {
            let position = usize::try_from(state.position).expect("position must be non-negative");
            let count = usize::try_from(state.m).expect("m must be non-negative");
            let dst = n_usize - usize::try_from(m).expect("m must be non-negative");
            sa.copy_within(position - count..position, dst);
        }

        for s in 0..4 * ALPHABET_SIZE {
            let a = buckets[s];
            let b = state.buckets[s];
            buckets[s] = a + b;
            state.buckets[s] = a;
        }
    }

    m
}

/// Internal helper: count and gather lms suffixes 32s 4k.
#[doc(hidden)]
pub fn count_and_gather_lms_suffixes_32s_4k(
    t: &[SaSint],
    sa: &mut [SaSint],
    n: SaSint,
    k: SaSint,
    buckets: &mut [SaSint],
    omp_block_start: FastSint,
    omp_block_size: FastSint,
) -> SaSint {
    buckets.fill(0);
    let n = n as FastSint;
    let _k = k as FastSint;
    let mut m = omp_block_start + omp_block_size - 1;

    if omp_block_size > 0 {
        let prefetch_distance = 64 as FastSint;
        let mut j = m + 1;
        let mut c0 = t[m as usize] as FastSint;
        let mut c1 = -1;

        while j < n {
            c1 = t[j as usize] as FastSint;
            if c1 != c0 {
                break;
            }
            j += 1;
        }

        let mut f0 = usize::from(c0 >= c1);
        let mut f1: usize;
        let mut i = m - 1;
        let limit = omp_block_start + prefetch_distance + 3;

        while i >= limit {
            let _prefetch_index = i - 2 * prefetch_distance;
            c1 = t[i as usize] as FastSint;
            f1 = usize::from(c1 > (c0 - f0 as FastSint));
            sa[m as usize] = (i + 1) as SaSint;
            m -= (f1 & !f0) as FastSint;
            buckets[buckets_index4(c0 as usize, f0 + f0 + f1)] += 1;

            c0 = t[(i - 1) as usize] as FastSint;
            f0 = usize::from(c0 > (c1 - f1 as FastSint));
            sa[m as usize] = i as SaSint;
            m -= (f0 & !f1) as FastSint;
            buckets[buckets_index4(c1 as usize, f1 + f1 + f0)] += 1;

            c1 = t[(i - 2) as usize] as FastSint;
            f1 = usize::from(c1 > (c0 - f0 as FastSint));
            sa[m as usize] = (i - 1) as SaSint;
            m -= (f1 & !f0) as FastSint;
            buckets[buckets_index4(c0 as usize, f0 + f0 + f1)] += 1;

            c0 = t[(i - 3) as usize] as FastSint;
            f0 = usize::from(c0 > (c1 - f1 as FastSint));
            sa[m as usize] = (i - 2) as SaSint;
            m -= (f0 & !f1) as FastSint;
            buckets[buckets_index4(c1 as usize, f1 + f1 + f0)] += 1;

            i -= 4;
        }

        let tail_limit = omp_block_start;
        while i >= tail_limit {
            c1 = c0;
            c0 = t[i as usize] as FastSint;
            f1 = f0;
            f0 = usize::from(c0 > (c1 - f1 as FastSint));
            sa[m as usize] = (i + 1) as SaSint;
            m -= (f0 & !f1) as FastSint;
            buckets[buckets_index4(c1 as usize, f1 + f1 + f0)] += 1;
            i -= 1;
        }

        c1 = if i >= 0 {
            t[i as usize] as FastSint
        } else {
            -1
        };
        f1 = usize::from(c1 > (c0 - f0 as FastSint));
        sa[m as usize] = (i + 1) as SaSint;
        m -= (f1 & !f0) as FastSint;
        buckets[buckets_index4(c0 as usize, f0 + f0 + f1)] += 1;
    }

    (omp_block_start + omp_block_size - 1 - m) as SaSint
}

/// Internal helper: count and gather lms suffixes 32s 2k.
#[doc(hidden)]
pub fn count_and_gather_lms_suffixes_32s_2k(
    t: &[SaSint],
    sa: &mut [SaSint],
    n: SaSint,
    k: SaSint,
    buckets: &mut [SaSint],
    omp_block_start: FastSint,
    omp_block_size: FastSint,
) -> SaSint {
    buckets.fill(0);
    let n = n as FastSint;
    let _k = k as FastSint;
    let mut m = omp_block_start + omp_block_size - 1;

    if omp_block_size > 0 {
        let prefetch_distance = 64 as FastSint;
        let mut j = m + 1;
        let mut c0 = t[m as usize] as FastSint;
        let mut c1 = -1;

        while j < n {
            c1 = t[j as usize] as FastSint;
            if c1 != c0 {
                break;
            }
            j += 1;
        }

        let mut f0 = usize::from(c0 >= c1);
        let mut f1: usize;
        let mut i = m - 1;
        let limit = omp_block_start + prefetch_distance + 3;

        while i >= limit {
            let _prefetch_index = i - 2 * prefetch_distance;
            c1 = t[i as usize] as FastSint;
            f1 = usize::from(c1 > (c0 - f0 as FastSint));
            sa[m as usize] = (i + 1) as SaSint;
            m -= (f1 & !f0) as FastSint;
            buckets[buckets_index2(c0 as usize, f1 & !f0)] += 1;

            c0 = t[(i - 1) as usize] as FastSint;
            f0 = usize::from(c0 > (c1 - f1 as FastSint));
            sa[m as usize] = i as SaSint;
            m -= (f0 & !f1) as FastSint;
            buckets[buckets_index2(c1 as usize, f0 & !f1)] += 1;

            c1 = t[(i - 2) as usize] as FastSint;
            f1 = usize::from(c1 > (c0 - f0 as FastSint));
            sa[m as usize] = (i - 1) as SaSint;
            m -= (f1 & !f0) as FastSint;
            buckets[buckets_index2(c0 as usize, f1 & !f0)] += 1;

            c0 = t[(i - 3) as usize] as FastSint;
            f0 = usize::from(c0 > (c1 - f1 as FastSint));
            sa[m as usize] = (i - 2) as SaSint;
            m -= (f0 & !f1) as FastSint;
            buckets[buckets_index2(c1 as usize, f0 & !f1)] += 1;

            i -= 4;
        }

        let tail_limit = omp_block_start;
        while i >= tail_limit {
            c1 = c0;
            c0 = t[i as usize] as FastSint;
            f1 = f0;
            f0 = usize::from(c0 > (c1 - f1 as FastSint));
            sa[m as usize] = (i + 1) as SaSint;
            m -= (f0 & !f1) as FastSint;
            buckets[buckets_index2(c1 as usize, f0 & !f1)] += 1;
            i -= 1;
        }

        c1 = if i >= 0 {
            t[i as usize] as FastSint
        } else {
            -1
        };
        f1 = usize::from(c1 > (c0 - f0 as FastSint));
        sa[m as usize] = (i + 1) as SaSint;
        m -= (f1 & !f0) as FastSint;
        buckets[buckets_index2(c0 as usize, f1 & !f0)] += 1;
    }

    (omp_block_start + omp_block_size - 1 - m) as SaSint
}

/// Internal helper: count and gather compacted lms suffixes 32s 2k.
#[doc(hidden)]
pub fn count_and_gather_compacted_lms_suffixes_32s_2k(
    t: &[SaSint],
    sa: &mut [SaSint],
    n: SaSint,
    k: SaSint,
    buckets: &mut [SaSint],
    omp_block_start: FastSint,
    omp_block_size: FastSint,
) -> SaSint {
    buckets.fill(0);
    let n_usize = usize::try_from(n).expect("n must be non-negative");
    let _k_usize = usize::try_from(k).expect("k must be non-negative");
    let block_start =
        usize::try_from(omp_block_start).expect("omp_block_start must be non-negative");
    let block_size = usize::try_from(omp_block_size).expect("omp_block_size must be non-negative");
    let mut m = block_start + block_size - 1;

    if omp_block_size > 0 {
        let mut j = m + 1;
        let mut c0 = t[m] as FastSint;
        let mut c1 = -1;

        while j < n_usize {
            c1 = t[j] as FastSint;
            if c1 != c0 {
                break;
            }
            j += 1;
        }

        let mut f0 = usize::from(c0 >= c1);
        let mut f1: usize;
        let mut i = m as FastSint - 1;
        let limit = block_start as FastSint + 3;

        while i >= limit {
            c1 = t[i as usize] as FastSint;
            f1 = usize::from(c1 > (c0 - f0 as FastSint));
            sa[m] = (i + 1) as SaSint;
            m -= f1 & !f0 & usize::from(c0 >= 0);
            buckets[buckets_index2((c0 as SaSint & SAINT_MAX) as usize, f1 & !f0)] += 1;

            c0 = t[(i - 1) as usize] as FastSint;
            f0 = usize::from(c0 > (c1 - f1 as FastSint));
            sa[m] = i as SaSint;
            m -= f0 & !f1 & usize::from(c1 >= 0);
            buckets[buckets_index2((c1 as SaSint & SAINT_MAX) as usize, f0 & !f1)] += 1;

            c1 = t[(i - 2) as usize] as FastSint;
            f1 = usize::from(c1 > (c0 - f0 as FastSint));
            sa[m] = (i - 1) as SaSint;
            m -= f1 & !f0 & usize::from(c0 >= 0);
            buckets[buckets_index2((c0 as SaSint & SAINT_MAX) as usize, f1 & !f0)] += 1;

            c0 = t[(i - 3) as usize] as FastSint;
            f0 = usize::from(c0 > (c1 - f1 as FastSint));
            sa[m] = (i - 2) as SaSint;
            m -= f0 & !f1 & usize::from(c1 >= 0);
            buckets[buckets_index2((c1 as SaSint & SAINT_MAX) as usize, f0 & !f1)] += 1;

            i -= 4;
        }

        let tail_limit = block_start as FastSint;
        while i >= tail_limit {
            c1 = c0;
            c0 = t[i as usize] as FastSint;
            f1 = f0;
            f0 = usize::from(c0 > (c1 - f1 as FastSint));
            sa[m] = (i + 1) as SaSint;
            m -= f0 & !f1 & usize::from(c1 >= 0);
            buckets[buckets_index2((c1 as SaSint & SAINT_MAX) as usize, f0 & !f1)] += 1;
            i -= 1;
        }

        c1 = if i >= 0 {
            t[i as usize] as FastSint
        } else {
            -1
        };
        f1 = usize::from(c1 > (c0 - f0 as FastSint));
        sa[m] = (i + 1) as SaSint;
        m -= f1 & !f0 & usize::from(c0 >= 0);
        buckets[buckets_index2((c0 as SaSint & SAINT_MAX) as usize, f1 & !f0)] += 1;
    }

    (block_start + block_size - 1 - m) as SaSint
}

/// Internal helper: get bucket stride.
#[doc(hidden)]
pub fn get_bucket_stride(
    free_space: FastSint,
    bucket_size: FastSint,
    num_buckets: FastSint,
) -> FastSint {
    let bucket_size_1024 = (bucket_size + 1023) & (-1024);
    if free_space / (num_buckets - 1) >= bucket_size_1024 {
        return bucket_size_1024;
    }
    let bucket_size_16 = (bucket_size + 15) & (-16);
    if free_space / (num_buckets - 1) >= bucket_size_16 {
        return bucket_size_16;
    }
    bucket_size
}

/// Internal helper: count and gather lms suffixes 32s 4k nofs (OpenMP variant).
#[doc(hidden)]
pub fn count_and_gather_lms_suffixes_32s_4k_nofs_omp(
    t: &[SaSint],
    sa: &mut [SaSint],
    n: SaSint,
    k: SaSint,
    buckets: &mut [SaSint],
    threads: SaSint,
) -> SaSint {
    let m;
    let omp_num_threads = if threads > 1 && n >= 65_536 { 2 } else { 1 };

    if omp_num_threads == 1 {
        m = count_and_gather_lms_suffixes_32s_4k(t, sa, n, k, buckets, 0, n as FastSint);
    } else {
        count_lms_suffixes_32s_4k(t, n, k, buckets);
        m = gather_lms_suffixes_32s(t, sa, n);
    }

    m
}

/// Internal helper: count and gather lms suffixes 32s 2k nofs (OpenMP variant).
#[doc(hidden)]
pub fn count_and_gather_lms_suffixes_32s_2k_nofs_omp(
    t: &[SaSint],
    sa: &mut [SaSint],
    n: SaSint,
    k: SaSint,
    buckets: &mut [SaSint],
    threads: SaSint,
) -> SaSint {
    let m;
    let omp_num_threads = if threads > 1 && n >= 65_536 { 2 } else { 1 };

    if omp_num_threads == 1 {
        m = count_and_gather_lms_suffixes_32s_2k(t, sa, n, k, buckets, 0, n as FastSint);
    } else {
        count_lms_suffixes_32s_2k(t, n, k, buckets);
        m = gather_lms_suffixes_32s(t, sa, n);
    }

    m
}

/// Internal helper: count and gather compacted lms suffixes 32s 2k nofs (OpenMP variant).
#[doc(hidden)]
pub fn count_and_gather_compacted_lms_suffixes_32s_2k_nofs_omp(
    t: &[SaSint],
    sa: &mut [SaSint],
    n: SaSint,
    k: SaSint,
    buckets: &mut [SaSint],
    threads: SaSint,
) -> SaSint {
    let m;
    let omp_num_threads = if threads > 1 && n >= 65_536 { 2 } else { 1 };

    if omp_num_threads == 1 {
        m = count_and_gather_compacted_lms_suffixes_32s_2k(t, sa, n, k, buckets, 0, n as FastSint);
    } else {
        count_compacted_lms_suffixes_32s_2k(t, n, k, buckets);
        m = gather_compacted_lms_suffixes_32s(t, sa, n);
    }

    m
}

/// Internal helper: count and gather lms suffixes 32s 4k fs (OpenMP variant).
#[doc(hidden)]
pub fn count_and_gather_lms_suffixes_32s_4k_fs_omp(
    t: &[SaSint],
    sa: &mut [SaSint],
    n: SaSint,
    k: SaSint,
    buckets: &mut [SaSint],
    local_buckets: SaSint,
    threads: SaSint,
    thread_state: &mut [ThreadState],
) -> SaSint {
    let n_usize = usize::try_from(n).expect("n must be non-negative");
    let k_usize = usize::try_from(k).expect("k must be non-negative");
    let omp_num_threads = usize::try_from(threads).expect("threads must be non-negative");
    let bucket_size = FastSint::try_from(4 * k_usize).expect("bucket size must fit FastSint");

    if omp_num_threads <= 1 || n < 65_536 {
        return count_and_gather_lms_suffixes_32s_4k(t, sa, n, k, buckets, 0, n as FastSint);
    }

    let omp_block_stride = (n_usize / omp_num_threads) & !15usize;
    let free_space = if local_buckets == 1 {
        FastSint::try_from(LIBSAIS_LOCAL_BUFFER_SIZE).expect("free space must fit FastSint")
    } else if local_buckets > 1 {
        FastSint::try_from(local_buckets).expect("free space must fit FastSint")
    } else {
        FastSint::try_from(buckets.len()).expect("free space must fit FastSint")
    };
    let bucket_stride = get_bucket_stride(
        free_space,
        bucket_size,
        FastSint::try_from(omp_num_threads).expect("thread count must fit FastSint"),
    );
    let bucket_size_usize = usize::try_from(bucket_size).expect("bucket size must be non-negative");
    let bucket_stride_usize =
        usize::try_from(bucket_stride).expect("bucket stride must be non-negative");
    let workspace_len =
        bucket_size_usize + bucket_stride_usize.saturating_mul(omp_num_threads.saturating_sub(1));
    let mut workspace = vec![0; workspace_len];

    {
        let sa_ptr = SyncMutPtr::new(sa);
        let ws_ptr = SyncMutPtr::new(&mut workspace);
        run_rayon_with_threads(omp_num_threads, || {
            thread_state[..omp_num_threads]
                .par_iter_mut()
                .enumerate()
                .for_each(|(omp_thread_num, state)| {
                    let omp_block_start = omp_thread_num * omp_block_stride;
                    let omp_block_size = if omp_thread_num + 1 < omp_num_threads {
                        omp_block_stride
                    } else {
                        n_usize - omp_block_start
                    };
                    let workspace_end = workspace_len - omp_thread_num * bucket_stride_usize;
                    let workspace_start = workspace_end - bucket_size_usize;
                    // SAFETY: per-thread disjoint sa and workspace ranges.
                    let sa = unsafe { sa_ptr.as_slice() };
                    let workspace = unsafe { ws_ptr.as_slice() };
                    let count = count_and_gather_lms_suffixes_32s_4k(
                        t,
                        sa,
                        n,
                        k,
                        &mut workspace[workspace_start..workspace_end],
                        omp_block_start as FastSint,
                        omp_block_size as FastSint,
                    );
                    state.position = (omp_block_start + omp_block_size) as FastSint;
                    state.count = count as FastSint;
                });
        });
    }

    let mut m = 0;
    for t in (0..omp_num_threads).rev() {
        m += thread_state[t].count as SaSint;

        if t + 1 != omp_num_threads && thread_state[t].count > 0 {
            let src_end =
                usize::try_from(thread_state[t].position).expect("position must be non-negative");
            let src_start = src_end
                - usize::try_from(thread_state[t].count).expect("count must be non-negative");
            let dst_start = usize::try_from(n - m).expect("destination must be non-negative");
            sa.copy_within(src_start..src_end, dst_start);
        }
    }

    let omp_num_threads = omp_num_threads - 1;
    let omp_block_stride = (bucket_size_usize / omp_num_threads) & !15usize;
    {
        let ws_ptr = SyncMutPtr::new(&mut workspace);
        run_rayon_with_threads(omp_num_threads, || {
            (0..omp_num_threads)
                .into_par_iter()
                .for_each(|omp_thread_num| {
                    let omp_block_start = omp_thread_num * omp_block_stride;
                    let omp_block_size = if omp_thread_num + 1 < omp_num_threads {
                        omp_block_stride
                    } else {
                        bucket_size_usize - omp_block_start
                    };
                    // SAFETY: per-thread disjoint absolute write ranges.
                    let workspace = unsafe { ws_ptr.as_slice() };
                    accumulate_counts_s32(
                        &mut workspace[omp_block_start..],
                        omp_block_size as FastSint,
                        bucket_stride,
                        FastSint::try_from(omp_num_threads + 1)
                            .expect("thread count must fit FastSint"),
                    );
                });
        });
    }

    let accumulated_start = omp_num_threads * bucket_stride_usize;
    buckets[..bucket_size_usize]
        .copy_from_slice(&workspace[accumulated_start..accumulated_start + bucket_size_usize]);
    m
}

/// Internal helper: count and gather lms suffixes 32s 2k fs (OpenMP variant).
#[doc(hidden)]
pub fn count_and_gather_lms_suffixes_32s_2k_fs_omp(
    t: &[SaSint],
    sa: &mut [SaSint],
    n: SaSint,
    k: SaSint,
    buckets: &mut [SaSint],
    local_buckets: SaSint,
    threads: SaSint,
    thread_state: &mut [ThreadState],
) -> SaSint {
    let n_usize = usize::try_from(n).expect("n must be non-negative");
    let k_usize = usize::try_from(k).expect("k must be non-negative");
    let omp_num_threads = usize::try_from(threads).expect("threads must be non-negative");
    let bucket_size = FastSint::try_from(2 * k_usize).expect("bucket size must fit FastSint");

    if omp_num_threads <= 1 || n < 65_536 {
        return count_and_gather_lms_suffixes_32s_2k(t, sa, n, k, buckets, 0, n as FastSint);
    }

    let omp_block_stride = (n_usize / omp_num_threads) & !15usize;
    let free_space = if local_buckets == 1 {
        FastSint::try_from(LIBSAIS_LOCAL_BUFFER_SIZE).expect("free space must fit FastSint")
    } else if local_buckets > 1 {
        FastSint::try_from(local_buckets).expect("free space must fit FastSint")
    } else {
        FastSint::try_from(buckets.len()).expect("free space must fit FastSint")
    };
    let bucket_stride = get_bucket_stride(
        free_space,
        bucket_size,
        FastSint::try_from(omp_num_threads).expect("thread count must fit FastSint"),
    );
    let bucket_size_usize = usize::try_from(bucket_size).expect("bucket size must be non-negative");
    let bucket_stride_usize =
        usize::try_from(bucket_stride).expect("bucket stride must be non-negative");
    let workspace_len =
        bucket_size_usize + bucket_stride_usize.saturating_mul(omp_num_threads.saturating_sub(1));
    let mut workspace = vec![0; workspace_len];

    {
        let sa_ptr = SyncMutPtr::new(sa);
        let ws_ptr = SyncMutPtr::new(&mut workspace);
        run_rayon_with_threads(omp_num_threads, || {
            thread_state[..omp_num_threads]
                .par_iter_mut()
                .enumerate()
                .for_each(|(omp_thread_num, state)| {
                    let omp_block_start = omp_thread_num * omp_block_stride;
                    let omp_block_size = if omp_thread_num + 1 < omp_num_threads {
                        omp_block_stride
                    } else {
                        n_usize - omp_block_start
                    };
                    let workspace_end = workspace_len - omp_thread_num * bucket_stride_usize;
                    let workspace_start = workspace_end - bucket_size_usize;
                    // SAFETY: per-thread disjoint sa and workspace ranges.
                    let sa = unsafe { sa_ptr.as_slice() };
                    let workspace = unsafe { ws_ptr.as_slice() };
                    let count = count_and_gather_lms_suffixes_32s_2k(
                        t,
                        sa,
                        n,
                        k,
                        &mut workspace[workspace_start..workspace_end],
                        omp_block_start as FastSint,
                        omp_block_size as FastSint,
                    );
                    state.position = (omp_block_start + omp_block_size) as FastSint;
                    state.count = count as FastSint;
                });
        });
    }

    let mut m = 0;
    for t in (0..omp_num_threads).rev() {
        m += thread_state[t].count as SaSint;
        if t + 1 != omp_num_threads && thread_state[t].count > 0 {
            let src_end =
                usize::try_from(thread_state[t].position).expect("position must be non-negative");
            let src_start = src_end
                - usize::try_from(thread_state[t].count).expect("count must be non-negative");
            let dst_start = usize::try_from(n - m).expect("destination must be non-negative");
            sa.copy_within(src_start..src_end, dst_start);
        }
    }

    let omp_num_threads = omp_num_threads - 1;
    let omp_block_stride = (bucket_size_usize / omp_num_threads) & !15usize;
    {
        let ws_ptr = SyncMutPtr::new(&mut workspace);
        run_rayon_with_threads(omp_num_threads, || {
            (0..omp_num_threads)
                .into_par_iter()
                .for_each(|omp_thread_num| {
                    let omp_block_start = omp_thread_num * omp_block_stride;
                    let omp_block_size = if omp_thread_num + 1 < omp_num_threads {
                        omp_block_stride
                    } else {
                        bucket_size_usize - omp_block_start
                    };
                    // SAFETY: per-thread disjoint absolute write ranges.
                    let workspace = unsafe { ws_ptr.as_slice() };
                    accumulate_counts_s32(
                        &mut workspace[omp_block_start..],
                        omp_block_size as FastSint,
                        bucket_stride,
                        FastSint::try_from(omp_num_threads + 1)
                            .expect("thread count must fit FastSint"),
                    );
                });
        });
    }

    let accumulated_start = omp_num_threads * bucket_stride_usize;
    buckets[..bucket_size_usize]
        .copy_from_slice(&workspace[accumulated_start..accumulated_start + bucket_size_usize]);
    m
}

/// Internal helper: count and gather compacted lms suffixes 32s 2k fs (OpenMP variant).
#[doc(hidden)]
pub fn count_and_gather_compacted_lms_suffixes_32s_2k_fs_omp(
    t: &[SaSint],
    sa: &mut [SaSint],
    n: SaSint,
    k: SaSint,
    buckets: &mut [SaSint],
    _local_buckets: SaSint,
    threads: SaSint,
    thread_state: &mut [ThreadState],
) {
    let n_usize = usize::try_from(n).expect("n must be non-negative");
    let k_usize = usize::try_from(k).expect("k must be non-negative");
    let thread_count = usize::try_from(threads).expect("threads must be non-negative");
    let bucket_size = 2 * k_usize;

    if thread_count <= 1 || n < 65_536 {
        let _ =
            count_and_gather_compacted_lms_suffixes_32s_2k(t, sa, n, k, buckets, 0, n as FastSint);
        return;
    }

    if thread_state.len() < thread_count || sa.len() < 2 * n_usize {
        let _ =
            count_and_gather_compacted_lms_suffixes_32s_2k_nofs_omp(t, sa, n, k, buckets, threads);
        return;
    }

    let omp_block_stride = (n_usize / thread_count) & !15usize;
    let free_space = if _local_buckets != 0 {
        FastSint::try_from(LIBSAIS_LOCAL_BUFFER_SIZE).expect("free space must fit FastSint")
    } else {
        FastSint::try_from(buckets.len()).expect("free space must fit FastSint")
    };
    let bucket_stride = get_bucket_stride(
        free_space,
        FastSint::try_from(bucket_size).expect("bucket size must fit FastSint"),
        FastSint::try_from(thread_count).expect("thread count must fit FastSint"),
    );
    let bucket_stride_usize =
        usize::try_from(bucket_stride).expect("bucket stride must be non-negative");
    let workspace_len =
        bucket_size + bucket_stride_usize.saturating_mul(thread_count.saturating_sub(1));
    let mut workspace = vec![0; workspace_len];

    let usable_thread_state_len = thread_count.min(thread_state.len());

    {
        let sa_ptr = SyncMutPtr::new(sa);
        let ws_ptr = SyncMutPtr::new(&mut workspace);
        let state_ptr = SyncMutPtr::new(thread_state);
        run_rayon_with_threads(thread_count, || {
            (0..thread_count)
                .into_par_iter()
                .for_each(|omp_thread_num| {
                    let omp_block_start = omp_thread_num * omp_block_stride;
                    let omp_block_size = if omp_thread_num + 1 < thread_count {
                        omp_block_stride
                    } else {
                        n_usize - omp_block_start
                    };

                    let workspace_end = workspace_len - omp_thread_num * bucket_stride_usize;
                    let workspace_start = workspace_end - bucket_size;
                    // SAFETY: per-thread disjoint sa block, workspace range, and thread_state index.
                    let sa = unsafe { sa_ptr.as_slice() };
                    let workspace = unsafe { ws_ptr.as_slice() };
                    let count = count_and_gather_compacted_lms_suffixes_32s_2k(
                        t,
                        &mut sa[n_usize..],
                        n,
                        k,
                        &mut workspace[workspace_start..workspace_end],
                        omp_block_start as FastSint,
                        omp_block_size as FastSint,
                    );

                    if omp_thread_num < usable_thread_state_len {
                        let states = unsafe { state_ptr.as_slice() };
                        states[omp_thread_num].position =
                            (omp_block_start + omp_block_size) as FastSint;
                        states[omp_thread_num].count = count as FastSint;
                    }
                });
        });
    }

    let mut m = 0usize;
    for omp_thread_num in (0..thread_count).rev() {
        let count = usize::try_from(thread_state[omp_thread_num].count)
            .expect("count must be non-negative");
        m += count;
        if count > 0 {
            let position = usize::try_from(thread_state[omp_thread_num].position)
                .expect("position must be non-negative");
            let src_start = n_usize + position - count;
            let src_end = n_usize + position;
            let dst_start = n_usize - m;
            sa.copy_within(src_start..src_end, dst_start);
        }
    }

    let accumulation_threads = thread_count;
    let omp_block_stride = (bucket_size / accumulation_threads) & !15usize;
    {
        let ws_ptr = SyncMutPtr::new(&mut workspace);
        run_rayon_with_threads(accumulation_threads, || {
            (0..accumulation_threads)
                .into_par_iter()
                .for_each(|omp_thread_num| {
                    let omp_block_start = omp_thread_num * omp_block_stride;
                    let omp_block_size = if omp_thread_num + 1 < accumulation_threads {
                        omp_block_stride
                    } else {
                        bucket_size - omp_block_start
                    };
                    // SAFETY: per-thread disjoint absolute write ranges.
                    let workspace = unsafe { ws_ptr.as_slice() };
                    accumulate_counts_s32(
                        &mut workspace[omp_block_start..],
                        omp_block_size as FastSint,
                        bucket_stride,
                        FastSint::try_from(thread_count).expect("thread count must fit FastSint"),
                    );
                });
        });
    }
    let accumulated_start = (accumulation_threads - 1) * bucket_stride_usize;
    buckets[..bucket_size]
        .copy_from_slice(&workspace[accumulated_start..accumulated_start + bucket_size]);
}

/// Internal helper: count and gather lms suffixes 32s 4k (OpenMP variant).
#[doc(hidden)]
pub fn count_and_gather_lms_suffixes_32s_4k_omp(
    t: &[SaSint],
    sa: &mut [SaSint],
    n: SaSint,
    k: SaSint,
    buckets: &mut [SaSint],
    local_buckets: SaSint,
    threads: SaSint,
    thread_state: &mut [ThreadState],
) -> SaSint {
    let free_space = if local_buckets != 0 {
        LIBSAIS_LOCAL_BUFFER_SIZE as FastSint
    } else {
        FastSint::try_from(buckets.len()).expect("bucket length must fit FastSint")
    };
    let threads_fast = threads as FastSint;
    let mut max_threads = (free_space / (((4 * k as FastSint) + 15) & -16)).min(threads_fast);

    if max_threads > 1 && n >= 65_536 && n / k >= 2 {
        let thread_cap = (n / (16 * k)) as FastSint;
        if max_threads > thread_cap {
            max_threads = thread_cap;
        }
        return count_and_gather_lms_suffixes_32s_4k_fs_omp(
            t,
            sa,
            n,
            k,
            buckets,
            local_buckets,
            max_threads.max(2) as SaSint,
            thread_state,
        );
    }

    if threads > 1 && n >= 65_536 {
        count_lms_suffixes_32s_4k(t, n, k, buckets);
        gather_lms_suffixes_32s(t, sa, n)
    } else {
        count_and_gather_lms_suffixes_32s_4k(t, sa, n, k, buckets, 0, n as FastSint)
    }
}

/// Internal helper: count and gather lms suffixes 32s 2k (OpenMP variant).
#[doc(hidden)]
pub fn count_and_gather_lms_suffixes_32s_2k_omp(
    t: &[SaSint],
    sa: &mut [SaSint],
    n: SaSint,
    k: SaSint,
    buckets: &mut [SaSint],
    local_buckets: SaSint,
    threads: SaSint,
    thread_state: &mut [ThreadState],
) -> SaSint {
    let free_space = if local_buckets != 0 {
        LIBSAIS_LOCAL_BUFFER_SIZE as FastSint
    } else {
        FastSint::try_from(buckets.len()).expect("bucket length must fit FastSint")
    };
    let threads_fast = threads as FastSint;
    let mut max_threads = (free_space / (((2 * k as FastSint) + 15) & -16)).min(threads_fast);

    if max_threads > 1 && n >= 65_536 && n / k >= 2 {
        let thread_cap = (n / (8 * k)) as FastSint;
        if max_threads > thread_cap {
            max_threads = thread_cap;
        }
        return count_and_gather_lms_suffixes_32s_2k_fs_omp(
            t,
            sa,
            n,
            k,
            buckets,
            local_buckets,
            max_threads.max(2) as SaSint,
            thread_state,
        );
    }

    if threads > 1 && n >= 65_536 {
        count_lms_suffixes_32s_2k(t, n, k, buckets);
        gather_lms_suffixes_32s(t, sa, n)
    } else {
        count_and_gather_lms_suffixes_32s_2k(t, sa, n, k, buckets, 0, n as FastSint)
    }
}

/// Internal helper: count and gather compacted lms suffixes 32s 2k (OpenMP variant).
#[doc(hidden)]
pub fn count_and_gather_compacted_lms_suffixes_32s_2k_omp(
    t: &[SaSint],
    sa: &mut [SaSint],
    n: SaSint,
    k: SaSint,
    buckets: &mut [SaSint],
    local_buckets: SaSint,
    threads: SaSint,
    thread_state: &mut [ThreadState],
) {
    let free_space = if local_buckets != 0 {
        LIBSAIS_LOCAL_BUFFER_SIZE as FastSint
    } else {
        FastSint::try_from(buckets.len()).expect("bucket length must fit FastSint")
    };
    let threads_fast = threads as FastSint;
    let mut max_threads = (free_space / (((2 * k as FastSint) + 15) & -16)).min(threads_fast);

    if local_buckets == 0 && max_threads > 1 && n >= 65_536 && n / k >= 2 {
        let thread_cap = (n / (8 * k)) as FastSint;
        if max_threads > thread_cap {
            max_threads = thread_cap;
        }
        count_and_gather_compacted_lms_suffixes_32s_2k_fs_omp(
            t,
            sa,
            n,
            k,
            buckets,
            local_buckets,
            max_threads.max(2) as SaSint,
            thread_state,
        );
        return;
    }

    let _ = count_and_gather_compacted_lms_suffixes_32s_2k_nofs_omp(t, sa, n, k, buckets, threads);
}

/// Internal helper: count suffixes 32s.
#[doc(hidden)]
pub fn count_suffixes_32s(t: &[SaSint], n: SaSint, k: SaSint, buckets: &mut [SaSint]) {
    let n_usize = usize::try_from(n).expect("n must be non-negative");
    let k_usize = usize::try_from(k).expect("k must be non-negative");
    buckets[..k_usize].fill(0);

    let mut i = 0usize;
    let mut j = n_usize.saturating_sub(7);
    while i < j {
        buckets[t[i] as usize] += 1;
        buckets[t[i + 1] as usize] += 1;
        buckets[t[i + 2] as usize] += 1;
        buckets[t[i + 3] as usize] += 1;
        buckets[t[i + 4] as usize] += 1;
        buckets[t[i + 5] as usize] += 1;
        buckets[t[i + 6] as usize] += 1;
        buckets[t[i + 7] as usize] += 1;
        i += 8;
    }

    j += 7;
    while i < j {
        buckets[t[i] as usize] += 1;
        i += 1;
    }
}

/// Internal helper: initialize buckets start and end 8u.
#[doc(hidden)]
pub fn initialize_buckets_start_and_end_8u(
    buckets: &mut [SaSint],
    freq: Option<&mut [SaSint]>,
) -> SaSint {
    let start_offset = 6 * ALPHABET_SIZE;
    let end_offset = 7 * ALPHABET_SIZE;
    let mut k = -1isize;
    let mut sum = 0;

    match freq {
        Some(freq) => {
            for j in 0..ALPHABET_SIZE {
                let i = buckets_index4(j, 0);
                let total = buckets[i] + buckets[i + 1] + buckets[i + 2] + buckets[i + 3];
                buckets[start_offset + j] = sum;
                sum += total;
                buckets[end_offset + j] = sum;
                if total > 0 {
                    k = j as isize;
                }
                freq[j] = total;
            }
        }
        None => {
            for j in 0..ALPHABET_SIZE {
                let i = buckets_index4(j, 0);
                let total = buckets[i] + buckets[i + 1] + buckets[i + 2] + buckets[i + 3];
                buckets[start_offset + j] = sum;
                sum += total;
                buckets[end_offset + j] = sum;
                if total > 0 {
                    k = j as isize;
                }
            }
        }
    }

    (k + 1) as SaSint
}

/// Internal helper: initialize buckets start and end 32s 6k.
#[doc(hidden)]
pub fn initialize_buckets_start_and_end_32s_6k(k: SaSint, buckets: &mut [SaSint]) {
    let k_usize = usize::try_from(k).expect("k must be non-negative");
    let start_offset = 4 * k_usize;
    let end_offset = 5 * k_usize;
    let mut sum = 0;
    for j in 0..k_usize {
        let i = buckets_index4(j, 0);
        buckets[start_offset + j] = sum;
        sum += buckets[i] + buckets[i + 1] + buckets[i + 2] + buckets[i + 3];
        buckets[end_offset + j] = sum;
    }
}

/// Internal helper: initialize buckets start and end 32s 4k.
#[doc(hidden)]
pub fn initialize_buckets_start_and_end_32s_4k(k: SaSint, buckets: &mut [SaSint]) {
    let k_usize = usize::try_from(k).expect("k must be non-negative");
    let start_offset = 2 * k_usize;
    let end_offset = 3 * k_usize;
    let mut sum = 0;
    for j in 0..k_usize {
        let i = buckets_index2(j, 0);
        buckets[start_offset + j] = sum;
        sum += buckets[i] + buckets[i + 1];
        buckets[end_offset + j] = sum;
    }
}

/// Internal helper: initialize buckets end 32s 2k.
#[doc(hidden)]
pub fn initialize_buckets_end_32s_2k(k: SaSint, buckets: &mut [SaSint]) {
    let k_usize = usize::try_from(k).expect("k must be non-negative");
    let mut sum0 = 0;
    for j in 0..k_usize {
        let i = buckets_index2(j, 0);
        sum0 += buckets[i] + buckets[i + 1];
        buckets[i] = sum0;
    }
}

/// Internal helper: initialize buckets start and end 32s 2k.
#[doc(hidden)]
pub fn initialize_buckets_start_and_end_32s_2k(k: SaSint, buckets: &mut [SaSint]) {
    let k_usize = usize::try_from(k).expect("k must be non-negative");
    for j in 0..k_usize {
        let i = buckets_index2(j, 0);
        buckets[j] = buckets[i];
    }
    buckets[k_usize] = 0;
    for j in 1..k_usize {
        buckets[k_usize + j] = buckets[j - 1];
    }
}

/// Internal helper: initialize buckets start 32s 1k.
#[doc(hidden)]
pub fn initialize_buckets_start_32s_1k(k: SaSint, buckets: &mut [SaSint]) {
    let k_usize = usize::try_from(k).expect("k must be non-negative");
    let mut sum = 0;
    for bucket in buckets.iter_mut().take(k_usize) {
        let tmp = *bucket;
        *bucket = sum;
        sum += tmp;
    }
}

/// Internal helper: initialize buckets end 32s 1k.
#[doc(hidden)]
pub fn initialize_buckets_end_32s_1k(k: SaSint, buckets: &mut [SaSint]) {
    let k_usize = usize::try_from(k).expect("k must be non-negative");
    let mut sum = 0;
    for bucket in buckets.iter_mut().take(k_usize) {
        sum += *bucket;
        *bucket = sum;
    }
}

/// Internal helper: initialize buckets for lms suffixes radix sort 8u.
#[doc(hidden)]
pub fn initialize_buckets_for_lms_suffixes_radix_sort_8u(
    t: &[u8],
    buckets: &mut [SaSint],
    mut first_lms_suffix: SaSint,
) -> SaSint {
    let mut f0 = 0usize;
    let mut f1: usize;
    let mut c0 = t[first_lms_suffix as usize] as FastSint;
    let mut c1: FastSint;

    while {
        first_lms_suffix -= 1;
        first_lms_suffix >= 0
    } {
        c1 = c0;
        c0 = t[first_lms_suffix as usize] as FastSint;
        f1 = f0;
        f0 = usize::from(c0 > (c1 - f1 as FastSint));
        let idx = 4 * c1 as usize + (f1 + f1 + f0);
        buckets[idx] -= 1;
    }
    buckets[4 * c0 as usize + (f0 + f0)] -= 1;

    let temp_offset = 4 * ALPHABET_SIZE;
    let mut sum = 0;
    for j in 0..ALPHABET_SIZE {
        let i = 4 * j;
        let tj = 2 * j;
        buckets[temp_offset + tj + 1] = sum;
        sum += buckets[i + 1] + buckets[i + 3];
        buckets[temp_offset + tj] = sum;
    }
    sum
}

/// Internal helper: initialize buckets for lms suffixes radix sort 32s 2k.
#[doc(hidden)]
pub fn initialize_buckets_for_lms_suffixes_radix_sort_32s_2k(
    t: &[SaSint],
    k: SaSint,
    buckets: &mut [SaSint],
    first_lms_suffix: SaSint,
) {
    let _k_usize = usize::try_from(k).expect("k must be non-negative");
    buckets[buckets_index2(t[first_lms_suffix as usize] as usize, 0)] += 1;
    buckets[buckets_index2(t[first_lms_suffix as usize] as usize, 1)] -= 1;

    let mut sum0 = 0;
    let mut sum1 = 0;
    for j in 0..usize::try_from(k).unwrap() {
        let i = buckets_index2(j, 0);
        sum0 += buckets[i] + buckets[i + 1];
        sum1 += buckets[i + 1];
        buckets[i] = sum0;
        buckets[i + 1] = sum1;
    }
}

/// Internal helper: initialize buckets for lms suffixes radix sort 32s 6k.
#[doc(hidden)]
pub fn initialize_buckets_for_lms_suffixes_radix_sort_32s_6k(
    t: &[SaSint],
    k: SaSint,
    buckets: &mut [SaSint],
    mut first_lms_suffix: SaSint,
) -> SaSint {
    let mut f0 = 0usize;
    let mut f1: usize;
    let mut c0 = t[first_lms_suffix as usize] as FastSint;
    let mut c1: FastSint;

    while {
        first_lms_suffix -= 1;
        first_lms_suffix >= 0
    } {
        c1 = c0;
        c0 = t[first_lms_suffix as usize] as FastSint;
        f1 = f0;
        f0 = usize::from(c0 > (c1 - f1 as FastSint));
        buckets[4 * c1 as usize + (f1 + f1 + f0)] -= 1;
    }
    buckets[4 * c0 as usize + (f0 + f0)] -= 1;

    let temp_offset = 4 * usize::try_from(k).unwrap();
    let mut sum = 0;
    for j in 0..usize::try_from(k).unwrap() {
        let i = 4 * j;
        sum += buckets[i + 1] + buckets[i + 3];
        buckets[temp_offset + j] = sum;
    }
    sum
}

/// Internal helper: initialize buckets for radix and partial sorting 32s 4k.
#[doc(hidden)]
pub fn initialize_buckets_for_radix_and_partial_sorting_32s_4k(
    t: &[SaSint],
    k: SaSint,
    buckets: &mut [SaSint],
    first_lms_suffix: SaSint,
) {
    let k_usize = usize::try_from(k).expect("k must be non-negative");
    let start_offset = 2 * k_usize;
    let end_offset = 3 * k_usize;

    buckets[buckets_index2(t[first_lms_suffix as usize] as usize, 0)] += 1;
    buckets[buckets_index2(t[first_lms_suffix as usize] as usize, 1)] -= 1;

    let mut sum0 = 0;
    let mut sum1 = 0;
    for j in 0..k_usize {
        let i = buckets_index2(j, 0);
        buckets[start_offset + j] = sum1;
        sum0 += buckets[i + 1];
        sum1 += buckets[i] + buckets[i + 1];
        buckets[i + 1] = sum0;
        buckets[end_offset + j] = sum1;
    }
}

/// Internal helper: radix sort lms suffixes 8u.
#[doc(hidden)]
pub fn radix_sort_lms_suffixes_8u(
    t: &[u8],
    sa: &mut [SaSint],
    induction_bucket: &mut [SaSint],
    omp_block_start: FastSint,
    omp_block_size: FastSint,
) {
    let prefetch_distance = 64 as FastSint;
    let mut i = omp_block_start + omp_block_size - 1;
    let mut j = omp_block_start + prefetch_distance + 3;

    while i >= j {
        let p0 = sa[i as usize];
        let idx0 = buckets_index2(t[p0 as usize] as usize, 0);
        induction_bucket[idx0] -= 1;
        sa[induction_bucket[idx0] as usize] = p0;

        let p1 = sa[(i - 1) as usize];
        let idx1 = buckets_index2(t[p1 as usize] as usize, 0);
        induction_bucket[idx1] -= 1;
        sa[induction_bucket[idx1] as usize] = p1;

        let p2 = sa[(i - 2) as usize];
        let idx2 = buckets_index2(t[p2 as usize] as usize, 0);
        induction_bucket[idx2] -= 1;
        sa[induction_bucket[idx2] as usize] = p2;

        let p3 = sa[(i - 3) as usize];
        let idx3 = buckets_index2(t[p3 as usize] as usize, 0);
        induction_bucket[idx3] -= 1;
        sa[induction_bucket[idx3] as usize] = p3;

        i -= 4;
    }

    j -= prefetch_distance + 3;
    while i >= j {
        let p = sa[i as usize];
        let idx = buckets_index2(t[p as usize] as usize, 0);
        induction_bucket[idx] -= 1;
        sa[induction_bucket[idx] as usize] = p;
        i -= 1;
    }
}

/// Internal helper: radix sort lms suffixes 8u (OpenMP variant).
#[doc(hidden)]
pub fn radix_sort_lms_suffixes_8u_omp(
    t: &[u8],
    sa: &mut [SaSint],
    n: SaSint,
    m: SaSint,
    flags: SaSint,
    buckets: &mut [SaSint],
    threads: SaSint,
    thread_state: &mut [ThreadState],
) {
    if (flags & LIBSAIS_FLAGS_GSA) != 0 {
        buckets[4 * ALPHABET_SIZE] -= 1;
    }

    let omp_num_threads = if threads > 1 && n >= 65_536 && m >= 65_536 {
        usize::try_from(threads)
            .expect("threads must be non-negative")
            .min(thread_state.len())
            .max(1)
    } else {
        1
    };

    if omp_num_threads == 1 {
        radix_sort_lms_suffixes_8u(
            t,
            sa,
            &mut buckets[4 * ALPHABET_SIZE..],
            n as FastSint - m as FastSint + 1,
            m as FastSint - 1,
        );
        return;
    }

    let src_bucket: &[SaSint] = &buckets[4 * ALPHABET_SIZE..];
    let m_values: Vec<FastSint> = thread_state[..omp_num_threads]
        .iter()
        .map(|s| s.m)
        .collect();
    let m_total = m as FastSint;
    let sa_ptr = SyncMutPtr::new(sa);

    run_rayon_with_threads(omp_num_threads, || {
        thread_state[..omp_num_threads]
            .par_iter_mut()
            .enumerate()
            .for_each(|(thread_num, state)| {
                for (i, j) in (0..=buckets_index2(ALPHABET_SIZE - 1, 0))
                    .step_by(buckets_index2(1, 0))
                    .zip((buckets_index4(0, 1)..).step_by(buckets_index4(1, 0)))
                {
                    state.buckets[i] = src_bucket[i] - state.buckets[j];
                }

                let mut omp_block_start: FastSint = 0;
                for &other_m in m_values[thread_num..omp_num_threads].iter().rev() {
                    omp_block_start += other_m;
                }

                let mut omp_block_size = m_values[thread_num];
                if omp_block_start == m_total && omp_block_size > 0 {
                    omp_block_start -= 1;
                    omp_block_size -= 1;
                }

                // SAFETY: per-thread disjoint sa block determined by `n - omp_block_start`.
                let sa = unsafe { sa_ptr.as_slice() };
                radix_sort_lms_suffixes_8u(
                    t,
                    sa,
                    &mut state.buckets,
                    n as FastSint - omp_block_start,
                    omp_block_size,
                );
            });
    });
}

/// Internal helper: radix sort lms suffixes 32s 6k.
#[doc(hidden)]
pub fn radix_sort_lms_suffixes_32s_6k(
    t: &[SaSint],
    sa: &mut [SaSint],
    induction_bucket: &mut [SaSint],
    omp_block_start: FastSint,
    omp_block_size: FastSint,
) {
    let prefetch_distance = 64 as FastSint;
    let mut i = omp_block_start + omp_block_size - 1;
    let mut j = omp_block_start + 2 * prefetch_distance + 3;

    while i >= j {
        let p0 = sa[i as usize];
        let idx0 = t[p0 as usize] as usize;
        induction_bucket[idx0] -= 1;
        sa[induction_bucket[idx0] as usize] = p0;

        let p1 = sa[(i - 1) as usize];
        let idx1 = t[p1 as usize] as usize;
        induction_bucket[idx1] -= 1;
        sa[induction_bucket[idx1] as usize] = p1;

        let p2 = sa[(i - 2) as usize];
        let idx2 = t[p2 as usize] as usize;
        induction_bucket[idx2] -= 1;
        sa[induction_bucket[idx2] as usize] = p2;

        let p3 = sa[(i - 3) as usize];
        let idx3 = t[p3 as usize] as usize;
        induction_bucket[idx3] -= 1;
        sa[induction_bucket[idx3] as usize] = p3;

        i -= 4;
    }

    j -= 2 * prefetch_distance + 3;
    while i >= j {
        let p = sa[i as usize];
        let idx = t[p as usize] as usize;
        induction_bucket[idx] -= 1;
        sa[induction_bucket[idx] as usize] = p;
        i -= 1;
    }
}

/// Internal helper: radix sort lms suffixes 32s 2k.
#[doc(hidden)]
pub fn radix_sort_lms_suffixes_32s_2k(
    t: &[SaSint],
    sa: &mut [SaSint],
    induction_bucket: &mut [SaSint],
    omp_block_start: FastSint,
    omp_block_size: FastSint,
) {
    let prefetch_distance = 64 as FastSint;
    let mut i = omp_block_start + omp_block_size - 1;
    let mut j = omp_block_start + 2 * prefetch_distance + 3;

    while i >= j {
        let p0 = sa[i as usize];
        let idx0 = buckets_index2(t[p0 as usize] as usize, 0);
        induction_bucket[idx0] -= 1;
        sa[induction_bucket[idx0] as usize] = p0;

        let p1 = sa[(i - 1) as usize];
        let idx1 = buckets_index2(t[p1 as usize] as usize, 0);
        induction_bucket[idx1] -= 1;
        sa[induction_bucket[idx1] as usize] = p1;

        let p2 = sa[(i - 2) as usize];
        let idx2 = buckets_index2(t[p2 as usize] as usize, 0);
        induction_bucket[idx2] -= 1;
        sa[induction_bucket[idx2] as usize] = p2;

        let p3 = sa[(i - 3) as usize];
        let idx3 = buckets_index2(t[p3 as usize] as usize, 0);
        induction_bucket[idx3] -= 1;
        sa[induction_bucket[idx3] as usize] = p3;

        i -= 4;
    }

    j -= 2 * prefetch_distance + 3;
    while i >= j {
        let p = sa[i as usize];
        let idx = buckets_index2(t[p as usize] as usize, 0);
        induction_bucket[idx] -= 1;
        sa[induction_bucket[idx] as usize] = p;
        i -= 1;
    }
}

/// Internal helper: radix sort lms suffixes 32s block gather.
#[doc(hidden)]
pub fn radix_sort_lms_suffixes_32s_block_gather(
    t: &[SaSint],
    sa: &[SaSint],
    cache: &mut [ThreadCache],
    omp_block_start: FastSint,
    omp_block_size: FastSint,
) {
    if omp_block_size <= 0 {
        return;
    }

    let start = usize::try_from(omp_block_start).expect("omp_block_start must be non-negative");
    let size = usize::try_from(omp_block_size).expect("omp_block_size must be non-negative");
    let mut i = start;
    let mut j = if size > 67 { start + size - 67 } else { start };

    while i < j {
        for current in [i, i + 1, i + 2, i + 3] {
            let ci = current - start;
            let index = sa[current];
            cache[ci].index = index;
            cache[ci].symbol = t[index as usize];
        }
        i += 4;
    }

    j = if size > 67 { j + 67 } else { start + size };
    while i < j {
        let ci = i - start;
        let index = sa[i];
        cache[ci].index = index;
        cache[ci].symbol = t[index as usize];
        i += 1;
    }
}

/// Internal helper: radix sort lms suffixes 32s 6k block sort.
#[doc(hidden)]
pub fn radix_sort_lms_suffixes_32s_6k_block_sort(
    induction_bucket: &mut [SaSint],
    cache: &mut [ThreadCache],
    omp_block_start: FastSint,
    omp_block_size: FastSint,
) {
    if omp_block_size <= 0 {
        return;
    }

    let start = usize::try_from(omp_block_start).expect("omp_block_start must be non-negative");
    let size = usize::try_from(omp_block_size).expect("omp_block_size must be non-negative");
    let mut i = start + size - 1;
    let mut j = start + 64 + 3;

    while i >= j {
        for current in [i, i - 1, i - 2, i - 3] {
            let ci = current - start;
            let v = cache[ci].symbol as usize;
            induction_bucket[v] -= 1;
            cache[ci].symbol = induction_bucket[v];
        }
        i -= 4;
    }

    j -= 64 + 3;
    while i >= j {
        let ci = i - start;
        let v = cache[ci].symbol as usize;
        induction_bucket[v] -= 1;
        cache[ci].symbol = induction_bucket[v];
        if i == 0 {
            break;
        }
        i -= 1;
    }
}

/// Internal helper: radix sort lms suffixes 32s 2k block sort.
#[doc(hidden)]
pub fn radix_sort_lms_suffixes_32s_2k_block_sort(
    induction_bucket: &mut [SaSint],
    cache: &mut [ThreadCache],
    omp_block_start: FastSint,
    omp_block_size: FastSint,
) {
    if omp_block_size <= 0 {
        return;
    }

    let start = usize::try_from(omp_block_start).expect("omp_block_start must be non-negative");
    let size = usize::try_from(omp_block_size).expect("omp_block_size must be non-negative");
    let mut i = start + size - 1;
    let mut j = start + 64 + 3;

    while i >= j {
        for current in [i, i - 1, i - 2, i - 3] {
            let ci = current - start;
            let v = buckets_index2(cache[ci].symbol as usize, 0);
            induction_bucket[v] -= 1;
            cache[ci].symbol = induction_bucket[v];
        }
        i -= 4;
    }

    j -= 64 + 3;
    while i >= j {
        let ci = i - start;
        let v = buckets_index2(cache[ci].symbol as usize, 0);
        induction_bucket[v] -= 1;
        cache[ci].symbol = induction_bucket[v];
        if i == 0 {
            break;
        }
        i -= 1;
    }
}

/// Internal helper: radix sort lms suffixes 32s 6k block (OpenMP variant).
#[doc(hidden)]
pub fn radix_sort_lms_suffixes_32s_6k_block_omp(
    t: &[SaSint],
    sa: &mut [SaSint],
    induction_bucket: &mut [SaSint],
    cache: &mut [ThreadCache],
    block_start: FastSint,
    block_size: FastSint,
    threads: SaSint,
) {
    if threads <= 1 || block_size < 16_384 {
        radix_sort_lms_suffixes_32s_6k(t, sa, induction_bucket, block_start, block_size);
        return;
    }

    let block_size_usize = usize::try_from(block_size).expect("block_size must be non-negative");
    let threads_usize = usize::try_from(threads)
        .expect("threads must be positive")
        .min(block_size_usize.max(1));
    let omp_block_stride = (block_size_usize / threads_usize) & !15usize;

    {
        let sa_ro: &[SaSint] = sa;
        let t_ro: &[SaSint] = t;
        let cache_ptr = SyncMutPtr::new(cache);
        run_rayon_with_threads(threads_usize, || {
            (0..threads_usize)
                .into_par_iter()
                .for_each(|omp_thread_num| {
                    let omp_block_start = omp_thread_num * omp_block_stride;
                    let omp_block_size = if omp_thread_num + 1 < threads_usize {
                        omp_block_stride
                    } else {
                        block_size_usize - omp_block_start
                    };
                    if omp_block_size > 0 {
                        // SAFETY: per-thread disjoint cache range.
                        let cache = unsafe { cache_ptr.as_slice() };
                        radix_sort_lms_suffixes_32s_block_gather(
                            t_ro,
                            sa_ro,
                            &mut cache[omp_block_start..],
                            block_start + omp_block_start as FastSint,
                            omp_block_size as FastSint,
                        );
                    }
                });
        });
    }

    radix_sort_lms_suffixes_32s_6k_block_sort(induction_bucket, cache, block_start, block_size);

    {
        let sa_ptr = SyncMutPtr::new(sa);
        let cache_ro: &[ThreadCache] = cache;
        run_rayon_with_threads(threads_usize, || {
            (0..threads_usize)
                .into_par_iter()
                .for_each(|omp_thread_num| {
                    let omp_block_start = omp_thread_num * omp_block_stride;
                    let omp_block_size = if omp_thread_num + 1 < threads_usize {
                        omp_block_stride
                    } else {
                        block_size_usize - omp_block_start
                    };
                    if omp_block_size > 0 {
                        // SAFETY: per-thread sa writes go to distinct symbol-indexed positions.
                        let sa = unsafe { sa_ptr.as_slice() };
                        place_cached_suffixes(
                            sa,
                            &cache_ro[omp_block_start..],
                            0,
                            omp_block_size as FastSint,
                        );
                    }
                });
        });
    }
}

/// Internal helper: radix sort lms suffixes 32s 2k block (OpenMP variant).
#[doc(hidden)]
pub fn radix_sort_lms_suffixes_32s_2k_block_omp(
    t: &[SaSint],
    sa: &mut [SaSint],
    induction_bucket: &mut [SaSint],
    cache: &mut [ThreadCache],
    block_start: FastSint,
    block_size: FastSint,
    threads: SaSint,
) {
    if threads <= 1 || block_size < 16_384 {
        radix_sort_lms_suffixes_32s_2k(t, sa, induction_bucket, block_start, block_size);
        return;
    }

    let block_size_usize = usize::try_from(block_size).expect("block_size must be non-negative");
    let threads_usize = usize::try_from(threads)
        .expect("threads must be positive")
        .min(block_size_usize.max(1));
    let omp_block_stride = (block_size_usize / threads_usize) & !15usize;

    {
        let sa_ro: &[SaSint] = sa;
        let t_ro: &[SaSint] = t;
        let cache_ptr = SyncMutPtr::new(cache);
        run_rayon_with_threads(threads_usize, || {
            (0..threads_usize)
                .into_par_iter()
                .for_each(|omp_thread_num| {
                    let omp_block_start = omp_thread_num * omp_block_stride;
                    let omp_block_size = if omp_thread_num + 1 < threads_usize {
                        omp_block_stride
                    } else {
                        block_size_usize - omp_block_start
                    };
                    if omp_block_size > 0 {
                        let cache = unsafe { cache_ptr.as_slice() };
                        radix_sort_lms_suffixes_32s_block_gather(
                            t_ro,
                            sa_ro,
                            &mut cache[omp_block_start..],
                            block_start + omp_block_start as FastSint,
                            omp_block_size as FastSint,
                        );
                    }
                });
        });
    }

    radix_sort_lms_suffixes_32s_2k_block_sort(induction_bucket, cache, block_start, block_size);

    {
        let sa_ptr = SyncMutPtr::new(sa);
        let cache_ro: &[ThreadCache] = cache;
        run_rayon_with_threads(threads_usize, || {
            (0..threads_usize)
                .into_par_iter()
                .for_each(|omp_thread_num| {
                    let omp_block_start = omp_thread_num * omp_block_stride;
                    let omp_block_size = if omp_thread_num + 1 < threads_usize {
                        omp_block_stride
                    } else {
                        block_size_usize - omp_block_start
                    };
                    if omp_block_size > 0 {
                        let sa = unsafe { sa_ptr.as_slice() };
                        place_cached_suffixes(
                            sa,
                            &cache_ro[omp_block_start..],
                            0,
                            omp_block_size as FastSint,
                        );
                    }
                });
        });
    }
}

/// Internal helper: radix sort lms suffixes 32s 6k (OpenMP variant).
#[doc(hidden)]
pub fn radix_sort_lms_suffixes_32s_6k_omp(
    t: &[SaSint],
    sa: &mut [SaSint],
    n: SaSint,
    m: SaSint,
    induction_bucket: &mut [SaSint],
    threads: SaSint,
    _thread_state: &mut [ThreadState],
) {
    if threads <= 1 || m < 65_536 {
        radix_sort_lms_suffixes_32s_6k(
            t,
            sa,
            induction_bucket,
            n as FastSint - m as FastSint + 1,
            m as FastSint - 1,
        );
        return;
    }

    let threads_usize = usize::try_from(threads).expect("threads must be positive");
    let mut cache = vec![ThreadCache::default(); threads_usize * LIBSAIS_PER_THREAD_CACHE_SIZE];
    let mut block_start = 0usize;
    let m_usize = usize::try_from(m).expect("m must be non-negative");
    let n_usize = usize::try_from(n).expect("n must be non-negative");
    let last = m_usize - 1;

    while block_start < last {
        let block_end = (block_start + threads_usize * LIBSAIS_PER_THREAD_CACHE_SIZE).min(last);
        radix_sort_lms_suffixes_32s_6k_block_omp(
            t,
            sa,
            induction_bucket,
            &mut cache,
            (n_usize - block_end) as FastSint,
            (block_end - block_start) as FastSint,
            threads,
        );
        block_start = block_end;
    }
}

/// Internal helper: radix sort lms suffixes 32s 2k (OpenMP variant).
#[doc(hidden)]
pub fn radix_sort_lms_suffixes_32s_2k_omp(
    t: &[SaSint],
    sa: &mut [SaSint],
    n: SaSint,
    m: SaSint,
    induction_bucket: &mut [SaSint],
    threads: SaSint,
    _thread_state: &mut [ThreadState],
) {
    if threads <= 1 || m < 65_536 {
        radix_sort_lms_suffixes_32s_2k(
            t,
            sa,
            induction_bucket,
            n as FastSint - m as FastSint + 1,
            m as FastSint - 1,
        );
        return;
    }

    let threads_usize = usize::try_from(threads).expect("threads must be positive");
    let mut cache = vec![ThreadCache::default(); threads_usize * LIBSAIS_PER_THREAD_CACHE_SIZE];
    let mut block_start = 0usize;
    let m_usize = usize::try_from(m).expect("m must be non-negative");
    let n_usize = usize::try_from(n).expect("n must be non-negative");
    let last = m_usize - 1;

    while block_start < last {
        let block_end = (block_start + threads_usize * LIBSAIS_PER_THREAD_CACHE_SIZE).min(last);
        radix_sort_lms_suffixes_32s_2k_block_omp(
            t,
            sa,
            induction_bucket,
            &mut cache,
            (n_usize - block_end) as FastSint,
            (block_end - block_start) as FastSint,
            threads,
        );
        block_start = block_end;
    }
}

/// Internal helper: radix sort lms suffixes 32s 1k.
#[doc(hidden)]
pub fn radix_sort_lms_suffixes_32s_1k(
    t: &[SaSint],
    sa: &mut [SaSint],
    n: SaSint,
    buckets: &mut [SaSint],
) -> SaSint {
    let n_usize = usize::try_from(n).expect("n must be non-negative");
    let mut i = n as FastSint - 2;
    let mut m = 0;
    let mut f0 = 1usize;
    let mut f1: usize;
    let mut c0 = t[n_usize - 1] as FastSint;
    let mut c1: FastSint;
    let mut c2 = 0 as FastSint;

    while i >= 67 {
        c1 = t[i as usize] as FastSint;
        f1 = usize::from(c1 > (c0 - f0 as FastSint));
        if (f1 & !f0) != 0 {
            c2 = c0;
            buckets[c2 as usize] -= 1;
            sa[buckets[c2 as usize] as usize] = (i + 1) as SaSint;
            m += 1;
        }

        c0 = t[(i - 1) as usize] as FastSint;
        f0 = usize::from(c0 > (c1 - f1 as FastSint));
        if (f0 & !f1) != 0 {
            c2 = c1;
            buckets[c2 as usize] -= 1;
            sa[buckets[c2 as usize] as usize] = i as SaSint;
            m += 1;
        }

        c1 = t[(i - 2) as usize] as FastSint;
        f1 = usize::from(c1 > (c0 - f0 as FastSint));
        if (f1 & !f0) != 0 {
            c2 = c0;
            buckets[c2 as usize] -= 1;
            sa[buckets[c2 as usize] as usize] = (i - 1) as SaSint;
            m += 1;
        }

        c0 = t[(i - 3) as usize] as FastSint;
        f0 = usize::from(c0 > (c1 - f1 as FastSint));
        if (f0 & !f1) != 0 {
            c2 = c1;
            buckets[c2 as usize] -= 1;
            sa[buckets[c2 as usize] as usize] = (i - 2) as SaSint;
            m += 1;
        }

        i -= 4;
    }

    while i >= 0 {
        c1 = c0;
        c0 = t[i as usize] as FastSint;
        f1 = f0;
        f0 = usize::from(c0 > (c1 - f1 as FastSint));
        if (f0 & !f1) != 0 {
            c2 = c1;
            buckets[c2 as usize] -= 1;
            sa[buckets[c2 as usize] as usize] = (i + 1) as SaSint;
            m += 1;
        }
        i -= 1;
    }

    if m > 1 {
        sa[buckets[c2 as usize] as usize] = 0;
    }

    m
}

/// Internal helper: radix sort set markers 32s 6k.
#[doc(hidden)]
pub fn radix_sort_set_markers_32s_6k(
    sa: &mut [SaSint],
    induction_bucket: &[SaSint],
    omp_block_start: FastSint,
    omp_block_size: FastSint,
) {
    let mut i = omp_block_start;
    let mut j = omp_block_start + omp_block_size - 67;

    while i < j {
        sa[induction_bucket[i as usize] as usize] |= SAINT_MIN;
        sa[induction_bucket[(i + 1) as usize] as usize] |= SAINT_MIN;
        sa[induction_bucket[(i + 2) as usize] as usize] |= SAINT_MIN;
        sa[induction_bucket[(i + 3) as usize] as usize] |= SAINT_MIN;
        i += 4;
    }

    j += 67;
    while i < j {
        sa[induction_bucket[i as usize] as usize] |= SAINT_MIN;
        i += 1;
    }
}

/// Internal helper: radix sort set markers 32s 4k.
#[doc(hidden)]
pub fn radix_sort_set_markers_32s_4k(
    sa: &mut [SaSint],
    induction_bucket: &[SaSint],
    omp_block_start: FastSint,
    omp_block_size: FastSint,
) {
    let mut i = omp_block_start;
    let mut j = omp_block_start + omp_block_size - 67;

    while i < j {
        sa[induction_bucket[buckets_index2(i as usize, 0)] as usize] |= SUFFIX_GROUP_MARKER;
        sa[induction_bucket[buckets_index2((i + 1) as usize, 0)] as usize] |= SUFFIX_GROUP_MARKER;
        sa[induction_bucket[buckets_index2((i + 2) as usize, 0)] as usize] |= SUFFIX_GROUP_MARKER;
        sa[induction_bucket[buckets_index2((i + 3) as usize, 0)] as usize] |= SUFFIX_GROUP_MARKER;
        i += 4;
    }

    j += 67;
    while i < j {
        sa[induction_bucket[buckets_index2(i as usize, 0)] as usize] |= SUFFIX_GROUP_MARKER;
        i += 1;
    }
}

/// Internal helper: radix sort set markers 32s 6k (OpenMP variant).
#[doc(hidden)]
pub fn radix_sort_set_markers_32s_6k_omp(
    sa: &mut [SaSint],
    k: SaSint,
    induction_bucket: &[SaSint],
    threads: SaSint,
) {
    if k <= 1 {
        return;
    }

    if threads <= 1 || k < 65_536 {
        radix_sort_set_markers_32s_6k(sa, induction_bucket, 0, k as FastSint - 1);
        return;
    }

    let threads_usize = usize::try_from(threads).expect("threads must be positive");
    let last = usize::try_from(k - 1).expect("k must be positive");
    let stride = (last / threads_usize) & !15usize;

    {
        let sa_ptr = SyncMutPtr::new(sa);
        run_rayon_with_threads(threads_usize, || {
            (0..threads_usize).into_par_iter().for_each(|thread| {
                let start = thread * stride;
                let end = if thread + 1 == threads_usize {
                    last
                } else {
                    start + stride
                };
                if end > start {
                    // SAFETY: per-thread disjoint sa[start..end] block.
                    let sa = unsafe { sa_ptr.as_slice() };
                    radix_sort_set_markers_32s_6k(
                        sa,
                        induction_bucket,
                        start as FastSint,
                        (end - start) as FastSint,
                    );
                }
            });
        });
    }
}

/// Internal helper: radix sort set markers 32s 4k (OpenMP variant).
#[doc(hidden)]
pub fn radix_sort_set_markers_32s_4k_omp(
    sa: &mut [SaSint],
    k: SaSint,
    induction_bucket: &[SaSint],
    threads: SaSint,
) {
    if k <= 1 {
        return;
    }

    if threads <= 1 || k < 65_536 {
        radix_sort_set_markers_32s_4k(sa, induction_bucket, 0, k as FastSint - 1);
        return;
    }

    let threads_usize = usize::try_from(threads).expect("threads must be positive");
    let last = usize::try_from(k - 1).expect("k must be positive");
    let stride = (last / threads_usize) & !15usize;

    {
        let sa_ptr = SyncMutPtr::new(sa);
        run_rayon_with_threads(threads_usize, || {
            (0..threads_usize).into_par_iter().for_each(|thread| {
                let start = thread * stride;
                let end = if thread + 1 == threads_usize {
                    last
                } else {
                    start + stride
                };
                if end > start {
                    let sa = unsafe { sa_ptr.as_slice() };
                    radix_sort_set_markers_32s_4k(
                        sa,
                        induction_bucket,
                        start as FastSint,
                        (end - start) as FastSint,
                    );
                }
            });
        });
    }
}

/// Internal helper: initialize buckets for partial sorting 8u.
#[doc(hidden)]
pub fn initialize_buckets_for_partial_sorting_8u(
    t: &[u8],
    buckets: &mut [SaSint],
    first_lms_suffix: SaSint,
    left_suffixes_count: SaSint,
) {
    let temp_offset = 4 * ALPHABET_SIZE;
    buckets[buckets_index4(t[first_lms_suffix as usize] as usize, 1)] += 1;

    let mut sum0 = left_suffixes_count + 1;
    let mut sum1 = 0;
    for j in 0..ALPHABET_SIZE {
        let i = buckets_index4(j, 0);
        let tj = buckets_index2(j, 0);
        buckets[temp_offset + tj] = sum0;
        sum0 += buckets[i] + buckets[i + 2];
        sum1 += buckets[i + 1];
        buckets[tj] = sum0;
        buckets[tj + 1] = sum1;
    }
}

/// Internal helper: initialize buckets for partial sorting 32s 6k.
#[doc(hidden)]
pub fn initialize_buckets_for_partial_sorting_32s_6k(
    t: &[SaSint],
    k: SaSint,
    buckets: &mut [SaSint],
    first_lms_suffix: SaSint,
    left_suffixes_count: SaSint,
) {
    let k_usize = usize::try_from(k).expect("k must be non-negative");
    let temp_offset = 4 * k_usize;
    let first_symbol = t[first_lms_suffix as usize] as usize;
    let mut sum0 = left_suffixes_count + 1;
    let mut sum1 = 0;
    let mut sum2 = 0;

    for j in 0..first_symbol {
        let i = buckets_index4(j, 0);
        let tj = buckets_index2(j, 0);
        let ss = buckets[i];
        let ls = buckets[i + 1];
        let sl = buckets[i + 2];
        let ll = buckets[i + 3];

        buckets[i] = sum0;
        buckets[i + 1] = sum2;
        buckets[i + 2] = 0;
        buckets[i + 3] = 0;

        sum0 += ss + sl;
        sum1 += ls;
        sum2 += ls + ll;

        buckets[temp_offset + tj] = sum0;
        buckets[temp_offset + tj + 1] = sum1;
    }

    sum1 += 1;
    for j in first_symbol..k_usize {
        let i = buckets_index4(j, 0);
        let tj = buckets_index2(j, 0);
        let ss = buckets[i];
        let ls = buckets[i + 1];
        let sl = buckets[i + 2];
        let ll = buckets[i + 3];

        buckets[i] = sum0;
        buckets[i + 1] = sum2;
        buckets[i + 2] = 0;
        buckets[i + 3] = 0;

        sum0 += ss + sl;
        sum1 += ls;
        sum2 += ls + ll;

        buckets[temp_offset + tj] = sum0;
        buckets[temp_offset + tj + 1] = sum1;
    }
}

/// Internal helper: partial sorting scan left to right 8u.
#[doc(hidden)]
pub fn partial_sorting_scan_left_to_right_8u(
    t: &[u8],
    sa: &mut [SaSint],
    buckets: &mut [SaSint],
    mut d: SaSint,
    omp_block_start: FastSint,
    omp_block_size: FastSint,
) -> SaSint {
    let induction_offset = 4 * ALPHABET_SIZE;
    let distinct_offset = 2 * ALPHABET_SIZE;
    let prefetch_distance = 64 as FastSint;
    let mut i = omp_block_start;
    let mut j = if omp_block_size > prefetch_distance + 1 {
        omp_block_start + omp_block_size - prefetch_distance - 1
    } else {
        omp_block_start
    };

    while i < j {
        let mut p0 = sa[i as usize];
        d += SaSint::from(p0 < 0);
        p0 &= SAINT_MAX;
        let v0 = buckets_index2(
            t[(p0 - 1) as usize] as usize,
            usize::from(t[(p0 - 2) as usize] >= t[(p0 - 1) as usize]),
        );
        let pos0 = buckets[induction_offset + v0] as usize;
        sa[pos0] = (p0 - 1) | (((buckets[distinct_offset + v0] != d) as SaSint) << (SAINT_BIT - 1));
        buckets[induction_offset + v0] += 1;
        buckets[distinct_offset + v0] = d;

        let mut p1 = sa[(i + 1) as usize];
        d += SaSint::from(p1 < 0);
        p1 &= SAINT_MAX;
        let v1 = buckets_index2(
            t[(p1 - 1) as usize] as usize,
            usize::from(t[(p1 - 2) as usize] >= t[(p1 - 1) as usize]),
        );
        let pos1 = buckets[induction_offset + v1] as usize;
        sa[pos1] = (p1 - 1) | (((buckets[distinct_offset + v1] != d) as SaSint) << (SAINT_BIT - 1));
        buckets[induction_offset + v1] += 1;
        buckets[distinct_offset + v1] = d;

        i += 2;
    }

    j = omp_block_start + omp_block_size;
    while i < j {
        let mut p = sa[i as usize];
        d += SaSint::from(p < 0);
        p &= SAINT_MAX;
        let v = buckets_index2(
            t[(p - 1) as usize] as usize,
            usize::from(t[(p - 2) as usize] >= t[(p - 1) as usize]),
        );
        let pos = buckets[induction_offset + v] as usize;
        sa[pos] = (p - 1) | (((buckets[distinct_offset + v] != d) as SaSint) << (SAINT_BIT - 1));
        buckets[induction_offset + v] += 1;
        buckets[distinct_offset + v] = d;
        i += 1;
    }

    d
}

/// Internal helper: partial sorting scan left to right 8u (OpenMP variant).
#[doc(hidden)]
pub fn partial_sorting_scan_left_to_right_8u_omp(
    t: &[u8],
    sa: &mut [SaSint],
    n: SaSint,
    k: SaSint,
    buckets: &mut [SaSint],
    left_suffixes_count: SaSint,
    mut d: SaSint,
    threads: SaSint,
    thread_state: &mut [ThreadState],
) -> SaSint {
    let v = buckets_index2(
        t[(n - 1) as usize] as usize,
        usize::from(t[(n - 2) as usize] >= t[(n - 1) as usize]),
    );
    let induction_offset = 4 * ALPHABET_SIZE;
    let distinct_offset = 2 * ALPHABET_SIZE;
    let pos = buckets[induction_offset + v] as usize;
    sa[pos] = (n - 1) | SAINT_MIN;
    buckets[induction_offset + v] += 1;
    d += 1;
    buckets[distinct_offset + v] = d;

    if threads == 1 || left_suffixes_count < 65_536 {
        return partial_sorting_scan_left_to_right_8u(
            t,
            sa,
            buckets,
            d,
            0,
            left_suffixes_count as FastSint,
        );
    }

    let mut block_start = 0usize;
    let left_suffixes_count =
        usize::try_from(left_suffixes_count).expect("left_suffixes_count must be non-negative");
    let threads_usize = usize::try_from(threads)
        .expect("threads must be non-negative")
        .min(thread_state.len())
        .max(1);
    while block_start < left_suffixes_count {
        if sa[block_start] == 0 {
            block_start += 1;
        } else {
            let mut block_max_end =
                block_start + threads_usize * (LIBSAIS_PER_THREAD_CACHE_SIZE - 16 * threads_usize);
            if block_max_end > left_suffixes_count {
                block_max_end = left_suffixes_count;
            }
            let mut block_end = block_start + 1;
            while block_end < block_max_end && sa[block_end] != 0 {
                block_end += 1;
            }
            let block_size = block_end - block_start;

            if block_size < 32 {
                while block_start < block_end {
                    let p = sa[block_start];
                    d += SaSint::from(p < 0);
                    let p = p & SAINT_MAX;
                    let v = buckets_index2(
                        t[(p - 1) as usize] as usize,
                        usize::from(t[(p - 2) as usize] >= t[(p - 1) as usize]),
                    );
                    let pos = buckets[induction_offset + v] as usize;
                    sa[pos] = (p - 1)
                        | (((buckets[distinct_offset + v] != d) as SaSint) << (SAINT_BIT - 1));
                    buckets[induction_offset + v] += 1;
                    buckets[distinct_offset + v] = d;
                    block_start += 1;
                }
            } else {
                d = partial_sorting_scan_left_to_right_8u_block_omp(
                    t,
                    sa,
                    k,
                    buckets,
                    d,
                    block_start as FastSint,
                    block_size as FastSint,
                    threads,
                    thread_state,
                );
                block_start = block_end;
            }
        }
    }

    d
}

/// Internal helper: partial sorting scan left to right 32s 6k.
#[doc(hidden)]
pub fn partial_sorting_scan_left_to_right_32s_6k(
    t: &[SaSint],
    sa: &mut [SaSint],
    buckets: &mut [SaSint],
    mut d: SaSint,
    omp_block_start: FastSint,
    omp_block_size: FastSint,
) -> SaSint {
    let prefetch_distance: FastSint = 64;

    let mut i = omp_block_start;
    let mut j = omp_block_start + omp_block_size - 2 * prefetch_distance - 1;
    while i < j {
        let mut p0 = sa[i as usize];
        d += SaSint::from(p0 < 0);
        p0 &= SAINT_MAX;
        let p0u = p0 as usize;
        let v0 = buckets_index4(t[p0u - 1] as usize, usize::from(t[p0u - 2] >= t[p0u - 1]));
        let pos0 = buckets[v0] as usize;
        sa[pos0] = (p0 - 1) | (((buckets[2 + v0] != d) as SaSint) << (SAINT_BIT - 1));
        buckets[v0] += 1;
        buckets[2 + v0] = d;

        let mut p1 = sa[(i + 1) as usize];
        d += SaSint::from(p1 < 0);
        p1 &= SAINT_MAX;
        let p1u = p1 as usize;
        let v1 = buckets_index4(t[p1u - 1] as usize, usize::from(t[p1u - 2] >= t[p1u - 1]));
        let pos1 = buckets[v1] as usize;
        sa[pos1] = (p1 - 1) | (((buckets[2 + v1] != d) as SaSint) << (SAINT_BIT - 1));
        buckets[v1] += 1;
        buckets[2 + v1] = d;

        i += 2;
    }

    j += 2 * prefetch_distance + 1;
    while i < j {
        let mut p = sa[i as usize];
        d += SaSint::from(p < 0);
        p &= SAINT_MAX;
        let pu = p as usize;
        let v = buckets_index4(t[pu - 1] as usize, usize::from(t[pu - 2] >= t[pu - 1]));
        let pos = buckets[v] as usize;
        sa[pos] = (p - 1) | (((buckets[2 + v] != d) as SaSint) << (SAINT_BIT - 1));
        buckets[v] += 1;
        buckets[2 + v] = d;
        i += 1;
    }

    d
}

/// Internal helper: partial sorting scan left to right 32s 4k.
#[doc(hidden)]
pub fn partial_sorting_scan_left_to_right_32s_4k(
    t: &[SaSint],
    sa: &mut [SaSint],
    k: SaSint,
    buckets: &mut [SaSint],
    mut d: SaSint,
    omp_block_start: FastSint,
    omp_block_size: FastSint,
) -> SaSint {
    let k_usize = usize::try_from(k).expect("k must be non-negative");
    let prefetch_distance: FastSint = 64;
    let induction_offset = 2 * k_usize;
    let mut i = omp_block_start;
    let mut j = omp_block_start + omp_block_size - 2 * prefetch_distance - 1;

    while i < j {
        let i0 = i as usize;
        let mut p0 = sa[i0];
        sa[i0] = p0 & SAINT_MAX;
        if p0 > 0 {
            sa[i0] = 0;
            d += p0 >> (SUFFIX_GROUP_BIT - 1);
            p0 &= !SUFFIX_GROUP_MARKER;
            let p0u = p0 as usize;
            let c0 = t[p0u - 1];
            let f0 = usize::from(t[p0u - 2] < c0);
            let v0 = buckets_index2(c0 as usize, f0);
            let c0u = c0 as usize;
            let pos0 = buckets[induction_offset + c0u] as usize;
            sa[pos0] = (p0 - 1)
                | ((f0 as SaSint) << (SAINT_BIT - 1))
                | (((buckets[v0] != d) as SaSint) << (SUFFIX_GROUP_BIT - 1));
            buckets[induction_offset + c0u] += 1;
            buckets[v0] = d;
        }

        let i1 = (i + 1) as usize;
        let mut p1 = sa[i1];
        sa[i1] = p1 & SAINT_MAX;
        if p1 > 0 {
            sa[i1] = 0;
            d += p1 >> (SUFFIX_GROUP_BIT - 1);
            p1 &= !SUFFIX_GROUP_MARKER;
            let p1u = p1 as usize;
            let c1 = t[p1u - 1];
            let f1 = usize::from(t[p1u - 2] < c1);
            let v1 = buckets_index2(c1 as usize, f1);
            let c1u = c1 as usize;
            let pos1 = buckets[induction_offset + c1u] as usize;
            sa[pos1] = (p1 - 1)
                | ((f1 as SaSint) << (SAINT_BIT - 1))
                | (((buckets[v1] != d) as SaSint) << (SUFFIX_GROUP_BIT - 1));
            buckets[induction_offset + c1u] += 1;
            buckets[v1] = d;
        }

        i += 2;
    }

    j += 2 * prefetch_distance + 1;
    while i < j {
        let iu = i as usize;
        let mut p = sa[iu];
        sa[iu] = p & SAINT_MAX;
        if p > 0 {
            sa[iu] = 0;
            d += p >> (SUFFIX_GROUP_BIT - 1);
            p &= !SUFFIX_GROUP_MARKER;
            let pu = p as usize;
            let c = t[pu - 1];
            let f = usize::from(t[pu - 2] < c);
            let v = buckets_index2(c as usize, f);
            let cu = c as usize;
            let pos = buckets[induction_offset + cu] as usize;
            sa[pos] = (p - 1)
                | ((f as SaSint) << (SAINT_BIT - 1))
                | (((buckets[v] != d) as SaSint) << (SUFFIX_GROUP_BIT - 1));
            buckets[induction_offset + cu] += 1;
            buckets[v] = d;
        }
        i += 1;
    }

    d
}

/// Internal helper: partial sorting scan left to right 32s 1k.
#[doc(hidden)]
pub fn partial_sorting_scan_left_to_right_32s_1k(
    t: &[SaSint],
    sa: &mut [SaSint],
    induction_bucket: &mut [SaSint],
    omp_block_start: FastSint,
    omp_block_size: FastSint,
) {
    let prefetch_distance = 64 as FastSint;
    let mut i = omp_block_start;
    let mut j = omp_block_start + omp_block_size - 2 * prefetch_distance - 1;

    while i < j {
        let p0 = sa[i as usize];
        sa[i as usize] = p0 & SAINT_MAX;
        if p0 > 0 {
            sa[i as usize] = 0;
            let c0 = t[(p0 - 1) as usize] as usize;
            let pos0 = induction_bucket[c0] as usize;
            induction_bucket[c0] += 1;
            sa[pos0] = (p0 - 1)
                | ((usize::from(t[(p0 - 2) as usize] < t[(p0 - 1) as usize]) as SaSint)
                    << (SAINT_BIT - 1));
        }

        let p1 = sa[(i + 1) as usize];
        sa[(i + 1) as usize] = p1 & SAINT_MAX;
        if p1 > 0 {
            sa[(i + 1) as usize] = 0;
            let c1 = t[(p1 - 1) as usize] as usize;
            let pos1 = induction_bucket[c1] as usize;
            induction_bucket[c1] += 1;
            sa[pos1] = (p1 - 1)
                | ((usize::from(t[(p1 - 2) as usize] < t[(p1 - 1) as usize]) as SaSint)
                    << (SAINT_BIT - 1));
        }

        i += 2;
    }

    j += 2 * prefetch_distance + 1;
    while i < j {
        let p = sa[i as usize];
        sa[i as usize] = p & SAINT_MAX;
        if p > 0 {
            sa[i as usize] = 0;
            let c = t[(p - 1) as usize] as usize;
            let pos = induction_bucket[c] as usize;
            induction_bucket[c] += 1;
            sa[pos] = (p - 1)
                | ((usize::from(t[(p - 2) as usize] < t[(p - 1) as usize]) as SaSint)
                    << (SAINT_BIT - 1));
        }
        i += 1;
    }
}

/// Internal helper: partial sorting scan left to right 32s 6k (OpenMP variant).
#[doc(hidden)]
pub fn partial_sorting_scan_left_to_right_32s_6k_omp(
    t: &[SaSint],
    sa: &mut [SaSint],
    n: SaSint,
    buckets: &mut [SaSint],
    left_suffixes_count: SaSint,
    mut d: SaSint,
    threads: SaSint,
    thread_state: &mut [ThreadState],
) -> SaSint {
    let v = buckets_index4(
        t[(n - 1) as usize] as usize,
        usize::from(t[(n - 2) as usize] >= t[(n - 1) as usize]),
    );
    let pos = buckets[v] as usize;
    sa[pos] = (n - 1) | SAINT_MIN;
    buckets[v] += 1;
    d += 1;
    buckets[2 + v] = d;
    if threads == 1 || left_suffixes_count < 65_536 {
        return partial_sorting_scan_left_to_right_32s_6k(
            t,
            sa,
            buckets,
            d,
            0,
            left_suffixes_count as FastSint,
        );
    }
    if thread_state.is_empty() {
        return partial_sorting_scan_left_to_right_32s_6k(
            t,
            sa,
            buckets,
            d,
            0,
            left_suffixes_count as FastSint,
        );
    }

    let left_suffixes_count =
        usize::try_from(left_suffixes_count).expect("left_suffixes_count must be non-negative");
    let threads_usize = usize::try_from(threads)
        .expect("threads must be non-negative")
        .max(1);
    let mut block_start = 0usize;
    let block_span = threads_usize * LIBSAIS_PER_THREAD_CACHE_SIZE;
    let mut cache = vec![ThreadCache::default(); block_span];
    while block_start < left_suffixes_count {
        let mut block_end = block_start + block_span;
        if block_end > left_suffixes_count {
            block_end = left_suffixes_count;
        }

        d = partial_sorting_scan_left_to_right_32s_6k_block_omp(
            t,
            sa,
            buckets,
            d,
            &mut cache,
            block_start as FastSint,
            (block_end - block_start) as FastSint,
            threads,
        );

        block_start = block_end;
    }

    d
}

/// Internal helper: partial sorting scan left to right 32s 4k (OpenMP variant).
#[doc(hidden)]
pub fn partial_sorting_scan_left_to_right_32s_4k_omp(
    t: &[SaSint],
    sa: &mut [SaSint],
    n: SaSint,
    k: SaSint,
    buckets: &mut [SaSint],
    mut d: SaSint,
    threads: SaSint,
    thread_state: &mut [ThreadState],
) -> SaSint {
    let k_usize = usize::try_from(k).expect("k must be non-negative");
    let induction_offset = 2 * k_usize;
    let distinct_offset = 0usize;
    let symbol = t[(n - 1) as usize] as usize;
    let is_s = usize::from(t[(n - 2) as usize] < t[(n - 1) as usize]);
    let pos = buckets[induction_offset + symbol] as usize;
    sa[pos] = (n - 1) | ((is_s as SaSint) << (SAINT_BIT - 1)) | SUFFIX_GROUP_MARKER;
    buckets[induction_offset + symbol] += 1;
    d += 1;
    buckets[distinct_offset + buckets_index2(symbol, is_s)] = d;

    if threads == 1 || n < 65_536 {
        d = partial_sorting_scan_left_to_right_32s_4k(t, sa, k, buckets, d, 0, n as FastSint);
    } else {
        if thread_state.is_empty() {
            return partial_sorting_scan_left_to_right_32s_4k(
                t,
                sa,
                k,
                buckets,
                d,
                0,
                n as FastSint,
            );
        }
        let mut block_start = 0usize;
        let n_usize = usize::try_from(n).expect("n must be non-negative");
        let threads_usize = usize::try_from(threads)
            .expect("threads must be non-negative")
            .max(1);
        let chunk_capacity = threads_usize * LIBSAIS_PER_THREAD_CACHE_SIZE;
        let mut cache = vec![ThreadCache::default(); chunk_capacity];

        while block_start < n_usize {
            let mut block_end = block_start + chunk_capacity;
            if block_end > n_usize {
                block_end = n_usize;
            }

            d = partial_sorting_scan_left_to_right_32s_4k_block_omp(
                t,
                sa,
                k,
                buckets,
                d,
                &mut cache,
                block_start as FastSint,
                (block_end - block_start) as FastSint,
                threads,
            );

            block_start = block_end;
        }
    }

    d
}

/// Internal helper: partial sorting scan left to right 32s 1k (OpenMP variant).
#[doc(hidden)]
pub fn partial_sorting_scan_left_to_right_32s_1k_omp(
    t: &[SaSint],
    sa: &mut [SaSint],
    n: SaSint,
    buckets: &mut [SaSint],
    threads: SaSint,
    thread_state: &mut [ThreadState],
) {
    let symbol = t[(n - 1) as usize] as usize;
    let pos = buckets[symbol] as usize;
    sa[pos] = (n - 1)
        | ((usize::from(t[(n - 2) as usize] < t[(n - 1) as usize]) as SaSint) << (SAINT_BIT - 1));
    buckets[symbol] += 1;
    if threads == 1 || n < 65_536 {
        partial_sorting_scan_left_to_right_32s_1k(t, sa, buckets, 0, n as FastSint);
    } else {
        if thread_state.is_empty() {
            partial_sorting_scan_left_to_right_32s_1k(t, sa, buckets, 0, n as FastSint);
            return;
        }
        let n_usize = usize::try_from(n).expect("n must be non-negative");
        let threads_usize = usize::try_from(threads)
            .expect("threads must be non-negative")
            .max(1);
        let mut block_start = 0usize;
        let block_span = threads_usize * LIBSAIS_PER_THREAD_CACHE_SIZE;
        let mut cache = vec![ThreadCache::default(); block_span];

        while block_start < n_usize {
            let mut block_end = block_start + block_span;
            if block_end > n_usize {
                block_end = n_usize;
            }

            partial_sorting_scan_left_to_right_32s_1k_block_omp(
                t,
                sa,
                buckets,
                &mut cache,
                block_start as FastSint,
                (block_end - block_start) as FastSint,
                threads,
            );

            block_start = block_end;
        }
    }
}

/// Internal helper: partial sorting scan left to right 8u block prepare.
#[doc(hidden)]
pub fn partial_sorting_scan_left_to_right_8u_block_prepare(
    t: &[u8],
    sa: &[SaSint],
    k: SaSint,
    buckets: &mut [SaSint],
    cache: &mut [ThreadCache],
    omp_block_start: FastSint,
    omp_block_size: FastSint,
) -> (FastSint, FastSint) {
    let k_usize = usize::try_from(k).expect("k must be non-negative");
    buckets[..2 * k_usize].fill(0);
    buckets[2 * k_usize..4 * k_usize].fill(0);

    let mut i = omp_block_start;
    let mut j = omp_block_start + omp_block_size - 65;
    let mut count = 0usize;
    let mut d: SaSint = 1;

    while i < j {
        let mut p0 = sa[i as usize];
        cache[count].index = p0;
        d += SaSint::from(p0 < 0);
        p0 &= SAINT_MAX;
        let v0 = buckets_index2(
            t[(p0 - 1) as usize] as usize,
            usize::from(t[(p0 - 2) as usize] >= t[(p0 - 1) as usize]),
        );
        cache[count].symbol = v0 as SaSint;
        count += 1;
        buckets[v0] += 1;
        buckets[2 * k_usize + v0] = d;

        let mut p1 = sa[(i + 1) as usize];
        cache[count].index = p1;
        d += SaSint::from(p1 < 0);
        p1 &= SAINT_MAX;
        let v1 = buckets_index2(
            t[(p1 - 1) as usize] as usize,
            usize::from(t[(p1 - 2) as usize] >= t[(p1 - 1) as usize]),
        );
        cache[count].symbol = v1 as SaSint;
        count += 1;
        buckets[v1] += 1;
        buckets[2 * k_usize + v1] = d;

        i += 2;
    }

    j += 65;
    while i < j {
        let mut p = sa[i as usize];
        cache[count].index = p;
        d += SaSint::from(p < 0);
        p &= SAINT_MAX;
        let v = buckets_index2(
            t[(p - 1) as usize] as usize,
            usize::from(t[(p - 2) as usize] >= t[(p - 1) as usize]),
        );
        cache[count].symbol = v as SaSint;
        count += 1;
        buckets[v] += 1;
        buckets[2 * k_usize + v] = d;
        i += 1;
    }

    (d as FastSint - 1, count as FastSint)
}

/// Internal helper: partial sorting scan left to right 8u block place.
#[doc(hidden)]
pub fn partial_sorting_scan_left_to_right_8u_block_place(
    sa: &mut [SaSint],
    buckets: &mut [SaSint],
    k: SaSint,
    cache: &[ThreadCache],
    count: FastSint,
    mut d: SaSint,
) {
    let split = 2 * usize::try_from(k).expect("k must be non-negative");
    let (induction_bucket, distinct_names) = buckets.split_at_mut(split);

    let mut i = 0usize;
    let mut j = usize::try_from(count)
        .expect("count must be non-negative")
        .saturating_sub(1);
    while i < j {
        let p0 = cache[i].index;
        d += SaSint::from(p0 < 0);
        let v0 = cache[i].symbol as usize;
        let pos0 = induction_bucket[v0] as usize;
        sa[pos0] = (p0 - 1) | (((distinct_names[v0] != d) as SaSint) << (SAINT_BIT - 1));
        induction_bucket[v0] += 1;
        distinct_names[v0] = d;

        let p1 = cache[i + 1].index;
        d += SaSint::from(p1 < 0);
        let v1 = cache[i + 1].symbol as usize;
        let pos1 = induction_bucket[v1] as usize;
        sa[pos1] = (p1 - 1) | (((distinct_names[v1] != d) as SaSint) << (SAINT_BIT - 1));
        induction_bucket[v1] += 1;
        distinct_names[v1] = d;

        i += 2;
    }

    j += 1;
    while i < j {
        let p = cache[i].index;
        d += SaSint::from(p < 0);
        let v = cache[i].symbol as usize;
        let pos = induction_bucket[v] as usize;
        sa[pos] = (p - 1) | (((distinct_names[v] != d) as SaSint) << (SAINT_BIT - 1));
        induction_bucket[v] += 1;
        distinct_names[v] = d;
        i += 1;
    }
}

/// Internal helper: partial sorting scan left to right 8u block (OpenMP variant).
#[doc(hidden)]
pub fn partial_sorting_scan_left_to_right_8u_block_omp(
    t: &[u8],
    sa: &mut [SaSint],
    k: SaSint,
    buckets: &mut [SaSint],
    d: SaSint,
    block_start: FastSint,
    block_size: FastSint,
    threads: SaSint,
    thread_state: &mut [ThreadState],
) -> SaSint {
    let mut d = d;
    let block_size_usize = usize::try_from(block_size).expect("block_size must be non-negative");
    let k_usize = usize::try_from(k).expect("k must be non-negative");
    let omp_num_threads = if threads > 1 && block_size_usize >= 64 * k_usize.max(256) {
        usize::try_from(threads)
            .expect("threads must be non-negative")
            .min(thread_state.len())
            .max(1)
    } else {
        1
    };
    let omp_block_stride = (block_size_usize / omp_num_threads) & !15usize;

    if omp_num_threads == 1 {
        return partial_sorting_scan_left_to_right_8u(t, sa, buckets, d, block_start, block_size);
    }

    {
        let sa_ro: &[SaSint] = sa;
        run_rayon_with_threads(omp_num_threads, || {
            thread_state[..omp_num_threads]
                .par_iter_mut()
                .enumerate()
                .for_each(|(omp_thread_num, state)| {
                    let mut omp_block_start = omp_thread_num * omp_block_stride;
                    let omp_block_size = if omp_thread_num + 1 < omp_num_threads {
                        omp_block_stride
                    } else {
                        block_size_usize - omp_block_start
                    };
                    omp_block_start +=
                        usize::try_from(block_start).expect("block_start must be non-negative");

                    let (position, count) = partial_sorting_scan_left_to_right_8u_block_prepare(
                        t,
                        sa_ro,
                        k,
                        &mut state.buckets,
                        &mut state.cache,
                        FastSint::try_from(omp_block_start).expect("block start must fit FastSint"),
                        FastSint::try_from(omp_block_size).expect("block size must fit FastSint"),
                    );
                    state.position = position;
                    state.count = count;
                });
        });
    }

    let induction_offset = 4 * ALPHABET_SIZE;
    let distinct_offset = 2 * ALPHABET_SIZE;
    let (prefix, induction_tail) = buckets.split_at_mut(induction_offset);
    let induction_bucket = &mut induction_tail[..2 * k_usize];
    let distinct_names = &mut prefix[distinct_offset..distinct_offset + 2 * k_usize];

    for tnum in 0..omp_num_threads {
        let state = &mut thread_state[tnum];
        let (temp_induction_bucket, temp_tail) = state.buckets.split_at_mut(2 * k_usize);
        let temp_distinct_names = &mut temp_tail[..2 * k_usize];

        for c in 0..2 * k_usize {
            let a = induction_bucket[c];
            let b = temp_induction_bucket[c];
            induction_bucket[c] = a + b;
            temp_induction_bucket[c] = a;
        }

        d -= 1;
        for c in 0..2 * k_usize {
            let a = distinct_names[c];
            let b = temp_distinct_names[c];
            let next_d = b + d;
            distinct_names[c] = if b > 0 { next_d } else { a };
            temp_distinct_names[c] = a;
        }
        d += 1 + SaSint::try_from(state.position).expect("position must fit SaSint");
        state.position = FastSint::try_from(d).expect("d must fit FastSint") - state.position;
    }

    {
        let sa_ptr = SyncMutPtr::new(sa);
        run_rayon_with_threads(omp_num_threads, || {
            thread_state[..omp_num_threads]
                .par_iter_mut()
                .for_each(|state| {
                    let sa = unsafe { sa_ptr.as_slice() };
                    partial_sorting_scan_left_to_right_8u_block_place(
                        sa,
                        &mut state.buckets,
                        k,
                        &state.cache,
                        state.count,
                        state.position as SaSint,
                    );
                });
        });
    }

    d
}

/// Internal helper: partial sorting shift markers 8u (OpenMP variant).
#[doc(hidden)]
pub fn partial_sorting_shift_markers_8u_omp(
    sa: &mut [SaSint],
    n: SaSint,
    buckets: &[SaSint],
    threads: SaSint,
) {
    let temp_bucket = &buckets[4 * ALPHABET_SIZE..];
    let thread_count = if threads > 1 && n >= 65536 {
        usize::try_from(threads).expect("threads must be positive")
    } else {
        1
    };
    let c_step = buckets_index2(1, 0) as isize;
    let c_min = buckets_index2(1, 0) as isize;
    let c_max = buckets_index2(ALPHABET_SIZE - 1, 0) as isize;
    {
        let sa_ptr = SyncMutPtr::new(sa);
        let buckets_ref: &[SaSint] = buckets;
        let temp_bucket_ref: &[SaSint] = temp_bucket;
        run_rayon_with_threads(thread_count, || {
            (0..thread_count).into_par_iter().for_each(|t| {
                let mut c = c_max - (t as isize * c_step);
                let sa = unsafe { sa_ptr.as_slice() };
                while c >= c_min {
                    let c_usize = c as usize;
                    let mut i = temp_bucket_ref[c_usize] as isize - 1;
                    let mut j = buckets_ref[c_usize - buckets_index2(1, 0)] as isize + 3;
                    let mut s = SAINT_MIN;

                    while i >= j {
                        let p0 = sa[i as usize];
                        let q0 = (p0 & SAINT_MIN) ^ s;
                        s ^= q0;
                        sa[i as usize] = p0 ^ q0;

                        let p1 = sa[(i - 1) as usize];
                        let q1 = (p1 & SAINT_MIN) ^ s;
                        s ^= q1;
                        sa[(i - 1) as usize] = p1 ^ q1;

                        let p2 = sa[(i - 2) as usize];
                        let q2 = (p2 & SAINT_MIN) ^ s;
                        s ^= q2;
                        sa[(i - 2) as usize] = p2 ^ q2;

                        let p3 = sa[(i - 3) as usize];
                        let q3 = (p3 & SAINT_MIN) ^ s;
                        s ^= q3;
                        sa[(i - 3) as usize] = p3 ^ q3;

                        i -= 4;
                    }

                    j -= 3;
                    while i >= j {
                        let p = sa[i as usize];
                        let q = (p & SAINT_MIN) ^ s;
                        s ^= q;
                        sa[i as usize] = p ^ q;
                        i -= 1;
                    }

                    c -= c_step * thread_count as isize;
                }
            });
        });
    }
}

/// Internal helper: partial sorting shift markers 32s 6k (OpenMP variant).
#[doc(hidden)]
pub fn partial_sorting_shift_markers_32s_6k_omp(
    sa: &mut [SaSint],
    k: SaSint,
    buckets: &[SaSint],
    threads: SaSint,
) {
    let k_usize = usize::try_from(k).expect("k must be non-negative");
    let temp_bucket = &buckets[4 * k_usize..];
    let thread_count = if threads > 1 && k >= 65536 {
        usize::try_from(threads).expect("threads must be positive")
    } else {
        1
    };
    {
        let sa_ptr = SyncMutPtr::new(sa);
        let buckets_ref: &[SaSint] = buckets;
        let temp_bucket_ref: &[SaSint] = temp_bucket;
        run_rayon_with_threads(thread_count, || {
            (0..thread_count).into_par_iter().for_each(|t| {
                let mut c = k_usize as isize - 1 - t as isize;
                let sa = unsafe { sa_ptr.as_slice() };
                while c >= 1 {
                    let c_usize = c as usize;
                    let mut i = buckets_ref[buckets_index4(c_usize, 0)] as isize - 1;
                    let mut j = temp_bucket_ref[buckets_index2(c_usize - 1, 0)] as isize + 3;
                    let mut s = SAINT_MIN;

                    while i >= j {
                        let p0 = sa[i as usize];
                        let q0 = (p0 & SAINT_MIN) ^ s;
                        s ^= q0;
                        sa[i as usize] = p0 ^ q0;

                        let p1 = sa[(i - 1) as usize];
                        let q1 = (p1 & SAINT_MIN) ^ s;
                        s ^= q1;
                        sa[(i - 1) as usize] = p1 ^ q1;

                        let p2 = sa[(i - 2) as usize];
                        let q2 = (p2 & SAINT_MIN) ^ s;
                        s ^= q2;
                        sa[(i - 2) as usize] = p2 ^ q2;

                        let p3 = sa[(i - 3) as usize];
                        let q3 = (p3 & SAINT_MIN) ^ s;
                        s ^= q3;
                        sa[(i - 3) as usize] = p3 ^ q3;

                        i -= 4;
                    }

                    j -= 3;
                    while i >= j {
                        let p = sa[i as usize];
                        let q = (p & SAINT_MIN) ^ s;
                        s ^= q;
                        sa[i as usize] = p ^ q;
                        i -= 1;
                    }

                    c -= thread_count as isize;
                }
            });
        });
    }
}

/// Internal helper: partial sorting shift markers 32s 4k.
#[doc(hidden)]
pub fn partial_sorting_shift_markers_32s_4k(sa: &mut [SaSint], n: SaSint) {
    let mut i = n as isize - 1;
    let mut s = SUFFIX_GROUP_MARKER;
    while i >= 3 {
        let p0 = sa[i as usize];
        let q0 =
            ((p0 & SUFFIX_GROUP_MARKER) ^ s) & (((p0 > 0) as SaSint) << (SUFFIX_GROUP_BIT - 1));
        s ^= q0;
        sa[i as usize] = p0 ^ q0;

        let p1 = sa[(i - 1) as usize];
        let q1 =
            ((p1 & SUFFIX_GROUP_MARKER) ^ s) & (((p1 > 0) as SaSint) << (SUFFIX_GROUP_BIT - 1));
        s ^= q1;
        sa[(i - 1) as usize] = p1 ^ q1;

        let p2 = sa[(i - 2) as usize];
        let q2 =
            ((p2 & SUFFIX_GROUP_MARKER) ^ s) & (((p2 > 0) as SaSint) << (SUFFIX_GROUP_BIT - 1));
        s ^= q2;
        sa[(i - 2) as usize] = p2 ^ q2;

        let p3 = sa[(i - 3) as usize];
        let q3 =
            ((p3 & SUFFIX_GROUP_MARKER) ^ s) & (((p3 > 0) as SaSint) << (SUFFIX_GROUP_BIT - 1));
        s ^= q3;
        sa[(i - 3) as usize] = p3 ^ q3;

        i -= 4;
    }

    while i >= 0 {
        let p = sa[i as usize];
        let q = ((p & SUFFIX_GROUP_MARKER) ^ s) & (((p > 0) as SaSint) << (SUFFIX_GROUP_BIT - 1));
        s ^= q;
        sa[i as usize] = p ^ q;
        i -= 1;
    }
}

/// Internal helper: partial sorting shift buckets 32s 6k.
#[doc(hidden)]
pub fn partial_sorting_shift_buckets_32s_6k(k: SaSint, buckets: &mut [SaSint]) {
    let k_usize = usize::try_from(k).expect("k must be non-negative");
    let temp_offset = 4 * k_usize;
    for i in 0..k_usize {
        let src = buckets_index2(i, 0);
        let dst = 2 * src;
        buckets[dst] = buckets[temp_offset + src];
        buckets[dst + 1] = buckets[temp_offset + src + 1];
    }
}

/// Internal helper: partial sorting scan right to left 8u.
#[doc(hidden)]
pub fn partial_sorting_scan_right_to_left_8u(
    t: &[u8],
    sa: &mut [SaSint],
    buckets: &mut [SaSint],
    mut d: SaSint,
    omp_block_start: FastSint,
    omp_block_size: FastSint,
) -> SaSint {
    if omp_block_size <= 0 {
        return d;
    }

    let prefetch_distance = 64usize;
    let (induction_bucket, distinct_names_all) = buckets.split_at_mut(2 * ALPHABET_SIZE);
    let distinct_names = &mut distinct_names_all[..2 * ALPHABET_SIZE];

    let start = usize::try_from(omp_block_start).expect("omp_block_start must be non-negative");
    let size = usize::try_from(omp_block_size).expect("omp_block_size must be non-negative");
    let mut i = start + size - 1;
    let mut j = start + prefetch_distance + 1;

    while i >= j {
        let mut p0 = sa[i];
        d += SaSint::from(p0 < 0);
        p0 &= SAINT_MAX;

        let p0_usize = p0 as usize;
        let v0 = buckets_index2(
            t[p0_usize - 1] as usize,
            usize::from(t[p0_usize - 2] > t[p0_usize - 1]),
        );

        induction_bucket[v0] -= 1;
        let slot0 = induction_bucket[v0] as usize;
        sa[slot0] = (p0 - 1) | (((distinct_names[v0] != d) as SaSint) << (SAINT_BIT - 1));
        distinct_names[v0] = d;

        let mut p1 = sa[i - 1];
        d += SaSint::from(p1 < 0);
        p1 &= SAINT_MAX;

        let p1_usize = p1 as usize;
        let v1 = buckets_index2(
            t[p1_usize - 1] as usize,
            usize::from(t[p1_usize - 2] > t[p1_usize - 1]),
        );

        induction_bucket[v1] -= 1;
        let slot1 = induction_bucket[v1] as usize;
        sa[slot1] = (p1 - 1) | (((distinct_names[v1] != d) as SaSint) << (SAINT_BIT - 1));
        distinct_names[v1] = d;

        i -= 2;
    }

    j = if start + prefetch_distance < start + size {
        start
    } else {
        start
    };
    while i >= j {
        let mut p = sa[i];
        d += SaSint::from(p < 0);
        p &= SAINT_MAX;

        let p_usize = p as usize;
        let v = buckets_index2(
            t[p_usize - 1] as usize,
            usize::from(t[p_usize - 2] > t[p_usize - 1]),
        );

        induction_bucket[v] -= 1;
        let slot = induction_bucket[v] as usize;
        sa[slot] = (p - 1) | (((distinct_names[v] != d) as SaSint) << (SAINT_BIT - 1));
        distinct_names[v] = d;

        if i == 0 {
            break;
        }
        i -= 1;
    }

    d
}

/// Internal helper: partial gsa scan right to left 8u.
#[doc(hidden)]
pub fn partial_gsa_scan_right_to_left_8u(
    t: &[u8],
    sa: &mut [SaSint],
    buckets: &mut [SaSint],
    mut d: SaSint,
    omp_block_start: FastSint,
    omp_block_size: FastSint,
) -> SaSint {
    if omp_block_size <= 0 {
        return d;
    }

    let prefetch_distance = 64usize;
    let (induction_bucket, distinct_names_all) = buckets.split_at_mut(2 * ALPHABET_SIZE);
    let distinct_names = &mut distinct_names_all[..2 * ALPHABET_SIZE];

    let start = usize::try_from(omp_block_start).expect("omp_block_start must be non-negative");
    let size = usize::try_from(omp_block_size).expect("omp_block_size must be non-negative");
    let mut i = start + size - 1;
    let mut j = start + prefetch_distance + 1;

    while i >= j {
        let mut p0 = sa[i];
        d += SaSint::from(p0 < 0);
        p0 &= SAINT_MAX;

        let p0_usize = p0 as usize;
        let v0 = buckets_index2(
            t[p0_usize - 1] as usize,
            usize::from(t[p0_usize - 2] > t[p0_usize - 1]),
        );

        if v0 != 1 {
            induction_bucket[v0] -= 1;
            let slot0 = induction_bucket[v0] as usize;
            sa[slot0] = (p0 - 1) | (((distinct_names[v0] != d) as SaSint) << (SAINT_BIT - 1));
            distinct_names[v0] = d;
        }

        let mut p1 = sa[i - 1];
        d += SaSint::from(p1 < 0);
        p1 &= SAINT_MAX;

        let p1_usize = p1 as usize;
        let v1 = buckets_index2(
            t[p1_usize - 1] as usize,
            usize::from(t[p1_usize - 2] > t[p1_usize - 1]),
        );

        if v1 != 1 {
            induction_bucket[v1] -= 1;
            let slot1 = induction_bucket[v1] as usize;
            sa[slot1] = (p1 - 1) | (((distinct_names[v1] != d) as SaSint) << (SAINT_BIT - 1));
            distinct_names[v1] = d;
        }

        i -= 2;
    }

    j = start;
    while i >= j {
        let mut p = sa[i];
        d += SaSint::from(p < 0);
        p &= SAINT_MAX;

        let p_usize = p as usize;
        let v = buckets_index2(
            t[p_usize - 1] as usize,
            usize::from(t[p_usize - 2] > t[p_usize - 1]),
        );

        if v != 1 {
            induction_bucket[v] -= 1;
            let slot = induction_bucket[v] as usize;
            sa[slot] = (p - 1) | (((distinct_names[v] != d) as SaSint) << (SAINT_BIT - 1));
            distinct_names[v] = d;
        }

        if i == 0 {
            break;
        }
        i -= 1;
    }

    d
}

/// Internal helper: partial sorting scan right to left 8u block prepare.
#[doc(hidden)]
pub fn partial_sorting_scan_right_to_left_8u_block_prepare(
    t: &[u8],
    sa: &[SaSint],
    k: SaSint,
    buckets: &mut [SaSint],
    cache: &mut [ThreadCache],
    omp_block_start: FastSint,
    omp_block_size: FastSint,
) -> (FastSint, FastSint) {
    let k_usize = usize::try_from(k).expect("k must be non-negative");
    let (induction_bucket, distinct_names_all) = buckets.split_at_mut(2 * k_usize);
    let distinct_names = &mut distinct_names_all[..2 * k_usize];
    induction_bucket.fill(0);
    distinct_names.fill(0);

    let start = usize::try_from(omp_block_start).expect("omp_block_start must be non-negative");
    let size = usize::try_from(omp_block_size).expect("omp_block_size must be non-negative");
    let mut count = 0usize;
    let mut d = 1;

    let mut i = start + size;
    while i > start {
        i -= 1;

        let mut p = sa[i];
        cache[count].index = p;
        d += SaSint::from(p < 0);
        p &= SAINT_MAX;

        let p_usize = usize::try_from(p).expect("suffix index must be non-negative");
        let v = buckets_index2(
            t[p_usize - 1] as usize,
            usize::from(t[p_usize - 2] > t[p_usize - 1]),
        );

        cache[count].symbol = v as SaSint;
        induction_bucket[v] += 1;
        distinct_names[v] = d;
        count += 1;
    }

    ((d - 1) as FastSint, count as FastSint)
}

/// Internal helper: partial sorting scan right to left 8u block place.
#[doc(hidden)]
pub fn partial_sorting_scan_right_to_left_8u_block_place(
    sa: &mut [SaSint],
    buckets: &mut [SaSint],
    k: SaSint,
    cache: &[ThreadCache],
    count: FastSint,
    mut d: SaSint,
) {
    let split = 2 * usize::try_from(k).expect("k must be non-negative");
    let (induction_bucket, distinct_names) = buckets.split_at_mut(split);

    let count = usize::try_from(count).expect("count must be non-negative");
    for entry in &cache[..count] {
        let p = entry.index;
        d += SaSint::from(p < 0);
        let v = usize::try_from(entry.symbol).expect("cache symbol must be non-negative");
        induction_bucket[v] -= 1;
        let slot = usize::try_from(induction_bucket[v]).expect("bucket slot must be non-negative");
        sa[slot] = (p - 1) | (((distinct_names[v] != d) as SaSint) << (SAINT_BIT - 1));
        distinct_names[v] = d;
    }
}

/// Internal helper: partial gsa scan right to left 8u block place.
#[doc(hidden)]
pub fn partial_gsa_scan_right_to_left_8u_block_place(
    sa: &mut [SaSint],
    buckets: &mut [SaSint],
    k: SaSint,
    cache: &[ThreadCache],
    count: FastSint,
    mut d: SaSint,
) {
    let split = 2 * usize::try_from(k).expect("k must be non-negative");
    let (induction_bucket, distinct_names) = buckets.split_at_mut(split);

    let count = usize::try_from(count).expect("count must be non-negative");
    for entry in &cache[..count] {
        let p = entry.index;
        d += SaSint::from(p < 0);
        let v = usize::try_from(entry.symbol).expect("cache symbol must be non-negative");
        if v != 1 {
            induction_bucket[v] -= 1;
            let slot =
                usize::try_from(induction_bucket[v]).expect("bucket slot must be non-negative");
            sa[slot] = (p - 1) | (((distinct_names[v] != d) as SaSint) << (SAINT_BIT - 1));
            distinct_names[v] = d;
        }
    }
}

/// Internal helper: partial sorting scan right to left 8u block (OpenMP variant).
#[doc(hidden)]
pub fn partial_sorting_scan_right_to_left_8u_block_omp(
    t: &[u8],
    sa: &mut [SaSint],
    k: SaSint,
    buckets: &mut [SaSint],
    d: SaSint,
    block_start: FastSint,
    block_size: FastSint,
    threads: SaSint,
    thread_state: &mut [ThreadState],
) -> SaSint {
    let mut d = d;
    let block_size_usize = usize::try_from(block_size).expect("block_size must be non-negative");
    let k_usize = usize::try_from(k).expect("k must be non-negative");
    let omp_num_threads = if threads > 1 && block_size_usize >= 64 * k_usize.max(256) {
        usize::try_from(threads)
            .expect("threads must be non-negative")
            .min(thread_state.len())
            .max(1)
    } else {
        1
    };
    let omp_block_stride = (block_size_usize / omp_num_threads) & !15usize;

    if omp_num_threads == 1 {
        return partial_sorting_scan_right_to_left_8u(t, sa, buckets, d, block_start, block_size);
    }

    {
        let sa_ro: &[SaSint] = sa;
        run_rayon_with_threads(omp_num_threads, || {
            thread_state[..omp_num_threads]
                .par_iter_mut()
                .enumerate()
                .for_each(|(omp_thread_num, state)| {
                    let mut omp_block_start = omp_thread_num * omp_block_stride;
                    let omp_block_size = if omp_thread_num + 1 < omp_num_threads {
                        omp_block_stride
                    } else {
                        block_size_usize - omp_block_start
                    };
                    omp_block_start +=
                        usize::try_from(block_start).expect("block_start must be non-negative");

                    let (position, count) = partial_sorting_scan_right_to_left_8u_block_prepare(
                        t,
                        sa_ro,
                        k,
                        &mut state.buckets,
                        &mut state.cache,
                        FastSint::try_from(omp_block_start).expect("block start must fit FastSint"),
                        FastSint::try_from(omp_block_size).expect("block size must fit FastSint"),
                    );
                    state.position = position;
                    state.count = count;
                });
        });
    }

    let distinct_offset = 2 * ALPHABET_SIZE;
    let (induction_bucket, distinct_tail) = buckets.split_at_mut(distinct_offset);
    let distinct_names = &mut distinct_tail[..2 * k_usize];

    for tnum in (0..omp_num_threads).rev() {
        let state = &mut thread_state[tnum];
        let (temp_induction_bucket, temp_tail) = state.buckets.split_at_mut(2 * k_usize);
        let temp_distinct_names = &mut temp_tail[..2 * k_usize];

        for c in 0..2 * k_usize {
            let a = induction_bucket[c];
            let b = temp_induction_bucket[c];
            induction_bucket[c] = a - b;
            temp_induction_bucket[c] = a;
        }

        d -= 1;
        for c in 0..2 * k_usize {
            let a = distinct_names[c];
            let b = temp_distinct_names[c];
            let next_d = b + d;
            distinct_names[c] = if b > 0 { next_d } else { a };
            temp_distinct_names[c] = a;
        }
        d += 1 + SaSint::try_from(state.position).expect("position must fit SaSint");
        state.position = FastSint::try_from(d).expect("d must fit FastSint") - state.position;
    }

    {
        let sa_ptr = SyncMutPtr::new(sa);
        run_rayon_with_threads(omp_num_threads, || {
            thread_state[..omp_num_threads]
                .par_iter_mut()
                .for_each(|state| {
                    let sa = unsafe { sa_ptr.as_slice() };
                    partial_sorting_scan_right_to_left_8u_block_place(
                        sa,
                        &mut state.buckets,
                        k,
                        &state.cache,
                        state.count,
                        state.position as SaSint,
                    );
                });
        });
    }

    d
}

/// Internal helper: partial gsa scan right to left 8u block (OpenMP variant).
#[doc(hidden)]
pub fn partial_gsa_scan_right_to_left_8u_block_omp(
    t: &[u8],
    sa: &mut [SaSint],
    k: SaSint,
    buckets: &mut [SaSint],
    d: SaSint,
    block_start: FastSint,
    block_size: FastSint,
    threads: SaSint,
    thread_state: &mut [ThreadState],
) -> SaSint {
    let mut d = d;
    let block_size_usize = usize::try_from(block_size).expect("block_size must be non-negative");
    let k_usize = usize::try_from(k).expect("k must be non-negative");
    let omp_num_threads = if threads > 1 && block_size_usize >= 64 * k_usize.max(256) {
        usize::try_from(threads)
            .expect("threads must be non-negative")
            .min(thread_state.len())
            .max(1)
    } else {
        1
    };
    let omp_block_stride = (block_size_usize / omp_num_threads) & !15usize;

    if omp_num_threads == 1 {
        return partial_gsa_scan_right_to_left_8u(t, sa, buckets, d, block_start, block_size);
    }

    {
        let sa_ro: &[SaSint] = sa;
        run_rayon_with_threads(omp_num_threads, || {
            thread_state[..omp_num_threads]
                .par_iter_mut()
                .enumerate()
                .for_each(|(omp_thread_num, state)| {
                    let mut omp_block_start = omp_thread_num * omp_block_stride;
                    let omp_block_size = if omp_thread_num + 1 < omp_num_threads {
                        omp_block_stride
                    } else {
                        block_size_usize - omp_block_start
                    };
                    omp_block_start +=
                        usize::try_from(block_start).expect("block_start must be non-negative");

                    let (position, count) = partial_sorting_scan_right_to_left_8u_block_prepare(
                        t,
                        sa_ro,
                        k,
                        &mut state.buckets,
                        &mut state.cache,
                        FastSint::try_from(omp_block_start).expect("block start must fit FastSint"),
                        FastSint::try_from(omp_block_size).expect("block size must fit FastSint"),
                    );
                    state.position = position;
                    state.count = count;
                });
        });
    }

    let distinct_offset = 2 * ALPHABET_SIZE;
    let (induction_bucket, distinct_tail) = buckets.split_at_mut(distinct_offset);
    let distinct_names = &mut distinct_tail[..2 * k_usize];

    for tnum in (0..omp_num_threads).rev() {
        let state = &mut thread_state[tnum];
        let (temp_induction_bucket, temp_tail) = state.buckets.split_at_mut(2 * k_usize);
        let temp_distinct_names = &mut temp_tail[..2 * k_usize];

        for c in 0..2 * k_usize {
            let a = induction_bucket[c];
            let b = temp_induction_bucket[c];
            induction_bucket[c] = a - b;
            temp_induction_bucket[c] = a;
        }

        d -= 1;
        for c in 0..2 * k_usize {
            let a = distinct_names[c];
            let b = temp_distinct_names[c];
            let next_d = b + d;
            distinct_names[c] = if b > 0 { next_d } else { a };
            temp_distinct_names[c] = a;
        }
        d += 1 + SaSint::try_from(state.position).expect("position must fit SaSint");
        state.position = FastSint::try_from(d).expect("d must fit FastSint") - state.position;
    }

    {
        let sa_ptr = SyncMutPtr::new(sa);
        run_rayon_with_threads(omp_num_threads, || {
            thread_state[..omp_num_threads]
                .par_iter_mut()
                .for_each(|state| {
                    let sa = unsafe { sa_ptr.as_slice() };
                    partial_gsa_scan_right_to_left_8u_block_place(
                        sa,
                        &mut state.buckets,
                        k,
                        &state.cache,
                        state.count,
                        state.position as SaSint,
                    );
                });
        });
    }

    d
}

/// Internal helper: partial sorting scan right to left 8u (OpenMP variant).
#[doc(hidden)]
pub fn partial_sorting_scan_right_to_left_8u_omp(
    t: &[u8],
    sa: &mut [SaSint],
    n: SaSint,
    k: SaSint,
    buckets: &mut [SaSint],
    first_lms_suffix: SaSint,
    left_suffixes_count: SaSint,
    mut d: SaSint,
    threads: SaSint,
    thread_state: &mut [ThreadState],
) {
    let scan_start = left_suffixes_count as FastSint + 1;
    let scan_end = n as FastSint - first_lms_suffix as FastSint;

    if threads == 1 || (scan_end - scan_start) < 65_536 {
        let _ = partial_sorting_scan_right_to_left_8u(
            t,
            sa,
            buckets,
            d,
            scan_start,
            scan_end - scan_start,
        );
        return;
    }

    let distinct_offset = 2 * ALPHABET_SIZE;

    let mut block_start = usize::try_from(scan_end - 1).expect("scan end must be positive");
    let scan_start_usize = usize::try_from(scan_start).expect("scan_start must be non-negative");
    let threads_usize = usize::try_from(threads)
        .expect("threads must be non-negative")
        .min(thread_state.len())
        .max(1);

    while block_start >= scan_start_usize {
        if sa[block_start] == 0 {
            if block_start == 0 {
                break;
            }
            block_start -= 1;
        } else {
            let mut block_max_end = block_start.saturating_sub(
                threads_usize * (LIBSAIS_PER_THREAD_CACHE_SIZE - 16 * threads_usize),
            );
            if block_max_end + 1 < scan_start_usize {
                block_max_end = scan_start_usize.saturating_sub(1);
            }
            let mut block_end = block_start - 1;
            while block_end > block_max_end && sa[block_end] != 0 {
                block_end -= 1;
            }
            let block_size = block_start - block_end;

            if block_size < 32 {
                while block_start > block_end {
                    let p = sa[block_start];
                    d += SaSint::from(p < 0);
                    let p = p & SAINT_MAX;
                    let v = buckets_index2(
                        t[(p - 1) as usize] as usize,
                        usize::from(t[(p - 2) as usize] > t[(p - 1) as usize]),
                    );
                    buckets[v] -= 1;
                    let slot =
                        usize::try_from(buckets[v]).expect("bucket slot must be non-negative");
                    sa[slot] = (p - 1)
                        | (((buckets[distinct_offset + v] != d) as SaSint) << (SAINT_BIT - 1));
                    buckets[distinct_offset + v] = d;

                    if block_start == 0 {
                        break;
                    }
                    block_start -= 1;
                }
            } else {
                d = partial_sorting_scan_right_to_left_8u_block_omp(
                    t,
                    sa,
                    k,
                    buckets,
                    d,
                    FastSint::try_from(block_end + 1).expect("block start must fit FastSint"),
                    FastSint::try_from(block_size).expect("block size must fit FastSint"),
                    threads,
                    thread_state,
                );
                block_start = block_end;
            }
        }
    }
}

/// Internal helper: partial gsa scan right to left 8u (OpenMP variant).
#[doc(hidden)]
pub fn partial_gsa_scan_right_to_left_8u_omp(
    t: &[u8],
    sa: &mut [SaSint],
    n: SaSint,
    k: SaSint,
    buckets: &mut [SaSint],
    first_lms_suffix: SaSint,
    left_suffixes_count: SaSint,
    mut d: SaSint,
    threads: SaSint,
    thread_state: &mut [ThreadState],
) {
    let scan_start = left_suffixes_count as FastSint + 1;
    let scan_end = n as FastSint - first_lms_suffix as FastSint;

    if threads == 1 || (scan_end - scan_start) < 65_536 {
        let _ =
            partial_gsa_scan_right_to_left_8u(t, sa, buckets, d, scan_start, scan_end - scan_start);
        return;
    }

    let distinct_offset = 2 * ALPHABET_SIZE;
    let mut block_start = usize::try_from(scan_end - 1).expect("scan end must be positive");
    let scan_start_usize = usize::try_from(scan_start).expect("scan_start must be non-negative");
    let threads_usize = usize::try_from(threads)
        .expect("threads must be non-negative")
        .min(thread_state.len())
        .max(1);

    while block_start >= scan_start_usize {
        if sa[block_start] == 0 {
            if block_start == 0 {
                break;
            }
            block_start -= 1;
        } else {
            let mut block_max_end = block_start.saturating_sub(
                threads_usize * (LIBSAIS_PER_THREAD_CACHE_SIZE - 16 * threads_usize),
            );
            if block_max_end + 1 < scan_start_usize {
                block_max_end = scan_start_usize.saturating_sub(1);
            }
            let mut block_end = block_start - 1;
            while block_end > block_max_end && sa[block_end] != 0 {
                block_end -= 1;
            }
            let block_size = block_start - block_end;

            if block_size < 32 {
                while block_start > block_end {
                    let p = sa[block_start];
                    d += SaSint::from(p < 0);
                    let p = p & SAINT_MAX;
                    let v = buckets_index2(
                        t[(p - 1) as usize] as usize,
                        usize::from(t[(p - 2) as usize] > t[(p - 1) as usize]),
                    );
                    if v != 1 {
                        buckets[v] -= 1;
                        let slot =
                            usize::try_from(buckets[v]).expect("bucket slot must be non-negative");
                        sa[slot] = (p - 1)
                            | (((buckets[distinct_offset + v] != d) as SaSint) << (SAINT_BIT - 1));
                        buckets[distinct_offset + v] = d;
                    }

                    if block_start == 0 {
                        break;
                    }
                    block_start -= 1;
                }
            } else {
                d = partial_gsa_scan_right_to_left_8u_block_omp(
                    t,
                    sa,
                    k,
                    buckets,
                    d,
                    FastSint::try_from(block_end + 1).expect("block start must fit FastSint"),
                    FastSint::try_from(block_size).expect("block size must fit FastSint"),
                    threads,
                    thread_state,
                );
                block_start = block_end;
            }
        }
    }
}

/// Internal helper: partial sorting scan right to left 32s 6k.
#[doc(hidden)]
pub fn partial_sorting_scan_right_to_left_32s_6k(
    t: &[SaSint],
    sa: &mut [SaSint],
    buckets: &mut [SaSint],
    mut d: SaSint,
    omp_block_start: FastSint,
    omp_block_size: FastSint,
) -> SaSint {
    if omp_block_size <= 0 {
        return d;
    }

    let prefetch_distance: FastSint = 64;
    let mut i = omp_block_start + omp_block_size - 1;
    let mut j = omp_block_start + 2 * prefetch_distance + 1;

    while i >= j {
        let mut p0 = sa[i as usize];
        d += SaSint::from(p0 < 0);
        p0 &= SAINT_MAX;
        let p0u = p0 as usize;
        let v0 = buckets_index4(t[p0u - 1] as usize, usize::from(t[p0u - 2] > t[p0u - 1]));
        buckets[v0] -= 1;
        let slot0 = buckets[v0] as usize;
        sa[slot0] = (p0 - 1) | (((buckets[2 + v0] != d) as SaSint) << (SAINT_BIT - 1));
        buckets[2 + v0] = d;

        let mut p1 = sa[(i - 1) as usize];
        d += SaSint::from(p1 < 0);
        p1 &= SAINT_MAX;
        let p1u = p1 as usize;
        let v1 = buckets_index4(t[p1u - 1] as usize, usize::from(t[p1u - 2] > t[p1u - 1]));
        buckets[v1] -= 1;
        let slot1 = buckets[v1] as usize;
        sa[slot1] = (p1 - 1) | (((buckets[2 + v1] != d) as SaSint) << (SAINT_BIT - 1));
        buckets[2 + v1] = d;

        i -= 2;
    }

    j -= 2 * prefetch_distance + 1;
    while i >= j {
        let mut p = sa[i as usize];
        d += SaSint::from(p < 0);
        p &= SAINT_MAX;
        let pu = p as usize;
        let v = buckets_index4(t[pu - 1] as usize, usize::from(t[pu - 2] > t[pu - 1]));

        buckets[v] -= 1;
        let slot = buckets[v] as usize;
        sa[slot] = (p - 1) | (((buckets[2 + v] != d) as SaSint) << (SAINT_BIT - 1));
        buckets[2 + v] = d;
        i -= 1;
    }

    d
}

/// Internal helper: partial sorting scan right to left 32s 4k.
#[doc(hidden)]
pub fn partial_sorting_scan_right_to_left_32s_4k(
    t: &[SaSint],
    sa: &mut [SaSint],
    k: SaSint,
    buckets: &mut [SaSint],
    mut d: SaSint,
    omp_block_start: FastSint,
    omp_block_size: FastSint,
) -> SaSint {
    if omp_block_size <= 0 {
        return d;
    }

    let k_usize = usize::try_from(k).expect("k must be non-negative");
    let prefetch_distance: FastSint = 64;
    let induction_offset = 3 * k_usize;

    let mut i = omp_block_start + omp_block_size - 1;
    let mut j = omp_block_start + 2 * prefetch_distance + 1;

    while i >= j {
        let i0 = i as usize;
        let mut p0 = sa[i0];
        if p0 > 0 {
            sa[i0] = 0;
            d += p0 >> (SUFFIX_GROUP_BIT - 1);
            p0 &= !SUFFIX_GROUP_MARKER;

            let p0u = p0 as usize;
            let c0 = t[p0u - 1];
            let f0 = usize::from(t[p0u - 2] > c0);
            let v0 = buckets_index2(c0 as usize, f0);
            let c0u = c0 as usize;
            buckets[induction_offset + c0u] -= 1;
            let slot0 = buckets[induction_offset + c0u] as usize;
            sa[slot0] = (p0 - 1)
                | ((f0 as SaSint) << (SAINT_BIT - 1))
                | (((buckets[v0] != d) as SaSint) << (SUFFIX_GROUP_BIT - 1));
            buckets[v0] = d;
        }

        let i1 = (i - 1) as usize;
        let mut p1 = sa[i1];
        if p1 > 0 {
            sa[i1] = 0;
            d += p1 >> (SUFFIX_GROUP_BIT - 1);
            p1 &= !SUFFIX_GROUP_MARKER;

            let p1u = p1 as usize;
            let c1 = t[p1u - 1];
            let f1 = usize::from(t[p1u - 2] > c1);
            let v1 = buckets_index2(c1 as usize, f1);
            let c1u = c1 as usize;
            buckets[induction_offset + c1u] -= 1;
            let slot1 = buckets[induction_offset + c1u] as usize;
            sa[slot1] = (p1 - 1)
                | ((f1 as SaSint) << (SAINT_BIT - 1))
                | (((buckets[v1] != d) as SaSint) << (SUFFIX_GROUP_BIT - 1));
            buckets[v1] = d;
        }

        i -= 2;
    }

    j -= 2 * prefetch_distance + 1;
    while i >= j {
        let iu = i as usize;
        let mut p = sa[iu];
        if p > 0 {
            sa[iu] = 0;
            d += p >> (SUFFIX_GROUP_BIT - 1);
            p &= !SUFFIX_GROUP_MARKER;

            let pu = p as usize;
            let c = t[pu - 1];
            let f = usize::from(t[pu - 2] > c);
            let v = buckets_index2(c as usize, f);
            let cu = c as usize;
            buckets[induction_offset + cu] -= 1;
            let slot = buckets[induction_offset + cu] as usize;
            sa[slot] = (p - 1)
                | ((f as SaSint) << (SAINT_BIT - 1))
                | (((buckets[v] != d) as SaSint) << (SUFFIX_GROUP_BIT - 1));
            buckets[v] = d;
        }
        i -= 1;
    }

    d
}

/// Internal helper: partial sorting scan right to left 32s 1k.
#[doc(hidden)]
pub fn partial_sorting_scan_right_to_left_32s_1k(
    t: &[SaSint],
    sa: &mut [SaSint],
    induction_bucket: &mut [SaSint],
    omp_block_start: FastSint,
    omp_block_size: FastSint,
) {
    if omp_block_size <= 0 {
        return;
    }

    let prefetch_distance = 64usize;
    let start = usize::try_from(omp_block_start).expect("omp_block_start must be non-negative");
    let size = usize::try_from(omp_block_size).expect("omp_block_size must be non-negative");
    let mut i = (start + size - 1) as isize;
    let mut j = (start + 2 * prefetch_distance + 1) as isize;

    while i >= j {
        let p0 = sa[i as usize];
        if p0 > 0 {
            sa[i as usize] = 0;
            let p0_usize = usize::try_from(p0).expect("suffix index must be non-negative");
            let bucket_index0 =
                usize::try_from(t[p0_usize - 1]).expect("bucket symbol must be non-negative");
            induction_bucket[bucket_index0] -= 1;
            let slot0 = usize::try_from(induction_bucket[bucket_index0])
                .expect("bucket slot must be non-negative");
            sa[slot0] = (p0 - 1)
                | ((usize::from(t[p0_usize - 2] > t[p0_usize - 1]) as SaSint) << (SAINT_BIT - 1));
        }
        let p1 = sa[(i - 1) as usize];
        if p1 > 0 {
            sa[(i - 1) as usize] = 0;
            let p1_usize = usize::try_from(p1).expect("suffix index must be non-negative");
            let bucket_index1 =
                usize::try_from(t[p1_usize - 1]).expect("bucket symbol must be non-negative");
            induction_bucket[bucket_index1] -= 1;
            let slot1 = usize::try_from(induction_bucket[bucket_index1])
                .expect("bucket slot must be non-negative");
            sa[slot1] = (p1 - 1)
                | ((usize::from(t[p1_usize - 2] > t[p1_usize - 1]) as SaSint) << (SAINT_BIT - 1));
        }

        i -= 2;
    }

    j -= (2 * prefetch_distance + 1) as isize;
    while i >= j {
        let p = sa[i as usize];
        if p > 0 {
            sa[i as usize] = 0;
            let p_usize = usize::try_from(p).expect("suffix index must be non-negative");
            let bucket_index =
                usize::try_from(t[p_usize - 1]).expect("bucket symbol must be non-negative");
            induction_bucket[bucket_index] -= 1;
            let slot = usize::try_from(induction_bucket[bucket_index])
                .expect("bucket slot must be non-negative");
            sa[slot] = (p - 1)
                | ((usize::from(t[p_usize - 2] > t[p_usize - 1]) as SaSint) << (SAINT_BIT - 1));
        }
        if i == 0 {
            break;
        }
        i -= 1;
    }
}

/// Internal helper: partial sorting scan right to left 32s 6k block gather.
#[doc(hidden)]
pub fn partial_sorting_scan_right_to_left_32s_6k_block_gather(
    t: &[SaSint],
    sa: &[SaSint],
    cache: &mut [ThreadCache],
    omp_block_start: FastSint,
    omp_block_size: FastSint,
) {
    if omp_block_size <= 0 {
        return;
    }

    let start = usize::try_from(omp_block_start).expect("omp_block_start must be non-negative");
    let size = usize::try_from(omp_block_size).expect("omp_block_size must be non-negative");
    for offset in 0..size {
        let i = start + offset;
        let mut p = sa[i];
        let mut symbol = 0usize;
        p &= SAINT_MAX;
        if p != 0 {
            let p_usize = usize::try_from(p).expect("suffix index must be non-negative");
            symbol = buckets_index4(
                usize::try_from(t[p_usize - 1]).expect("bucket symbol must be non-negative"),
                usize::from(t[p_usize - 2] > t[p_usize - 1]),
            );
        }
        cache[offset].index = sa[i];
        cache[offset].symbol = symbol as SaSint;
    }
}

/// Internal helper: partial sorting scan right to left 32s 4k block gather.
#[doc(hidden)]
pub fn partial_sorting_scan_right_to_left_32s_4k_block_gather(
    t: &[SaSint],
    sa: &mut [SaSint],
    cache: &mut [ThreadCache],
    omp_block_start: FastSint,
    omp_block_size: FastSint,
) {
    if omp_block_size <= 0 {
        return;
    }

    let start = usize::try_from(omp_block_start).expect("omp_block_start must be non-negative");
    let size = usize::try_from(omp_block_size).expect("omp_block_size must be non-negative");
    for offset in 0..size {
        let i = start + offset;
        let mut symbol = SAINT_MIN;
        let mut p = sa[i];
        if p > 0 {
            sa[i] = 0;
            cache[offset].index = p;
            p &= !SUFFIX_GROUP_MARKER;
            let p_usize = usize::try_from(p).expect("suffix index must be non-negative");
            symbol = buckets_index2(
                usize::try_from(t[p_usize - 1]).expect("bucket symbol must be non-negative"),
                usize::from(t[p_usize - 2] > t[p_usize - 1]),
            ) as SaSint;
        }
        cache[offset].symbol = symbol;
    }
}

/// Internal helper: partial sorting scan right to left 32s 1k block gather.
#[doc(hidden)]
pub fn partial_sorting_scan_right_to_left_32s_1k_block_gather(
    t: &[SaSint],
    sa: &mut [SaSint],
    cache: &mut [ThreadCache],
    omp_block_start: FastSint,
    omp_block_size: FastSint,
) {
    if omp_block_size <= 0 {
        return;
    }
    let start = usize::try_from(omp_block_start).expect("omp_block_start must be non-negative");
    let size = usize::try_from(omp_block_size).expect("omp_block_size must be non-negative");
    for offset in 0..size {
        let i = start + offset;
        let mut symbol = SAINT_MIN;
        let p = sa[i];
        if p > 0 {
            sa[i] = 0;
            cache[offset].index = (p - 1)
                | ((usize::from(t[p as usize - 2] > t[p as usize - 1]) as SaSint)
                    << (SAINT_BIT - 1));
            symbol = t[p as usize - 1];
        }
        cache[offset].symbol = symbol;
    }
}

/// Internal helper: partial sorting scan right to left 32s 6k block sort.
#[doc(hidden)]
pub fn partial_sorting_scan_right_to_left_32s_6k_block_sort(
    t: &[SaSint],
    buckets: &mut [SaSint],
    mut d: SaSint,
    cache: &mut [ThreadCache],
    omp_block_start: FastSint,
    omp_block_size: FastSint,
) -> SaSint {
    if omp_block_size <= 0 {
        return d;
    }

    let size = usize::try_from(omp_block_size).expect("omp_block_size must be non-negative");
    let mut i = size;
    while i > 0 {
        i -= 1;

        let v = usize::try_from(cache[i].symbol).expect("cache symbol must be non-negative");
        let p = cache[i].index;
        d += SaSint::from(p < 0);
        buckets[v] -= 1;
        let target = buckets[v];
        cache[i].symbol = target;
        cache[i].index = (p - 1) | (((buckets[2 + v] != d) as SaSint) << (SAINT_BIT - 1));
        buckets[2 + v] = d;

        if target >= omp_block_start as SaSint
            && target < (omp_block_start + omp_block_size) as SaSint
        {
            let s = usize::try_from(target - omp_block_start as SaSint)
                .expect("cache slot must be non-negative");
            let q = cache[i].index & SAINT_MAX;
            let q_usize = usize::try_from(q).expect("suffix index must be non-negative");
            cache[s].index = cache[i].index;
            cache[s].symbol = buckets_index4(
                usize::try_from(t[q_usize - 1]).expect("bucket symbol must be non-negative"),
                usize::from(t[q_usize - 2] > t[q_usize - 1]),
            ) as SaSint;
        }
    }

    d
}

/// Internal helper: partial sorting scan right to left 32s 4k block sort.
#[doc(hidden)]
pub fn partial_sorting_scan_right_to_left_32s_4k_block_sort(
    t: &[SaSint],
    k: SaSint,
    buckets: &mut [SaSint],
    mut d: SaSint,
    cache: &mut [ThreadCache],
    omp_block_start: FastSint,
    omp_block_size: FastSint,
) -> SaSint {
    if omp_block_size <= 0 {
        return d;
    }

    let k_usize = usize::try_from(k).expect("k must be non-negative");
    let (distinct_names, tail) = buckets.split_at_mut(2 * k_usize);
    let induction_bucket = &mut tail[k_usize..2 * k_usize];

    let size = usize::try_from(omp_block_size).expect("omp_block_size must be non-negative");
    let mut i = size;
    while i > 0 {
        i -= 1;

        let v = cache[i].symbol;
        if v >= 0 {
            let p = cache[i].index;
            d += p >> (SUFFIX_GROUP_BIT - 1);
            let bucket_index = usize::try_from(v >> 1).expect("bucket symbol must be non-negative");
            induction_bucket[bucket_index] -= 1;
            let target = induction_bucket[bucket_index];
            cache[i].symbol = target;
            cache[i].index = (p - 1)
                | ((v & 1) << (SAINT_BIT - 1))
                | (((distinct_names
                    [usize::try_from(v).expect("bucket symbol must be non-negative")]
                    != d) as SaSint)
                    << (SUFFIX_GROUP_BIT - 1));
            distinct_names[usize::try_from(v).expect("bucket symbol must be non-negative")] = d;

            if target >= omp_block_start as SaSint
                && target < (omp_block_start + omp_block_size) as SaSint
            {
                let ni = usize::try_from(target - omp_block_start as SaSint)
                    .expect("cache slot must be non-negative");
                let mut np = cache[i].index;
                if np > 0 {
                    cache[i].index = 0;
                    cache[ni].index = np;
                    np &= !SUFFIX_GROUP_MARKER;
                    let np_usize = usize::try_from(np).expect("suffix index must be non-negative");
                    cache[ni].symbol = buckets_index2(
                        usize::try_from(t[np_usize - 1])
                            .expect("bucket symbol must be non-negative"),
                        usize::from(t[np_usize - 2] > t[np_usize - 1]),
                    ) as SaSint;
                }
            }
        }
    }

    d
}

/// Internal helper: partial sorting scan right to left 32s 1k block sort.
#[doc(hidden)]
pub fn partial_sorting_scan_right_to_left_32s_1k_block_sort(
    t: &[SaSint],
    induction_bucket: &mut [SaSint],
    cache: &mut [ThreadCache],
    omp_block_start: FastSint,
    omp_block_size: FastSint,
) {
    if omp_block_size <= 0 {
        return;
    }
    let size = usize::try_from(omp_block_size).expect("omp_block_size must be non-negative");
    let mut offset = size;

    while offset > 0 {
        offset -= 1;
        let v = cache[offset].symbol;
        if v >= 0 {
            let bucket_index = v as usize;
            induction_bucket[bucket_index] -= 1;
            let target = induction_bucket[bucket_index];
            cache[offset].symbol = target;
            let block_end = omp_block_start as SaSint + omp_block_size as SaSint;
            if target >= omp_block_start as SaSint && target < block_end {
                let ni = usize::try_from(target - omp_block_start as SaSint)
                    .expect("cache slot must be non-negative");
                let np = cache[offset].index;
                if np > 0 {
                    cache[offset].index = 0;
                    cache[ni].index = (np - 1)
                        | ((usize::from(t[np as usize - 2] > t[np as usize - 1]) as SaSint)
                            << (SAINT_BIT - 1));
                    cache[ni].symbol = t[np as usize - 1];
                }
            }
        }
    }
}

/// Internal helper: partial sorting scan right to left 32s 6k block (OpenMP variant).
#[doc(hidden)]
pub fn partial_sorting_scan_right_to_left_32s_6k_block_omp(
    t: &[SaSint],
    sa: &mut [SaSint],
    buckets: &mut [SaSint],
    mut d: SaSint,
    cache: &mut [ThreadCache],
    block_start: FastSint,
    block_size: FastSint,
    threads: SaSint,
) -> SaSint {
    if block_size <= 0 {
        return d;
    }
    if threads == 1 || block_size < 16_384 {
        return partial_sorting_scan_right_to_left_32s_6k(
            t,
            sa,
            buckets,
            d,
            block_start,
            block_size,
        );
    }

    let threads_usize = usize::try_from(threads)
        .expect("threads must be non-negative")
        .max(1);
    let block_size_usize = usize::try_from(block_size).expect("block_size must be non-negative");
    let omp_num_threads = threads_usize.min(block_size_usize.max(1));
    let omp_block_stride = (block_size_usize / omp_num_threads) & !15usize;

    {
        let sa_ro: &[SaSint] = sa;
        let t_ro: &[SaSint] = t;
        let cache_ptr = SyncMutPtr::new(cache);
        run_rayon_with_threads(omp_num_threads, || {
            (0..omp_num_threads)
                .into_par_iter()
                .for_each(|omp_thread_num| {
                    let omp_block_size = if omp_thread_num + 1 < omp_num_threads {
                        omp_block_stride
                    } else {
                        block_size_usize - omp_thread_num * omp_block_stride
                    };
                    let omp_block_start = usize::try_from(block_start)
                        .expect("block_start must be non-negative")
                        + omp_thread_num * omp_block_stride;
                    if omp_block_size > 0 {
                        let cache = unsafe { cache_ptr.as_slice() };
                        partial_sorting_scan_right_to_left_32s_6k_block_gather(
                            t_ro,
                            sa_ro,
                            &mut cache[omp_thread_num * omp_block_stride
                                ..omp_thread_num * omp_block_stride + omp_block_size],
                            omp_block_start as FastSint,
                            omp_block_size as FastSint,
                        );
                    }
                });
        });
    }

    d = partial_sorting_scan_right_to_left_32s_6k_block_sort(
        t,
        buckets,
        d,
        &mut cache[..block_size_usize],
        block_start,
        block_size,
    );

    {
        let sa_ptr = SyncMutPtr::new(sa);
        let cache_ro: &[ThreadCache] = cache;
        run_rayon_with_threads(omp_num_threads, || {
            (0..omp_num_threads)
                .into_par_iter()
                .for_each(|omp_thread_num| {
                    let omp_block_size = if omp_thread_num + 1 < omp_num_threads {
                        omp_block_stride
                    } else {
                        block_size_usize - omp_thread_num * omp_block_stride
                    };
                    let cache_start = omp_thread_num * omp_block_stride;
                    if omp_block_size > 0 {
                        let sa = unsafe { sa_ptr.as_slice() };
                        place_cached_suffixes(
                            sa,
                            &cache_ro[cache_start..],
                            0,
                            omp_block_size as FastSint,
                        );
                    }
                });
        });
    }

    d
}

/// Internal helper: partial sorting scan right to left 32s 4k block (OpenMP variant).
#[doc(hidden)]
pub fn partial_sorting_scan_right_to_left_32s_4k_block_omp(
    t: &[SaSint],
    sa: &mut [SaSint],
    k: SaSint,
    buckets: &mut [SaSint],
    mut d: SaSint,
    cache: &mut [ThreadCache],
    block_start: FastSint,
    block_size: FastSint,
    threads: SaSint,
) -> SaSint {
    if block_size <= 0 {
        return d;
    }
    if threads == 1 || block_size < 16_384 {
        return partial_sorting_scan_right_to_left_32s_4k(
            t,
            sa,
            k,
            buckets,
            d,
            block_start,
            block_size,
        );
    }

    let threads_usize = usize::try_from(threads)
        .expect("threads must be non-negative")
        .max(1);
    let block_size_usize = usize::try_from(block_size).expect("block_size must be non-negative");
    let omp_num_threads = threads_usize.min(block_size_usize.max(1));
    let omp_block_stride = (block_size_usize / omp_num_threads) & !15usize;

    {
        let sa_ptr = SyncMutPtr::new(sa);
        let t_ro: &[SaSint] = t;
        let cache_ptr = SyncMutPtr::new(cache);
        run_rayon_with_threads(omp_num_threads, || {
            (0..omp_num_threads)
                .into_par_iter()
                .for_each(|omp_thread_num| {
                    let omp_block_size = if omp_thread_num + 1 < omp_num_threads {
                        omp_block_stride
                    } else {
                        block_size_usize - omp_thread_num * omp_block_stride
                    };
                    let omp_block_start = usize::try_from(block_start)
                        .expect("block_start must be non-negative")
                        + omp_thread_num * omp_block_stride;
                    if omp_block_size > 0 {
                        let cache = unsafe { cache_ptr.as_slice() };
                        let sa = unsafe { sa_ptr.as_slice() };
                        partial_sorting_scan_right_to_left_32s_4k_block_gather(
                            t_ro,
                            sa,
                            &mut cache[omp_thread_num * omp_block_stride
                                ..omp_thread_num * omp_block_stride + omp_block_size],
                            omp_block_start as FastSint,
                            omp_block_size as FastSint,
                        );
                    }
                });
        });
    }

    d = partial_sorting_scan_right_to_left_32s_4k_block_sort(
        t,
        k,
        buckets,
        d,
        &mut cache[..block_size_usize],
        block_start,
        block_size,
    );

    {
        let sa_ptr = SyncMutPtr::new(sa);
        let cache_ptr = SyncMutPtr::new(cache);
        run_rayon_with_threads(omp_num_threads, || {
            (0..omp_num_threads)
                .into_par_iter()
                .for_each(|omp_thread_num| {
                    let omp_block_start = omp_thread_num * omp_block_stride;
                    let omp_block_size = if omp_thread_num + 1 < omp_num_threads {
                        omp_block_stride
                    } else {
                        block_size_usize - omp_block_start
                    };
                    if omp_block_size > 0 {
                        let sa = unsafe { sa_ptr.as_slice() };
                        let cache = unsafe { cache_ptr.as_slice() };
                        compact_and_place_cached_suffixes(
                            sa,
                            &mut cache[omp_block_start..],
                            0,
                            omp_block_size as FastSint,
                        );
                    }
                });
        });
    }

    d
}

/// Internal helper: partial sorting scan right to left 32s 1k block (OpenMP variant).
#[doc(hidden)]
pub fn partial_sorting_scan_right_to_left_32s_1k_block_omp(
    t: &[SaSint],
    sa: &mut [SaSint],
    buckets: &mut [SaSint],
    cache: &mut [ThreadCache],
    block_start: FastSint,
    block_size: FastSint,
    threads: SaSint,
) {
    if block_size <= 0 {
        return;
    }
    if threads == 1 || block_size < 16_384 {
        partial_sorting_scan_right_to_left_32s_1k(t, sa, buckets, block_start, block_size);
        return;
    }

    let threads_usize = usize::try_from(threads)
        .expect("threads must be non-negative")
        .max(1);
    let block_size_usize = usize::try_from(block_size).expect("block_size must be non-negative");
    let block_start_usize = usize::try_from(block_start).expect("block_start must be non-negative");
    let omp_num_threads = threads_usize.min(block_size_usize.max(1));
    let omp_block_stride = (block_size_usize / omp_num_threads) & !15usize;

    {
        let sa_ptr = SyncMutPtr::new(sa);
        let t_ro: &[SaSint] = t;
        let cache_ptr = SyncMutPtr::new(cache);
        run_rayon_with_threads(omp_num_threads, || {
            (0..omp_num_threads)
                .into_par_iter()
                .for_each(|omp_thread_num| {
                    let omp_block_size = if omp_thread_num + 1 < omp_num_threads {
                        omp_block_stride
                    } else {
                        block_size_usize - omp_thread_num * omp_block_stride
                    };
                    let omp_block_start = block_start_usize + omp_thread_num * omp_block_stride;
                    if omp_block_size > 0 {
                        let cache = unsafe { cache_ptr.as_slice() };
                        let sa = unsafe { sa_ptr.as_slice() };
                        partial_sorting_scan_right_to_left_32s_1k_block_gather(
                            t_ro,
                            sa,
                            &mut cache[omp_thread_num * omp_block_stride
                                ..omp_thread_num * omp_block_stride + omp_block_size],
                            omp_block_start as FastSint,
                            omp_block_size as FastSint,
                        );
                    }
                });
        });
    }

    let cache = &mut cache[..block_size_usize];
    partial_sorting_scan_right_to_left_32s_1k_block_sort(
        t,
        buckets,
        cache,
        block_start,
        block_size,
    );
    {
        let sa_ptr = SyncMutPtr::new(sa);
        let cache_ptr = SyncMutPtr::new(cache);
        run_rayon_with_threads(omp_num_threads, || {
            (0..omp_num_threads)
                .into_par_iter()
                .for_each(|omp_thread_num| {
                    let omp_block_start = omp_thread_num * omp_block_stride;
                    let omp_block_size = if omp_thread_num + 1 < omp_num_threads {
                        omp_block_stride
                    } else {
                        block_size_usize - omp_block_start
                    };
                    if omp_block_size > 0 {
                        let sa = unsafe { sa_ptr.as_slice() };
                        let cache = unsafe { cache_ptr.as_slice() };
                        compact_and_place_cached_suffixes(
                            sa,
                            &mut cache[omp_block_start..],
                            0,
                            omp_block_size as FastSint,
                        );
                    }
                });
        });
    }
}

/// Internal helper: partial sorting scan left to right 32s 6k block gather.
#[doc(hidden)]
pub fn partial_sorting_scan_left_to_right_32s_6k_block_gather(
    t: &[SaSint],
    sa: &mut [SaSint],
    cache: &mut [ThreadCache],
    omp_block_start: FastSint,
    omp_block_size: FastSint,
) {
    if omp_block_size <= 0 {
        return;
    }

    let start = usize::try_from(omp_block_start).expect("omp_block_start must be non-negative");
    let size = usize::try_from(omp_block_size).expect("omp_block_size must be non-negative");
    for offset in 0..size {
        let i = start + offset;
        let p = sa[i];
        cache[offset].index = p;
        let q = p & SAINT_MAX;
        cache[offset].symbol = if q != 0 {
            buckets_index4(
                usize::try_from(t[q as usize - 1]).expect("bucket symbol must be non-negative"),
                usize::from(t[q as usize - 2] >= t[q as usize - 1]),
            ) as SaSint
        } else {
            0
        };
    }
}

/// Internal helper: partial sorting scan left to right 32s 4k block gather.
#[doc(hidden)]
pub fn partial_sorting_scan_left_to_right_32s_4k_block_gather(
    t: &[SaSint],
    sa: &mut [SaSint],
    cache: &mut [ThreadCache],
    omp_block_start: FastSint,
    omp_block_size: FastSint,
) {
    if omp_block_size <= 0 {
        return;
    }

    let start = usize::try_from(omp_block_start).expect("omp_block_start must be non-negative");
    let size = usize::try_from(omp_block_size).expect("omp_block_size must be non-negative");
    for offset in 0..size {
        let i = start + offset;
        let mut symbol = SAINT_MIN;
        let mut p = sa[i];
        if p > 0 {
            cache[offset].index = p;
            p &= !SUFFIX_GROUP_MARKER;
            let p_usize = usize::try_from(p).expect("suffix index must be non-negative");
            symbol = buckets_index2(
                usize::try_from(t[p_usize - 1]).expect("bucket symbol must be non-negative"),
                usize::from(t[p_usize - 2] < t[p_usize - 1]),
            ) as SaSint;
            p = 0;
        }
        cache[offset].symbol = symbol;
        sa[i] = p & SAINT_MAX;
    }
}

/// Internal helper: partial sorting scan left to right 32s 1k block gather.
#[doc(hidden)]
pub fn partial_sorting_scan_left_to_right_32s_1k_block_gather(
    t: &[SaSint],
    sa: &mut [SaSint],
    cache: &mut [ThreadCache],
    omp_block_start: FastSint,
    omp_block_size: FastSint,
) {
    if omp_block_size <= 0 {
        return;
    }
    let start = usize::try_from(omp_block_start).expect("omp_block_start must be non-negative");
    let size = usize::try_from(omp_block_size).expect("omp_block_size must be non-negative");
    for offset in 0..size {
        let i = start + offset;
        let mut symbol = SAINT_MIN;
        let mut p = sa[i];
        if p > 0 {
            cache[offset].index = (p - 1)
                | ((usize::from(t[p as usize - 2] < t[p as usize - 1]) as SaSint)
                    << (SAINT_BIT - 1));
            symbol = t[p as usize - 1];
            p = 0;
        }
        cache[offset].symbol = symbol;
        sa[i] = p & SAINT_MAX;
    }
}

/// Internal helper: partial sorting scan left to right 32s 6k block sort.
#[doc(hidden)]
pub fn partial_sorting_scan_left_to_right_32s_6k_block_sort(
    t: &[SaSint],
    buckets: &mut [SaSint],
    mut d: SaSint,
    cache: &mut [ThreadCache],
    omp_block_start: FastSint,
    omp_block_size: FastSint,
) -> SaSint {
    if omp_block_size <= 0 {
        return d;
    }

    let start = usize::try_from(omp_block_start).expect("omp_block_start must be non-negative");
    let block_end =
        start + usize::try_from(omp_block_size).expect("omp_block_size must be non-negative");

    let mut i = start;
    let mut j = block_end.saturating_sub(65);
    while i < j {
        let cache_i0 = i - start;
        let cache_i1 = cache_i0 + 1;

        let v0 =
            usize::try_from(cache[cache_i0].symbol).expect("cache symbol must be non-negative");
        let p0 = cache[cache_i0].index;
        d += SaSint::from(p0 < 0);
        cache[cache_i0].symbol = buckets[v0];
        buckets[v0] += 1;
        cache[cache_i0].index =
            (p0 - 1) | ((SaSint::from(buckets[2 + v0] != d)) << (SAINT_BIT - 1));
        buckets[2 + v0] = d;
        if cache[cache_i0].symbol >= omp_block_start as SaSint
            && cache[cache_i0].symbol < block_end as SaSint
        {
            let s = usize::try_from(cache[cache_i0].symbol - omp_block_start as SaSint)
                .expect("cache slot must be non-negative");
            let q = cache[cache_i0].index & SAINT_MAX;
            cache[s].index = cache[cache_i0].index;
            let q_usize = usize::try_from(q).expect("suffix index must be non-negative");
            cache[s].symbol = buckets_index4(
                usize::try_from(t[q_usize - 1]).expect("bucket symbol must be non-negative"),
                usize::from(t[q_usize - 2] >= t[q_usize - 1]),
            ) as SaSint;
        }

        let v1 =
            usize::try_from(cache[cache_i1].symbol).expect("cache symbol must be non-negative");
        let p1 = cache[cache_i1].index;
        d += SaSint::from(p1 < 0);
        cache[cache_i1].symbol = buckets[v1];
        buckets[v1] += 1;
        cache[cache_i1].index =
            (p1 - 1) | ((SaSint::from(buckets[2 + v1] != d)) << (SAINT_BIT - 1));
        buckets[2 + v1] = d;
        if cache[cache_i1].symbol >= omp_block_start as SaSint
            && cache[cache_i1].symbol < block_end as SaSint
        {
            let s = usize::try_from(cache[cache_i1].symbol - omp_block_start as SaSint)
                .expect("cache slot must be non-negative");
            let q = cache[cache_i1].index & SAINT_MAX;
            cache[s].index = cache[cache_i1].index;
            let q_usize = usize::try_from(q).expect("suffix index must be non-negative");
            cache[s].symbol = buckets_index4(
                usize::try_from(t[q_usize - 1]).expect("bucket symbol must be non-negative"),
                usize::from(t[q_usize - 2] >= t[q_usize - 1]),
            ) as SaSint;
        }

        i += 2;
    }

    j += 65;
    while i < j {
        let cache_i = i - start;
        let v = usize::try_from(cache[cache_i].symbol).expect("cache symbol must be non-negative");
        let p = cache[cache_i].index;
        d += SaSint::from(p < 0);
        cache[cache_i].symbol = buckets[v];
        buckets[v] += 1;
        cache[cache_i].index = (p - 1) | ((SaSint::from(buckets[2 + v] != d)) << (SAINT_BIT - 1));
        buckets[2 + v] = d;
        if cache[cache_i].symbol >= omp_block_start as SaSint
            && cache[cache_i].symbol < block_end as SaSint
        {
            let s = usize::try_from(cache[cache_i].symbol - omp_block_start as SaSint)
                .expect("cache slot must be non-negative");
            let q = cache[cache_i].index & SAINT_MAX;
            cache[s].index = cache[cache_i].index;
            let q_usize = usize::try_from(q).expect("suffix index must be non-negative");
            cache[s].symbol = buckets_index4(
                usize::try_from(t[q_usize - 1]).expect("bucket symbol must be non-negative"),
                usize::from(t[q_usize - 2] >= t[q_usize - 1]),
            ) as SaSint;
        }
        i += 1;
    }

    d
}

/// Internal helper: partial sorting scan left to right 32s 4k block sort.
#[doc(hidden)]
pub fn partial_sorting_scan_left_to_right_32s_4k_block_sort(
    t: &[SaSint],
    k: SaSint,
    buckets: &mut [SaSint],
    mut d: SaSint,
    cache: &mut [ThreadCache],
    omp_block_start: FastSint,
    omp_block_size: FastSint,
) -> SaSint {
    if omp_block_size <= 0 {
        return d;
    }

    let k_usize = usize::try_from(k).expect("k must be non-negative");
    let (distinct_names, tail) = buckets.split_at_mut(2 * k_usize);
    let induction_bucket = &mut tail[..k_usize];

    let start = usize::try_from(omp_block_start).expect("omp_block_start must be non-negative");
    let size = usize::try_from(omp_block_size).expect("omp_block_size must be non-negative");
    let block_end = start + size;

    for offset in 0..size {
        let v = cache[offset].symbol;
        if v >= 0 {
            let p = cache[offset].index;
            d += p >> (SUFFIX_GROUP_BIT - 1);

            let bucket_index = usize::try_from(v >> 1).expect("bucket index must be non-negative");
            let v_usize = usize::try_from(v).expect("cache symbol must be non-negative");
            let target = induction_bucket[bucket_index];
            induction_bucket[bucket_index] += 1;

            cache[offset].symbol = target;
            cache[offset].index = (p - 1)
                | ((v & 1) << (SAINT_BIT - 1))
                | (((distinct_names[v_usize] != d) as SaSint) << (SUFFIX_GROUP_BIT - 1));
            distinct_names[v_usize] = d;

            if target >= omp_block_start as SaSint && target < block_end as SaSint {
                let ni = usize::try_from(target - omp_block_start as SaSint)
                    .expect("cache slot must be non-negative");
                let mut np = cache[offset].index;
                if np > 0 {
                    cache[ni].index = np;
                    np &= !SUFFIX_GROUP_MARKER;
                    let np_usize = usize::try_from(np).expect("suffix index must be non-negative");
                    cache[ni].symbol = buckets_index2(
                        usize::try_from(t[np_usize - 1])
                            .expect("bucket symbol must be non-negative"),
                        usize::from(t[np_usize - 2] < t[np_usize - 1]),
                    ) as SaSint;
                    np = 0;
                }
                cache[offset].index = np & SAINT_MAX;
            }
        }
    }

    d
}

/// Internal helper: partial sorting scan left to right 32s 1k block sort.
#[doc(hidden)]
pub fn partial_sorting_scan_left_to_right_32s_1k_block_sort(
    t: &[SaSint],
    induction_bucket: &mut [SaSint],
    cache: &mut [ThreadCache],
    omp_block_start: FastSint,
    omp_block_size: FastSint,
) {
    if omp_block_size <= 0 {
        return;
    }
    let start = usize::try_from(omp_block_start).expect("omp_block_start must be non-negative");
    let size = usize::try_from(omp_block_size).expect("omp_block_size must be non-negative");
    let block_end = start + size;

    for offset in 0..size {
        let v = cache[offset].symbol;
        if v >= 0 {
            let v_usize = v as usize;
            let target = induction_bucket[v_usize];
            cache[offset].symbol = target;
            induction_bucket[v_usize] += 1;
            if target >= omp_block_start as SaSint && target < block_end as SaSint {
                let ni = usize::try_from(target - omp_block_start as SaSint)
                    .expect("cache slot must be non-negative");
                let mut np = cache[offset].index;
                if np > 0 {
                    cache[ni].index = (np - 1)
                        | ((usize::from(t[np as usize - 2] < t[np as usize - 1]) as SaSint)
                            << (SAINT_BIT - 1));
                    cache[ni].symbol = t[np as usize - 1];
                    np = 0;
                }
                cache[offset].index = np & SAINT_MAX;
            }
        }
    }
}

/// Internal helper: partial sorting scan left to right 32s 6k block (OpenMP variant).
#[doc(hidden)]
pub fn partial_sorting_scan_left_to_right_32s_6k_block_omp(
    t: &[SaSint],
    sa: &mut [SaSint],
    buckets: &mut [SaSint],
    d: SaSint,
    cache: &mut [ThreadCache],
    block_start: FastSint,
    block_size: FastSint,
    threads: SaSint,
) -> SaSint {
    if block_size <= 0 {
        return d;
    }
    if threads == 1 || block_size < 16_384 {
        return partial_sorting_scan_left_to_right_32s_6k(
            t,
            sa,
            buckets,
            d,
            block_start,
            block_size,
        );
    }

    let threads_usize = usize::try_from(threads)
        .expect("threads must be non-negative")
        .max(1);
    let block_size_usize = usize::try_from(block_size).expect("block_size must be non-negative");
    let block_start_usize = usize::try_from(block_start).expect("block_start must be non-negative");
    let omp_num_threads = threads_usize.min(block_size_usize.max(1));
    let omp_block_stride = (block_size_usize / omp_num_threads) & !15usize;

    {
        let sa_ptr = SyncMutPtr::new(sa);
        let t_ro: &[SaSint] = t;
        let cache_ptr = SyncMutPtr::new(cache);
        run_rayon_with_threads(omp_num_threads, || {
            (0..omp_num_threads)
                .into_par_iter()
                .for_each(|omp_thread_num| {
                    let omp_block_size = if omp_thread_num + 1 < omp_num_threads {
                        omp_block_stride
                    } else {
                        block_size_usize - omp_thread_num * omp_block_stride
                    };
                    let omp_block_start = block_start_usize + omp_thread_num * omp_block_stride;
                    if omp_block_size > 0 {
                        let cache = unsafe { cache_ptr.as_slice() };
                        let sa = unsafe { sa_ptr.as_slice() };
                        partial_sorting_scan_left_to_right_32s_6k_block_gather(
                            t_ro,
                            sa,
                            &mut cache[omp_thread_num * omp_block_stride
                                ..omp_thread_num * omp_block_stride + omp_block_size],
                            omp_block_start as FastSint,
                            omp_block_size as FastSint,
                        );
                    }
                });
        });
    }

    let d = partial_sorting_scan_left_to_right_32s_6k_block_sort(
        t,
        buckets,
        d,
        &mut cache[..block_size_usize],
        block_start,
        block_size,
    );

    {
        let sa_ptr = SyncMutPtr::new(sa);
        let cache_ro: &[ThreadCache] = cache;
        run_rayon_with_threads(omp_num_threads, || {
            (0..omp_num_threads)
                .into_par_iter()
                .for_each(|omp_thread_num| {
                    let omp_block_size = if omp_thread_num + 1 < omp_num_threads {
                        omp_block_stride
                    } else {
                        block_size_usize - omp_thread_num * omp_block_stride
                    };
                    if omp_block_size > 0 {
                        let sa = unsafe { sa_ptr.as_slice() };
                        place_cached_suffixes(
                            sa,
                            &cache_ro[omp_thread_num * omp_block_stride..],
                            0,
                            omp_block_size as FastSint,
                        );
                    }
                });
        });
    }
    d
}

/// Internal helper: partial sorting scan left to right 32s 4k block (OpenMP variant).
#[doc(hidden)]
pub fn partial_sorting_scan_left_to_right_32s_4k_block_omp(
    t: &[SaSint],
    sa: &mut [SaSint],
    k: SaSint,
    buckets: &mut [SaSint],
    d: SaSint,
    cache: &mut [ThreadCache],
    block_start: FastSint,
    block_size: FastSint,
    threads: SaSint,
) -> SaSint {
    if block_size <= 0 {
        return d;
    }
    if threads == 1 || block_size < 16_384 {
        return partial_sorting_scan_left_to_right_32s_4k(
            t,
            sa,
            k,
            buckets,
            d,
            block_start,
            block_size,
        );
    }

    let threads_usize = usize::try_from(threads)
        .expect("threads must be non-negative")
        .max(1);
    let block_size_usize = usize::try_from(block_size).expect("block_size must be non-negative");
    let block_start_usize = usize::try_from(block_start).expect("block_start must be non-negative");
    let omp_num_threads = threads_usize.min(block_size_usize.max(1));
    let omp_block_stride = (block_size_usize / omp_num_threads) & !15usize;

    {
        let sa_ptr = SyncMutPtr::new(sa);
        let t_ro: &[SaSint] = t;
        let cache_ptr = SyncMutPtr::new(cache);
        run_rayon_with_threads(omp_num_threads, || {
            (0..omp_num_threads)
                .into_par_iter()
                .for_each(|omp_thread_num| {
                    let omp_block_size = if omp_thread_num + 1 < omp_num_threads {
                        omp_block_stride
                    } else {
                        block_size_usize - omp_thread_num * omp_block_stride
                    };
                    let omp_block_start = block_start_usize + omp_thread_num * omp_block_stride;
                    if omp_block_size > 0 {
                        let cache = unsafe { cache_ptr.as_slice() };
                        let sa = unsafe { sa_ptr.as_slice() };
                        partial_sorting_scan_left_to_right_32s_4k_block_gather(
                            t_ro,
                            sa,
                            &mut cache[omp_thread_num * omp_block_stride
                                ..omp_thread_num * omp_block_stride + omp_block_size],
                            omp_block_start as FastSint,
                            omp_block_size as FastSint,
                        );
                    }
                });
        });
    }

    let cache = &mut cache[..block_size_usize];
    let d = partial_sorting_scan_left_to_right_32s_4k_block_sort(
        t,
        k,
        buckets,
        d,
        cache,
        block_start,
        block_size,
    );

    {
        let sa_ptr = SyncMutPtr::new(sa);
        let cache_ptr = SyncMutPtr::new(cache);
        run_rayon_with_threads(omp_num_threads, || {
            (0..omp_num_threads)
                .into_par_iter()
                .for_each(|omp_thread_num| {
                    let omp_block_start = omp_thread_num * omp_block_stride;
                    let omp_block_size = if omp_thread_num + 1 < omp_num_threads {
                        omp_block_stride
                    } else {
                        block_size_usize - omp_block_start
                    };
                    if omp_block_size > 0 {
                        let sa = unsafe { sa_ptr.as_slice() };
                        let cache = unsafe { cache_ptr.as_slice() };
                        compact_and_place_cached_suffixes(
                            sa,
                            &mut cache[omp_block_start..],
                            0,
                            omp_block_size as FastSint,
                        );
                    }
                });
        });
    }

    d
}

/// Internal helper: partial sorting scan left to right 32s 1k block (OpenMP variant).
#[doc(hidden)]
pub fn partial_sorting_scan_left_to_right_32s_1k_block_omp(
    t: &[SaSint],
    sa: &mut [SaSint],
    buckets: &mut [SaSint],
    cache: &mut [ThreadCache],
    block_start: FastSint,
    block_size: FastSint,
    threads: SaSint,
) {
    if block_size <= 0 {
        return;
    }
    if threads == 1 || block_size < 16_384 {
        partial_sorting_scan_left_to_right_32s_1k(t, sa, buckets, block_start, block_size);
        return;
    }

    let threads_usize = usize::try_from(threads)
        .expect("threads must be non-negative")
        .max(1);
    let block_size_usize = usize::try_from(block_size).expect("block_size must be non-negative");
    let block_start_usize = usize::try_from(block_start).expect("block_start must be non-negative");
    let omp_num_threads = threads_usize.min(block_size_usize.max(1));
    let omp_block_stride = (block_size_usize / omp_num_threads) & !15usize;

    {
        let sa_ptr = SyncMutPtr::new(sa);
        let t_ro: &[SaSint] = t;
        let cache_ptr = SyncMutPtr::new(cache);
        run_rayon_with_threads(omp_num_threads, || {
            (0..omp_num_threads)
                .into_par_iter()
                .for_each(|omp_thread_num| {
                    let omp_block_size = if omp_thread_num + 1 < omp_num_threads {
                        omp_block_stride
                    } else {
                        block_size_usize - omp_thread_num * omp_block_stride
                    };
                    let omp_block_start = block_start_usize + omp_thread_num * omp_block_stride;
                    if omp_block_size > 0 {
                        let cache = unsafe { cache_ptr.as_slice() };
                        let sa = unsafe { sa_ptr.as_slice() };
                        partial_sorting_scan_left_to_right_32s_1k_block_gather(
                            t_ro,
                            sa,
                            &mut cache[omp_thread_num * omp_block_stride
                                ..omp_thread_num * omp_block_stride + omp_block_size],
                            omp_block_start as FastSint,
                            omp_block_size as FastSint,
                        );
                    }
                });
        });
    }

    let cache = &mut cache[..block_size_usize];
    partial_sorting_scan_left_to_right_32s_1k_block_sort(
        t,
        buckets,
        cache,
        block_start,
        block_size,
    );
    {
        let sa_ptr = SyncMutPtr::new(sa);
        let cache_ptr = SyncMutPtr::new(cache);
        run_rayon_with_threads(omp_num_threads, || {
            (0..omp_num_threads)
                .into_par_iter()
                .for_each(|omp_thread_num| {
                    let omp_block_start = omp_thread_num * omp_block_stride;
                    let omp_block_size = if omp_thread_num + 1 < omp_num_threads {
                        omp_block_stride
                    } else {
                        block_size_usize - omp_block_start
                    };
                    if omp_block_size > 0 {
                        let sa = unsafe { sa_ptr.as_slice() };
                        let cache = unsafe { cache_ptr.as_slice() };
                        compact_and_place_cached_suffixes(
                            sa,
                            &mut cache[omp_block_start..],
                            0,
                            omp_block_size as FastSint,
                        );
                    }
                });
        });
    }
}

/// Internal helper: partial sorting scan right to left 32s 6k (OpenMP variant).
#[doc(hidden)]
pub fn partial_sorting_scan_right_to_left_32s_6k_omp(
    t: &[SaSint],
    sa: &mut [SaSint],
    n: SaSint,
    buckets: &mut [SaSint],
    first_lms_suffix: SaSint,
    left_suffixes_count: SaSint,
    mut d: SaSint,
    threads: SaSint,
    thread_state: &mut [ThreadState],
) -> SaSint {
    let scan_start = left_suffixes_count as FastSint + 1;
    let scan_end = n as FastSint - first_lms_suffix as FastSint;
    if threads == 1 || (scan_end - scan_start) < 65_536 {
        return partial_sorting_scan_right_to_left_32s_6k(
            t,
            sa,
            buckets,
            d,
            scan_start,
            scan_end - scan_start,
        );
    }
    if thread_state.is_empty() {
        return partial_sorting_scan_right_to_left_32s_6k(
            t,
            sa,
            buckets,
            d,
            scan_start,
            scan_end - scan_start,
        );
    }

    let threads_usize = usize::try_from(threads)
        .expect("threads must be non-negative")
        .max(1);
    let mut cache = vec![ThreadCache::default(); threads_usize * LIBSAIS_PER_THREAD_CACHE_SIZE];
    let mut block_start = scan_end - 1;
    let block_span = FastSint::try_from(threads_usize * LIBSAIS_PER_THREAD_CACHE_SIZE)
        .expect("block span must fit FastSint");
    while block_start >= scan_start {
        let mut block_end = block_start - block_span;
        if block_end < scan_start {
            block_end = scan_start - 1;
        }

        d = partial_sorting_scan_right_to_left_32s_6k_block_omp(
            t,
            sa,
            buckets,
            d,
            &mut cache,
            block_end + 1,
            block_start - block_end,
            threads,
        );

        if block_end < scan_start {
            break;
        }
        block_start = block_end;
    }

    d
}

/// Internal helper: partial sorting scan right to left 32s 4k (OpenMP variant).
#[doc(hidden)]
pub fn partial_sorting_scan_right_to_left_32s_4k_omp(
    t: &[SaSint],
    sa: &mut [SaSint],
    n: SaSint,
    k: SaSint,
    buckets: &mut [SaSint],
    mut d: SaSint,
    threads: SaSint,
    thread_state: &mut [ThreadState],
) -> SaSint {
    if threads == 1 || n < 65_536 {
        return partial_sorting_scan_right_to_left_32s_4k(t, sa, k, buckets, d, 0, n as FastSint);
    }
    if thread_state.is_empty() {
        return partial_sorting_scan_right_to_left_32s_4k(t, sa, k, buckets, d, 0, n as FastSint);
    }
    let threads_usize = usize::try_from(threads)
        .expect("threads must be non-negative")
        .max(1);
    let mut cache = vec![ThreadCache::default(); threads_usize * LIBSAIS_PER_THREAD_CACHE_SIZE];
    let mut block_start = FastSint::try_from(n).expect("n must fit FastSint") - 1;
    let block_span = FastSint::try_from(threads_usize * LIBSAIS_PER_THREAD_CACHE_SIZE)
        .expect("block span must fit FastSint");
    while block_start >= 0 {
        let mut block_end = block_start - block_span;
        if block_end < 0 {
            block_end = -1;
        }

        d = partial_sorting_scan_right_to_left_32s_4k_block_omp(
            t,
            sa,
            k,
            buckets,
            d,
            &mut cache,
            block_end + 1,
            block_start - block_end,
            threads,
        );

        if block_end < 0 {
            break;
        }
        block_start = block_end;
    }

    d
}

/// Internal helper: partial sorting scan right to left 32s 1k (OpenMP variant).
#[doc(hidden)]
pub fn partial_sorting_scan_right_to_left_32s_1k_omp(
    t: &[SaSint],
    sa: &mut [SaSint],
    n: SaSint,
    buckets: &mut [SaSint],
    threads: SaSint,
    thread_state: &mut [ThreadState],
) {
    if threads == 1 || n < 65_536 {
        partial_sorting_scan_right_to_left_32s_1k(t, sa, buckets, 0, n as FastSint);
        return;
    }
    if thread_state.is_empty() {
        partial_sorting_scan_right_to_left_32s_1k(t, sa, buckets, 0, n as FastSint);
        return;
    }

    let threads_usize = usize::try_from(threads)
        .expect("threads must be non-negative")
        .max(1);
    let mut cache = vec![ThreadCache::default(); threads_usize * LIBSAIS_PER_THREAD_CACHE_SIZE];
    let mut block_start = FastSint::try_from(n).expect("n must fit FastSint") - 1;
    let block_span = FastSint::try_from(threads_usize * LIBSAIS_PER_THREAD_CACHE_SIZE)
        .expect("block span must fit FastSint");
    while block_start >= 0 {
        let mut block_end = block_start - block_span;
        if block_end < 0 {
            block_end = -1;
        }

        partial_sorting_scan_right_to_left_32s_1k_block_omp(
            t,
            sa,
            buckets,
            &mut cache,
            block_end + 1,
            block_start - block_end,
            threads,
        );

        if block_end < 0 {
            break;
        }
        block_start = block_end;
    }
}

/// Internal helper: partial sorting gather lms suffixes 32s 4k.
#[doc(hidden)]
pub fn partial_sorting_gather_lms_suffixes_32s_4k(
    sa: &mut [SaSint],
    omp_block_start: FastSint,
    omp_block_size: FastSint,
) -> FastSint {
    if omp_block_size <= 0 {
        return omp_block_start;
    }

    let start = usize::try_from(omp_block_start).expect("omp_block_start must be non-negative");
    let size = usize::try_from(omp_block_size).expect("omp_block_size must be non-negative");
    let mut l = start;

    for i in start..start + size {
        let s = sa[i] as SaUint;
        sa[l] = ((s.wrapping_sub(SUFFIX_GROUP_MARKER as SaUint)) & !(SUFFIX_GROUP_MARKER as SaUint))
            as SaSint;
        l += usize::from((s as SaSint) < 0);
    }

    l as FastSint
}

/// Internal helper: partial sorting gather lms suffixes 32s 1k.
#[doc(hidden)]
pub fn partial_sorting_gather_lms_suffixes_32s_1k(
    sa: &mut [SaSint],
    omp_block_start: FastSint,
    omp_block_size: FastSint,
) -> FastSint {
    if omp_block_size <= 0 {
        return omp_block_start;
    }

    let start = usize::try_from(omp_block_start).expect("omp_block_start must be non-negative");
    let size = usize::try_from(omp_block_size).expect("omp_block_size must be non-negative");
    let mut l = start;

    for i in start..start + size {
        let s = sa[i];
        sa[l] = s & SAINT_MAX;
        l += usize::from(s < 0);
    }

    l as FastSint
}

/// Internal helper: partial sorting gather lms suffixes 32s 4k (OpenMP variant).
#[doc(hidden)]
pub fn partial_sorting_gather_lms_suffixes_32s_4k_omp(
    sa: &mut [SaSint],
    n: SaSint,
    threads: SaSint,
    thread_state: &mut [ThreadState],
) {
    let n_usize = usize::try_from(n).expect("n must be non-negative");
    let omp_num_threads = if threads > 1 && n >= 65_536 {
        usize::try_from(threads)
            .expect("threads must be non-negative")
            .min(thread_state.len())
            .max(1)
    } else {
        1
    };

    if omp_num_threads == 1 {
        let _ = partial_sorting_gather_lms_suffixes_32s_4k(sa, 0, n as FastSint);
        return;
    }

    let omp_block_stride = (n_usize / omp_num_threads) & !15usize;
    for (thread_num, state) in thread_state.iter_mut().take(omp_num_threads).enumerate() {
        let block_start = thread_num * omp_block_stride;
        let block_size = if thread_num + 1 < omp_num_threads {
            omp_block_stride
        } else {
            n_usize - block_start
        };
        state.position = block_start as FastSint;
        state.count = partial_sorting_gather_lms_suffixes_32s_4k(
            sa,
            block_start as FastSint,
            block_size as FastSint,
        ) - block_start as FastSint;
    }

    let mut position = 0usize;
    for (thread_num, state) in thread_state.iter().take(omp_num_threads).enumerate() {
        let count = usize::try_from(state.count).expect("count must be non-negative");
        let src = usize::try_from(state.position).expect("position must be non-negative");
        if thread_num > 0 && count > 0 {
            sa.copy_within(src..src + count, position);
        }
        position += count;
    }
}

/// Internal helper: partial sorting gather lms suffixes 32s 1k (OpenMP variant).
#[doc(hidden)]
pub fn partial_sorting_gather_lms_suffixes_32s_1k_omp(
    sa: &mut [SaSint],
    n: SaSint,
    threads: SaSint,
    thread_state: &mut [ThreadState],
) {
    let n_usize = usize::try_from(n).expect("n must be non-negative");
    let omp_num_threads = if threads > 1 && n >= 65_536 {
        usize::try_from(threads)
            .expect("threads must be non-negative")
            .min(thread_state.len())
            .max(1)
    } else {
        1
    };

    if omp_num_threads == 1 {
        let _ = partial_sorting_gather_lms_suffixes_32s_1k(sa, 0, n as FastSint);
        return;
    }

    let omp_block_stride = (n_usize / omp_num_threads) & !15usize;
    for (thread_num, state) in thread_state.iter_mut().take(omp_num_threads).enumerate() {
        let block_start = thread_num * omp_block_stride;
        let block_size = if thread_num + 1 < omp_num_threads {
            omp_block_stride
        } else {
            n_usize - block_start
        };
        state.position = block_start as FastSint;
        state.count = partial_sorting_gather_lms_suffixes_32s_1k(
            sa,
            block_start as FastSint,
            block_size as FastSint,
        ) - block_start as FastSint;
    }

    let mut position = 0usize;
    for (thread_num, state) in thread_state.iter().take(omp_num_threads).enumerate() {
        let count = usize::try_from(state.count).expect("count must be non-negative");
        let src = usize::try_from(state.position).expect("position must be non-negative");
        if thread_num > 0 && count > 0 {
            sa.copy_within(src..src + count, position);
        }
        position += count;
    }
}

/// Internal helper: induce partial order 8u (OpenMP variant).
#[doc(hidden)]
pub fn induce_partial_order_8u_omp(
    t: &[u8],
    sa: &mut [SaSint],
    n: SaSint,
    k: SaSint,
    flags: SaSint,
    buckets: &mut [SaSint],
    first_lms_suffix: SaSint,
    left_suffixes_count: SaSint,
    threads: SaSint,
    thread_state: &mut [ThreadState],
) {
    buckets[2 * ALPHABET_SIZE..4 * ALPHABET_SIZE].fill(0);

    if (flags & LIBSAIS_FLAGS_GSA) != 0 {
        let left = 4 * ALPHABET_SIZE + buckets_index2(0, 1);
        let right = 4 * ALPHABET_SIZE + buckets_index2(1, 1);
        buckets[left] = buckets[right] - 1;
        flip_suffix_markers_omp(sa, buckets[left], threads);
    }

    let d = partial_sorting_scan_left_to_right_8u_omp(
        t,
        sa,
        n,
        k,
        buckets,
        left_suffixes_count,
        0,
        threads,
        thread_state,
    );
    partial_sorting_shift_markers_8u_omp(sa, n, buckets, threads);

    if (flags & LIBSAIS_FLAGS_GSA) != 0 {
        partial_gsa_scan_right_to_left_8u_omp(
            t,
            sa,
            n,
            k,
            buckets,
            first_lms_suffix,
            left_suffixes_count,
            d,
            threads,
            thread_state,
        );

        if t[usize::try_from(first_lms_suffix).expect("first_lms_suffix must be non-negative")] == 0
        {
            let count = usize::try_from(buckets[buckets_index2(1, 1)] - 1)
                .expect("count must be non-negative");
            sa.copy_within(0..count, 1);
            sa[0] = first_lms_suffix | SAINT_MIN;
        }

        buckets[buckets_index2(0, 1)] = 0;
    } else {
        partial_sorting_scan_right_to_left_8u_omp(
            t,
            sa,
            n,
            k,
            buckets,
            first_lms_suffix,
            left_suffixes_count,
            d,
            threads,
            thread_state,
        );
    }
}

/// Internal helper: induce partial order 32s 6k (OpenMP variant).
#[doc(hidden)]
pub fn induce_partial_order_32s_6k_omp(
    t: &[SaSint],
    sa: &mut [SaSint],
    n: SaSint,
    k: SaSint,
    buckets: &mut [SaSint],
    first_lms_suffix: SaSint,
    left_suffixes_count: SaSint,
    threads: SaSint,
    thread_state: &mut [ThreadState],
) {
    let d = partial_sorting_scan_left_to_right_32s_6k_omp(
        t,
        sa,
        n,
        buckets,
        left_suffixes_count,
        0,
        threads,
        thread_state,
    );
    partial_sorting_shift_markers_32s_6k_omp(sa, k, buckets, threads);
    partial_sorting_shift_buckets_32s_6k(k, buckets);
    let _ = partial_sorting_scan_right_to_left_32s_6k_omp(
        t,
        sa,
        n,
        buckets,
        first_lms_suffix,
        left_suffixes_count,
        d,
        threads,
        thread_state,
    );
}

/// Internal helper: induce partial order 32s 4k (OpenMP variant).
#[doc(hidden)]
pub fn induce_partial_order_32s_4k_omp(
    t: &[SaSint],
    sa: &mut [SaSint],
    n: SaSint,
    k: SaSint,
    buckets: &mut [SaSint],
    threads: SaSint,
    thread_state: &mut [ThreadState],
) {
    let zero_len = 2 * usize::try_from(k).expect("k must be non-negative");
    buckets[..zero_len].fill(0);

    let d = partial_sorting_scan_left_to_right_32s_4k_omp(
        t,
        sa,
        n,
        k,
        buckets,
        0,
        threads,
        thread_state,
    );
    partial_sorting_shift_markers_32s_4k(sa, n);
    let _ = partial_sorting_scan_right_to_left_32s_4k_omp(
        t,
        sa,
        n,
        k,
        buckets,
        d,
        threads,
        thread_state,
    );
    partial_sorting_gather_lms_suffixes_32s_4k_omp(sa, n, threads, thread_state);
}

/// Internal helper: induce partial order 32s 2k (OpenMP variant).
#[doc(hidden)]
pub fn induce_partial_order_32s_2k_omp(
    t: &[SaSint],
    sa: &mut [SaSint],
    n: SaSint,
    k: SaSint,
    buckets: &mut [SaSint],
    threads: SaSint,
    thread_state: &mut [ThreadState],
) {
    let k_usize = usize::try_from(k).expect("k must be non-negative");
    let (left, right) = buckets.split_at_mut(k_usize);
    partial_sorting_scan_left_to_right_32s_1k_omp(t, sa, n, right, threads, thread_state);
    partial_sorting_scan_right_to_left_32s_1k_omp(t, sa, n, left, threads, thread_state);
    partial_sorting_gather_lms_suffixes_32s_1k_omp(sa, n, threads, thread_state);
}

/// Internal helper: induce partial order 32s 1k (OpenMP variant).
#[doc(hidden)]
pub fn induce_partial_order_32s_1k_omp(
    t: &[SaSint],
    sa: &mut [SaSint],
    n: SaSint,
    k: SaSint,
    buckets: &mut [SaSint],
    threads: SaSint,
    thread_state: &mut [ThreadState],
) {
    count_suffixes_32s(t, n, k, buckets);
    initialize_buckets_start_32s_1k(k, buckets);
    partial_sorting_scan_left_to_right_32s_1k_omp(t, sa, n, buckets, threads, thread_state);

    count_suffixes_32s(t, n, k, buckets);
    initialize_buckets_end_32s_1k(k, buckets);
    partial_sorting_scan_right_to_left_32s_1k_omp(t, sa, n, buckets, threads, thread_state);

    partial_sorting_gather_lms_suffixes_32s_1k_omp(sa, n, threads, thread_state);
}

/// Internal helper: renumber lms suffixes 8u.
#[doc(hidden)]
pub fn renumber_lms_suffixes_8u(
    sa: &mut [SaSint],
    m: SaSint,
    mut name: SaSint,
    omp_block_start: FastSint,
    omp_block_size: FastSint,
) -> SaSint {
    if omp_block_size <= 0 {
        return name;
    }

    let m_usize = usize::try_from(m).expect("m must be non-negative");
    let (sa_head, sam) = sa.split_at_mut(m_usize);
    let mut i = omp_block_start;
    let mut j = omp_block_start + omp_block_size - 64 - 3;

    while i < j {
        let i0 = i as usize;
        let p0 = sa_head[i0];
        let d0 = ((p0 & SAINT_MAX) >> 1) as usize;
        sam[d0] = name | SAINT_MIN;
        name += SaSint::from(p0 < 0);

        let p1 = sa_head[i0 + 1];
        let d1 = ((p1 & SAINT_MAX) >> 1) as usize;
        sam[d1] = name | SAINT_MIN;
        name += SaSint::from(p1 < 0);

        let p2 = sa_head[i0 + 2];
        let d2 = ((p2 & SAINT_MAX) >> 1) as usize;
        sam[d2] = name | SAINT_MIN;
        name += SaSint::from(p2 < 0);

        let p3 = sa_head[i0 + 3];
        let d3 = ((p3 & SAINT_MAX) >> 1) as usize;
        sam[d3] = name | SAINT_MIN;
        name += SaSint::from(p3 < 0);

        i += 4;
    }

    j += 64 + 3;
    while i < j {
        let p = sa_head[i as usize];
        let d = ((p & SAINT_MAX) >> 1) as usize;
        sam[d] = name | SAINT_MIN;
        name += SaSint::from(p < 0);
        i += 1;
    }

    name
}

/// Internal helper: gather marked lms suffixes.
#[doc(hidden)]
pub fn gather_marked_lms_suffixes(
    sa: &mut [SaSint],
    m: SaSint,
    l: FastSint,
    omp_block_start: FastSint,
    omp_block_size: FastSint,
) -> FastSint {
    if omp_block_size <= 0 {
        return l;
    }

    let mut l = l - 1;
    let mut i = m as FastSint + omp_block_start + omp_block_size - 1;
    let mut j = m as FastSint + omp_block_start + 3;

    while i >= j {
        let i0 = i as usize;
        let s0 = sa[i0];
        sa[l as usize] = s0 & SAINT_MAX;
        l -= FastSint::from(s0 < 0);

        let s1 = sa[i0 - 1];
        sa[l as usize] = s1 & SAINT_MAX;
        l -= FastSint::from(s1 < 0);

        let s2 = sa[i0 - 2];
        sa[l as usize] = s2 & SAINT_MAX;
        l -= FastSint::from(s2 < 0);

        let s3 = sa[i0 - 3];
        sa[l as usize] = s3 & SAINT_MAX;
        l -= FastSint::from(s3 < 0);

        i -= 4;
    }

    j -= 3;
    while i >= j {
        let s = sa[i as usize];
        sa[l as usize] = s & SAINT_MAX;
        l -= FastSint::from(s < 0);
        i -= 1;
    }

    l + 1
}

/// Internal helper: renumber lms suffixes 8u (OpenMP variant).
#[doc(hidden)]
pub fn renumber_lms_suffixes_8u_omp(
    sa: &mut [SaSint],
    m: SaSint,
    threads: SaSint,
    thread_state: &mut [ThreadState],
) -> SaSint {
    let omp_num_threads = if threads > 1 && m >= 65_536 {
        usize::try_from(threads)
            .expect("threads must be non-negative")
            .min(thread_state.len())
            .max(1)
    } else {
        1
    };
    let omp_block_stride = (m as FastSint / omp_num_threads as FastSint) & !15;

    let name = if omp_num_threads == 1 {
        renumber_lms_suffixes_8u(sa, m, 0, 0, m as FastSint)
    } else {
        {
            let sa_ro: &[SaSint] = sa;
            run_rayon_with_threads(omp_num_threads, || {
                thread_state[..omp_num_threads]
                    .par_iter_mut()
                    .enumerate()
                    .for_each(|(omp_thread_num, state)| {
                        let omp_block_start = omp_thread_num as FastSint * omp_block_stride;
                        let omp_block_size = if omp_thread_num + 1 < omp_num_threads {
                            omp_block_stride
                        } else {
                            m as FastSint - omp_block_start
                        };
                        state.count =
                            count_negative_marked_suffixes(sa_ro, omp_block_start, omp_block_size)
                                as FastSint;
                    });
            });
        }

        let counts: Vec<FastSint> = thread_state[..omp_num_threads]
            .iter()
            .map(|s| s.count)
            .collect();
        let name = counts.iter().sum::<FastSint>() as SaSint;

        {
            let sa_ptr = SyncMutPtr::new(sa);
            let counts_ref: &[FastSint] = &counts;
            run_rayon_with_threads(omp_num_threads, || {
                (0..omp_num_threads)
                    .into_par_iter()
                    .for_each(|omp_thread_num| {
                        let omp_block_start = omp_thread_num as FastSint * omp_block_stride;
                        let omp_block_size = if omp_thread_num + 1 < omp_num_threads {
                            omp_block_stride
                        } else {
                            m as FastSint - omp_block_start
                        };
                        let mut count: FastSint = 0;
                        for tt in 0..omp_thread_num {
                            count += counts_ref[tt];
                        }
                        let sa = unsafe { sa_ptr.as_slice() };
                        let _ = renumber_lms_suffixes_8u(
                            sa,
                            m,
                            count as SaSint,
                            omp_block_start,
                            omp_block_size,
                        );
                    });
            });
        }
        name
    };

    name
}

/// Internal helper: gather marked lms suffixes (OpenMP variant).
#[doc(hidden)]
pub fn gather_marked_lms_suffixes_omp(
    sa: &mut [SaSint],
    n: SaSint,
    m: SaSint,
    fs: SaSint,
    threads: SaSint,
    thread_state: &mut [ThreadState],
) {
    let n_fast = n as FastSint;
    let m_fast = m as FastSint;
    let omp_num_threads = if threads > 1 && n >= 131_072 {
        usize::try_from(threads)
            .expect("threads must be non-negative")
            .min(thread_state.len())
            .max(1)
    } else {
        1
    };
    let omp_block_stride = ((n_fast >> 1) / omp_num_threads as FastSint) & !15;

    if omp_num_threads == 1 {
        let _ = gather_marked_lms_suffixes(sa, m, n_fast + fs as FastSint, 0, n_fast >> 1);
    } else {
        {
            let sa_ptr = SyncMutPtr::new(sa);
            run_rayon_with_threads(omp_num_threads, || {
                thread_state[..omp_num_threads]
                    .par_iter_mut()
                    .enumerate()
                    .for_each(|(omp_thread_num, state)| {
                        let omp_block_start = omp_thread_num as FastSint * omp_block_stride;
                        let omp_block_size = if omp_thread_num + 1 < omp_num_threads {
                            omp_block_stride
                        } else {
                            (n_fast >> 1) - omp_block_start
                        };

                        let sa = unsafe { sa_ptr.as_slice() };
                        if omp_thread_num < omp_num_threads - 1 {
                            state.position = gather_marked_lms_suffixes(
                                sa,
                                m,
                                m_fast + omp_block_start + omp_block_size,
                                omp_block_start,
                                omp_block_size,
                            );
                            state.count =
                                m_fast + omp_block_start + omp_block_size - state.position;
                        } else {
                            state.position = gather_marked_lms_suffixes(
                                sa,
                                m,
                                n_fast + fs as FastSint,
                                omp_block_start,
                                omp_block_size,
                            );
                            state.count = n_fast + fs as FastSint - state.position;
                        }
                    });
            });
        }

        let mut position = n_fast + fs as FastSint;
        for t in (0..omp_num_threads).rev() {
            position -= thread_state[t].count;
            if t + 1 != omp_num_threads && thread_state[t].count > 0 {
                let src = usize::try_from(thread_state[t].position)
                    .expect("position must be non-negative");
                let len =
                    usize::try_from(thread_state[t].count).expect("count must be non-negative");
                let dst = usize::try_from(position).expect("position must be non-negative");
                sa.copy_within(src..src + len, dst);
            }
        }
    }
}

/// Internal helper: renumber and gather lms suffixes (OpenMP variant).
#[doc(hidden)]
pub fn renumber_and_gather_lms_suffixes_omp(
    sa: &mut [SaSint],
    n: SaSint,
    m: SaSint,
    fs: SaSint,
    threads: SaSint,
    thread_state: &mut [ThreadState],
) -> SaSint {
    let m_usize = usize::try_from(m).expect("m must be non-negative");
    let half_n = usize::try_from(n >> 1).expect("n must be non-negative");
    sa[m_usize..m_usize + half_n].fill(0);

    let name = renumber_lms_suffixes_8u_omp(sa, m, threads, thread_state);
    if name < m {
        gather_marked_lms_suffixes_omp(sa, n, m, fs, threads, thread_state);
    } else {
        let mut i = 0;
        while i < m_usize {
            sa[i] &= SAINT_MAX;
            i += 1;
        }
    }

    name
}

/// Internal helper: renumber distinct lms suffixes 32s 4k.
#[doc(hidden)]
pub fn renumber_distinct_lms_suffixes_32s_4k(
    sa: &mut [SaSint],
    m: SaSint,
    mut name: SaSint,
    omp_block_start: FastSint,
    omp_block_size: FastSint,
) -> SaSint {
    if omp_block_size <= 0 {
        return name;
    }

    let prefetch_distance = 64usize;
    let m_usize = usize::try_from(m).expect("m must be non-negative");
    let start = usize::try_from(omp_block_start).expect("omp_block_start must be non-negative");
    let size = usize::try_from(omp_block_size).expect("omp_block_size must be non-negative");
    let (sa_head, sam) = sa.split_at_mut(m_usize);
    let mut i = start;
    let mut j = start
        .saturating_add(size)
        .saturating_sub(prefetch_distance + 3);
    let mut p0;
    let mut p1;
    let mut p2;
    let mut p3 = 0;

    while i < j {
        p0 = sa_head[i];
        sa_head[i] = p0 & SAINT_MAX;
        sam[(sa_head[i] >> 1) as usize] = name | (p0 & p3 & SAINT_MIN);
        name += SaSint::from(p0 < 0);

        p1 = sa_head[i + 1];
        sa_head[i + 1] = p1 & SAINT_MAX;
        sam[(sa_head[i + 1] >> 1) as usize] = name | (p1 & p0 & SAINT_MIN);
        name += SaSint::from(p1 < 0);

        p2 = sa_head[i + 2];
        sa_head[i + 2] = p2 & SAINT_MAX;
        sam[(sa_head[i + 2] >> 1) as usize] = name | (p2 & p1 & SAINT_MIN);
        name += SaSint::from(p2 < 0);

        p3 = sa_head[i + 3];
        sa_head[i + 3] = p3 & SAINT_MAX;
        sam[(sa_head[i + 3] >> 1) as usize] = name | (p3 & p2 & SAINT_MIN);
        name += SaSint::from(p3 < 0);

        i += 4;
    }

    j = start + size;
    while i < j {
        p2 = p3;
        p3 = sa_head[i];
        sa_head[i] = p3 & SAINT_MAX;
        sam[(sa_head[i] >> 1) as usize] = name | (p3 & p2 & SAINT_MIN);
        name += SaSint::from(p3 < 0);
        i += 1;
    }

    name
}

/// Internal helper: mark distinct lms suffixes 32s.
#[doc(hidden)]
pub fn mark_distinct_lms_suffixes_32s(
    sa: &mut [SaSint],
    m: SaSint,
    omp_block_start: FastSint,
    omp_block_size: FastSint,
) {
    if omp_block_size <= 0 {
        return;
    }

    let m_usize = usize::try_from(m).expect("m must be non-negative");
    let start = usize::try_from(omp_block_start).expect("omp_block_start must be non-negative");
    let size = usize::try_from(omp_block_size).expect("omp_block_size must be non-negative");
    let mut i = m_usize + start;
    let mut j = m_usize + start + size.saturating_sub(3);
    let mut p3 = 0;

    while i < j {
        let mut p0 = sa[i];
        sa[i] = p0 & (p3 | SAINT_MAX);
        p0 = if p0 == 0 { p3 } else { p0 };

        let mut p1 = sa[i + 1];
        sa[i + 1] = p1 & (p0 | SAINT_MAX);
        p1 = if p1 == 0 { p0 } else { p1 };

        let mut p2 = sa[i + 2];
        sa[i + 2] = p2 & (p1 | SAINT_MAX);
        p2 = if p2 == 0 { p1 } else { p2 };

        p3 = sa[i + 3];
        sa[i + 3] = p3 & (p2 | SAINT_MAX);
        p3 = if p3 == 0 { p2 } else { p3 };

        i += 4;
    }

    j = m_usize + start + size;
    while i < j {
        let p2 = p3;
        p3 = sa[i];
        sa[i] = p3 & (p2 | SAINT_MAX);
        p3 = if p3 == 0 { p2 } else { p3 };
        i += 1;
    }
}

/// Internal helper: clamp lms suffixes length 32s.
#[doc(hidden)]
pub fn clamp_lms_suffixes_length_32s(
    sa: &mut [SaSint],
    m: SaSint,
    omp_block_start: FastSint,
    omp_block_size: FastSint,
) {
    if omp_block_size <= 0 {
        return;
    }

    let m_usize = usize::try_from(m).expect("m must be non-negative");
    let start = usize::try_from(omp_block_start).expect("omp_block_start must be non-negative");
    let size = usize::try_from(omp_block_size).expect("omp_block_size must be non-negative");
    let mut i = m_usize + start;
    let mut j = m_usize + start + size.saturating_sub(3);

    while i < j {
        let s0 = sa[i];
        sa[i] = if s0 < 0 { s0 } else { 0 } & SAINT_MAX;

        let s1 = sa[i + 1];
        sa[i + 1] = if s1 < 0 { s1 } else { 0 } & SAINT_MAX;

        let s2 = sa[i + 2];
        sa[i + 2] = if s2 < 0 { s2 } else { 0 } & SAINT_MAX;

        let s3 = sa[i + 3];
        sa[i + 3] = if s3 < 0 { s3 } else { 0 } & SAINT_MAX;

        i += 4;
    }

    j = m_usize + start + size;
    while i < j {
        let s = sa[i];
        sa[i] = if s < 0 { s } else { 0 } & SAINT_MAX;
        i += 1;
    }
}

/// Internal helper: renumber distinct lms suffixes 32s 4k (OpenMP variant).
#[doc(hidden)]
pub fn renumber_distinct_lms_suffixes_32s_4k_omp(
    sa: &mut [SaSint],
    m: SaSint,
    threads: SaSint,
    thread_state: &mut [ThreadState],
) -> SaSint {
    let m_usize = usize::try_from(m).expect("m must be non-negative");
    let omp_num_threads = if threads > 1 && m >= 65_536 {
        usize::try_from(threads)
            .expect("threads must be non-negative")
            .min(thread_state.len())
            .max(1)
    } else {
        1
    };
    let omp_block_stride = (m_usize / omp_num_threads) & !15usize;

    let name = if omp_num_threads == 1 {
        let omp_block_start = 0usize;
        let omp_block_size = m_usize - omp_block_start;
        renumber_distinct_lms_suffixes_32s_4k(
            sa,
            m,
            1,
            omp_block_start as FastSint,
            omp_block_size as FastSint,
        )
    } else {
        {
            let sa_ro: &[SaSint] = sa;
            run_rayon_with_threads(omp_num_threads, || {
                thread_state[..omp_num_threads]
                    .par_iter_mut()
                    .enumerate()
                    .for_each(|(omp_thread_num, state)| {
                        let omp_block_start = omp_thread_num * omp_block_stride;
                        let omp_block_size = if omp_thread_num + 1 < omp_num_threads {
                            omp_block_stride
                        } else {
                            m_usize - omp_block_start
                        };
                        state.count = count_negative_marked_suffixes(
                            sa_ro,
                            omp_block_start as FastSint,
                            omp_block_size as FastSint,
                        ) as FastSint;
                    });
            });
        }

        let counts: Vec<FastSint> = thread_state[..omp_num_threads]
            .iter()
            .map(|s| s.count)
            .collect();
        let name = (1 + counts.iter().sum::<FastSint>()) as SaSint;

        {
            let sa_ptr = SyncMutPtr::new(sa);
            let counts_ref: &[FastSint] = &counts;
            run_rayon_with_threads(omp_num_threads, || {
                (0..omp_num_threads)
                    .into_par_iter()
                    .for_each(|omp_thread_num| {
                        let omp_block_start = omp_thread_num * omp_block_stride;
                        let omp_block_size = if omp_thread_num + 1 < omp_num_threads {
                            omp_block_stride
                        } else {
                            m_usize - omp_block_start
                        };
                        let mut count: FastSint = 1;
                        for tt in 0..omp_thread_num {
                            count += counts_ref[tt];
                        }
                        let sa = unsafe { sa_ptr.as_slice() };
                        let _ = renumber_distinct_lms_suffixes_32s_4k(
                            sa,
                            m,
                            count as SaSint,
                            omp_block_start as FastSint,
                            omp_block_size as FastSint,
                        );
                    });
            });
        }
        name
    };

    name - 1
}

/// Internal helper: mark distinct lms suffixes 32s (OpenMP variant).
#[doc(hidden)]
pub fn mark_distinct_lms_suffixes_32s_omp(
    sa: &mut [SaSint],
    n: SaSint,
    m: SaSint,
    threads: SaSint,
) {
    let half_n = usize::try_from(n >> 1).expect("n must be non-negative");
    let omp_num_threads = if threads > 1 && n >= 131_072 {
        usize::try_from(threads)
            .expect("threads must be non-negative")
            .max(1)
    } else {
        1
    };
    let omp_block_stride = (half_n / omp_num_threads) & !15usize;

    {
        let sa_ptr = SyncMutPtr::new(sa);
        run_rayon_with_threads(omp_num_threads, || {
            (0..omp_num_threads)
                .into_par_iter()
                .for_each(|omp_thread_num| {
                    let omp_block_start = omp_thread_num * omp_block_stride;
                    let omp_block_size = if omp_thread_num + 1 < omp_num_threads {
                        omp_block_stride
                    } else {
                        half_n - omp_block_start
                    };
                    let sa = unsafe { sa_ptr.as_slice() };
                    mark_distinct_lms_suffixes_32s(
                        sa,
                        m,
                        omp_block_start as FastSint,
                        omp_block_size as FastSint,
                    );
                });
        });
    }
}

/// Internal helper: clamp lms suffixes length 32s (OpenMP variant).
#[doc(hidden)]
pub fn clamp_lms_suffixes_length_32s_omp(sa: &mut [SaSint], n: SaSint, m: SaSint, threads: SaSint) {
    let half_n = usize::try_from(n >> 1).expect("n must be non-negative");
    let omp_num_threads = if threads > 1 && n >= 131_072 {
        usize::try_from(threads)
            .expect("threads must be non-negative")
            .max(1)
    } else {
        1
    };
    let omp_block_stride = (half_n / omp_num_threads) & !15usize;

    {
        let sa_ptr = SyncMutPtr::new(sa);
        run_rayon_with_threads(omp_num_threads, || {
            (0..omp_num_threads)
                .into_par_iter()
                .for_each(|omp_thread_num| {
                    let omp_block_start = omp_thread_num * omp_block_stride;
                    let omp_block_size = if omp_thread_num + 1 < omp_num_threads {
                        omp_block_stride
                    } else {
                        half_n - omp_block_start
                    };
                    let sa = unsafe { sa_ptr.as_slice() };
                    clamp_lms_suffixes_length_32s(
                        sa,
                        m,
                        omp_block_start as FastSint,
                        omp_block_size as FastSint,
                    );
                });
        });
    }
}

/// Internal helper: renumber and mark distinct lms suffixes 32s 4k (OpenMP variant).
#[doc(hidden)]
pub fn renumber_and_mark_distinct_lms_suffixes_32s_4k_omp(
    sa: &mut [SaSint],
    n: SaSint,
    m: SaSint,
    threads: SaSint,
    thread_state: &mut [ThreadState],
) -> SaSint {
    let m_usize = usize::try_from(m).expect("m must be non-negative");
    let half_n = usize::try_from(n >> 1).expect("n must be non-negative");
    sa[m_usize..m_usize + half_n].fill(0);

    let name = renumber_distinct_lms_suffixes_32s_4k_omp(sa, m, threads, thread_state);
    if name < m {
        mark_distinct_lms_suffixes_32s_omp(sa, n, m, threads);
    }

    name
}

/// Internal helper: renumber and mark distinct lms suffixes 32s 1k (OpenMP variant).
#[doc(hidden)]
pub fn renumber_and_mark_distinct_lms_suffixes_32s_1k_omp(
    t: &[SaSint],
    sa: &mut [SaSint],
    n: SaSint,
    m: SaSint,
    threads: SaSint,
) -> SaSint {
    let m_usize = usize::try_from(m).expect("m must be non-negative");
    let n_usize = usize::try_from(n).expect("n must be non-negative");

    let _ = gather_lms_suffixes_32s(t, sa, n);

    let zero_len = n_usize
        .checked_sub(m_usize)
        .and_then(|v| v.checked_sub(m_usize))
        .expect("n must be at least 2*m");
    sa[m_usize..m_usize + zero_len].fill(0);

    {
        let prefetch_distance: FastSint = 64;
        let mut i = n as FastSint - m as FastSint;
        let mut j = n as FastSint - 1 - prefetch_distance - 3;

        while i < j {
            let iu = i as usize;
            let s0 = (sa[iu] as SaUint >> 1) as usize;
            let s1 = (sa[iu + 1] as SaUint >> 1) as usize;
            let s2 = (sa[iu + 2] as SaUint >> 1) as usize;
            let s3 = (sa[iu + 3] as SaUint >> 1) as usize;

            sa[m_usize + s0] = sa[iu + 1] - sa[iu] + 1 + SAINT_MIN;
            sa[m_usize + s1] = sa[iu + 2] - sa[iu + 1] + 1 + SAINT_MIN;
            sa[m_usize + s2] = sa[iu + 3] - sa[iu + 2] + 1 + SAINT_MIN;
            sa[m_usize + s3] = sa[iu + 4] - sa[iu + 3] + 1 + SAINT_MIN;
            i += 4;
        }

        j += prefetch_distance + 3;
        while i < j {
            let iu = i as usize;
            let s = (sa[iu] as SaUint >> 1) as usize;
            sa[m_usize + s] = sa[iu + 1] - sa[iu] + 1 + SAINT_MIN;
            i += 1;
        }

        let tail = (sa[n_usize - 1] as SaUint >> 1) as usize;
        sa[m_usize + tail] = 1 + SAINT_MIN;
    }

    clamp_lms_suffixes_length_32s_omp(sa, n, m, threads);

    let mut name = 1;
    if m_usize > 0 {
        let (sa_head, sam) = sa.split_at_mut(m_usize);
        let mut i = 1usize;
        let prefetch_distance = 64usize;
        let mut j = m_usize.saturating_sub(prefetch_distance + 1);
        let mut p = usize::try_from(sa_head[0]).expect("suffix index must be non-negative");
        let mut plen = sam[p >> 1];
        let mut pdiff = SAINT_MIN;

        while i < j {
            let q = usize::try_from(sa_head[i]).expect("suffix index must be non-negative");
            let qlen = sam[q >> 1];
            let mut qdiff = SAINT_MIN;
            if plen == qlen {
                let mut l = 0usize;
                while l < qlen as usize {
                    if t[p + l] != t[q + l] {
                        break;
                    }
                    l += 1;
                }
                qdiff = ((l as SaSint) - qlen) & SAINT_MIN;
            }
            sam[p >> 1] = name | (pdiff & qdiff);
            name += SaSint::from(qdiff < 0);

            p = usize::try_from(sa_head[i + 1]).expect("suffix index must be non-negative");
            plen = sam[p >> 1];
            pdiff = SAINT_MIN;
            if qlen == plen {
                let mut l = 0usize;
                while l < plen as usize {
                    if t[q + l] != t[p + l] {
                        break;
                    }
                    l += 1;
                }
                pdiff = ((l as SaSint) - plen) & SAINT_MIN;
            }
            sam[q >> 1] = name | (qdiff & pdiff);
            name += SaSint::from(pdiff < 0);
            i += 2;
        }

        j = m_usize;
        while i < j {
            let q = usize::try_from(sa_head[i]).expect("suffix index must be non-negative");
            let qlen = sam[q >> 1];
            let mut qdiff = SAINT_MIN;
            if plen == qlen {
                let mut l = 0usize;
                while l < plen as usize {
                    if t[p + l] != t[q + l] {
                        break;
                    }
                    l += 1;
                }
                qdiff = ((l as SaSint) - plen) & SAINT_MIN;
            }
            sam[p >> 1] = name | (pdiff & qdiff);
            name += SaSint::from(qdiff < 0);

            p = q;
            plen = qlen;
            pdiff = qdiff;
            i += 1;
        }

        sam[p >> 1] = name | pdiff;
        name += 1;
    }

    if name <= m {
        mark_distinct_lms_suffixes_32s_omp(sa, n, m, threads);
    }

    name - 1
}

/// Internal helper: reconstruct lms suffixes.
#[doc(hidden)]
pub fn reconstruct_lms_suffixes(
    sa: &mut [SaSint],
    n: SaSint,
    m: SaSint,
    omp_block_start: FastSint,
    omp_block_size: FastSint,
) {
    if omp_block_size <= 0 {
        return;
    }

    let prefetch_distance: FastSint = 64;
    let base = (n - m) as usize;
    let mut i = omp_block_start;
    let mut j = omp_block_start + omp_block_size - prefetch_distance - 3;

    while i < j {
        let iu = i as usize;
        let s0 = sa[iu] as usize;
        let s1 = sa[iu + 1] as usize;
        let s2 = sa[iu + 2] as usize;
        let s3 = sa[iu + 3] as usize;
        sa[iu] = sa[base + s0];
        sa[iu + 1] = sa[base + s1];
        sa[iu + 2] = sa[base + s2];
        sa[iu + 3] = sa[base + s3];
        i += 4;
    }

    j += prefetch_distance + 3;
    while i < j {
        let iu = i as usize;
        let s = sa[iu] as usize;
        sa[iu] = sa[base + s];
        i += 1;
    }
}

/// Internal helper: reconstruct lms suffixes (OpenMP variant).
#[doc(hidden)]
pub fn reconstruct_lms_suffixes_omp(sa: &mut [SaSint], n: SaSint, m: SaSint, threads: SaSint) {
    let m_usize = usize::try_from(m).expect("m must be non-negative");
    let omp_num_threads = if threads > 1 && m >= 65_536 {
        usize::try_from(threads)
            .expect("threads must be non-negative")
            .max(1)
    } else {
        1
    };
    let omp_block_stride = (m_usize / omp_num_threads) & !15usize;

    {
        let sa_ptr = SyncMutPtr::new(sa);
        run_rayon_with_threads(omp_num_threads, || {
            (0..omp_num_threads)
                .into_par_iter()
                .for_each(|omp_thread_num| {
                    let omp_block_start = omp_thread_num * omp_block_stride;
                    let omp_block_size = if omp_thread_num + 1 < omp_num_threads {
                        omp_block_stride
                    } else {
                        m_usize - omp_block_start
                    };
                    let sa = unsafe { sa_ptr.as_slice() };
                    reconstruct_lms_suffixes(
                        sa,
                        n,
                        m,
                        omp_block_start as FastSint,
                        omp_block_size as FastSint,
                    );
                });
        });
    }
}

/// Internal helper: place lms suffixes interval 8u.
#[doc(hidden)]
pub fn place_lms_suffixes_interval_8u(
    sa: &mut [SaSint],
    n: SaSint,
    mut m: SaSint,
    flags: SaSint,
    buckets: &mut [SaSint],
) {
    let bucket_end_base = 7 * ALPHABET_SIZE;
    if (flags & LIBSAIS_FLAGS_GSA) != 0 {
        buckets[bucket_end_base] -= 1;
    }

    let mut j = usize::try_from(n).expect("n must be non-negative");
    for c in (0..ALPHABET_SIZE - 1).rev() {
        let l = usize::try_from(
            buckets[buckets_index2(c, 1) + buckets_index2(1, 0)] - buckets[buckets_index2(c, 1)],
        )
        .expect("interval length must be non-negative");
        if l > 0 {
            let i = usize::try_from(buckets[bucket_end_base + c])
                .expect("bucket end must be non-negative");
            if j > i {
                sa[i..j].fill(0);
            }

            let new_j = i - l;
            let src_end = usize::try_from(m).expect("m must be non-negative");
            let src_start = src_end - l;
            sa.copy_within(src_start..src_end, new_j);
            m -= l as SaSint;
            j = new_j;
        }
    }

    sa[..j].fill(0);

    if (flags & LIBSAIS_FLAGS_GSA) != 0 {
        buckets[bucket_end_base] += 1;
    }
}

/// Internal helper: place lms suffixes interval 32s 4k.
#[doc(hidden)]
pub fn place_lms_suffixes_interval_32s_4k(
    sa: &mut [SaSint],
    n: SaSint,
    k: SaSint,
    mut m: SaSint,
    buckets: &[SaSint],
) {
    let k_usize = usize::try_from(k).expect("k must be non-negative");
    let bucket_end = &buckets[3 * k_usize..4 * k_usize];

    let mut j = usize::try_from(n).expect("n must be non-negative");
    for c in (0..k_usize - 1).rev() {
        let l = usize::try_from(
            buckets[buckets_index2(c, 1) + buckets_index2(1, 0)] - buckets[buckets_index2(c, 1)],
        )
        .expect("interval length must be non-negative");
        if l > 0 {
            let i = usize::try_from(bucket_end[c]).expect("bucket end must be non-negative");
            if j > i {
                sa[i..j].fill(0);
            }

            let new_j = i - l;
            let src_end = usize::try_from(m).expect("m must be non-negative");
            let src_start = src_end - l;
            sa.copy_within(src_start..src_end, new_j);
            m -= l as SaSint;
            j = new_j;
        }
    }

    sa[..j].fill(0);
}

/// Internal helper: place lms suffixes interval 32s 2k.
#[doc(hidden)]
pub fn place_lms_suffixes_interval_32s_2k(
    sa: &mut [SaSint],
    n: SaSint,
    k: SaSint,
    mut m: SaSint,
    buckets: &[SaSint],
) {
    let k_usize = usize::try_from(k).expect("k must be non-negative");
    let mut j = usize::try_from(n).expect("n must be non-negative");

    if k_usize > 1 {
        let mut c = buckets_index2(k_usize - 2, 0) as isize;
        while c >= buckets_index2(0, 0) as isize {
            let c_usize = c as usize;
            let l = usize::try_from(
                buckets[c_usize + buckets_index2(1, 1)] - buckets[c_usize + buckets_index2(0, 1)],
            )
            .expect("interval length must be non-negative");
            if l > 0 {
                let i =
                    usize::try_from(buckets[c_usize]).expect("bucket start must be non-negative");
                if j > i {
                    sa[i..j].fill(0);
                }

                let new_j = i - l;
                let src_end = usize::try_from(m).expect("m must be non-negative");
                let src_start = src_end - l;
                sa.copy_within(src_start..src_end, new_j);
                m -= l as SaSint;
                j = new_j;
            }
            c -= buckets_index2(1, 0) as isize;
        }
    }

    sa[..j].fill(0);
}

/// Internal helper: place lms suffixes interval 32s 1k.
#[doc(hidden)]
pub fn place_lms_suffixes_interval_32s_1k(
    t: &[SaSint],
    sa: &mut [SaSint],
    k: SaSint,
    m: SaSint,
    buckets: &[SaSint],
) {
    let mut c = k - 1;
    let c_usize = usize::try_from(c).expect("k must be positive");
    let mut l = usize::try_from(buckets[c_usize]).expect("bucket end must be non-negative");

    let m_usize = usize::try_from(m).expect("m must be non-negative");
    for i in (0..m_usize).rev() {
        let p = usize::try_from(sa[i]).expect("suffix index must be non-negative");
        let tp = t[p];
        if tp != c {
            c = tp;
            let bucket = usize::try_from(c).expect("bucket index must be non-negative");
            let bucket_pos =
                usize::try_from(buckets[bucket]).expect("bucket end must be non-negative");
            if l > bucket_pos {
                sa[bucket_pos..l].fill(0);
            }
            l = bucket_pos;
        }
        l -= 1;
        sa[l] = p as SaSint;
    }

    sa[..l].fill(0);
}

/// Internal helper: place lms suffixes histogram 32s 6k.
#[doc(hidden)]
pub fn place_lms_suffixes_histogram_32s_6k(
    sa: &mut [SaSint],
    n: SaSint,
    k: SaSint,
    mut m: SaSint,
    buckets: &[SaSint],
) {
    let k_usize = usize::try_from(k).expect("k must be non-negative");
    let bucket_end = &buckets[5 * k_usize..6 * k_usize];

    let mut j = usize::try_from(n).expect("n must be non-negative");
    for c in (0..k_usize - 1).rev() {
        let l = usize::try_from(buckets[buckets_index4(c, 1)])
            .expect("histogram length must be non-negative");
        if l > 0 {
            let i = usize::try_from(bucket_end[c]).expect("bucket end must be non-negative");
            if j > i {
                sa[i..j].fill(0);
            }

            let new_j = i - l;
            let src_end = usize::try_from(m).expect("m must be non-negative");
            let src_start = src_end - l;
            sa.copy_within(src_start..src_end, new_j);
            m -= l as SaSint;
            j = new_j;
        }
    }

    sa[..j].fill(0);
}

/// Internal helper: place lms suffixes histogram 32s 4k.
#[doc(hidden)]
pub fn place_lms_suffixes_histogram_32s_4k(
    sa: &mut [SaSint],
    n: SaSint,
    k: SaSint,
    mut m: SaSint,
    buckets: &[SaSint],
) {
    let k_usize = usize::try_from(k).expect("k must be non-negative");
    let bucket_end = &buckets[3 * k_usize..4 * k_usize];

    let mut j = usize::try_from(n).expect("n must be non-negative");
    for c in (0..k_usize - 1).rev() {
        let l = usize::try_from(buckets[buckets_index2(c, 1)])
            .expect("histogram length must be non-negative");
        if l > 0 {
            let i = usize::try_from(bucket_end[c]).expect("bucket end must be non-negative");
            if j > i {
                sa[i..j].fill(0);
            }

            let new_j = i - l;
            let src_end = usize::try_from(m).expect("m must be non-negative");
            let src_start = src_end - l;
            sa.copy_within(src_start..src_end, new_j);
            m -= l as SaSint;
            j = new_j;
        }
    }

    sa[..j].fill(0);
}

/// Internal helper: place lms suffixes histogram 32s 2k.
#[doc(hidden)]
pub fn place_lms_suffixes_histogram_32s_2k(
    sa: &mut [SaSint],
    n: SaSint,
    k: SaSint,
    mut m: SaSint,
    buckets: &[SaSint],
) {
    let k_usize = usize::try_from(k).expect("k must be non-negative");
    let mut j = usize::try_from(n).expect("n must be non-negative");

    if k_usize > 1 {
        let mut c = buckets_index2(k_usize - 2, 0) as isize;
        while c >= buckets_index2(0, 0) as isize {
            let c_usize = c as usize;
            let l = usize::try_from(buckets[c_usize + buckets_index2(0, 1)])
                .expect("histogram length must be non-negative");
            if l > 0 {
                let i =
                    usize::try_from(buckets[c_usize]).expect("bucket start must be non-negative");
                if j > i {
                    sa[i..j].fill(0);
                }

                let new_j = i - l;
                let src_end = usize::try_from(m).expect("m must be non-negative");
                let src_start = src_end - l;
                sa.copy_within(src_start..src_end, new_j);
                m -= l as SaSint;
                j = new_j;
            }
            c -= buckets_index2(1, 0) as isize;
        }
    }

    sa[..j].fill(0);
}

/// Internal helper: final bwt scan left to right 8u.
#[doc(hidden)]
pub fn final_bwt_scan_left_to_right_8u(
    t: &[u8],
    sa: &mut [SaSint],
    induction_bucket: &mut [SaSint],
    omp_block_start: FastSint,
    omp_block_size: FastSint,
) {
    if omp_block_size <= 0 {
        return;
    }

    let start = usize::try_from(omp_block_start).expect("omp_block_start must be non-negative");
    let size = usize::try_from(omp_block_size).expect("omp_block_size must be non-negative");
    for i in start..start + size {
        let mut p = sa[i];
        sa[i] = p & SAINT_MAX;
        if p > 0 {
            p -= 1;
            let p_usize = usize::try_from(p).expect("suffix index must be non-negative");
            sa[i] = t[p_usize] as SaSint | SAINT_MIN;
            let bucket = t[p_usize] as usize;
            let slot = usize::try_from(induction_bucket[bucket])
                .expect("bucket slot must be non-negative");
            sa[slot] = p
                | ((usize::from(t[p_usize - usize::from(p > 0)] < t[p_usize]) as SaSint)
                    << (SAINT_BIT - 1));
            induction_bucket[bucket] += 1;
        }
    }
}

/// Internal helper: final bwt aux scan left to right 8u.
#[doc(hidden)]
pub fn final_bwt_aux_scan_left_to_right_8u(
    t: &[u8],
    sa: &mut [SaSint],
    rm: SaSint,
    i_out: &mut [SaSint],
    induction_bucket: &mut [SaSint],
    omp_block_start: FastSint,
    omp_block_size: FastSint,
) {
    if omp_block_size <= 0 {
        return;
    }

    let start = usize::try_from(omp_block_start).expect("omp_block_start must be non-negative");
    let size = usize::try_from(omp_block_size).expect("omp_block_size must be non-negative");
    for i in start..start + size {
        let mut p = sa[i];
        sa[i] = p & SAINT_MAX;
        if p > 0 {
            p -= 1;
            let p_usize = usize::try_from(p).expect("suffix index must be non-negative");
            sa[i] = t[p_usize] as SaSint | SAINT_MIN;
            let bucket = t[p_usize] as usize;
            let slot = usize::try_from(induction_bucket[bucket])
                .expect("bucket slot must be non-negative");
            sa[slot] = p
                | ((usize::from(t[p_usize - usize::from(p > 0)] < t[p_usize]) as SaSint)
                    << (SAINT_BIT - 1));
            induction_bucket[bucket] += 1;
            if (p & rm) == 0 {
                let out_idx =
                    usize::try_from(p / (rm + 1)).expect("sample index must be non-negative");
                i_out[out_idx] = induction_bucket[bucket];
            }
        }
    }
}

/// Internal helper: final sorting scan left to right 8u.
#[doc(hidden)]
pub fn final_sorting_scan_left_to_right_8u(
    t: &[u8],
    sa: &mut [SaSint],
    induction_bucket: &mut [SaSint],
    omp_block_start: FastSint,
    omp_block_size: FastSint,
) {
    if omp_block_size <= 0 {
        return;
    }

    let prefetch_distance = 64usize;
    let start = usize::try_from(omp_block_start).expect("omp_block_start must be non-negative");
    let size = usize::try_from(omp_block_size).expect("omp_block_size must be non-negative");

    let mut i = start;
    let mut j = if size > prefetch_distance + 1 {
        start + size - (prefetch_distance + 1)
    } else {
        start
    };
    while i < j {
        let mut p0 = sa[i];
        sa[i] = p0 ^ SAINT_MIN;
        if p0 > 0 {
            p0 -= 1;
            let p0_usize = p0 as usize;
            let bucket0 = t[p0_usize] as usize;
            let slot0 = induction_bucket[bucket0] as usize;
            sa[slot0] = p0
                | ((usize::from(t[p0_usize - usize::from(p0 > 0)] < t[p0_usize]) as SaSint)
                    << (SAINT_BIT - 1));
            induction_bucket[bucket0] += 1;
        }

        let mut p1 = sa[i + 1];
        sa[i + 1] = p1 ^ SAINT_MIN;
        if p1 > 0 {
            p1 -= 1;
            let p1_usize = p1 as usize;
            let bucket1 = t[p1_usize] as usize;
            let slot1 = induction_bucket[bucket1] as usize;
            sa[slot1] = p1
                | ((usize::from(t[p1_usize - usize::from(p1 > 0)] < t[p1_usize]) as SaSint)
                    << (SAINT_BIT - 1));
            induction_bucket[bucket1] += 1;
        }

        i += 2;
    }

    j = start + size;
    while i < j {
        let mut p = sa[i];
        sa[i] = p ^ SAINT_MIN;
        if p > 0 {
            p -= 1;
            let p_usize = p as usize;
            let bucket = t[p_usize] as usize;
            let slot = induction_bucket[bucket] as usize;
            sa[slot] = p
                | ((usize::from(t[p_usize - usize::from(p > 0)] < t[p_usize]) as SaSint)
                    << (SAINT_BIT - 1));
            induction_bucket[bucket] += 1;
        }
        i += 1;
    }
}

/// Internal helper: final sorting scan left to right 32s.
#[doc(hidden)]
pub fn final_sorting_scan_left_to_right_32s(
    t: &[SaSint],
    sa: &mut [SaSint],
    induction_bucket: &mut [SaSint],
    omp_block_start: FastSint,
    omp_block_size: FastSint,
) {
    if omp_block_size <= 0 {
        return;
    }

    let prefetch_distance: FastSint = 64;
    let mut i = omp_block_start;
    let mut j = omp_block_start + omp_block_size - 2 * prefetch_distance - 1;

    while i < j {
        let i0 = i as usize;
        let mut p0 = sa[i0];
        sa[i0] = p0 ^ SAINT_MIN;
        if p0 > 0 {
            p0 -= 1;
            let p0u = p0 as usize;
            let bucket0 = t[p0u] as usize;
            let slot0 = induction_bucket[bucket0] as usize;
            sa[slot0] = p0
                | ((usize::from(t[p0u - usize::from(p0 > 0)] < t[p0u]) as SaSint)
                    << (SAINT_BIT - 1));
            induction_bucket[bucket0] += 1;
        }

        let i1 = (i + 1) as usize;
        let mut p1 = sa[i1];
        sa[i1] = p1 ^ SAINT_MIN;
        if p1 > 0 {
            p1 -= 1;
            let p1u = p1 as usize;
            let bucket1 = t[p1u] as usize;
            let slot1 = induction_bucket[bucket1] as usize;
            sa[slot1] = p1
                | ((usize::from(t[p1u - usize::from(p1 > 0)] < t[p1u]) as SaSint)
                    << (SAINT_BIT - 1));
            induction_bucket[bucket1] += 1;
        }
        i += 2;
    }

    j += 2 * prefetch_distance + 1;
    while i < j {
        let iu = i as usize;
        let mut p = sa[iu];
        sa[iu] = p ^ SAINT_MIN;
        if p > 0 {
            p -= 1;
            let pu = p as usize;
            let bucket = t[pu] as usize;
            let slot = induction_bucket[bucket] as usize;
            sa[slot] = p
                | ((usize::from(t[pu - usize::from(p > 0)] < t[pu]) as SaSint) << (SAINT_BIT - 1));
            induction_bucket[bucket] += 1;
        }
        i += 1;
    }
}

/// Internal helper: final bwt scan left to right 8u block prepare.
#[doc(hidden)]
pub fn final_bwt_scan_left_to_right_8u_block_prepare(
    t: &[u8],
    sa: &mut [SaSint],
    k: SaSint,
    buckets: &mut [SaSint],
    cache: &mut [ThreadCache],
    omp_block_start: FastSint,
    omp_block_size: FastSint,
) -> FastSint {
    if omp_block_size <= 0 {
        return 0;
    }

    let k_usize = usize::try_from(k).expect("k must be non-negative");
    buckets[..k_usize].fill(0);

    let start = usize::try_from(omp_block_start).expect("omp_block_start must be non-negative");
    let size = usize::try_from(omp_block_size).expect("omp_block_size must be non-negative");
    let mut count = 0usize;
    for i in start..start + size {
        let mut p = sa[i];
        sa[i] = p & SAINT_MAX;
        if p > 0 {
            p -= 1;
            let p_usize = usize::try_from(p).expect("suffix index must be non-negative");
            let symbol = t[p_usize] as usize;
            sa[i] = t[p_usize] as SaSint | SAINT_MIN;
            buckets[symbol] += 1;
            cache[count].symbol = symbol as SaSint;
            cache[count].index = p
                | ((usize::from(t[p_usize - usize::from(p > 0)] < t[p_usize]) as SaSint)
                    << (SAINT_BIT - 1));
            count += 1;
        }
    }

    count as FastSint
}

/// Internal helper: final sorting scan left to right 8u block prepare.
#[doc(hidden)]
pub fn final_sorting_scan_left_to_right_8u_block_prepare(
    t: &[u8],
    sa: &mut [SaSint],
    k: SaSint,
    buckets: &mut [SaSint],
    cache: &mut [ThreadCache],
    omp_block_start: FastSint,
    omp_block_size: FastSint,
) -> FastSint {
    if omp_block_size <= 0 {
        return 0;
    }

    let k_usize = usize::try_from(k).expect("k must be non-negative");
    buckets[..k_usize].fill(0);

    let start = usize::try_from(omp_block_start).expect("omp_block_start must be non-negative");
    let size = usize::try_from(omp_block_size).expect("omp_block_size must be non-negative");
    let mut count = 0usize;
    for i in start..start + size {
        let mut p = sa[i];
        sa[i] = p ^ SAINT_MIN;
        if p > 0 {
            p -= 1;
            let p_usize = usize::try_from(p).expect("suffix index must be non-negative");
            let symbol = t[p_usize] as usize;
            buckets[symbol] += 1;
            cache[count].symbol = symbol as SaSint;
            cache[count].index = p
                | ((usize::from(t[p_usize - usize::from(p > 0)] < t[p_usize]) as SaSint)
                    << (SAINT_BIT - 1));
            count += 1;
        }
    }

    count as FastSint
}

/// Internal helper: final order scan left to right 8u block place.
#[doc(hidden)]
pub fn final_order_scan_left_to_right_8u_block_place(
    sa: &mut [SaSint],
    buckets: &mut [SaSint],
    cache: &[ThreadCache],
    count: FastSint,
) {
    if count <= 0 {
        return;
    }

    let count_usize = usize::try_from(count).expect("count must be non-negative");
    for entry in &cache[..count_usize] {
        let symbol = usize::try_from(entry.symbol).expect("cache symbol must be non-negative");
        let slot = usize::try_from(buckets[symbol]).expect("bucket slot must be non-negative");
        sa[slot] = entry.index;
        buckets[symbol] += 1;
    }
}

/// Internal helper: final bwt aux scan left to right 8u block place.
#[doc(hidden)]
pub fn final_bwt_aux_scan_left_to_right_8u_block_place(
    sa: &mut [SaSint],
    rm: SaSint,
    i_out: &mut [SaSint],
    buckets: &mut [SaSint],
    cache: &[ThreadCache],
    count: FastSint,
) {
    if count <= 0 {
        return;
    }

    let count_usize = usize::try_from(count).expect("count must be non-negative");
    for entry in &cache[..count_usize] {
        let symbol = usize::try_from(entry.symbol).expect("cache symbol must be non-negative");
        let slot = usize::try_from(buckets[symbol]).expect("bucket slot must be non-negative");
        sa[slot] = entry.index;
        buckets[symbol] += 1;
        if (entry.index & rm) == 0 {
            let sample_index = usize::try_from((entry.index & SAINT_MAX) / (rm + 1))
                .expect("sample index must be non-negative");
            i_out[sample_index] = buckets[symbol];
        }
    }
}

/// Internal helper: final sorting scan left to right 32s block gather.
#[doc(hidden)]
pub fn final_sorting_scan_left_to_right_32s_block_gather(
    t: &[SaSint],
    sa: &mut [SaSint],
    cache: &mut [ThreadCache],
    omp_block_start: FastSint,
    omp_block_size: FastSint,
) {
    if omp_block_size <= 0 {
        return;
    }
    let start = usize::try_from(omp_block_start).expect("omp_block_start must be non-negative");
    let size = usize::try_from(omp_block_size).expect("omp_block_size must be non-negative");
    for offset in 0..size {
        let i = start + offset;
        let mut symbol = SAINT_MIN;
        let mut p = sa[i];
        sa[i] = p ^ SAINT_MIN;
        if p > 0 {
            p -= 1;
            let p_usize = p as usize;
            cache[offset].index = p
                | ((usize::from(t[p_usize - usize::from(p > 0)] < t[p_usize]) as SaSint)
                    << (SAINT_BIT - 1));
            symbol = t[p_usize];
        }
        cache[offset].symbol = symbol;
    }
}

/// Internal helper: final sorting scan left to right 32s block sort.
#[doc(hidden)]
pub fn final_sorting_scan_left_to_right_32s_block_sort(
    t: &[SaSint],
    induction_bucket: &mut [SaSint],
    cache: &mut [ThreadCache],
    omp_block_start: FastSint,
    omp_block_size: FastSint,
) {
    if omp_block_size <= 0 {
        return;
    }
    let prefetch_distance = 64usize;
    let start = omp_block_start as usize;
    let block_end = start + omp_block_size as usize;
    let mut i = start;
    let mut j = start + (omp_block_size as usize).saturating_sub(prefetch_distance + 1);

    while i < j {
        let ci = i - start;
        let v0 = cache[ci].symbol;
        if v0 >= 0 {
            let bucket_index0 = v0 as usize;
            cache[ci].symbol = induction_bucket[bucket_index0];
            induction_bucket[bucket_index0] += 1;
            if cache[ci].symbol < block_end as SaSint {
                let ni = cache[ci].symbol as usize;
                let cni = ni - start;
                let mut np = cache[ci].index;
                cache[ci].index = np ^ SAINT_MIN;
                if np > 0 {
                    np -= 1;
                    let np_usize = np as usize;
                    cache[cni].index = np
                        | ((usize::from(t[np_usize - usize::from(np > 0)] < t[np_usize])
                            as SaSint)
                            << (SAINT_BIT - 1));
                    cache[cni].symbol = t[np_usize];
                }
            }
        }

        let i1 = i + 1;
        let ci1 = i1 - start;
        let v1 = cache[ci1].symbol;
        if v1 >= 0 {
            let bucket_index1 = v1 as usize;
            cache[ci1].symbol = induction_bucket[bucket_index1];
            induction_bucket[bucket_index1] += 1;
            if cache[ci1].symbol < block_end as SaSint {
                let ni = cache[ci1].symbol as usize;
                let cni = ni - start;
                let mut np = cache[ci1].index;
                cache[ci1].index = np ^ SAINT_MIN;
                if np > 0 {
                    np -= 1;
                    let np_usize = np as usize;
                    cache[cni].index = np
                        | ((usize::from(t[np_usize - usize::from(np > 0)] < t[np_usize])
                            as SaSint)
                            << (SAINT_BIT - 1));
                    cache[cni].symbol = t[np_usize];
                }
            }
        }

        i += 2;
    }

    j = block_end;
    while i < j {
        let ci = i - start;
        let v = cache[ci].symbol;
        if v >= 0 {
            let bucket_index = v as usize;
            cache[ci].symbol = induction_bucket[bucket_index];
            induction_bucket[bucket_index] += 1;
            if cache[ci].symbol < block_end as SaSint {
                let ni = cache[ci].symbol as usize;
                let cni = ni - start;
                let mut np = cache[ci].index;
                cache[ci].index = np ^ SAINT_MIN;
                if np > 0 {
                    np -= 1;
                    let np_usize = np as usize;
                    cache[cni].index = np
                        | ((usize::from(t[np_usize - usize::from(np > 0)] < t[np_usize])
                            as SaSint)
                            << (SAINT_BIT - 1));
                    cache[cni].symbol = t[np_usize];
                }
            }
        }
        i += 1;
    }
}

/// Internal helper: final bwt scan left to right 8u block (OpenMP variant).
#[doc(hidden)]
pub fn final_bwt_scan_left_to_right_8u_block_omp(
    t: &[u8],
    sa: &mut [SaSint],
    k: SaSint,
    induction_bucket: &mut [SaSint],
    block_start: FastSint,
    block_size: FastSint,
    threads: SaSint,
    thread_state: &mut [ThreadState],
) {
    if block_size <= 0 {
        return;
    }

    let k_usize = usize::try_from(k).expect("k must be non-negative");
    let block_size_usize = usize::try_from(block_size).expect("block_size must be non-negative");
    let omp_num_threads = if threads > 1 && block_size_usize >= 64 * k_usize.max(256) {
        usize::try_from(threads)
            .expect("threads must be non-negative")
            .min(thread_state.len())
            .max(1)
    } else {
        1
    };

    if omp_num_threads == 1 {
        final_bwt_scan_left_to_right_8u(t, sa, induction_bucket, block_start, block_size);
        return;
    }

    let block_start_usize = usize::try_from(block_start).expect("block_start must be non-negative");
    let omp_block_stride = (block_size_usize / omp_num_threads) & !15usize;
    for (thread_num, state) in thread_state.iter_mut().take(omp_num_threads).enumerate() {
        let relative_start = thread_num * omp_block_stride;
        let size = if thread_num + 1 < omp_num_threads {
            omp_block_stride
        } else {
            block_size_usize - relative_start
        };
        state.count = final_bwt_scan_left_to_right_8u_block_prepare(
            t,
            sa,
            k,
            &mut state.buckets,
            &mut state.cache,
            (block_start_usize + relative_start) as FastSint,
            size as FastSint,
        );
    }

    for state in thread_state.iter_mut().take(omp_num_threads) {
        for (c, bucket) in induction_bucket.iter_mut().take(k_usize).enumerate() {
            let a = *bucket;
            let b = state.buckets[c];
            *bucket = a + b;
            state.buckets[c] = a;
        }
    }

    for state in thread_state.iter_mut().take(omp_num_threads) {
        final_order_scan_left_to_right_8u_block_place(
            sa,
            &mut state.buckets,
            &state.cache,
            state.count,
        );
    }
}

/// Internal helper: final bwt aux scan left to right 8u block (OpenMP variant).
#[doc(hidden)]
pub fn final_bwt_aux_scan_left_to_right_8u_block_omp(
    t: &[u8],
    sa: &mut [SaSint],
    k: SaSint,
    rm: SaSint,
    i_out: &mut [SaSint],
    induction_bucket: &mut [SaSint],
    block_start: FastSint,
    block_size: FastSint,
    threads: SaSint,
    thread_state: &mut [ThreadState],
) {
    if block_size <= 0 {
        return;
    }

    let k_usize = usize::try_from(k).expect("k must be non-negative");
    let block_size_usize = usize::try_from(block_size).expect("block_size must be non-negative");
    let omp_num_threads = if threads > 1 && block_size_usize >= 64 * k_usize.max(256) {
        usize::try_from(threads)
            .expect("threads must be non-negative")
            .min(thread_state.len())
            .max(1)
    } else {
        1
    };

    if omp_num_threads == 1 {
        final_bwt_aux_scan_left_to_right_8u(
            t,
            sa,
            rm,
            i_out,
            induction_bucket,
            block_start,
            block_size,
        );
        return;
    }

    let block_start_usize = usize::try_from(block_start).expect("block_start must be non-negative");
    let omp_block_stride = (block_size_usize / omp_num_threads) & !15usize;
    for (thread_num, state) in thread_state.iter_mut().take(omp_num_threads).enumerate() {
        let relative_start = thread_num * omp_block_stride;
        let size = if thread_num + 1 < omp_num_threads {
            omp_block_stride
        } else {
            block_size_usize - relative_start
        };
        state.count = final_bwt_scan_left_to_right_8u_block_prepare(
            t,
            sa,
            k,
            &mut state.buckets,
            &mut state.cache,
            (block_start_usize + relative_start) as FastSint,
            size as FastSint,
        );
    }

    for state in thread_state.iter_mut().take(omp_num_threads) {
        for (c, bucket) in induction_bucket.iter_mut().take(k_usize).enumerate() {
            let a = *bucket;
            let b = state.buckets[c];
            *bucket = a + b;
            state.buckets[c] = a;
        }
    }

    for state in thread_state.iter_mut().take(omp_num_threads) {
        final_bwt_aux_scan_left_to_right_8u_block_place(
            sa,
            rm,
            i_out,
            &mut state.buckets,
            &state.cache,
            state.count,
        );
    }
}

/// Internal helper: final sorting scan left to right 8u block (OpenMP variant).
#[doc(hidden)]
pub fn final_sorting_scan_left_to_right_8u_block_omp(
    t: &[u8],
    sa: &mut [SaSint],
    k: SaSint,
    induction_bucket: &mut [SaSint],
    block_start: FastSint,
    block_size: FastSint,
    threads: SaSint,
    thread_state: &mut [ThreadState],
) {
    if block_size <= 0 {
        return;
    }

    let k_usize = usize::try_from(k).expect("k must be non-negative");
    let block_size_usize = usize::try_from(block_size).expect("block_size must be non-negative");
    let omp_num_threads = if threads > 1 && block_size_usize >= 64 * k_usize.max(256) {
        usize::try_from(threads)
            .expect("threads must be non-negative")
            .min(thread_state.len())
            .max(1)
    } else {
        1
    };

    if omp_num_threads == 1 {
        final_sorting_scan_left_to_right_8u(t, sa, induction_bucket, block_start, block_size);
        return;
    }

    let block_start_usize = usize::try_from(block_start).expect("block_start must be non-negative");
    let omp_block_stride = (block_size_usize / omp_num_threads) & !15usize;
    for (thread_num, state) in thread_state.iter_mut().take(omp_num_threads).enumerate() {
        let relative_start = thread_num * omp_block_stride;
        let size = if thread_num + 1 < omp_num_threads {
            omp_block_stride
        } else {
            block_size_usize - relative_start
        };
        state.count = final_sorting_scan_left_to_right_8u_block_prepare(
            t,
            sa,
            k,
            &mut state.buckets,
            &mut state.cache,
            (block_start_usize + relative_start) as FastSint,
            size as FastSint,
        );
    }

    for state in thread_state.iter_mut().take(omp_num_threads) {
        for (c, bucket) in induction_bucket.iter_mut().take(k_usize).enumerate() {
            let a = *bucket;
            let b = state.buckets[c];
            *bucket = a + b;
            state.buckets[c] = a;
        }
    }

    for state in thread_state.iter_mut().take(omp_num_threads) {
        final_order_scan_left_to_right_8u_block_place(
            sa,
            &mut state.buckets,
            &state.cache,
            state.count,
        );
    }
}

/// Internal helper: final sorting scan left to right 32s block (OpenMP variant).
#[doc(hidden)]
pub fn final_sorting_scan_left_to_right_32s_block_omp(
    t: &[SaSint],
    sa: &mut [SaSint],
    buckets: &mut [SaSint],
    cache: &mut [ThreadCache],
    block_start: FastSint,
    block_size: FastSint,
    threads: SaSint,
) {
    if threads <= 1 || block_size < 16_384 {
        final_sorting_scan_left_to_right_32s(t, sa, buckets, block_start, block_size);
        return;
    }

    final_sorting_scan_left_to_right_32s_block_gather(t, sa, cache, block_start, block_size);
    final_sorting_scan_left_to_right_32s_block_sort(t, buckets, cache, block_start, block_size);
    let block_size_usize = usize::try_from(block_size).expect("block_size must be non-negative");
    let threads_usize = usize::try_from(threads.max(1)).expect("threads must be positive");
    let omp_num_threads = threads_usize.min(block_size_usize);
    let omp_block_stride = (block_size_usize / omp_num_threads) & !15usize;
    {
        let sa_ptr = SyncMutPtr::new(sa);
        let cache_ptr = SyncMutPtr::new(cache);
        run_rayon_with_threads(omp_num_threads, || {
            (0..omp_num_threads)
                .into_par_iter()
                .for_each(|omp_thread_num| {
                    let omp_block_start = omp_thread_num * omp_block_stride;
                    let omp_block_size = if omp_thread_num + 1 < omp_num_threads {
                        omp_block_stride
                    } else {
                        block_size_usize - omp_block_start
                    };
                    let sa = unsafe { sa_ptr.as_slice() };
                    let cache = unsafe { cache_ptr.as_slice() };
                    compact_and_place_cached_suffixes(
                        sa,
                        cache,
                        omp_block_start as FastSint,
                        omp_block_size as FastSint,
                    );
                });
        });
    }
}

/// Internal helper: final bwt scan left to right 8u (OpenMP variant).
#[doc(hidden)]
pub fn final_bwt_scan_left_to_right_8u_omp(
    t: &[u8],
    sa: &mut [SaSint],
    n: FastSint,
    k: SaSint,
    induction_bucket: &mut [SaSint],
    threads: SaSint,
    thread_state: &mut [ThreadState],
) {
    let n_usize = usize::try_from(n).expect("n must be non-negative");
    let last = n_usize - 1;
    let bucket = t[last] as usize;
    let slot = usize::try_from(induction_bucket[bucket]).expect("bucket slot must be non-negative");
    sa[slot] =
        (n as SaSint - 1) | ((usize::from(t[last - 1] < t[last]) as SaSint) << (SAINT_BIT - 1));
    induction_bucket[bucket] += 1;

    if threads == 1 || n < 65_536 {
        final_bwt_scan_left_to_right_8u(t, sa, induction_bucket, 0, n);
        return;
    }

    let mut block_start = 0usize;
    while block_start < n_usize {
        if sa[block_start] == 0 {
            block_start += 1;
        } else {
            let threads_usize = usize::try_from(threads)
                .expect("threads must be non-negative")
                .min(thread_state.len())
                .max(1);
            let max_span = threads_usize * (LIBSAIS_PER_THREAD_CACHE_SIZE - 16 * threads_usize);
            let block_max_end = (block_start + max_span).min(n_usize);
            let mut block_end = block_start + 1;
            while block_end < block_max_end && sa[block_end] != 0 {
                block_end += 1;
            }
            let size = block_end - block_start;

            if size < 32 {
                final_bwt_scan_left_to_right_8u(
                    t,
                    sa,
                    induction_bucket,
                    block_start as FastSint,
                    size as FastSint,
                );
            } else {
                final_bwt_scan_left_to_right_8u_block_omp(
                    t,
                    sa,
                    k,
                    induction_bucket,
                    block_start as FastSint,
                    size as FastSint,
                    threads,
                    thread_state,
                );
            }
            block_start = block_end;
        }
    }
}

/// Internal helper: final bwt aux scan left to right 8u (OpenMP variant).
#[doc(hidden)]
pub fn final_bwt_aux_scan_left_to_right_8u_omp(
    t: &[u8],
    sa: &mut [SaSint],
    n: FastSint,
    k: SaSint,
    rm: SaSint,
    i_out: &mut [SaSint],
    induction_bucket: &mut [SaSint],
    threads: SaSint,
    thread_state: &mut [ThreadState],
) {
    let n_usize = usize::try_from(n).expect("n must be non-negative");
    let last = n_usize - 1;
    let bucket = t[last] as usize;
    let slot = usize::try_from(induction_bucket[bucket]).expect("bucket slot must be non-negative");
    sa[slot] =
        (n as SaSint - 1) | ((usize::from(t[last - 1] < t[last]) as SaSint) << (SAINT_BIT - 1));
    induction_bucket[bucket] += 1;
    if (((n as SaSint) - 1) & rm) == 0 {
        i_out[last / usize::try_from(rm + 1).expect("rm must allow positive step")] =
            induction_bucket[bucket];
    }

    if threads == 1 || n < 65_536 {
        final_bwt_aux_scan_left_to_right_8u(t, sa, rm, i_out, induction_bucket, 0, n);
        return;
    }

    let mut block_start = 0usize;
    while block_start < n_usize {
        if sa[block_start] == 0 {
            block_start += 1;
        } else {
            let threads_usize = usize::try_from(threads)
                .expect("threads must be non-negative")
                .min(thread_state.len())
                .max(1);
            let max_span = threads_usize * (LIBSAIS_PER_THREAD_CACHE_SIZE - 16 * threads_usize);
            let block_max_end = (block_start + max_span).min(n_usize);
            let mut block_end = block_start + 1;
            while block_end < block_max_end && sa[block_end] != 0 {
                block_end += 1;
            }
            let size = block_end - block_start;

            if size < 32 {
                final_bwt_aux_scan_left_to_right_8u(
                    t,
                    sa,
                    rm,
                    i_out,
                    induction_bucket,
                    block_start as FastSint,
                    size as FastSint,
                );
            } else {
                final_bwt_aux_scan_left_to_right_8u_block_omp(
                    t,
                    sa,
                    k,
                    rm,
                    i_out,
                    induction_bucket,
                    block_start as FastSint,
                    size as FastSint,
                    threads,
                    thread_state,
                );
            }
            block_start = block_end;
        }
    }
}

/// Internal helper: final sorting scan left to right 8u (OpenMP variant).
#[doc(hidden)]
pub fn final_sorting_scan_left_to_right_8u_omp(
    t: &[u8],
    sa: &mut [SaSint],
    n: FastSint,
    k: SaSint,
    induction_bucket: &mut [SaSint],
    threads: SaSint,
    thread_state: &mut [ThreadState],
) {
    let n_usize = usize::try_from(n).expect("n must be non-negative");
    let last = n_usize - 1;
    let bucket = t[last] as usize;
    let slot = usize::try_from(induction_bucket[bucket]).expect("bucket slot must be non-negative");
    sa[slot] =
        (n as SaSint - 1) | ((usize::from(t[last - 1] < t[last]) as SaSint) << (SAINT_BIT - 1));
    induction_bucket[bucket] += 1;

    if threads == 1 || n < 65_536 {
        final_sorting_scan_left_to_right_8u(t, sa, induction_bucket, 0, n);
        return;
    }

    let mut block_start = 0usize;
    while block_start < n_usize {
        if sa[block_start] == 0 {
            block_start += 1;
        } else {
            let threads_usize = usize::try_from(threads)
                .expect("threads must be non-negative")
                .min(thread_state.len())
                .max(1);
            let max_span = threads_usize * (LIBSAIS_PER_THREAD_CACHE_SIZE - 16 * threads_usize);
            let block_max_end = (block_start + max_span).min(n_usize);
            let mut block_end = block_start + 1;
            while block_end < block_max_end && sa[block_end] != 0 {
                block_end += 1;
            }
            let size = block_end - block_start;

            if size < 32 {
                final_sorting_scan_left_to_right_8u(
                    t,
                    sa,
                    induction_bucket,
                    block_start as FastSint,
                    size as FastSint,
                );
            } else {
                final_sorting_scan_left_to_right_8u_block_omp(
                    t,
                    sa,
                    k,
                    induction_bucket,
                    block_start as FastSint,
                    size as FastSint,
                    threads,
                    thread_state,
                );
            }
            block_start = block_end;
        }
    }
}

/// Internal helper: final sorting scan left to right 32s (OpenMP variant).
#[doc(hidden)]
pub fn final_sorting_scan_left_to_right_32s_omp(
    t: &[SaSint],
    sa: &mut [SaSint],
    n: SaSint,
    induction_bucket: &mut [SaSint],
    threads: SaSint,
    thread_state: &mut [ThreadState],
) {
    let n_usize = usize::try_from(n).expect("n must be non-negative");
    let last = n_usize - 1;
    let bucket = usize::try_from(t[last]).expect("bucket symbol must be non-negative");
    let slot = usize::try_from(induction_bucket[bucket]).expect("bucket slot must be non-negative");
    sa[slot] = (n - 1) | ((usize::from(t[last - 1] < t[last]) as SaSint) << (SAINT_BIT - 1));
    induction_bucket[bucket] += 1;

    if threads == 1 || n < 65_536 {
        final_sorting_scan_left_to_right_32s(t, sa, induction_bucket, 0, n as FastSint);
        return;
    }

    if thread_state.is_empty() {
        final_sorting_scan_left_to_right_32s(t, sa, induction_bucket, 0, n as FastSint);
        return;
    }

    let threads_usize = usize::try_from(threads)
        .expect("threads must be non-negative")
        .max(1);
    let mut cache = vec![ThreadCache::default(); threads_usize * LIBSAIS_PER_THREAD_CACHE_SIZE];
    let mut block_start = 0usize;
    while block_start < n_usize {
        let block_end = (block_start + threads_usize * LIBSAIS_PER_THREAD_CACHE_SIZE).min(n_usize);
        final_sorting_scan_left_to_right_32s_block_omp(
            t,
            sa,
            induction_bucket,
            &mut cache,
            block_start as FastSint,
            (block_end - block_start) as FastSint,
            threads,
        );
        block_start = block_end;
    }
}

/// Internal helper: final bwt scan right to left 8u.
#[doc(hidden)]
pub fn final_bwt_scan_right_to_left_8u(
    t: &[u8],
    sa: &mut [SaSint],
    induction_bucket: &mut [SaSint],
    omp_block_start: FastSint,
    omp_block_size: FastSint,
) -> SaSint {
    if omp_block_size <= 0 {
        return -1;
    }

    let mut index = -1;

    let start =
        usize::try_from(omp_block_start).expect("omp_block_start must be non-negative") as FastSint;
    let mut i = omp_block_start + omp_block_size - 1;
    let mut j = start + 1;
    while i >= j {
        let i0 = usize::try_from(i).expect("loop index must be non-negative");
        let i1 = usize::try_from(i - 1).expect("loop index must be non-negative");

        let mut p0 = sa[i0];
        if p0 == 0 {
            index = i0 as SaSint;
        }
        sa[i0] = p0 & SAINT_MAX;
        if p0 > 0 {
            p0 -= 1;
            let p0_usize = usize::try_from(p0).expect("suffix index must be non-negative");
            let c0 = t[p0_usize - usize::from(p0 > 0)] as SaSint;
            let c1 = t[p0_usize] as SaSint;
            sa[i0] = c1;
            induction_bucket[c1 as usize] -= 1;
            let slot = usize::try_from(induction_bucket[c1 as usize])
                .expect("bucket slot must be non-negative");
            let marked = c0 | SAINT_MIN;
            sa[slot] = if c0 <= c1 { p0 } else { marked };
        }

        let mut p1 = sa[i1];
        if p1 == 0 {
            index = i1 as SaSint;
        }
        sa[i1] = p1 & SAINT_MAX;
        if p1 > 0 {
            p1 -= 1;
            let p1_usize = usize::try_from(p1).expect("suffix index must be non-negative");
            let c0 = t[p1_usize - usize::from(p1 > 0)] as SaSint;
            let c1 = t[p1_usize] as SaSint;
            sa[i1] = c1;
            induction_bucket[c1 as usize] -= 1;
            let slot = usize::try_from(induction_bucket[c1 as usize])
                .expect("bucket slot must be non-negative");
            let marked = c0 | SAINT_MIN;
            sa[slot] = if c0 <= c1 { p1 } else { marked };
        }

        i -= 2;
    }

    j -= 1;
    while i >= j {
        let idx = usize::try_from(i).expect("loop index must be non-negative");
        let mut p = sa[idx];
        if p == 0 {
            index = idx as SaSint;
        }
        sa[idx] = p & SAINT_MAX;
        if p > 0 {
            p -= 1;
            let p_usize = usize::try_from(p).expect("suffix index must be non-negative");
            let c0 = t[p_usize - usize::from(p > 0)] as SaSint;
            let c1 = t[p_usize] as SaSint;
            sa[idx] = c1;
            induction_bucket[c1 as usize] -= 1;
            let slot = usize::try_from(induction_bucket[c1 as usize])
                .expect("bucket slot must be non-negative");
            let marked = c0 | SAINT_MIN;
            sa[slot] = if c0 <= c1 { p } else { marked };
        }

        i -= 1;
    }

    index
}

/// Internal helper: final bwt aux scan right to left 8u.
#[doc(hidden)]
pub fn final_bwt_aux_scan_right_to_left_8u(
    t: &[u8],
    sa: &mut [SaSint],
    rm: SaSint,
    i_out: &mut [SaSint],
    induction_bucket: &mut [SaSint],
    omp_block_start: FastSint,
    omp_block_size: FastSint,
) {
    if omp_block_size <= 0 {
        return;
    }

    let start =
        usize::try_from(omp_block_start).expect("omp_block_start must be non-negative") as FastSint;
    let mut i = omp_block_start + omp_block_size - 1;
    let mut j = start + 1;
    while i >= j {
        let i0 = usize::try_from(i).expect("loop index must be non-negative");
        let i1 = usize::try_from(i - 1).expect("loop index must be non-negative");

        let mut p0 = sa[i0];
        sa[i0] = p0 & SAINT_MAX;
        if p0 > 0 {
            p0 -= 1;
            let p0_usize = usize::try_from(p0).expect("suffix index must be non-negative");
            let c0 = t[p0_usize - usize::from(p0 > 0)] as SaSint;
            let c1 = t[p0_usize] as SaSint;
            sa[i0] = c1;
            induction_bucket[c1 as usize] -= 1;
            let slot = usize::try_from(induction_bucket[c1 as usize])
                .expect("bucket slot must be non-negative");
            let marked = c0 | SAINT_MIN;
            sa[slot] = if c0 <= c1 { p0 } else { marked };
            if (p0 & rm) == 0 {
                let out_idx =
                    usize::try_from(p0 / (rm + 1)).expect("sample index must be non-negative");
                i_out[out_idx] = induction_bucket[t[p0_usize] as usize] + 1;
            }
        }

        let mut p1 = sa[i1];
        sa[i1] = p1 & SAINT_MAX;
        if p1 > 0 {
            p1 -= 1;
            let p1_usize = usize::try_from(p1).expect("suffix index must be non-negative");
            let c0 = t[p1_usize - usize::from(p1 > 0)] as SaSint;
            let c1 = t[p1_usize] as SaSint;
            sa[i1] = c1;
            induction_bucket[c1 as usize] -= 1;
            let slot = usize::try_from(induction_bucket[c1 as usize])
                .expect("bucket slot must be non-negative");
            let marked = c0 | SAINT_MIN;
            sa[slot] = if c0 <= c1 { p1 } else { marked };
            if (p1 & rm) == 0 {
                let out_idx =
                    usize::try_from(p1 / (rm + 1)).expect("sample index must be non-negative");
                i_out[out_idx] = induction_bucket[t[p1_usize] as usize] + 1;
            }
        }

        i -= 2;
    }

    j -= 1;
    while i >= j {
        let idx = usize::try_from(i).expect("loop index must be non-negative");
        let mut p = sa[idx];
        sa[idx] = p & SAINT_MAX;
        if p > 0 {
            p -= 1;
            let p_usize = usize::try_from(p).expect("suffix index must be non-negative");
            let c0 = t[p_usize - usize::from(p > 0)] as SaSint;
            let c1 = t[p_usize] as SaSint;
            sa[idx] = c1;
            induction_bucket[c1 as usize] -= 1;
            let slot = usize::try_from(induction_bucket[c1 as usize])
                .expect("bucket slot must be non-negative");
            let marked = c0 | SAINT_MIN;
            sa[slot] = if c0 <= c1 { p } else { marked };
            if (p & rm) == 0 {
                let out_idx =
                    usize::try_from(p / (rm + 1)).expect("sample index must be non-negative");
                i_out[out_idx] = induction_bucket[t[p_usize] as usize] + 1;
            }
        }

        i -= 1;
    }
}

/// Internal helper: final sorting scan right to left 8u.
#[doc(hidden)]
pub fn final_sorting_scan_right_to_left_8u(
    t: &[u8],
    sa: &mut [SaSint],
    induction_bucket: &mut [SaSint],
    omp_block_start: FastSint,
    omp_block_size: FastSint,
) {
    if omp_block_size <= 0 {
        return;
    }

    let prefetch_distance = 64usize;
    let start = usize::try_from(omp_block_start).expect("omp_block_start must be non-negative");
    let size = usize::try_from(omp_block_size).expect("omp_block_size must be non-negative");
    let mut i = start + size - 1;
    let mut j = start + prefetch_distance + 1;

    while i >= j {
        let mut p0 = sa[i];
        sa[i] = p0 & SAINT_MAX;
        if p0 > 0 {
            p0 -= 1;
            let p0_usize = p0 as usize;
            let bucket0 = t[p0_usize] as usize;
            induction_bucket[bucket0] -= 1;
            let slot0 = induction_bucket[bucket0] as usize;
            sa[slot0] = p0
                | ((usize::from(t[p0_usize - usize::from(p0 > 0)] > t[p0_usize]) as SaSint)
                    << (SAINT_BIT - 1));
        }

        let mut p1 = sa[i - 1];
        sa[i - 1] = p1 & SAINT_MAX;
        if p1 > 0 {
            p1 -= 1;
            let p1_usize = p1 as usize;
            let bucket1 = t[p1_usize] as usize;
            induction_bucket[bucket1] -= 1;
            let slot1 = induction_bucket[bucket1] as usize;
            sa[slot1] = p1
                | ((usize::from(t[p1_usize - usize::from(p1 > 0)] > t[p1_usize]) as SaSint)
                    << (SAINT_BIT - 1));
        }

        i -= 2;
    }

    j -= prefetch_distance + 1;
    while i >= j {
        let mut p = sa[i];
        sa[i] = p & SAINT_MAX;
        if p > 0 {
            p -= 1;
            let p_usize = p as usize;
            let bucket = t[p_usize] as usize;
            induction_bucket[bucket] -= 1;
            let slot = induction_bucket[bucket] as usize;
            sa[slot] = p
                | ((usize::from(t[p_usize - usize::from(p > 0)] > t[p_usize]) as SaSint)
                    << (SAINT_BIT - 1));
        }

        if i == 0 {
            break;
        }
        i -= 1;
    }
}

/// Internal helper: final gsa scan right to left 8u.
#[doc(hidden)]
pub fn final_gsa_scan_right_to_left_8u(
    t: &[u8],
    sa: &mut [SaSint],
    induction_bucket: &mut [SaSint],
    omp_block_start: FastSint,
    omp_block_size: FastSint,
) {
    if omp_block_size <= 0 {
        return;
    }

    let start = usize::try_from(omp_block_start).expect("omp_block_start must be non-negative");
    let size = usize::try_from(omp_block_size).expect("omp_block_size must be non-negative");
    let mut i = start + size;
    while i > start {
        i -= 1;
        let mut p = sa[i];
        sa[i] = p & SAINT_MAX;
        if p > 0 {
            let p_usize = usize::try_from(p).expect("suffix index must be non-negative");
            if t[p_usize - 1] > 0 {
                p -= 1;
                let bucket =
                    t[usize::try_from(p).expect("suffix index must be non-negative")] as usize;
                induction_bucket[bucket] -= 1;
                let slot = usize::try_from(induction_bucket[bucket])
                    .expect("bucket slot must be non-negative");
                let p_usize = usize::try_from(p).expect("suffix index must be non-negative");
                sa[slot] = p
                    | ((usize::from(t[p_usize - usize::from(p > 0)] > t[p_usize]) as SaSint)
                        << (SAINT_BIT - 1));
            }
        }
    }
}

/// Internal helper: final sorting scan right to left 32s.
#[doc(hidden)]
pub fn final_sorting_scan_right_to_left_32s(
    t: &[SaSint],
    sa: &mut [SaSint],
    induction_bucket: &mut [SaSint],
    omp_block_start: FastSint,
    omp_block_size: FastSint,
) {
    if omp_block_size <= 0 {
        return;
    }

    let prefetch_distance: FastSint = 64;
    let mut i = omp_block_start + omp_block_size - 1;
    let mut j = omp_block_start + 2 * prefetch_distance + 1;

    while i >= j {
        let i0 = i as usize;
        let mut p0 = sa[i0];
        sa[i0] = p0 & SAINT_MAX;
        if p0 > 0 {
            p0 -= 1;
            let p0u = p0 as usize;
            let bucket0 = t[p0u] as usize;
            induction_bucket[bucket0] -= 1;
            let slot0 = induction_bucket[bucket0] as usize;
            sa[slot0] = p0
                | ((usize::from(t[p0u - usize::from(p0 > 0)] > t[p0u]) as SaSint)
                    << (SAINT_BIT - 1));
        }

        let i1 = (i - 1) as usize;
        let mut p1 = sa[i1];
        sa[i1] = p1 & SAINT_MAX;
        if p1 > 0 {
            p1 -= 1;
            let p1u = p1 as usize;
            let bucket1 = t[p1u] as usize;
            induction_bucket[bucket1] -= 1;
            let slot1 = induction_bucket[bucket1] as usize;
            sa[slot1] = p1
                | ((usize::from(t[p1u - usize::from(p1 > 0)] > t[p1u]) as SaSint)
                    << (SAINT_BIT - 1));
        }
        i -= 2;
    }

    j -= 2 * prefetch_distance + 1;
    while i >= j {
        let iu = i as usize;
        let mut p = sa[iu];
        sa[iu] = p & SAINT_MAX;
        if p > 0 {
            p -= 1;
            let pu = p as usize;
            let bucket = t[pu] as usize;
            induction_bucket[bucket] -= 1;
            let slot = induction_bucket[bucket] as usize;
            sa[slot] = p
                | ((usize::from(t[pu - usize::from(p > 0)] > t[pu]) as SaSint) << (SAINT_BIT - 1));
        }
        i -= 1;
    }
}

/// Internal helper: final bwt scan right to left 8u block prepare.
#[doc(hidden)]
pub fn final_bwt_scan_right_to_left_8u_block_prepare(
    t: &[u8],
    sa: &mut [SaSint],
    k: SaSint,
    buckets: &mut [SaSint],
    cache: &mut [ThreadCache],
    omp_block_start: FastSint,
    omp_block_size: FastSint,
) -> FastSint {
    if omp_block_size <= 0 {
        return 0;
    }
    let k_usize = usize::try_from(k).expect("k must be non-negative");
    buckets[..k_usize].fill(0);
    let start = usize::try_from(omp_block_start).expect("omp_block_start must be non-negative");
    let size = usize::try_from(omp_block_size).expect("omp_block_size must be non-negative");
    let mut count = 0usize;
    let mut i = start + size;
    while i > start {
        i -= 1;
        let mut p = sa[i];
        sa[i] = p & SAINT_MAX;
        if p > 0 {
            p -= 1;
            let p_usize = usize::try_from(p).expect("suffix index must be non-negative");
            let c0 = t[p_usize - usize::from(p > 0)] as SaSint;
            let c1 = t[p_usize] as SaSint;
            sa[i] = c1;
            buckets[c1 as usize] += 1;
            cache[count].symbol = c1;
            cache[count].index = if c0 <= c1 { p } else { c0 | SAINT_MIN };
            count += 1;
        }
    }
    count as FastSint
}

/// Internal helper: final bwt aux scan right to left 8u block prepare.
#[doc(hidden)]
pub fn final_bwt_aux_scan_right_to_left_8u_block_prepare(
    t: &[u8],
    sa: &mut [SaSint],
    k: SaSint,
    buckets: &mut [SaSint],
    cache: &mut [ThreadCache],
    omp_block_start: FastSint,
    omp_block_size: FastSint,
) -> FastSint {
    if omp_block_size <= 0 {
        return 0;
    }
    let k_usize = usize::try_from(k).expect("k must be non-negative");
    buckets[..k_usize].fill(0);
    let start = usize::try_from(omp_block_start).expect("omp_block_start must be non-negative");
    let size = usize::try_from(omp_block_size).expect("omp_block_size must be non-negative");
    let mut count = 0usize;
    let mut i = start + size;
    while i > start {
        i -= 1;
        let mut p = sa[i];
        sa[i] = p & SAINT_MAX;
        if p > 0 {
            p -= 1;
            let p_usize = usize::try_from(p).expect("suffix index must be non-negative");
            let c0 = t[p_usize - usize::from(p > 0)] as SaSint;
            let c1 = t[p_usize] as SaSint;
            sa[i] = c1;
            buckets[c1 as usize] += 1;
            cache[count].symbol = c1;
            cache[count].index = if c0 <= c1 { p } else { c0 | SAINT_MIN };
            cache[count + 1].index = p;
            count += 2;
        }
    }
    count as FastSint
}

/// Internal helper: final sorting scan right to left 8u block prepare.
#[doc(hidden)]
pub fn final_sorting_scan_right_to_left_8u_block_prepare(
    t: &[u8],
    sa: &mut [SaSint],
    k: SaSint,
    buckets: &mut [SaSint],
    cache: &mut [ThreadCache],
    omp_block_start: FastSint,
    omp_block_size: FastSint,
) -> FastSint {
    if omp_block_size <= 0 {
        return 0;
    }

    let k_usize = usize::try_from(k).expect("k must be non-negative");
    buckets[..k_usize].fill(0);

    let start =
        usize::try_from(omp_block_start).expect("omp_block_start must be non-negative") as FastSint;
    let mut i = omp_block_start + omp_block_size - 1;
    let mut j = start + 1;
    let mut count = 0usize;

    while i >= j {
        let i0 = usize::try_from(i).expect("loop index must be non-negative");
        let i1 = usize::try_from(i - 1).expect("loop index must be non-negative");

        let mut p0 = sa[i0];
        sa[i0] = p0 & SAINT_MAX;
        if p0 > 0 {
            p0 -= 1;
            let p0_usize = usize::try_from(p0).expect("suffix index must be non-negative");
            let c0 = t[p0_usize] as SaSint;
            buckets[c0 as usize] += 1;
            cache[count].symbol = c0;
            cache[count].index = p0
                | ((usize::from(t[p0_usize - usize::from(p0 > 0)] > t[p0_usize]) as SaSint)
                    << (SAINT_BIT - 1));
            count += 1;
        }

        let mut p1 = sa[i1];
        sa[i1] = p1 & SAINT_MAX;
        if p1 > 0 {
            p1 -= 1;
            let p1_usize = usize::try_from(p1).expect("suffix index must be non-negative");
            let c1 = t[p1_usize] as SaSint;
            buckets[c1 as usize] += 1;
            cache[count].symbol = c1;
            cache[count].index = p1
                | ((usize::from(t[p1_usize - usize::from(p1 > 0)] > t[p1_usize]) as SaSint)
                    << (SAINT_BIT - 1));
            count += 1;
        }

        i -= 2;
    }

    j -= 1;
    while i >= j {
        let idx = usize::try_from(i).expect("loop index must be non-negative");
        let mut p = sa[idx];
        sa[idx] = p & SAINT_MAX;
        if p > 0 {
            p -= 1;
            let p_usize = usize::try_from(p).expect("suffix index must be non-negative");
            let c = t[p_usize] as SaSint;
            buckets[c as usize] += 1;
            cache[count].symbol = c;
            cache[count].index = p
                | ((usize::from(t[p_usize - usize::from(p > 0)] > t[p_usize]) as SaSint)
                    << (SAINT_BIT - 1));
            count += 1;
        }

        i -= 1;
    }

    count as FastSint
}

/// Internal helper: final order scan right to left 8u block place.
#[doc(hidden)]
pub fn final_order_scan_right_to_left_8u_block_place(
    sa: &mut [SaSint],
    buckets: &mut [SaSint],
    cache: &[ThreadCache],
    count: FastSint,
) {
    if count <= 0 {
        return;
    }
    let count_usize = usize::try_from(count).expect("count must be non-negative");
    for entry in &cache[..count_usize] {
        let symbol = usize::try_from(entry.symbol).expect("cache symbol must be non-negative");
        buckets[symbol] -= 1;
        let slot = usize::try_from(buckets[symbol]).expect("bucket slot must be non-negative");
        sa[slot] = entry.index;
    }
}

/// Internal helper: final gsa scan right to left 8u block place.
#[doc(hidden)]
pub fn final_gsa_scan_right_to_left_8u_block_place(
    sa: &mut [SaSint],
    buckets: &mut [SaSint],
    cache: &[ThreadCache],
    count: FastSint,
) {
    if count <= 0 {
        return;
    }
    let count_usize = usize::try_from(count).expect("count must be non-negative");
    for entry in &cache[..count_usize] {
        if entry.symbol > 0 {
            let symbol = usize::try_from(entry.symbol).expect("cache symbol must be non-negative");
            buckets[symbol] -= 1;
            let slot = usize::try_from(buckets[symbol]).expect("bucket slot must be non-negative");
            sa[slot] = entry.index;
        }
    }
}

/// Internal helper: final bwt aux scan right to left 8u block place.
#[doc(hidden)]
pub fn final_bwt_aux_scan_right_to_left_8u_block_place(
    sa: &mut [SaSint],
    rm: SaSint,
    i_out: &mut [SaSint],
    buckets: &mut [SaSint],
    cache: &[ThreadCache],
    count: FastSint,
) {
    if count <= 0 {
        return;
    }
    let count_usize = usize::try_from(count).expect("count must be non-negative");
    let mut i = 0usize;
    while i < count_usize {
        let symbol = usize::try_from(cache[i].symbol).expect("cache symbol must be non-negative");
        buckets[symbol] -= 1;
        let slot = usize::try_from(buckets[symbol]).expect("bucket slot must be non-negative");
        sa[slot] = cache[i].index;
        if (cache[i + 1].index & rm) == 0 {
            let sample_index = usize::try_from((cache[i + 1].index & SAINT_MAX) / (rm + 1))
                .expect("sample index must be non-negative");
            i_out[sample_index] = buckets[symbol] + 1;
        }
        i += 2;
    }
}

/// Internal helper: final sorting scan right to left 32s block gather.
#[doc(hidden)]
pub fn final_sorting_scan_right_to_left_32s_block_gather(
    t: &[SaSint],
    sa: &mut [SaSint],
    cache: &mut [ThreadCache],
    omp_block_start: FastSint,
    omp_block_size: FastSint,
) {
    if omp_block_size <= 0 {
        return;
    }
    let start = usize::try_from(omp_block_start).expect("omp_block_start must be non-negative");
    let size = usize::try_from(omp_block_size).expect("omp_block_size must be non-negative");
    for offset in 0..size {
        let i = start + offset;
        let mut symbol = SAINT_MIN;
        let mut p = sa[i];
        sa[i] = p & SAINT_MAX;
        if p > 0 {
            p -= 1;
            let p_usize = p as usize;
            cache[offset].index = p
                | ((usize::from(t[p_usize - usize::from(p > 0)] > t[p_usize]) as SaSint)
                    << (SAINT_BIT - 1));
            symbol = t[p_usize];
        }
        cache[offset].symbol = symbol;
    }
}

/// Internal helper: final sorting scan right to left 32s block sort.
#[doc(hidden)]
pub fn final_sorting_scan_right_to_left_32s_block_sort(
    t: &[SaSint],
    induction_bucket: &mut [SaSint],
    cache: &mut [ThreadCache],
    omp_block_start: FastSint,
    omp_block_size: FastSint,
) {
    if omp_block_size <= 0 {
        return;
    }
    let prefetch_distance = 64usize;
    let start = omp_block_start as usize;
    let mut i = start + omp_block_size as usize - 1;
    let mut j = start + prefetch_distance + 1;

    while i >= j {
        let ci = i - start;
        let v0 = cache[ci].symbol;
        if v0 >= 0 {
            let bucket_index0 = v0 as usize;
            induction_bucket[bucket_index0] -= 1;
            cache[ci].symbol = induction_bucket[bucket_index0];
            if cache[ci].symbol >= omp_block_start as SaSint {
                let ni = cache[ci].symbol as usize;
                let cni = ni - start;
                let mut np = cache[ci].index;
                cache[ci].index = np & SAINT_MAX;
                if np > 0 {
                    np -= 1;
                    let np_usize = np as usize;
                    cache[cni].index = np
                        | ((usize::from(t[np_usize - usize::from(np > 0)] > t[np_usize])
                            as SaSint)
                            << (SAINT_BIT - 1));
                    cache[cni].symbol = t[np_usize];
                }
            }
        }

        let i1 = i - 1;
        let ci1 = i1 - start;
        let v1 = cache[ci1].symbol;
        if v1 >= 0 {
            let bucket_index1 = v1 as usize;
            induction_bucket[bucket_index1] -= 1;
            cache[ci1].symbol = induction_bucket[bucket_index1];
            if cache[ci1].symbol >= omp_block_start as SaSint {
                let ni = cache[ci1].symbol as usize;
                let cni = ni - start;
                let mut np = cache[ci1].index;
                cache[ci1].index = np & SAINT_MAX;
                if np > 0 {
                    np -= 1;
                    let np_usize = np as usize;
                    cache[cni].index = np
                        | ((usize::from(t[np_usize - usize::from(np > 0)] > t[np_usize])
                            as SaSint)
                            << (SAINT_BIT - 1));
                    cache[cni].symbol = t[np_usize];
                }
            }
        }

        i -= 2;
    }

    j -= prefetch_distance + 1;
    while i >= j {
        let ci = i - start;
        let v = cache[ci].symbol;
        if v >= 0 {
            let bucket_index = v as usize;
            induction_bucket[bucket_index] -= 1;
            cache[ci].symbol = induction_bucket[bucket_index];
            if cache[ci].symbol >= omp_block_start as SaSint {
                let ni = cache[ci].symbol as usize;
                let cni = ni - start;
                let mut np = cache[ci].index;
                cache[ci].index = np & SAINT_MAX;
                if np > 0 {
                    np -= 1;
                    let np_usize = np as usize;
                    cache[cni].index = np
                        | ((usize::from(t[np_usize - usize::from(np > 0)] > t[np_usize])
                            as SaSint)
                            << (SAINT_BIT - 1));
                    cache[cni].symbol = t[np_usize];
                }
            }
        }

        if i == 0 {
            break;
        }
        i -= 1;
    }
}

/// Internal helper: final bwt scan right to left 8u block (OpenMP variant).
#[doc(hidden)]
pub fn final_bwt_scan_right_to_left_8u_block_omp(
    t: &[u8],
    sa: &mut [SaSint],
    k: SaSint,
    induction_bucket: &mut [SaSint],
    block_start: FastSint,
    block_size: FastSint,
    threads: SaSint,
    thread_state: &mut [ThreadState],
) {
    if block_size <= 0 {
        return;
    }
    let k_usize = usize::try_from(k).expect("k must be non-negative");
    let block_size_usize = usize::try_from(block_size).expect("block_size must be non-negative");
    let threads_usize = usize::try_from(threads.max(1)).expect("threads must be positive");
    let omp_num_threads = threads_usize.min(thread_state.len()).min(block_size_usize);
    if omp_num_threads <= 1 || block_size < 64 * k.max(256) as FastSint {
        let _ = final_bwt_scan_right_to_left_8u(t, sa, induction_bucket, block_start, block_size);
        return;
    }

    let omp_block_stride = (block_size_usize / omp_num_threads) & !15usize;
    for (omp_thread_num, state) in thread_state.iter_mut().take(omp_num_threads).enumerate() {
        let omp_block_start = omp_thread_num * omp_block_stride;
        let omp_block_size = if omp_thread_num + 1 < omp_num_threads {
            omp_block_stride
        } else {
            block_size_usize - omp_block_start
        };
        state.count = final_bwt_scan_right_to_left_8u_block_prepare(
            t,
            sa,
            k,
            &mut state.buckets,
            &mut state.cache,
            block_start + omp_block_start as FastSint,
            omp_block_size as FastSint,
        );
    }
    for state in thread_state.iter_mut().take(omp_num_threads).rev() {
        for c in 0..k_usize {
            let a = induction_bucket[c];
            let b = state.buckets[c];
            induction_bucket[c] = a - b;
            state.buckets[c] = a;
        }
    }
    for state in thread_state.iter_mut().take(omp_num_threads) {
        final_order_scan_right_to_left_8u_block_place(
            sa,
            &mut state.buckets,
            &state.cache,
            state.count,
        );
    }
}

/// Internal helper: final bwt aux scan right to left 8u block (OpenMP variant).
#[doc(hidden)]
pub fn final_bwt_aux_scan_right_to_left_8u_block_omp(
    t: &[u8],
    sa: &mut [SaSint],
    k: SaSint,
    rm: SaSint,
    i_out: &mut [SaSint],
    induction_bucket: &mut [SaSint],
    block_start: FastSint,
    block_size: FastSint,
    threads: SaSint,
    thread_state: &mut [ThreadState],
) {
    if block_size <= 0 {
        return;
    }
    let k_usize = usize::try_from(k).expect("k must be non-negative");
    let block_size_usize = usize::try_from(block_size).expect("block_size must be non-negative");
    let threads_usize = usize::try_from(threads.max(1)).expect("threads must be positive");
    let omp_num_threads = threads_usize.min(thread_state.len()).min(block_size_usize);
    if omp_num_threads <= 1 || block_size < 64 * k.max(256) as FastSint {
        final_bwt_aux_scan_right_to_left_8u(
            t,
            sa,
            rm,
            i_out,
            induction_bucket,
            block_start,
            block_size,
        );
        return;
    }

    let omp_block_stride = (block_size_usize / omp_num_threads) & !15usize;
    for (omp_thread_num, state) in thread_state.iter_mut().take(omp_num_threads).enumerate() {
        let omp_block_start = omp_thread_num * omp_block_stride;
        let omp_block_size = if omp_thread_num + 1 < omp_num_threads {
            omp_block_stride
        } else {
            block_size_usize - omp_block_start
        };
        state.count = final_bwt_aux_scan_right_to_left_8u_block_prepare(
            t,
            sa,
            k,
            &mut state.buckets,
            &mut state.cache,
            block_start + omp_block_start as FastSint,
            omp_block_size as FastSint,
        );
    }
    for state in thread_state.iter_mut().take(omp_num_threads).rev() {
        for c in 0..k_usize {
            let a = induction_bucket[c];
            let b = state.buckets[c];
            induction_bucket[c] = a - b;
            state.buckets[c] = a;
        }
    }
    for state in thread_state.iter_mut().take(omp_num_threads) {
        final_bwt_aux_scan_right_to_left_8u_block_place(
            sa,
            rm,
            i_out,
            &mut state.buckets,
            &state.cache,
            state.count,
        );
    }
}

/// Internal helper: final sorting scan right to left 8u block (OpenMP variant).
#[doc(hidden)]
pub fn final_sorting_scan_right_to_left_8u_block_omp(
    t: &[u8],
    sa: &mut [SaSint],
    k: SaSint,
    induction_bucket: &mut [SaSint],
    block_start: FastSint,
    block_size: FastSint,
    threads: SaSint,
    thread_state: &mut [ThreadState],
) {
    if block_size <= 0 {
        return;
    }
    let k_usize = usize::try_from(k).expect("k must be non-negative");
    let block_size_usize = usize::try_from(block_size).expect("block_size must be non-negative");
    let threads_usize = usize::try_from(threads.max(1)).expect("threads must be positive");
    let omp_num_threads = threads_usize.min(thread_state.len()).min(block_size_usize);
    if omp_num_threads <= 1 || block_size < 64 * k.max(256) as FastSint {
        final_sorting_scan_right_to_left_8u(t, sa, induction_bucket, block_start, block_size);
        return;
    }

    let omp_block_stride = (block_size_usize / omp_num_threads) & !15usize;
    for (omp_thread_num, state) in thread_state.iter_mut().take(omp_num_threads).enumerate() {
        let omp_block_start = omp_thread_num * omp_block_stride;
        let omp_block_size = if omp_thread_num + 1 < omp_num_threads {
            omp_block_stride
        } else {
            block_size_usize - omp_block_start
        };
        state.count = final_sorting_scan_right_to_left_8u_block_prepare(
            t,
            sa,
            k,
            &mut state.buckets,
            &mut state.cache,
            block_start + omp_block_start as FastSint,
            omp_block_size as FastSint,
        );
    }
    for state in thread_state.iter_mut().take(omp_num_threads).rev() {
        for c in 0..k_usize {
            let a = induction_bucket[c];
            let b = state.buckets[c];
            induction_bucket[c] = a - b;
            state.buckets[c] = a;
        }
    }
    for state in thread_state.iter_mut().take(omp_num_threads) {
        final_order_scan_right_to_left_8u_block_place(
            sa,
            &mut state.buckets,
            &state.cache,
            state.count,
        );
    }
}

/// Internal helper: final gsa scan right to left 8u block (OpenMP variant).
#[doc(hidden)]
pub fn final_gsa_scan_right_to_left_8u_block_omp(
    t: &[u8],
    sa: &mut [SaSint],
    k: SaSint,
    induction_bucket: &mut [SaSint],
    block_start: FastSint,
    block_size: FastSint,
    threads: SaSint,
    thread_state: &mut [ThreadState],
) {
    if block_size <= 0 {
        return;
    }
    let k_usize = usize::try_from(k).expect("k must be non-negative");
    let block_size_usize = usize::try_from(block_size).expect("block_size must be non-negative");
    let threads_usize = usize::try_from(threads.max(1)).expect("threads must be positive");
    let omp_num_threads = threads_usize.min(thread_state.len()).min(block_size_usize);
    if omp_num_threads <= 1 || block_size < 64 * k.max(256) as FastSint {
        final_gsa_scan_right_to_left_8u(t, sa, induction_bucket, block_start, block_size);
        return;
    }

    let omp_block_stride = (block_size_usize / omp_num_threads) & !15usize;
    for (omp_thread_num, state) in thread_state.iter_mut().take(omp_num_threads).enumerate() {
        let omp_block_start = omp_thread_num * omp_block_stride;
        let omp_block_size = if omp_thread_num + 1 < omp_num_threads {
            omp_block_stride
        } else {
            block_size_usize - omp_block_start
        };
        state.count = final_sorting_scan_right_to_left_8u_block_prepare(
            t,
            sa,
            k,
            &mut state.buckets,
            &mut state.cache,
            block_start + omp_block_start as FastSint,
            omp_block_size as FastSint,
        );
    }
    for state in thread_state.iter_mut().take(omp_num_threads).rev() {
        for c in 0..k_usize {
            let a = induction_bucket[c];
            let b = state.buckets[c];
            induction_bucket[c] = a - b;
            state.buckets[c] = a;
        }
    }
    for state in thread_state.iter_mut().take(omp_num_threads) {
        final_gsa_scan_right_to_left_8u_block_place(
            sa,
            &mut state.buckets,
            &state.cache,
            state.count,
        );
    }
}

/// Internal helper: final sorting scan right to left 32s block (OpenMP variant).
#[doc(hidden)]
pub fn final_sorting_scan_right_to_left_32s_block_omp(
    t: &[SaSint],
    sa: &mut [SaSint],
    buckets: &mut [SaSint],
    cache: &mut [ThreadCache],
    block_start: FastSint,
    block_size: FastSint,
    threads: SaSint,
) {
    if threads <= 1 || block_size < 16_384 {
        final_sorting_scan_right_to_left_32s(t, sa, buckets, block_start, block_size);
        return;
    }

    final_sorting_scan_right_to_left_32s_block_gather(t, sa, cache, block_start, block_size);
    final_sorting_scan_right_to_left_32s_block_sort(t, buckets, cache, block_start, block_size);
    let block_size_usize = usize::try_from(block_size).expect("block_size must be non-negative");
    let threads_usize = usize::try_from(threads.max(1)).expect("threads must be positive");
    let omp_num_threads = threads_usize.min(block_size_usize);
    let omp_block_stride = (block_size_usize / omp_num_threads) & !15usize;
    {
        let sa_ptr = SyncMutPtr::new(sa);
        let cache_ptr = SyncMutPtr::new(cache);
        run_rayon_with_threads(omp_num_threads, || {
            (0..omp_num_threads)
                .into_par_iter()
                .for_each(|omp_thread_num| {
                    let omp_block_start = omp_thread_num * omp_block_stride;
                    let omp_block_size = if omp_thread_num + 1 < omp_num_threads {
                        omp_block_stride
                    } else {
                        block_size_usize - omp_block_start
                    };
                    let sa = unsafe { sa_ptr.as_slice() };
                    let cache = unsafe { cache_ptr.as_slice() };
                    compact_and_place_cached_suffixes(
                        sa,
                        cache,
                        omp_block_start as FastSint,
                        omp_block_size as FastSint,
                    );
                });
        });
    }
}

/// Internal helper: final bwt scan right to left 8u (OpenMP variant).
#[doc(hidden)]
pub fn final_bwt_scan_right_to_left_8u_omp(
    t: &[u8],
    sa: &mut [SaSint],
    n: SaSint,
    k: SaSint,
    induction_bucket: &mut [SaSint],
    threads: SaSint,
    thread_state: &mut [ThreadState],
) -> SaSint {
    if threads == 1 || n < 65_536 {
        return final_bwt_scan_right_to_left_8u(t, sa, induction_bucket, 0, n as FastSint);
    }
    let mut index = -1;
    let mut block_start = usize::try_from(n).expect("n must be non-negative");
    while block_start > 0 {
        block_start -= 1;
        if sa[block_start] == 0 {
            index = block_start as SaSint;
        } else {
            let threads_usize = usize::try_from(threads)
                .expect("threads must be non-negative")
                .min(thread_state.len())
                .max(1);
            let max_back =
                threads_usize * LIBSAIS_PER_THREAD_CACHE_SIZE.saturating_sub(16 * threads_usize);
            let block_max_end = block_start.saturating_sub(max_back);
            let mut block_end = block_start;
            while block_end > block_max_end && sa[block_end - 1] != 0 {
                block_end -= 1;
            }
            let size = block_start - block_end + 1;
            if size < 32 {
                let res = final_bwt_scan_right_to_left_8u(
                    t,
                    sa,
                    induction_bucket,
                    block_end as FastSint,
                    size as FastSint,
                );
                if res >= 0 {
                    index = res;
                }
            } else {
                final_bwt_scan_right_to_left_8u_block_omp(
                    t,
                    sa,
                    k,
                    induction_bucket,
                    block_end as FastSint,
                    size as FastSint,
                    threads,
                    thread_state,
                );
            }
            block_start = block_end;
        }
    }
    index
}

/// Internal helper: final bwt aux scan right to left 8u (OpenMP variant).
#[doc(hidden)]
pub fn final_bwt_aux_scan_right_to_left_8u_omp(
    t: &[u8],
    sa: &mut [SaSint],
    n: SaSint,
    k: SaSint,
    rm: SaSint,
    i_out: &mut [SaSint],
    induction_bucket: &mut [SaSint],
    threads: SaSint,
    thread_state: &mut [ThreadState],
) {
    if threads == 1 || n < 65_536 {
        final_bwt_aux_scan_right_to_left_8u(t, sa, rm, i_out, induction_bucket, 0, n as FastSint);
        return;
    }
    let mut block_start = usize::try_from(n).expect("n must be non-negative");
    while block_start > 0 {
        block_start -= 1;
        if sa[block_start] != 0 {
            let threads_usize = usize::try_from(threads)
                .expect("threads must be non-negative")
                .min(thread_state.len())
                .max(1);
            let max_back = threads_usize
                * (LIBSAIS_PER_THREAD_CACHE_SIZE.saturating_sub(16 * threads_usize) / 2);
            let block_max_end = block_start.saturating_sub(max_back);
            let mut block_end = block_start;
            while block_end > block_max_end && sa[block_end - 1] != 0 {
                block_end -= 1;
            }
            let size = block_start - block_end + 1;
            if size < 32 {
                final_bwt_aux_scan_right_to_left_8u(
                    t,
                    sa,
                    rm,
                    i_out,
                    induction_bucket,
                    block_end as FastSint,
                    size as FastSint,
                );
            } else {
                final_bwt_aux_scan_right_to_left_8u_block_omp(
                    t,
                    sa,
                    k,
                    rm,
                    i_out,
                    induction_bucket,
                    block_end as FastSint,
                    size as FastSint,
                    threads,
                    thread_state,
                );
            }
            block_start = block_end;
        }
    }
}

/// Internal helper: final sorting scan right to left 8u (OpenMP variant).
#[doc(hidden)]
pub fn final_sorting_scan_right_to_left_8u_omp(
    t: &[u8],
    sa: &mut [SaSint],
    omp_block_start: FastSint,
    omp_block_size: FastSint,
    k: SaSint,
    induction_bucket: &mut [SaSint],
    threads: SaSint,
    thread_state: &mut [ThreadState],
) {
    if threads == 1 || omp_block_size < 65_536 {
        final_sorting_scan_right_to_left_8u(
            t,
            sa,
            induction_bucket,
            omp_block_start,
            omp_block_size,
        );
        return;
    }
    let start = usize::try_from(omp_block_start).expect("omp_block_start must be non-negative");
    let size = usize::try_from(omp_block_size).expect("omp_block_size must be non-negative");
    let mut block_start = start + size;
    while block_start > start {
        block_start -= 1;
        if sa[block_start] != 0 {
            let threads_usize = usize::try_from(threads)
                .expect("threads must be non-negative")
                .min(thread_state.len())
                .max(1);
            let max_back =
                threads_usize * LIBSAIS_PER_THREAD_CACHE_SIZE.saturating_sub(16 * threads_usize);
            let block_max_end = block_start.saturating_sub(max_back).max(start);
            let mut block_end = block_start;
            while block_end > block_max_end && sa[block_end - 1] != 0 {
                block_end -= 1;
            }
            let span = block_start - block_end + 1;
            if span < 32 {
                final_sorting_scan_right_to_left_8u(
                    t,
                    sa,
                    induction_bucket,
                    block_end as FastSint,
                    span as FastSint,
                );
            } else {
                final_sorting_scan_right_to_left_8u_block_omp(
                    t,
                    sa,
                    k,
                    induction_bucket,
                    block_end as FastSint,
                    span as FastSint,
                    threads,
                    thread_state,
                );
            }
            block_start = block_end;
        }
    }
}

/// Internal helper: final gsa scan right to left 8u (OpenMP variant).
#[doc(hidden)]
pub fn final_gsa_scan_right_to_left_8u_omp(
    t: &[u8],
    sa: &mut [SaSint],
    omp_block_start: FastSint,
    omp_block_size: FastSint,
    k: SaSint,
    induction_bucket: &mut [SaSint],
    threads: SaSint,
    thread_state: &mut [ThreadState],
) {
    if threads == 1 || omp_block_size < 65_536 {
        final_gsa_scan_right_to_left_8u(t, sa, induction_bucket, omp_block_start, omp_block_size);
        return;
    }
    let start = usize::try_from(omp_block_start).expect("omp_block_start must be non-negative");
    let size = usize::try_from(omp_block_size).expect("omp_block_size must be non-negative");
    let mut block_start = start + size;
    while block_start > start {
        block_start -= 1;
        if sa[block_start] != 0 {
            let threads_usize = usize::try_from(threads)
                .expect("threads must be non-negative")
                .min(thread_state.len())
                .max(1);
            let max_back =
                threads_usize * LIBSAIS_PER_THREAD_CACHE_SIZE.saturating_sub(16 * threads_usize);
            let block_max_end = block_start.saturating_sub(max_back).max(start);
            let mut block_end = block_start;
            while block_end > block_max_end && sa[block_end - 1] != 0 {
                block_end -= 1;
            }
            let span = block_start - block_end + 1;
            if span < 32 {
                final_gsa_scan_right_to_left_8u(
                    t,
                    sa,
                    induction_bucket,
                    block_end as FastSint,
                    span as FastSint,
                );
            } else {
                final_gsa_scan_right_to_left_8u_block_omp(
                    t,
                    sa,
                    k,
                    induction_bucket,
                    block_end as FastSint,
                    span as FastSint,
                    threads,
                    thread_state,
                );
            }
            block_start = block_end;
        }
    }
}

/// Internal helper: final sorting scan right to left 32s (OpenMP variant).
#[doc(hidden)]
pub fn final_sorting_scan_right_to_left_32s_omp(
    t: &[SaSint],
    sa: &mut [SaSint],
    n: SaSint,
    induction_bucket: &mut [SaSint],
    threads: SaSint,
    thread_state: &mut [ThreadState],
) {
    if threads == 1 || n < 65_536 {
        final_sorting_scan_right_to_left_32s(t, sa, induction_bucket, 0, n as FastSint);
        return;
    }
    if thread_state.is_empty() {
        final_sorting_scan_right_to_left_32s(t, sa, induction_bucket, 0, n as FastSint);
        return;
    }
    let threads_usize = usize::try_from(threads)
        .expect("threads must be non-negative")
        .max(1);
    let mut cache = vec![ThreadCache::default(); threads_usize * LIBSAIS_PER_THREAD_CACHE_SIZE];
    let mut block_start = isize::try_from(n).expect("n must fit isize") - 1;
    while block_start >= 0 {
        let block_end = (block_start
            - isize::try_from(threads_usize * LIBSAIS_PER_THREAD_CACHE_SIZE)
                .expect("block span must fit isize"))
        .max(-1);
        final_sorting_scan_right_to_left_32s_block_omp(
            t,
            sa,
            induction_bucket,
            &mut cache,
            (block_end + 1) as FastSint,
            (block_start - block_end) as FastSint,
            threads,
        );
        block_start = block_end;
    }
}

/// Internal helper: clear lms suffixes (OpenMP variant).
#[doc(hidden)]
pub fn clear_lms_suffixes_omp(
    sa: &mut [SaSint],
    n: SaSint,
    k: SaSint,
    bucket_start: &[SaSint],
    bucket_end: &[SaSint],
    threads: SaSint,
) {
    let k_usize = usize::try_from(k).expect("k must be non-negative");
    let thread_count = if threads > 1 && n >= 65536 {
        usize::try_from(threads).expect("threads must be positive")
    } else {
        1
    };
    {
        let sa_ptr = SyncMutPtr::new(sa);
        let bucket_start_ref: &[SaSint] = bucket_start;
        let bucket_end_ref: &[SaSint] = bucket_end;
        run_rayon_with_threads(thread_count, || {
            (0..thread_count).into_par_iter().for_each(|t| {
                let mut c = t;
                let sa = unsafe { sa_ptr.as_slice() };
                while c < k_usize {
                    if bucket_end_ref[c] > bucket_start_ref[c] {
                        let start = usize::try_from(bucket_start_ref[c])
                            .expect("bucket start must be non-negative");
                        let end = usize::try_from(bucket_end_ref[c])
                            .expect("bucket end must be non-negative");
                        sa[start..end].fill(0);
                    }
                    c += thread_count;
                }
            });
        });
    }
}

/// Internal helper: induce final order 8u (OpenMP variant).
#[doc(hidden)]
pub fn induce_final_order_8u_omp(
    t: &[u8],
    sa: &mut [SaSint],
    n: SaSint,
    k: SaSint,
    flags: SaSint,
    r: SaSint,
    i_out: Option<&mut [SaSint]>,
    buckets: &mut [SaSint],
    threads: SaSint,
    thread_state: &mut [ThreadState],
) -> SaSint {
    if (flags & LIBSAIS_FLAGS_BWT) == 0 {
        if (flags & LIBSAIS_FLAGS_GSA) != 0 {
            buckets[6 * ALPHABET_SIZE] = buckets[7 * ALPHABET_SIZE] - 1;
        }

        let (left_buckets, right_tail) = buckets.split_at_mut(7 * ALPHABET_SIZE);
        let bucket_start = &mut left_buckets[6 * ALPHABET_SIZE..7 * ALPHABET_SIZE];
        let bucket_end = &mut right_tail[..ALPHABET_SIZE];

        final_sorting_scan_left_to_right_8u_omp(
            t,
            sa,
            n as FastSint,
            k,
            bucket_start,
            threads,
            thread_state,
        );
        if threads > 1 && n >= 65_536 {
            clear_lms_suffixes_omp(
                sa,
                n,
                ALPHABET_SIZE as SaSint,
                bucket_start,
                bucket_end,
                threads,
            );
        }

        if (flags & LIBSAIS_FLAGS_GSA) != 0 {
            flip_suffix_markers_omp(sa, bucket_end[0], threads);
            final_gsa_scan_right_to_left_8u_omp(
                t,
                sa,
                bucket_end[0] as FastSint,
                n as FastSint - bucket_end[0] as FastSint,
                k,
                bucket_end,
                threads,
                thread_state,
            );
        } else {
            final_sorting_scan_right_to_left_8u_omp(
                t,
                sa,
                0,
                n as FastSint,
                k,
                bucket_end,
                threads,
                thread_state,
            );
        }

        0
    } else if let Some(i_out) = i_out {
        let (left_buckets, right_tail) = buckets.split_at_mut(7 * ALPHABET_SIZE);
        let bucket_start = &mut left_buckets[6 * ALPHABET_SIZE..7 * ALPHABET_SIZE];
        let bucket_end = &mut right_tail[..ALPHABET_SIZE];

        final_bwt_aux_scan_left_to_right_8u_omp(
            t,
            sa,
            n as FastSint,
            k,
            r - 1,
            i_out,
            bucket_start,
            threads,
            thread_state,
        );
        if threads > 1 && n >= 65_536 {
            clear_lms_suffixes_omp(
                sa,
                n,
                ALPHABET_SIZE as SaSint,
                bucket_start,
                bucket_end,
                threads,
            );
        }
        final_bwt_aux_scan_right_to_left_8u_omp(
            t,
            sa,
            n,
            k,
            r - 1,
            i_out,
            bucket_end,
            threads,
            thread_state,
        );
        0
    } else {
        let (left_buckets, right_tail) = buckets.split_at_mut(7 * ALPHABET_SIZE);
        let bucket_start = &mut left_buckets[6 * ALPHABET_SIZE..7 * ALPHABET_SIZE];
        let bucket_end = &mut right_tail[..ALPHABET_SIZE];

        final_bwt_scan_left_to_right_8u_omp(
            t,
            sa,
            n as FastSint,
            k,
            bucket_start,
            threads,
            thread_state,
        );
        if threads > 1 && n >= 65_536 {
            clear_lms_suffixes_omp(
                sa,
                n,
                ALPHABET_SIZE as SaSint,
                bucket_start,
                bucket_end,
                threads,
            );
        }
        final_bwt_scan_right_to_left_8u_omp(t, sa, n, k, bucket_end, threads, thread_state)
    }
}

/// Internal helper: induce final order 32s 6k.
#[doc(hidden)]
pub fn induce_final_order_32s_6k(
    t: &[SaSint],
    sa: &mut [SaSint],
    n: SaSint,
    k: SaSint,
    buckets: &mut [SaSint],
    threads: SaSint,
    thread_state: &mut [ThreadState],
) {
    let k_usize = usize::try_from(k).expect("k must be non-negative");
    let (_head, tail) = buckets.split_at_mut(4 * k_usize);
    let (left, right) = tail.split_at_mut(k_usize);
    final_sorting_scan_left_to_right_32s_omp(t, sa, n, left, threads, thread_state);
    final_sorting_scan_right_to_left_32s_omp(t, sa, n, right, threads, thread_state);
}

/// Internal helper: induce final order 32s 4k.
#[doc(hidden)]
pub fn induce_final_order_32s_4k(
    t: &[SaSint],
    sa: &mut [SaSint],
    n: SaSint,
    k: SaSint,
    buckets: &mut [SaSint],
    threads: SaSint,
    thread_state: &mut [ThreadState],
) {
    let k_usize = usize::try_from(k).expect("k must be non-negative");
    let (_head, tail) = buckets.split_at_mut(2 * k_usize);
    let (left, right) = tail.split_at_mut(k_usize);
    final_sorting_scan_left_to_right_32s_omp(t, sa, n, left, threads, thread_state);
    final_sorting_scan_right_to_left_32s_omp(t, sa, n, right, threads, thread_state);
}

/// Internal helper: induce final order 32s 2k.
#[doc(hidden)]
pub fn induce_final_order_32s_2k(
    t: &[SaSint],
    sa: &mut [SaSint],
    n: SaSint,
    k: SaSint,
    buckets: &mut [SaSint],
    threads: SaSint,
    thread_state: &mut [ThreadState],
) {
    let k_usize = usize::try_from(k).expect("k must be non-negative");
    let (right, left) = buckets.split_at_mut(k_usize);
    final_sorting_scan_left_to_right_32s_omp(t, sa, n, left, threads, thread_state);
    final_sorting_scan_right_to_left_32s_omp(t, sa, n, right, threads, thread_state);
}

/// Internal helper: induce final order 32s 1k.
#[doc(hidden)]
pub fn induce_final_order_32s_1k(
    t: &[SaSint],
    sa: &mut [SaSint],
    n: SaSint,
    k: SaSint,
    buckets: &mut [SaSint],
    threads: SaSint,
    thread_state: &mut [ThreadState],
) {
    count_suffixes_32s(t, n, k, buckets);
    initialize_buckets_start_32s_1k(k, buckets);
    final_sorting_scan_left_to_right_32s_omp(t, sa, n, buckets, threads, thread_state);

    count_suffixes_32s(t, n, k, buckets);
    initialize_buckets_end_32s_1k(k, buckets);
    final_sorting_scan_right_to_left_32s_omp(t, sa, n, buckets, threads, thread_state);
}

/// Internal helper: renumber unique and nonunique lms suffixes 32s.
#[doc(hidden)]
pub fn renumber_unique_and_nonunique_lms_suffixes_32s(
    t: &mut [SaSint],
    sa: &mut [SaSint],
    m: SaSint,
    mut f: SaSint,
    omp_block_start: FastSint,
    omp_block_size: FastSint,
) -> SaSint {
    if omp_block_size <= 0 {
        return f;
    }

    let prefetch_distance = 64 as SaSint;
    let m_usize = usize::try_from(m).expect("m must be non-negative");
    let (sa_head, sam) = sa.split_at_mut(m_usize);
    let mut i = omp_block_start as SaSint;
    let mut j = omp_block_start as SaSint + omp_block_size as SaSint - 2 * prefetch_distance - 3;

    while i < j {
        let p0 = sa_head[i as usize] as SaUint;
        let p0_half = (p0 >> 1) as usize;
        let mut s0 = sam[p0_half];
        if s0 < 0 {
            t[p0 as usize] |= SAINT_MIN;
            f += 1;
            s0 = i + SAINT_MIN + f;
        }
        sam[p0_half] = s0 - f;

        let p1 = sa_head[(i + 1) as usize] as SaUint;
        let p1_half = (p1 >> 1) as usize;
        let mut s1 = sam[p1_half];
        if s1 < 0 {
            t[p1 as usize] |= SAINT_MIN;
            f += 1;
            s1 = i + 1 + SAINT_MIN + f;
        }
        sam[p1_half] = s1 - f;

        let p2 = sa_head[(i + 2) as usize] as SaUint;
        let p2_half = (p2 >> 1) as usize;
        let mut s2 = sam[p2_half];
        if s2 < 0 {
            t[p2 as usize] |= SAINT_MIN;
            f += 1;
            s2 = i + 2 + SAINT_MIN + f;
        }
        sam[p2_half] = s2 - f;

        let p3 = sa_head[(i + 3) as usize] as SaUint;
        let p3_half = (p3 >> 1) as usize;
        let mut s3 = sam[p3_half];
        if s3 < 0 {
            t[p3 as usize] |= SAINT_MIN;
            f += 1;
            s3 = i + 3 + SAINT_MIN + f;
        }
        sam[p3_half] = s3 - f;

        i += 4;
    }

    j += 2 * prefetch_distance + 3;
    while i < j {
        let p = sa_head[i as usize] as SaUint;
        let p_half = (p >> 1) as usize;
        let mut s = sam[p_half];
        if s < 0 {
            t[p as usize] |= SAINT_MIN;
            f += 1;
            s = i + SAINT_MIN + f;
        }
        sam[p_half] = s - f;
        i += 1;
    }

    f
}

/// Internal helper: compact unique and nonunique lms suffixes 32s.
#[doc(hidden)]
pub fn compact_unique_and_nonunique_lms_suffixes_32s(
    sa: &mut [SaSint],
    m: SaSint,
    pl: &mut FastSint,
    pr: &mut FastSint,
    omp_block_start: FastSint,
    omp_block_size: FastSint,
) {
    if omp_block_size <= 0 {
        return;
    }

    let m_usize = usize::try_from(m).expect("m must be non-negative");
    let start = usize::try_from(omp_block_start).expect("omp_block_start must be non-negative");
    let size = usize::try_from(omp_block_size).expect("omp_block_size must be non-negative");

    let source: Vec<SaSint> = sa[m_usize + start..m_usize + start + size].to_vec();
    let mut l = usize::try_from(*pl - 1).expect("left position must be positive");
    let mut r = usize::try_from(*pr - 1).expect("right position must be positive");

    for &p in source.iter().rev() {
        let pu = p as SaUint;
        sa[l] = (pu & SAINT_MAX as SaUint) as SaSint;
        l = l.saturating_sub(usize::from((pu as SaSint) < 0));

        sa[r] = pu.wrapping_sub(1) as SaSint;
        r = r.saturating_sub(usize::from((pu as SaSint) > 0));
    }

    *pl = l as FastSint + 1;
    *pr = r as FastSint + 1;
}

/// Internal helper: count unique suffixes.
#[doc(hidden)]
pub fn count_unique_suffixes(
    sa: &[SaSint],
    m: SaSint,
    omp_block_start: FastSint,
    omp_block_size: FastSint,
) -> SaSint {
    if omp_block_size <= 0 {
        return 0;
    }

    let m_usize = usize::try_from(m).expect("m must be non-negative");
    let sam = &sa[m_usize..];
    let mut i = usize::try_from(omp_block_start).expect("omp_block_start must be non-negative");
    let block_end =
        i + usize::try_from(omp_block_size).expect("omp_block_size must be non-negative");
    let j = block_end.saturating_sub(67);
    let mut f0 = 0;
    let mut f1 = 0;
    let mut f2 = 0;
    let mut f3 = 0;

    while i < j {
        f0 += SaSint::from(
            sam[usize::try_from((sa[i] as SaUint) >> 1).expect("name slot must fit usize")] < 0,
        );
        f1 += SaSint::from(
            sam[usize::try_from((sa[i + 1] as SaUint) >> 1).expect("name slot must fit usize")] < 0,
        );
        f2 += SaSint::from(
            sam[usize::try_from((sa[i + 2] as SaUint) >> 1).expect("name slot must fit usize")] < 0,
        );
        f3 += SaSint::from(
            sam[usize::try_from((sa[i + 3] as SaUint) >> 1).expect("name slot must fit usize")] < 0,
        );
        i += 4;
    }

    while i < block_end {
        f0 += SaSint::from(
            sam[usize::try_from((sa[i] as SaUint) >> 1).expect("name slot must fit usize")] < 0,
        );
        i += 1;
    }

    f0 + f1 + f2 + f3
}

/// Internal helper: renumber unique and nonunique lms suffixes 32s (OpenMP variant).
#[doc(hidden)]
pub fn renumber_unique_and_nonunique_lms_suffixes_32s_omp(
    t: &mut [SaSint],
    sa: &mut [SaSint],
    m: SaSint,
    threads: SaSint,
    thread_state: &mut [ThreadState],
) -> SaSint {
    let f = if threads == 1 || m < 65_536 {
        renumber_unique_and_nonunique_lms_suffixes_32s(t, sa, m, 0, 0, m as FastSint)
    } else {
        let threads_usize = usize::try_from(threads)
            .expect("threads must be non-negative")
            .max(1);
        let m_usize = usize::try_from(m).expect("m must be non-negative");
        let omp_num_threads = threads_usize.min(m_usize.max(1));
        let omp_block_stride = (m_usize / omp_num_threads) & !15usize;

        {
            let sa_ro: &[SaSint] = sa;
            run_rayon_with_threads(omp_num_threads, || {
                thread_state[..omp_num_threads]
                    .par_iter_mut()
                    .enumerate()
                    .for_each(|(omp_thread_num, state)| {
                        let omp_block_start = omp_thread_num * omp_block_stride;
                        let omp_block_size = if omp_thread_num + 1 < omp_num_threads {
                            omp_block_stride
                        } else {
                            m_usize - omp_block_start
                        };
                        state.count = count_unique_suffixes(
                            sa_ro,
                            m,
                            omp_block_start as FastSint,
                            omp_block_size as FastSint,
                        ) as FastSint;
                    });
            });
        }

        let counts: Vec<FastSint> = thread_state[..omp_num_threads]
            .iter()
            .map(|s| s.count)
            .collect();
        let f = counts.iter().sum::<FastSint>() as SaSint;

        {
            let sa_ptr = SyncMutPtr::new(sa);
            let t_ptr = SyncMutPtr::new(t);
            let counts_ref: &[FastSint] = &counts;
            run_rayon_with_threads(omp_num_threads, || {
                (0..omp_num_threads)
                    .into_par_iter()
                    .for_each(|omp_thread_num| {
                        let omp_block_start = omp_thread_num * omp_block_stride;
                        let omp_block_size = if omp_thread_num + 1 < omp_num_threads {
                            omp_block_stride
                        } else {
                            m_usize - omp_block_start
                        };

                        let mut count: FastSint = 0;
                        for tt in 0..omp_thread_num {
                            count += counts_ref[tt];
                        }

                        let sa = unsafe { sa_ptr.as_slice() };
                        let t = unsafe { t_ptr.as_slice() };
                        renumber_unique_and_nonunique_lms_suffixes_32s(
                            t,
                            sa,
                            m,
                            count as SaSint,
                            omp_block_start as FastSint,
                            omp_block_size as FastSint,
                        );
                    });
            });
        }
        f
    };

    f
}

/// Internal helper: compact unique and nonunique lms suffixes 32s (OpenMP variant).
#[doc(hidden)]
pub fn compact_unique_and_nonunique_lms_suffixes_32s_omp(
    sa: &mut [SaSint],
    n: SaSint,
    m: SaSint,
    fs: SaSint,
    f: SaSint,
    threads: SaSint,
    thread_state: &mut [ThreadState],
) {
    let half_n = (n as FastSint) >> 1;
    if threads == 1 || n < 131_072 || m >= fs {
        let mut l = m as FastSint;
        let mut r = n as FastSint + fs as FastSint;
        compact_unique_and_nonunique_lms_suffixes_32s(sa, m, &mut l, &mut r, 0, half_n);
    } else {
        let threads_usize = usize::try_from(threads)
            .expect("threads must be non-negative")
            .max(1);
        let half_n_usize = usize::try_from(half_n).expect("half_n must be non-negative");
        let omp_num_threads = threads_usize.min(half_n_usize.max(1));
        let omp_block_stride = (half_n_usize / omp_num_threads) & !15usize;

        {
            let sa_ptr = SyncMutPtr::new(sa);
            run_rayon_with_threads(omp_num_threads, || {
                thread_state[..omp_num_threads]
                    .par_iter_mut()
                    .enumerate()
                    .for_each(|(omp_thread_num, state)| {
                        let omp_block_start = omp_thread_num * omp_block_stride;
                        let omp_block_size = if omp_thread_num + 1 < omp_num_threads {
                            omp_block_stride
                        } else {
                            half_n_usize - omp_block_start
                        };

                        let mut position = m as FastSint
                            + half_n
                            + omp_block_start as FastSint
                            + omp_block_size as FastSint;
                        let mut count = m as FastSint
                            + omp_block_start as FastSint
                            + omp_block_size as FastSint;

                        let sa = unsafe { sa_ptr.as_slice() };
                        compact_unique_and_nonunique_lms_suffixes_32s(
                            sa,
                            m,
                            &mut position,
                            &mut count,
                            omp_block_start as FastSint,
                            omp_block_size as FastSint,
                        );
                        state.position = position;
                        state.count = count;
                    });
            });
        }

        let mut position = m as FastSint;
        for t in (0..omp_num_threads).rev() {
            let omp_block_end = if t + 1 < omp_num_threads {
                omp_block_stride * (t + 1)
            } else {
                half_n_usize
            };
            let count =
                m as FastSint + half_n + omp_block_end as FastSint - thread_state[t].position;
            if count > 0 {
                position -= count;
                let dst = usize::try_from(position).expect("destination must be non-negative");
                let src =
                    usize::try_from(thread_state[t].position).expect("source must be non-negative");
                let len = usize::try_from(count).expect("length must be non-negative");
                sa.copy_within(src..src + len, dst);
            }
        }

        let mut position = n as FastSint + fs as FastSint;
        for t in (0..omp_num_threads).rev() {
            let omp_block_end = if t + 1 < omp_num_threads {
                omp_block_stride * (t + 1)
            } else {
                half_n_usize
            };
            let count = m as FastSint + omp_block_end as FastSint - thread_state[t].count;
            if count > 0 {
                position -= count;
                let dst = usize::try_from(position).expect("destination must be non-negative");
                let src =
                    usize::try_from(thread_state[t].count).expect("source must be non-negative");
                let len = usize::try_from(count).expect("length must be non-negative");
                sa.copy_within(src..src + len, dst);
            }
        }
    }

    let copy_dst = usize::try_from(n + fs - m).expect("copy destination must be non-negative");
    let copy_src = usize::try_from(m - f).expect("copy source must be non-negative");
    let copy_len = usize::try_from(f).expect("copy length must be non-negative");
    sa.copy_within(copy_src..copy_src + copy_len, copy_dst);
}

/// Internal helper: compact lms suffixes 32s (OpenMP variant).
#[doc(hidden)]
pub fn compact_lms_suffixes_32s_omp(
    t: &mut [SaSint],
    sa: &mut [SaSint],
    n: SaSint,
    m: SaSint,
    fs: SaSint,
    threads: SaSint,
    thread_state: &mut [ThreadState],
) -> SaSint {
    let f = renumber_unique_and_nonunique_lms_suffixes_32s_omp(t, sa, m, threads, thread_state);
    compact_unique_and_nonunique_lms_suffixes_32s_omp(sa, n, m, fs, f, threads, thread_state);
    f
}

/// Internal helper: merge unique lms suffixes 32s.
#[doc(hidden)]
pub fn merge_unique_lms_suffixes_32s(
    t: &mut [SaSint],
    sa: &mut [SaSint],
    n: SaSint,
    m: SaSint,
    l: FastSint,
    omp_block_start: FastSint,
    omp_block_size: FastSint,
) {
    if omp_block_size <= 0 {
        return;
    }

    let n_usize = usize::try_from(n).expect("n must be non-negative");
    let m_usize = usize::try_from(m).expect("m must be non-negative");
    let mut src_index = n_usize - m_usize - 1 + usize::try_from(l).expect("l must be non-negative");
    let mut tmp = sa[src_index] as FastSint;
    src_index += 1;

    let mut i = usize::try_from(omp_block_start).expect("omp_block_start must be non-negative");
    let block_end =
        i + usize::try_from(omp_block_size).expect("omp_block_size must be non-negative");
    let j = block_end.saturating_sub(6);
    while i < j {
        let c0 = t[i];
        if c0 < 0 {
            t[i] = c0 & SAINT_MAX;
            sa[usize::try_from(tmp).expect("target slot must be non-negative")] = i as SaSint;
            i += 1;
            tmp = sa[src_index] as FastSint;
            src_index += 1;
        }

        let c1 = t[i + 1];
        if c1 < 0 {
            t[i + 1] = c1 & SAINT_MAX;
            sa[usize::try_from(tmp).expect("target slot must be non-negative")] = i as SaSint + 1;
            i += 1;
            tmp = sa[src_index] as FastSint;
            src_index += 1;
        }

        let c2 = t[i + 2];
        if c2 < 0 {
            t[i + 2] = c2 & SAINT_MAX;
            sa[usize::try_from(tmp).expect("target slot must be non-negative")] = i as SaSint + 2;
            i += 1;
            tmp = sa[src_index] as FastSint;
            src_index += 1;
        }

        let c3 = t[i + 3];
        if c3 < 0 {
            t[i + 3] = c3 & SAINT_MAX;
            sa[usize::try_from(tmp).expect("target slot must be non-negative")] = i as SaSint + 3;
            i += 1;
            tmp = sa[src_index] as FastSint;
            src_index += 1;
        }

        i += 4;
    }

    while i < block_end {
        let c = t[i];
        if c < 0 {
            t[i] = c & SAINT_MAX;
            sa[usize::try_from(tmp).expect("target slot must be non-negative")] = i as SaSint;
            i += 1;
            tmp = sa[src_index] as FastSint;
            src_index += 1;
        }
        i += 1;
    }
}

/// Internal helper: merge nonunique lms suffixes 32s.
#[doc(hidden)]
pub fn merge_nonunique_lms_suffixes_32s(
    sa: &mut [SaSint],
    n: SaSint,
    m: SaSint,
    l: FastSint,
    omp_block_start: FastSint,
    omp_block_size: FastSint,
) {
    if omp_block_size <= 0 {
        return;
    }

    let n_usize = usize::try_from(n).expect("n must be non-negative");
    let m_usize = usize::try_from(m).expect("m must be non-negative");
    let mut src_index = n_usize - m_usize - 1 + usize::try_from(l).expect("l must be non-negative");
    let mut tmp = sa[src_index];
    src_index += 1;

    let mut i = usize::try_from(omp_block_start).expect("omp_block_start must be non-negative");
    let block_end =
        i + usize::try_from(omp_block_size).expect("omp_block_size must be non-negative");
    let j = block_end.saturating_sub(3);
    while i < j {
        if sa[i] == 0 {
            sa[i] = tmp;
            tmp = sa[src_index];
            src_index += 1;
        }
        if sa[i + 1] == 0 {
            sa[i + 1] = tmp;
            tmp = sa[src_index];
            src_index += 1;
        }
        if sa[i + 2] == 0 {
            sa[i + 2] = tmp;
            tmp = sa[src_index];
            src_index += 1;
        }
        if sa[i + 3] == 0 {
            sa[i + 3] = tmp;
            tmp = sa[src_index];
            src_index += 1;
        }
        i += 4;
    }

    while i < block_end {
        if sa[i] == 0 {
            sa[i] = tmp;
            tmp = sa[src_index];
            src_index += 1;
        }
        i += 1;
    }
}

/// Internal helper: merge unique lms suffixes 32s (OpenMP variant).
#[doc(hidden)]
pub fn merge_unique_lms_suffixes_32s_omp(
    t: &mut [SaSint],
    sa: &mut [SaSint],
    n: SaSint,
    m: SaSint,
    threads: SaSint,
    thread_state: &mut [ThreadState],
) {
    if threads == 1 || n < 65_536 {
        merge_unique_lms_suffixes_32s(t, sa, n, m, 0, 0, n as FastSint);
        return;
    }

    let threads_usize = usize::try_from(threads)
        .expect("threads must be non-negative")
        .max(1);
    let n_usize = usize::try_from(n).expect("n must be non-negative");
    let omp_num_threads = threads_usize.min(n_usize.max(1));
    let omp_block_stride = (n_usize / omp_num_threads) & !15usize;

    {
        let t_ro: &[SaSint] = t;
        run_rayon_with_threads(omp_num_threads, || {
            thread_state[..omp_num_threads]
                .par_iter_mut()
                .enumerate()
                .for_each(|(omp_thread_num, state)| {
                    let omp_block_start = omp_thread_num * omp_block_stride;
                    let omp_block_size = if omp_thread_num + 1 < omp_num_threads {
                        omp_block_stride
                    } else {
                        n_usize - omp_block_start
                    };
                    state.count = count_negative_marked_suffixes(
                        t_ro,
                        omp_block_start as FastSint,
                        omp_block_size as FastSint,
                    ) as FastSint;
                });
        });
    }

    let counts: Vec<FastSint> = thread_state[..omp_num_threads]
        .iter()
        .map(|s| s.count)
        .collect();

    {
        let sa_ptr = SyncMutPtr::new(sa);
        let t_ptr = SyncMutPtr::new(t);
        let counts_ref: &[FastSint] = &counts;
        run_rayon_with_threads(omp_num_threads, || {
            (0..omp_num_threads)
                .into_par_iter()
                .for_each(|omp_thread_num| {
                    let omp_block_start = omp_thread_num * omp_block_stride;
                    let omp_block_size = if omp_thread_num + 1 < omp_num_threads {
                        omp_block_stride
                    } else {
                        n_usize - omp_block_start
                    };

                    let mut count: FastSint = 0;
                    for tt in 0..omp_thread_num {
                        count += counts_ref[tt];
                    }

                    let sa = unsafe { sa_ptr.as_slice() };
                    let t = unsafe { t_ptr.as_slice() };
                    merge_unique_lms_suffixes_32s(
                        t,
                        sa,
                        n,
                        m,
                        count,
                        omp_block_start as FastSint,
                        omp_block_size as FastSint,
                    );
                });
        });
    }
}

/// Internal helper: merge nonunique lms suffixes 32s (OpenMP variant).
#[doc(hidden)]
pub fn merge_nonunique_lms_suffixes_32s_omp(
    sa: &mut [SaSint],
    n: SaSint,
    m: SaSint,
    f: SaSint,
    threads: SaSint,
    thread_state: &mut [ThreadState],
) {
    if threads == 1 || m < 65_536 {
        merge_nonunique_lms_suffixes_32s(sa, n, m, f as FastSint, 0, m as FastSint);
        return;
    }

    let threads_usize = usize::try_from(threads)
        .expect("threads must be non-negative")
        .max(1);
    let m_usize = usize::try_from(m).expect("m must be non-negative");
    let omp_num_threads = threads_usize.min(m_usize.max(1));
    let omp_block_stride = (m_usize / omp_num_threads) & !15usize;

    {
        let sa_ro: &[SaSint] = sa;
        run_rayon_with_threads(omp_num_threads, || {
            thread_state[..omp_num_threads]
                .par_iter_mut()
                .enumerate()
                .for_each(|(omp_thread_num, state)| {
                    let omp_block_start = omp_thread_num * omp_block_stride;
                    let omp_block_size = if omp_thread_num + 1 < omp_num_threads {
                        omp_block_stride
                    } else {
                        m_usize - omp_block_start
                    };
                    state.count = count_zero_marked_suffixes(
                        sa_ro,
                        omp_block_start as FastSint,
                        omp_block_size as FastSint,
                    ) as FastSint;
                });
        });
    }

    let counts: Vec<FastSint> = thread_state[..omp_num_threads]
        .iter()
        .map(|s| s.count)
        .collect();

    {
        let sa_ptr = SyncMutPtr::new(sa);
        let counts_ref: &[FastSint] = &counts;
        run_rayon_with_threads(omp_num_threads, || {
            (0..omp_num_threads)
                .into_par_iter()
                .for_each(|omp_thread_num| {
                    let omp_block_start = omp_thread_num * omp_block_stride;
                    let omp_block_size = if omp_thread_num + 1 < omp_num_threads {
                        omp_block_stride
                    } else {
                        m_usize - omp_block_start
                    };

                    let mut count: FastSint = f as FastSint;
                    for tt in 0..omp_thread_num {
                        count += counts_ref[tt];
                    }

                    let sa = unsafe { sa_ptr.as_slice() };
                    merge_nonunique_lms_suffixes_32s(
                        sa,
                        n,
                        m,
                        count,
                        omp_block_start as FastSint,
                        omp_block_size as FastSint,
                    );
                });
        });
    }
}

/// Internal helper: merge compacted lms suffixes 32s (OpenMP variant).
#[doc(hidden)]
pub fn merge_compacted_lms_suffixes_32s_omp(
    t: &mut [SaSint],
    sa: &mut [SaSint],
    n: SaSint,
    m: SaSint,
    f: SaSint,
    threads: SaSint,
    thread_state: &mut [ThreadState],
) {
    merge_unique_lms_suffixes_32s_omp(t, sa, n, m, threads, thread_state);
    merge_nonunique_lms_suffixes_32s_omp(sa, n, m, f, threads, thread_state);
}

/// Internal helper: reconstruct compacted lms suffixes 32s 2k (OpenMP variant).
#[doc(hidden)]
pub fn reconstruct_compacted_lms_suffixes_32s_2k_omp(
    t: &mut [SaSint],
    sa: &mut [SaSint],
    n: SaSint,
    k: SaSint,
    m: SaSint,
    fs: SaSint,
    f: SaSint,
    buckets: &mut [SaSint],
    local_buckets: SaSint,
    threads: SaSint,
    thread_state: &mut [ThreadState],
) {
    if f > 0 {
        let dst = usize::try_from(n - m - 1).expect("destination must be non-negative");
        let src = usize::try_from(n + fs - m).expect("source must be non-negative");
        let len = usize::try_from(f).expect("length must be non-negative");
        sa.copy_within(src..src + len, dst);

        let _ = count_and_gather_compacted_lms_suffixes_32s_2k_omp(
            t,
            sa,
            n,
            k,
            buckets,
            local_buckets,
            threads,
            thread_state,
        );
        reconstruct_lms_suffixes_omp(sa, n, m - f, threads);

        let src_copy = 0usize;
        let dst_copy = usize::try_from(n - m - 1 + f).expect("destination must be non-negative");
        let copy_len = usize::try_from(m - f).expect("copy length must be non-negative");
        sa.copy_within(src_copy..src_copy + copy_len, dst_copy);
        sa[..usize::try_from(m).expect("m must be non-negative")].fill(0);

        merge_compacted_lms_suffixes_32s_omp(t, sa, n, m, f, threads, thread_state);
    } else {
        let _ = count_and_gather_lms_suffixes_32s_2k(t, sa, n, k, buckets, 0, n as FastSint);
        reconstruct_lms_suffixes_omp(sa, n, m, threads);
    }
}

/// Internal helper: reconstruct compacted lms suffixes 32s 1k (OpenMP variant).
#[doc(hidden)]
pub fn reconstruct_compacted_lms_suffixes_32s_1k_omp(
    t: &mut [SaSint],
    sa: &mut [SaSint],
    n: SaSint,
    m: SaSint,
    fs: SaSint,
    f: SaSint,
    threads: SaSint,
    thread_state: &mut [ThreadState],
) {
    if f > 0 {
        let dst = usize::try_from(n - m - 1).expect("destination must be non-negative");
        let src = usize::try_from(n + fs - m).expect("source must be non-negative");
        let len = usize::try_from(f).expect("length must be non-negative");
        sa.copy_within(src..src + len, dst);

        let _ = gather_compacted_lms_suffixes_32s(t, sa, n);
        reconstruct_lms_suffixes_omp(sa, n, m - f, threads);

        let dst_copy = usize::try_from(n - m - 1 + f).expect("destination must be non-negative");
        let copy_len = usize::try_from(m - f).expect("copy length must be non-negative");
        sa.copy_within(0..copy_len, dst_copy);
        sa[..usize::try_from(m).expect("m must be non-negative")].fill(0);

        merge_compacted_lms_suffixes_32s_omp(t, sa, n, m, f, threads, thread_state);
    } else {
        let _ = gather_lms_suffixes_32s(t, sa, n);
        reconstruct_lms_suffixes_omp(sa, n, m, threads);
    }
}

fn normalize_omp_threads(threads: SaSint) -> SaSint {
    if threads > 0 {
        threads
    } else {
        std::thread::available_parallelism()
            .map(|value| value.get() as SaSint)
            .unwrap_or(1)
            .max(1)
    }
}

fn libsais64_main_32s_recursion(
    t_ptr: *mut SaSint,
    sa_ptr: *mut SaSint,
    sa_capacity: usize,
    n: SaSint,
    k: SaSint,
    fs: SaSint,
    threads: SaSint,
    thread_state: &mut [ThreadState],
    _local_buffer: &mut [SaSint],
) -> SaSint {
    let fs = fs.min(SAINT_MAX - n);
    let local_buffer_size = SaSint::try_from(LIBSAIS_LOCAL_BUFFER_SIZE).expect("fits");
    let n_usize = usize::try_from(n).expect("n must be non-negative");
    let fs_usize = usize::try_from(fs).expect("fs must be non-negative");
    let total_len = n_usize + fs_usize;
    assert!(total_len <= sa_capacity);

    if k > 0 && n <= i32::MAX as SaSint {
        let int32_max = i32::MAX as SaSint;
        let expanded_space = fs as i128 + fs as i128 + n as i128 + n as i128;
        let new_fs = if expanded_space <= int32_max as i128 {
            fs + fs + n
        } else {
            int32_max - n
        };

        if (new_fs / k >= 6)
            || (new_fs / k >= 4 && n <= int32_max / 2)
            || (new_fs / k < 4 && new_fs >= fs)
        {
            let mut t32 = unsafe { std::slice::from_raw_parts(t_ptr, n_usize) }
                .iter()
                .map(|&value| (value as u64 as u32) as i32)
                .collect::<Vec<_>>();
            let mut sa32 = vec![0i32; n_usize + new_fs as usize];

            let index = crate::libsais_int_omp(
                &mut t32,
                &mut sa32,
                k as i32,
                new_fs as i32,
                threads as i32,
            );
            if index >= 0 {
                unsafe {
                    let t = std::slice::from_raw_parts_mut(t_ptr, n_usize);
                    for (dst, src) in t.iter_mut().zip(t32.iter()) {
                        *dst = (*src as u32) as SaSint;
                    }

                    let sa = std::slice::from_raw_parts_mut(sa_ptr, n_usize);
                    for (dst, src) in sa.iter_mut().zip(sa32.iter()) {
                        *dst = (*src as u32) as SaSint;
                    }
                }
            }

            return index as SaSint;
        }
    }

    if k > 0 && ((fs / k) >= 6 || (local_buffer_size / k) >= 6) {
        let k_usize = usize::try_from(k).expect("k must be non-negative");
        let alignment = if fs >= 1024 && ((fs - 1024) / k) >= 6 {
            1024usize
        } else {
            16usize
        };
        let need = 6 * k_usize;
        let use_local_buffer = local_buffer_size > fs;
        let buckets_ptr = if use_local_buffer {
            _local_buffer.as_mut_ptr()
        } else {
            unsafe {
                let sa = std::slice::from_raw_parts_mut(sa_ptr, total_len);
                let start =
                    if fs_usize >= need + alignment && ((fs_usize - alignment) / k_usize) >= 6 {
                        let byte_ptr = sa[total_len - need - alignment..].as_mut_ptr() as usize;
                        let aligned = align_up(byte_ptr, alignment * mem::size_of::<SaSint>());
                        (aligned - sa_ptr as usize) / mem::size_of::<SaSint>()
                    } else {
                        total_len - need
                    };
                sa[start..].as_mut_ptr()
            }
        };

        let m = unsafe {
            let t = std::slice::from_raw_parts_mut(t_ptr, n_usize);
            let sa = std::slice::from_raw_parts_mut(sa_ptr, total_len);
            let buckets = std::slice::from_raw_parts_mut(buckets_ptr, need);
            count_and_gather_lms_suffixes_32s_4k_omp(
                t,
                sa,
                n,
                k,
                buckets,
                SaSint::from(use_local_buffer),
                threads,
                thread_state,
            )
        };
        if m > 1 {
            let m_usize = usize::try_from(m).expect("m must be non-negative");
            unsafe {
                let sa = std::slice::from_raw_parts_mut(sa_ptr, total_len);
                sa[..n_usize - m_usize].fill(0);
            }

            let first_lms_suffix = unsafe {
                let sa = std::slice::from_raw_parts_mut(sa_ptr, total_len);
                sa[n_usize - m_usize]
            };
            let left_suffixes_count = unsafe {
                let buckets = std::slice::from_raw_parts_mut(buckets_ptr, need);
                initialize_buckets_for_lms_suffixes_radix_sort_32s_6k(
                    std::slice::from_raw_parts_mut(t_ptr, n_usize),
                    k,
                    buckets,
                    first_lms_suffix,
                )
            };

            unsafe {
                let t = std::slice::from_raw_parts_mut(t_ptr, n_usize);
                let sa = std::slice::from_raw_parts_mut(sa_ptr, total_len);
                let buckets = std::slice::from_raw_parts_mut(buckets_ptr, need);
                let (_, induction_bucket) = buckets.split_at_mut(4 * k_usize);
                radix_sort_lms_suffixes_32s_6k_omp(
                    t,
                    sa,
                    n,
                    m,
                    induction_bucket,
                    threads,
                    thread_state,
                );
                if (n / 8192) < k {
                    radix_sort_set_markers_32s_6k_omp(sa, k, induction_bucket, threads);
                }
                if threads > 1 && n >= 65_536 {
                    sa[n_usize - m_usize..n_usize].fill(0);
                }
                initialize_buckets_for_partial_sorting_32s_6k(
                    t,
                    k,
                    buckets,
                    first_lms_suffix,
                    left_suffixes_count,
                );
                induce_partial_order_32s_6k_omp(
                    t,
                    sa,
                    n,
                    k,
                    buckets,
                    first_lms_suffix,
                    left_suffixes_count,
                    threads,
                    thread_state,
                );
            }

            let names = unsafe {
                let sa = std::slice::from_raw_parts_mut(sa_ptr, total_len);
                if (n / 8192) < k {
                    renumber_and_mark_distinct_lms_suffixes_32s_4k_omp(
                        sa,
                        n,
                        m,
                        threads,
                        thread_state,
                    )
                } else {
                    renumber_and_gather_lms_suffixes_omp(sa, n, m, fs, threads, thread_state)
                }
            };

            if names < m {
                let f = if (n / 8192) < k {
                    unsafe {
                        let t = std::slice::from_raw_parts_mut(t_ptr, n_usize);
                        let sa = std::slice::from_raw_parts_mut(sa_ptr, total_len);
                        compact_lms_suffixes_32s_omp(t, sa, n, m, fs, threads, thread_state)
                    }
                } else {
                    0
                };

                let new_t_start =
                    total_len - usize::try_from(m - f).expect("m - f must be non-negative");
                if libsais64_main_32s_recursion(
                    unsafe {
                        std::slice::from_raw_parts_mut(sa_ptr, total_len)[new_t_start..]
                            .as_mut_ptr()
                    },
                    sa_ptr,
                    sa_capacity,
                    m - f,
                    names - f,
                    fs + n - 2 * m + f,
                    threads,
                    thread_state,
                    _local_buffer,
                ) != 0
                {
                    return -2;
                }

                unsafe {
                    let t = std::slice::from_raw_parts_mut(t_ptr, n_usize);
                    let sa = std::slice::from_raw_parts_mut(sa_ptr, total_len);
                    let buckets = std::slice::from_raw_parts_mut(buckets_ptr, need);
                    reconstruct_compacted_lms_suffixes_32s_2k_omp(
                        t,
                        sa,
                        n,
                        k,
                        m,
                        fs,
                        f,
                        buckets,
                        SaSint::from(use_local_buffer),
                        threads,
                        thread_state,
                    );
                }
            } else {
                unsafe {
                    let t = std::slice::from_raw_parts_mut(t_ptr, n_usize);
                    let buckets = std::slice::from_raw_parts_mut(buckets_ptr, need);
                    count_lms_suffixes_32s_2k(t, n, k, buckets);
                }
            }

            unsafe {
                let buckets = std::slice::from_raw_parts_mut(buckets_ptr, need);
                initialize_buckets_start_and_end_32s_4k(k, buckets);
                let sa = std::slice::from_raw_parts_mut(sa_ptr, total_len);
                place_lms_suffixes_histogram_32s_4k(sa, n, k, m, buckets);
                let t = std::slice::from_raw_parts_mut(t_ptr, n_usize);
                induce_final_order_32s_4k(t, sa, n, k, buckets, threads, thread_state);
            }
        } else {
            unsafe {
                let sa = std::slice::from_raw_parts_mut(sa_ptr, total_len);
                sa[0] = sa[n_usize - 1];
            }

            unsafe {
                let buckets = std::slice::from_raw_parts_mut(buckets_ptr, need);
                initialize_buckets_start_and_end_32s_6k(k, buckets);
                let sa = std::slice::from_raw_parts_mut(sa_ptr, total_len);
                place_lms_suffixes_histogram_32s_6k(sa, n, k, m, buckets);
                let t = std::slice::from_raw_parts_mut(t_ptr, n_usize);
                induce_final_order_32s_6k(t, sa, n, k, buckets, threads, thread_state);
            }
        }

        return 0;
    } else if k > 0 && n <= SAINT_MAX / 2 && ((fs / k) >= 4 || (local_buffer_size / k) >= 4) {
        let k_usize = usize::try_from(k).expect("k must be non-negative");
        let alignment = if fs >= 1024 && ((fs - 1024) / k) >= 4 {
            1024usize
        } else {
            16usize
        };
        let need = 4 * k_usize;
        let use_local_buffer = local_buffer_size > fs;
        let buckets_ptr = if use_local_buffer {
            _local_buffer.as_mut_ptr()
        } else {
            unsafe {
                let sa = std::slice::from_raw_parts_mut(sa_ptr, total_len);
                let start =
                    if fs_usize >= need + alignment && ((fs_usize - alignment) / k_usize) >= 4 {
                        let byte_ptr = sa[total_len - need - alignment..].as_mut_ptr() as usize;
                        let aligned = align_up(byte_ptr, alignment * mem::size_of::<SaSint>());
                        (aligned - sa_ptr as usize) / mem::size_of::<SaSint>()
                    } else {
                        total_len - need
                    };
                sa[start..].as_mut_ptr()
            }
        };

        let m = unsafe {
            let t = std::slice::from_raw_parts_mut(t_ptr, n_usize);
            let sa = std::slice::from_raw_parts_mut(sa_ptr, total_len);
            let buckets = std::slice::from_raw_parts_mut(buckets_ptr, need);
            count_and_gather_lms_suffixes_32s_2k_omp(
                t,
                sa,
                n,
                k,
                buckets,
                SaSint::from(use_local_buffer),
                threads,
                thread_state,
            )
        };
        if m > 1 {
            let m_usize = usize::try_from(m).expect("m must be non-negative");
            unsafe {
                let t = std::slice::from_raw_parts_mut(t_ptr, n_usize);
                let buckets = std::slice::from_raw_parts_mut(buckets_ptr, need);
                let sa = std::slice::from_raw_parts_mut(sa_ptr, total_len);
                initialize_buckets_for_radix_and_partial_sorting_32s_4k(
                    t,
                    k,
                    buckets,
                    sa[n_usize - m_usize],
                );
                let (_, induction_bucket) = buckets.split_at_mut(1);
                radix_sort_lms_suffixes_32s_2k_omp(
                    t,
                    sa,
                    n,
                    m,
                    induction_bucket,
                    threads,
                    thread_state,
                );
                radix_sort_set_markers_32s_4k_omp(sa, k, induction_bucket, threads);
                place_lms_suffixes_interval_32s_4k(sa, n, k, m - 1, buckets);
                induce_partial_order_32s_4k_omp(t, sa, n, k, buckets, threads, thread_state);
            }

            let names = unsafe {
                let sa = std::slice::from_raw_parts_mut(sa_ptr, total_len);
                renumber_and_mark_distinct_lms_suffixes_32s_4k_omp(sa, n, m, threads, thread_state)
            };
            if names < m {
                let f = unsafe {
                    let t = std::slice::from_raw_parts_mut(t_ptr, n_usize);
                    let sa = std::slice::from_raw_parts_mut(sa_ptr, total_len);
                    compact_lms_suffixes_32s_omp(t, sa, n, m, fs, threads, thread_state)
                };

                let new_t_start =
                    total_len - usize::try_from(m - f).expect("m - f must be non-negative");
                if libsais64_main_32s_recursion(
                    unsafe {
                        std::slice::from_raw_parts_mut(sa_ptr, total_len)[new_t_start..]
                            .as_mut_ptr()
                    },
                    sa_ptr,
                    sa_capacity,
                    m - f,
                    names - f,
                    fs + n - 2 * m + f,
                    threads,
                    thread_state,
                    _local_buffer,
                ) != 0
                {
                    return -2;
                }

                unsafe {
                    let t = std::slice::from_raw_parts_mut(t_ptr, n_usize);
                    let sa = std::slice::from_raw_parts_mut(sa_ptr, total_len);
                    let buckets = std::slice::from_raw_parts_mut(buckets_ptr, need);
                    reconstruct_compacted_lms_suffixes_32s_2k_omp(
                        t,
                        sa,
                        n,
                        k,
                        m,
                        fs,
                        f,
                        buckets,
                        SaSint::from(use_local_buffer),
                        threads,
                        thread_state,
                    );
                }
            } else {
                unsafe {
                    let t = std::slice::from_raw_parts_mut(t_ptr, n_usize);
                    let buckets = std::slice::from_raw_parts_mut(buckets_ptr, need);
                    count_lms_suffixes_32s_2k(t, n, k, buckets);
                }
            }
        } else {
            unsafe {
                let sa = std::slice::from_raw_parts_mut(sa_ptr, total_len);
                sa[0] = sa[n_usize - 1];
            }
        }

        unsafe {
            let buckets = std::slice::from_raw_parts_mut(buckets_ptr, need);
            initialize_buckets_start_and_end_32s_4k(k, buckets);
            let sa = std::slice::from_raw_parts_mut(sa_ptr, total_len);
            place_lms_suffixes_histogram_32s_4k(sa, n, k, m, buckets);
            let t = std::slice::from_raw_parts_mut(t_ptr, n_usize);
            induce_final_order_32s_4k(t, sa, n, k, buckets, threads, thread_state);
        }

        return 0;
    } else if k > 0 && ((fs / k) >= 2 || (local_buffer_size / k) >= 2) {
        let k_usize = usize::try_from(k).expect("k must be non-negative");
        let alignment = if fs >= 1024 && ((fs - 1024) / k) >= 2 {
            1024usize
        } else {
            16usize
        };
        let need = 2 * k_usize;
        let use_local_buffer = local_buffer_size > fs;
        let buckets_ptr = if use_local_buffer {
            _local_buffer.as_mut_ptr()
        } else {
            unsafe {
                let sa = std::slice::from_raw_parts_mut(sa_ptr, total_len);
                let start =
                    if fs_usize >= need + alignment && ((fs_usize - alignment) / k_usize) >= 2 {
                        let byte_ptr = sa[total_len - need - alignment..].as_mut_ptr() as usize;
                        let aligned = align_up(byte_ptr, alignment * mem::size_of::<SaSint>());
                        (aligned - sa_ptr as usize) / mem::size_of::<SaSint>()
                    } else {
                        total_len - need
                    };
                sa[start..].as_mut_ptr()
            }
        };

        let m = unsafe {
            let t = std::slice::from_raw_parts_mut(t_ptr, n_usize);
            let sa = std::slice::from_raw_parts_mut(sa_ptr, total_len);
            let buckets = std::slice::from_raw_parts_mut(buckets_ptr, need);
            count_and_gather_lms_suffixes_32s_2k_omp(
                t,
                sa,
                n,
                k,
                buckets,
                SaSint::from(use_local_buffer),
                threads,
                thread_state,
            )
        };
        if m > 1 {
            let m_usize = usize::try_from(m).expect("m must be non-negative");
            unsafe {
                let t = std::slice::from_raw_parts_mut(t_ptr, n_usize);
                let buckets = std::slice::from_raw_parts_mut(buckets_ptr, need);
                let sa = std::slice::from_raw_parts_mut(sa_ptr, total_len);
                initialize_buckets_for_lms_suffixes_radix_sort_32s_2k(
                    t,
                    k,
                    buckets,
                    sa[n_usize - m_usize],
                );
                let (_, induction_bucket) = buckets.split_at_mut(1);
                radix_sort_lms_suffixes_32s_2k_omp(
                    t,
                    sa,
                    n,
                    m,
                    induction_bucket,
                    threads,
                    thread_state,
                );
                place_lms_suffixes_interval_32s_2k(sa, n, k, m - 1, buckets);
            }

            unsafe {
                let buckets = std::slice::from_raw_parts_mut(buckets_ptr, need);
                initialize_buckets_start_and_end_32s_2k(k, buckets);
                let t = std::slice::from_raw_parts_mut(t_ptr, n_usize);
                let sa = std::slice::from_raw_parts_mut(sa_ptr, total_len);
                induce_partial_order_32s_2k_omp(t, sa, n, k, buckets, threads, thread_state);
            }

            let names = unsafe {
                let t = std::slice::from_raw_parts_mut(t_ptr, n_usize);
                let sa = std::slice::from_raw_parts_mut(sa_ptr, total_len);
                renumber_and_mark_distinct_lms_suffixes_32s_1k_omp(t, sa, n, m, threads)
            };
            if names < m {
                let f = unsafe {
                    let t = std::slice::from_raw_parts_mut(t_ptr, n_usize);
                    let sa = std::slice::from_raw_parts_mut(sa_ptr, total_len);
                    compact_lms_suffixes_32s_omp(t, sa, n, m, fs, threads, thread_state)
                };

                let new_t_start =
                    total_len - usize::try_from(m - f).expect("m - f must be non-negative");
                if libsais64_main_32s_recursion(
                    unsafe {
                        std::slice::from_raw_parts_mut(sa_ptr, total_len)[new_t_start..]
                            .as_mut_ptr()
                    },
                    sa_ptr,
                    sa_capacity,
                    m - f,
                    names - f,
                    fs + n - 2 * m + f,
                    threads,
                    thread_state,
                    _local_buffer,
                ) != 0
                {
                    return -2;
                }

                unsafe {
                    let t = std::slice::from_raw_parts_mut(t_ptr, n_usize);
                    let sa = std::slice::from_raw_parts_mut(sa_ptr, total_len);
                    let buckets = std::slice::from_raw_parts_mut(buckets_ptr, need);
                    reconstruct_compacted_lms_suffixes_32s_2k_omp(
                        t,
                        sa,
                        n,
                        k,
                        m,
                        fs,
                        f,
                        buckets,
                        SaSint::from(use_local_buffer),
                        threads,
                        thread_state,
                    );
                }
            } else {
                unsafe {
                    let t = std::slice::from_raw_parts_mut(t_ptr, n_usize);
                    let buckets = std::slice::from_raw_parts_mut(buckets_ptr, need);
                    count_lms_suffixes_32s_2k(t, n, k, buckets);
                }
            }
        } else {
            unsafe {
                let sa = std::slice::from_raw_parts_mut(sa_ptr, total_len);
                sa[0] = sa[n_usize - 1];
            }
        }

        unsafe {
            let buckets = std::slice::from_raw_parts_mut(buckets_ptr, need);
            initialize_buckets_end_32s_2k(k, buckets);
            let sa = std::slice::from_raw_parts_mut(sa_ptr, total_len);
            place_lms_suffixes_histogram_32s_2k(sa, n, k, m, buckets);
        }

        unsafe {
            let buckets = std::slice::from_raw_parts_mut(buckets_ptr, need);
            initialize_buckets_start_and_end_32s_2k(k, buckets);
            let t = std::slice::from_raw_parts_mut(t_ptr, n_usize);
            let sa = std::slice::from_raw_parts_mut(sa_ptr, total_len);
            induce_final_order_32s_2k(t, sa, n, k, buckets, threads, thread_state);
        }

        return 0;
    } else {
        let k_usize = usize::try_from(k).expect("k must be non-negative");
        let mut heap_buckets = if fs < k { Some(vec![0; k_usize]) } else { None };
        let alignment = if fs >= 1024 && (fs - 1024) >= k {
            1024usize
        } else {
            16usize
        };
        let mut buckets_ptr = if let Some(ref mut heap) = heap_buckets {
            heap.as_mut_ptr()
        } else {
            unsafe {
                let sa = std::slice::from_raw_parts_mut(sa_ptr, total_len);
                let start = if fs_usize >= k_usize + alignment {
                    let byte_ptr = sa[total_len - k_usize - alignment..].as_mut_ptr() as usize;
                    let aligned = align_up(byte_ptr, alignment * mem::size_of::<SaSint>());
                    (aligned - sa_ptr as usize) / mem::size_of::<SaSint>()
                } else {
                    total_len - k_usize
                };
                sa[start..].as_mut_ptr()
            }
        };

        if buckets_ptr.is_null() {
            return -2;
        }

        unsafe {
            let sa = std::slice::from_raw_parts_mut(sa_ptr, total_len);
            sa[..n_usize].fill(0);
        }

        unsafe {
            let t = std::slice::from_raw_parts_mut(t_ptr, n_usize);
            let buckets = std::slice::from_raw_parts_mut(buckets_ptr, k_usize);
            count_suffixes_32s(t, n, k, buckets);
        }
        unsafe {
            let buckets = std::slice::from_raw_parts_mut(buckets_ptr, k_usize);
            initialize_buckets_end_32s_1k(k, buckets);
        }

        let m = unsafe {
            let t = std::slice::from_raw_parts_mut(t_ptr, n_usize);
            let sa = std::slice::from_raw_parts_mut(sa_ptr, total_len);
            let buckets = std::slice::from_raw_parts_mut(buckets_ptr, k_usize);
            radix_sort_lms_suffixes_32s_1k(t, sa, n, buckets)
        };
        if m > 1 {
            unsafe {
                let t = std::slice::from_raw_parts_mut(t_ptr, n_usize);
                let sa = std::slice::from_raw_parts_mut(sa_ptr, total_len);
                let buckets = std::slice::from_raw_parts_mut(buckets_ptr, k_usize);
                induce_partial_order_32s_1k_omp(t, sa, n, k, buckets, threads, thread_state);
            }

            let names = unsafe {
                let t = std::slice::from_raw_parts_mut(t_ptr, n_usize);
                let sa = std::slice::from_raw_parts_mut(sa_ptr, total_len);
                renumber_and_mark_distinct_lms_suffixes_32s_1k_omp(t, sa, n, m, threads)
            };
            if names < m {
                if heap_buckets.is_some() {
                    let _ = heap_buckets.take();
                    buckets_ptr = std::ptr::null_mut();
                }

                let f = unsafe {
                    let t = std::slice::from_raw_parts_mut(t_ptr, n_usize);
                    let sa = std::slice::from_raw_parts_mut(sa_ptr, total_len);
                    compact_lms_suffixes_32s_omp(t, sa, n, m, fs, threads, thread_state)
                };

                let new_t_start =
                    total_len - usize::try_from(m - f).expect("m - f must be non-negative");
                if libsais64_main_32s_recursion(
                    unsafe {
                        std::slice::from_raw_parts_mut(sa_ptr, total_len)[new_t_start..]
                            .as_mut_ptr()
                    },
                    sa_ptr,
                    sa_capacity,
                    m - f,
                    names - f,
                    fs + n - 2 * m + f,
                    threads,
                    thread_state,
                    _local_buffer,
                ) != 0
                {
                    return -2;
                }

                unsafe {
                    let t = std::slice::from_raw_parts_mut(t_ptr, n_usize);
                    let sa = std::slice::from_raw_parts_mut(sa_ptr, total_len);
                    reconstruct_compacted_lms_suffixes_32s_1k_omp(
                        t,
                        sa,
                        n,
                        m,
                        fs,
                        f,
                        threads,
                        thread_state,
                    );
                }

                if buckets_ptr.is_null() {
                    heap_buckets = Some(vec![0; k_usize]);
                    buckets_ptr = heap_buckets
                        .as_mut()
                        .expect("heap buckets must exist")
                        .as_mut_ptr();
                    if buckets_ptr.is_null() {
                        return -2;
                    }
                }
            }

            unsafe {
                let t = std::slice::from_raw_parts_mut(t_ptr, n_usize);
                let buckets = std::slice::from_raw_parts_mut(buckets_ptr, k_usize);
                count_suffixes_32s(t, n, k, buckets);
            }
            unsafe {
                let buckets = std::slice::from_raw_parts_mut(buckets_ptr, k_usize);
                initialize_buckets_end_32s_1k(k, buckets);
            }
            unsafe {
                let t = std::slice::from_raw_parts_mut(t_ptr, n_usize);
                let sa = std::slice::from_raw_parts_mut(sa_ptr, total_len);
                let buckets = std::slice::from_raw_parts_mut(buckets_ptr, k_usize);
                place_lms_suffixes_interval_32s_1k(t, sa, k, m, buckets);
            }
        }

        unsafe {
            let t = std::slice::from_raw_parts_mut(t_ptr, n_usize);
            let sa = std::slice::from_raw_parts_mut(sa_ptr, total_len);
            let buckets = std::slice::from_raw_parts_mut(buckets_ptr, k_usize);
            induce_final_order_32s_1k(t, sa, n, k, buckets, threads, thread_state);
        }

        0
    }
}

fn libsais64_main_32s_entry(
    t: &mut [SaSint],
    sa: &mut [SaSint],
    n: SaSint,
    k: SaSint,
    fs: SaSint,
    threads: SaSint,
    thread_state: &mut [ThreadState],
) -> SaSint {
    let mut local_buffer = [0; 2 * LIBSAIS_LOCAL_BUFFER_SIZE];
    libsais64_main_32s_recursion(
        t.as_mut_ptr(),
        sa.as_mut_ptr(),
        sa.len(),
        n,
        k,
        fs,
        threads,
        thread_state,
        &mut local_buffer[LIBSAIS_LOCAL_BUFFER_SIZE..],
    )
}

fn libsais64_main_8u(
    t: &[u8],
    sa: &mut [SaSint],
    buckets: &mut [SaSint],
    flags: SaSint,
    r: SaSint,
    i: Option<&mut [SaSint]>,
    fs: SaSint,
    freq: Option<&mut [SaSint]>,
    threads: SaSint,
    thread_state: &mut [ThreadState],
) -> SaSint {
    let n = SaSint::try_from(t.len()).expect("input length must fit SaSint");
    let n_usize = usize::try_from(n).expect("n must be non-negative");
    let fs = fs.min(SAINT_MAX - n);

    let m = count_and_gather_lms_suffixes_8u_omp(t, sa, n, buckets, threads, thread_state);
    let k = initialize_buckets_start_and_end_8u(buckets, freq);

    if (flags & LIBSAIS_FLAGS_GSA) != 0 && (buckets[0] != 0 || buckets[2] != 0 || buckets[3] != 1) {
        return -1;
    }

    if m > 0 {
        let m_usize = usize::try_from(m).expect("m must be non-negative");
        let first_lms_suffix = sa[n_usize - m_usize];
        let left_suffixes_count =
            initialize_buckets_for_lms_suffixes_radix_sort_8u(t, buckets, first_lms_suffix);

        if threads > 1 && n >= 65_536 {
            sa[..n_usize - m_usize].fill(0);
        }
        radix_sort_lms_suffixes_8u_omp(t, sa, n, m, flags, buckets, threads, thread_state);
        if threads > 1 && n >= 65_536 {
            sa[n_usize - m_usize..n_usize].fill(0);
        }

        initialize_buckets_for_partial_sorting_8u(
            t,
            buckets,
            first_lms_suffix,
            left_suffixes_count,
        );
        induce_partial_order_8u_omp(
            t,
            sa,
            n,
            k,
            flags,
            buckets,
            first_lms_suffix,
            left_suffixes_count,
            threads,
            thread_state,
        );

        let names = renumber_and_gather_lms_suffixes_omp(sa, n, m, fs, threads, thread_state);
        if names < m {
            if libsais64_main_32s_entry(
                unsafe {
                    std::slice::from_raw_parts_mut(
                        sa[n_usize + usize::try_from(fs).expect("fs must be non-negative")
                            - m_usize..]
                            .as_mut_ptr(),
                        m_usize,
                    )
                },
                sa,
                m,
                names,
                fs + n - 2 * m,
                threads,
                thread_state,
            ) != 0
            {
                return -2;
            }

            gather_lms_suffixes_8u_omp(t, sa, n, threads, thread_state);
            reconstruct_lms_suffixes_omp(sa, n, m, threads);
        }

        place_lms_suffixes_interval_8u(sa, n, m, flags, buckets);
    } else {
        sa[..n_usize].fill(0);
    }

    induce_final_order_8u_omp(t, sa, n, k, flags, r, i, buckets, threads, thread_state)
}

fn libsais64_main(
    t: &[u8],
    sa: &mut [SaSint],
    flags: SaSint,
    r: SaSint,
    i: Option<&mut [SaSint]>,
    fs: SaSint,
    freq: Option<&mut [SaSint]>,
    threads: SaSint,
) -> SaSint {
    let threads = normalize_omp_threads(threads);
    if threads > 1 {
        let mut thread_state = match alloc_thread_state(threads) {
            Some(thread_state) => thread_state,
            None => return -2,
        };
        let mut buckets = vec![0; 8 * ALPHABET_SIZE];

        libsais64_main_8u(
            t,
            sa,
            &mut buckets,
            flags,
            r,
            i,
            fs,
            freq,
            threads,
            &mut thread_state,
        )
    } else {
        let mut thread_state = [];
        let mut buckets = [0; 8 * ALPHABET_SIZE];

        libsais64_main_8u(
            t,
            sa,
            &mut buckets,
            flags,
            r,
            i,
            fs,
            freq,
            threads,
            &mut thread_state,
        )
    }
}

fn libsais64_main_int(
    t: &mut [SaSint],
    sa: &mut [SaSint],
    k: SaSint,
    fs: SaSint,
    threads: SaSint,
) -> SaSint {
    let threads = normalize_omp_threads(threads);
    let mut thread_state = if threads > 1 {
        match alloc_thread_state(threads) {
            Some(thread_state) => thread_state,
            None => return -2,
        }
    } else {
        Vec::new()
    };

    libsais64_main_32s_entry(
        t,
        sa,
        SaSint::try_from(t.len()).expect("input length must fit SaSint"),
        k,
        fs,
        threads,
        &mut thread_state,
    )
}

#[allow(dead_code)]
fn convert_32u_to_64u(s: &[u32], d: &mut [u64], block_start: usize, block_size: usize) {
    for i in block_start..block_start + block_size {
        d[i] = s[i] as u64;
    }
}

#[allow(dead_code)]
fn convert_inplace_32u_to_64u(v: &mut [u32], block_start: usize, block_size: usize) {
    for i in (block_start..block_start + block_size).rev() {
        let value = v[i];
        v[2 * i] = value;
        v[2 * i + 1] = 0;
    }
}

#[allow(dead_code)]
fn convert_inplace_64u_to_32u(v: &mut [u32], block_start: usize, block_size: usize) {
    for i in block_start..block_start + block_size {
        v[i] = v[2 * i];
    }
}

#[allow(dead_code)]
fn convert_inplace_32u_to_64u_omp(v: &mut [u32], n: SaSint, threads: SaSint) {
    let mut n = usize::try_from(n).expect("n must be non-negative");
    let threads = usize::try_from(threads.max(1)).expect("threads must be non-negative");

    while n >= 65_536 {
        let block_size = n >> 1;
        n -= block_size;

        let omp_block_stride = (block_size / threads) & !15usize;
        for thread in 0..threads {
            let block_start = thread * omp_block_stride;
            let size = if thread + 1 < threads {
                omp_block_stride
            } else {
                block_size - block_start
            };
            convert_inplace_32u_to_64u(v, n + block_start, size);
        }
    }

    convert_inplace_32u_to_64u(v, 0, n);
}

fn libsais64_main_ctx(
    ctx: &mut Context,
    t: &[u8],
    sa: &mut [SaSint],
    flags: SaSint,
    r: SaSint,
    i: Option<&mut [SaSint]>,
    fs: SaSint,
    freq: Option<&mut [SaSint]>,
) -> SaSint {
    if ctx.threads <= 0 || ctx.buckets.len() != 8 * ALPHABET_SIZE {
        return -2;
    }

    let mut empty_thread_state = [];
    let thread_state = if ctx.threads > 1 {
        match ctx.thread_state.as_deref_mut() {
            Some(thread_state) if thread_state.len() >= ctx.threads as usize => thread_state,
            None => return -2,
            Some(_) => return -2,
        }
    } else {
        &mut empty_thread_state
    };

    libsais64_main_8u(
        t,
        sa,
        &mut ctx.buckets,
        flags,
        r,
        i,
        fs,
        freq,
        ctx.threads as SaSint,
        thread_state,
    )
}

/// Constructs the suffix array of a given string.
///
/// # Arguments
/// - `t` (`[0..n-1]`): The input string.
/// - `sa` (`[0..n-1+fs]`): The output array of suffixes.
/// - `fs`: The extra space available at the end of SA array (0 should be enough for most cases).
/// - `freq` (`[0..255]`): The output symbol frequency table (can be `None`).
///
/// # Returns
/// 0 if no error occurred, -1 or -2 otherwise.
pub fn libsais64(t: &[u8], sa: &mut [SaSint], fs: SaSint, freq: Option<&mut [SaSint]>) -> SaSint {
    if fs < 0
        || sa.len()
            < t.len()
                .saturating_add(usize::try_from(fs).unwrap_or(usize::MAX))
    {
        return -1;
    }
    if let Some(freq) = freq.as_ref() {
        if freq.len() < ALPHABET_SIZE {
            return -1;
        }
    }

    let n = t.len();
    if n <= 1 {
        if let Some(freq) = freq {
            freq[..ALPHABET_SIZE].fill(0);
            if n == 1 {
                freq[t[0] as usize] += 1;
            }
        }
        if n == 1 {
            sa[0] = 0;
        }
        return 0;
    }

    if n <= i32::MAX as usize {
        return libsais64_run_32bit_omp(t, sa, fs, freq, 1, false)
            .expect("n <= INT32_MAX must have 32-bit workspace");
    }

    libsais64_main(t, sa, LIBSAIS_FLAGS_NONE, 0, None, fs, freq, 1)
}

#[cfg(feature = "upstream-c")]
unsafe extern "C" {
    fn probe_public_libsais64_omp_freq(
        t: *const u8,
        sa: *mut SaSint,
        n: SaSint,
        fs: SaSint,
        freq: *mut SaSint,
        threads: SaSint,
    ) -> SaSint;
}

/// Wrapper around the bundled upstream C `libsais64_omp` implementation.
///
/// Available only with the `upstream-c` feature. Same semantics as the Rust [`libsais64_omp`] function but defers all work to the C library; intended for the differential test suite and benchmarks.
///
/// - `t` (`[0..n-1]`): the input string.
/// - `sa` (`[0..n-1+fs]`): the output array of suffixes.
/// - `fs`: extra space available at the end of `sa`.
/// - `freq` (`[0..255]`): optional output symbol frequency table.
/// - `threads`: number of worker threads (can be 0 for the implementation default).
///
/// Returns 0 on success, -1 or -2 on error.
#[cfg(feature = "upstream-c")]
pub fn libsais64_upstream_c_omp(
    t: &[u8],
    sa: &mut [SaSint],
    fs: SaSint,
    freq: Option<&mut [SaSint]>,
    threads: SaSint,
) -> SaSint {
    if threads < 0 {
        return -1;
    }
    if fs < 0
        || t.len() > SaSint::MAX as usize
        || sa.len()
            < t.len()
                .saturating_add(usize::try_from(fs).unwrap_or(usize::MAX))
    {
        return -1;
    }
    if let Some(freq) = freq.as_ref() {
        if freq.len() < ALPHABET_SIZE {
            return -1;
        }
    }

    let n = t.len() as SaSint;
    let freq_ptr = freq.map_or(std::ptr::null_mut(), |freq| freq.as_mut_ptr());
    unsafe {
        probe_public_libsais64_omp_freq(
            t.as_ptr(),
            sa.as_mut_ptr(),
            n,
            fs,
            freq_ptr,
            threads.max(1),
        )
    }
}

/// `MaybeUninit` variant of [`libsais64_upstream_c_omp`].
///
/// Available only with the `upstream-c` feature. Identical semantics, except that the `sa` output may be uninitialised on entry — the C implementation always writes every required slot, so the caller may treat the slice as initialised on success.
#[cfg(feature = "upstream-c")]
pub fn libsais64_upstream_c_omp_uninit(
    t: &[u8],
    sa: &mut [MaybeUninit<SaSint>],
    fs: SaSint,
    freq: Option<&mut [SaSint]>,
    threads: SaSint,
) -> SaSint {
    if threads < 0 {
        return -1;
    }
    if fs < 0
        || t.len() > SaSint::MAX as usize
        || sa.len()
            < t.len()
                .saturating_add(usize::try_from(fs).unwrap_or(usize::MAX))
    {
        return -1;
    }
    if let Some(freq) = freq.as_ref() {
        if freq.len() < ALPHABET_SIZE {
            return -1;
        }
    }

    let n = t.len() as SaSint;
    let freq_ptr = freq.map_or(std::ptr::null_mut(), |freq| freq.as_mut_ptr());
    unsafe {
        probe_public_libsais64_omp_freq(
            t.as_ptr(),
            sa.as_mut_ptr().cast::<SaSint>(),
            n,
            fs,
            freq_ptr,
            threads.max(1),
        )
    }
}

/// Constructs the generalized suffix array (GSA) of given string set.
///
/// # Arguments
/// - `t` (`[0..n-1]`): The input string set using 0 as separators (`t[n-1]` must be 0).
/// - `sa` (`[0..n-1+fs]`): The output array of suffixes.
/// - `fs`: The extra space available at the end of SA array (0 should be enough for most cases).
/// - `freq` (`[0..255]`): The output symbol frequency table (can be `None`).
///
/// # Returns
/// 0 if no error occurred, -1 or -2 otherwise.
pub fn libsais64_gsa(
    t: &[u8],
    sa: &mut [SaSint],
    fs: SaSint,
    freq: Option<&mut [SaSint]>,
) -> SaSint {
    if fs < 0
        || sa.len()
            < t.len()
                .saturating_add(usize::try_from(fs).unwrap_or(usize::MAX))
    {
        return -1;
    }
    if let Some(freq) = freq.as_ref() {
        if freq.len() < ALPHABET_SIZE {
            return -1;
        }
    }

    let n = t.len();
    if n > 0 && t[n - 1] != 0 {
        return -1;
    }

    if n <= 1 {
        if let Some(freq) = freq {
            freq[..ALPHABET_SIZE].fill(0);
            if n == 1 {
                freq[t[0] as usize] += 1;
            }
        }
        if n == 1 {
            sa[0] = 0;
        }
        return 0;
    }

    if n <= i32::MAX as usize {
        return libsais64_run_32bit_omp(t, sa, fs, freq, 1, true)
            .expect("n <= INT32_MAX must have 32-bit workspace");
    }

    libsais64_main(t, sa, LIBSAIS_FLAGS_GSA, 0, None, fs, freq, 1)
}

/// Alias for `libsais64_long`. See its documentation.
pub fn libsais64_int(t: &mut [SaSint], sa: &mut [SaSint], k: SaSint, fs: SaSint) -> SaSint {
    if fs < 0
        || sa.len()
            < t.len()
                .saturating_add(usize::try_from(fs).unwrap_or(usize::MAX))
    {
        return -1;
    }

    if t.len() <= 1 {
        if t.len() == 1 {
            sa[0] = 0;
        }
        return 0;
    }

    libsais64_main_int(t, sa, k, fs, 1)
}

/// Constructs the suffix array of a given integer array.
///
/// Note, during construction input array will be modified, but restored at the end if no errors occurred.
///
/// # Arguments
/// - `t` (`[0..n-1]`): The input integer array.
/// - `sa` (`[0..n-1+fs]`): The output array of suffixes.
/// - `k`: The alphabet size of the input integer array.
/// - `fs`: Extra space available at the end of SA array (can be 0, but 4k or better 6k is recommended for optimal performance).
///
/// # Returns
/// 0 if no error occurred, -1 or -2 otherwise.
pub fn libsais64_long(t: &mut [SaSint], sa: &mut [SaSint], k: SaSint, fs: SaSint) -> SaSint {
    libsais64_int(t, sa, k, fs)
}

/// Constructs the suffix array of a given string using a libsais64 context.
///
/// - `ctx`: the libsais64 context.
/// - `t` (`[0..n-1]`): the input string.
/// - `sa` (`[0..n-1+fs]`): the output array of suffixes.
/// - `fs`: extra space available at the end of `sa` (0 should be enough for most cases).
/// - `freq` (`[0..255]`): optional output symbol frequency table.
///
/// Returns 0 on success, -1 or -2 on error.
pub fn libsais64_ctx(
    ctx: &mut Context,
    t: &[u8],
    sa: &mut [SaSint],
    fs: SaSint,
    freq: Option<&mut [SaSint]>,
) -> SaSint {
    if fs < 0
        || sa.len()
            < t.len()
                .saturating_add(usize::try_from(fs).unwrap_or(usize::MAX))
    {
        return -1;
    }
    if let Some(freq) = freq.as_ref() {
        if freq.len() < ALPHABET_SIZE {
            return -1;
        }
    }

    let n = t.len();
    if n <= 1 {
        if let Some(freq) = freq {
            freq[..ALPHABET_SIZE].fill(0);
            if n == 1 {
                freq[t[0] as usize] += 1;
            }
        }
        if n == 1 {
            sa[0] = 0;
        }
        return 0;
    }

    libsais64_main_ctx(ctx, t, sa, LIBSAIS_FLAGS_NONE, 0, None, fs, freq)
}

/// Constructs the generalized suffix array (GSA) of a given string set using a libsais64 context.
///
/// - `ctx`: the libsais64 context.
/// - `t` (`[0..n-1]`): the input string set using 0 as separators (`t[n-1]` must be 0).
/// - `sa` (`[0..n-1+fs]`): the output array of suffixes.
/// - `fs`: extra space available at the end of `sa` (0 should be enough for most cases).
/// - `freq` (`[0..255]`): optional output symbol frequency table.
///
/// Returns 0 on success, -1 or -2 on error.
pub fn libsais64_gsa_ctx(
    ctx: &mut Context,
    t: &[u8],
    sa: &mut [SaSint],
    fs: SaSint,
    freq: Option<&mut [SaSint]>,
) -> SaSint {
    if fs < 0
        || sa.len()
            < t.len()
                .saturating_add(usize::try_from(fs).unwrap_or(usize::MAX))
    {
        return -1;
    }
    if let Some(freq) = freq.as_ref() {
        if freq.len() < ALPHABET_SIZE {
            return -1;
        }
    }

    let n = t.len();
    if n > 0 && t[n - 1] != 0 {
        return -1;
    }

    if n <= 1 {
        if let Some(freq) = freq {
            freq[..ALPHABET_SIZE].fill(0);
            if n == 1 {
                freq[t[0] as usize] += 1;
            }
        }
        if n == 1 {
            sa[0] = 0;
        }
        return 0;
    }

    libsais64_main_ctx(ctx, t, sa, LIBSAIS_FLAGS_GSA, 0, None, fs, freq)
}

/// Constructs the burrows-wheeler transformed string (BWT) of a given string.
///
/// # Arguments
/// - `t` (`[0..n-1]`): The input string.
/// - `u` (`[0..n-1]`): The output string (can be `t`).
/// - `a` (`[0..n-1+fs]`): The temporary array.
/// - `fs`: The extra space available at the end of A array (0 should be enough for most cases).
/// - `freq` (`[0..255]`): The output symbol frequency table (can be `None`).
///
/// # Returns
/// The primary index if no error occurred, -1 or -2 otherwise.
pub fn libsais64_bwt(
    t: &[u8],
    u: &mut [u8],
    a: &mut [SaSint],
    fs: SaSint,
    freq: Option<&mut [SaSint]>,
) -> SaSint {
    if fs < 0
        || u.len() < t.len()
        || a.len()
            < t.len()
                .saturating_add(usize::try_from(fs).unwrap_or(usize::MAX))
    {
        return -1;
    }
    if let Some(freq) = freq.as_ref() {
        if freq.len() < ALPHABET_SIZE {
            return -1;
        }
    }

    let n = t.len();
    if n <= 1 {
        if let Some(freq) = freq {
            freq[..ALPHABET_SIZE].fill(0);
            if n == 1 {
                u[0] = t[0];
                freq[t[0] as usize] += 1;
            }
        } else if n == 1 {
            u[0] = t[0];
        }
        return n as SaSint;
    }

    if n <= i32::MAX as usize {
        return libsais64_bwt_run_32bit_omp(t, u, fs, freq, 1)
            .expect("n <= INT32_MAX must have 32-bit workspace");
    }

    let mut index = libsais64_main(t, a, LIBSAIS_FLAGS_BWT, 0, None, fs, freq, 1);
    if index >= 0 {
        index += 1;
        let split = usize::try_from(index).expect("index must be non-negative");
        u[0] = t[n - 1];
        bwt_copy_8u_omp(&mut u[1..split], &a[..split - 1], index - 1, 1);
        bwt_copy_8u_omp(
            &mut u[split..n],
            &a[split..n],
            SaSint::try_from(n - split).expect("fits"),
            1,
        );
    }
    index
}

/// Constructs the burrows-wheeler transformed string (BWT) of a given string with auxiliary indexes.
///
/// # Arguments
/// - `t` (`[0..n-1]`): The input string.
/// - `u` (`[0..n-1]`): The output string (can be `t`).
/// - `a` (`[0..n-1+fs]`): The temporary array.
/// - `fs`: The extra space available at the end of A array (0 should be enough for most cases).
/// - `freq` (`[0..255]`): The output symbol frequency table (can be `None`).
/// - `r`: The sampling rate for auxiliary indexes (must be power of 2).
/// - `i` (`[0..(n-1)/r]`): The output auxiliary indexes.
///
/// # Returns
/// 0 if no error occurred, -1 or -2 otherwise.
pub fn libsais64_bwt_aux(
    t: &[u8],
    u: &mut [u8],
    a: &mut [SaSint],
    fs: SaSint,
    freq: Option<&mut [SaSint]>,
    r: SaSint,
    i: &mut [SaSint],
) -> SaSint {
    let n = t.len();
    if fs < 0
        || r < 2
        || (r & (r - 1)) != 0
        || u.len() < n
        || a.len() < n.saturating_add(usize::try_from(fs).unwrap_or(usize::MAX))
        || freq.as_ref().is_some_and(|freq| freq.len() < ALPHABET_SIZE)
    {
        return -1;
    }
    let sample_count = if n == 0 {
        1
    } else {
        usize::try_from((SaSint::try_from(n).expect("input length must fit SaSint") - 1) / r)
            .expect("sample count must be non-negative")
            + 1
    };
    if i.len() < sample_count {
        return -1;
    }

    if n <= 1 {
        if let Some(freq) = freq {
            freq[..ALPHABET_SIZE].fill(0);
            if n == 1 {
                u[0] = t[0];
                freq[t[0] as usize] += 1;
            }
        } else if n == 1 {
            u[0] = t[0];
        }
        i[0] = n as SaSint;
        return 0;
    }

    if n <= i32::MAX as usize && r <= i32::MAX as SaSint {
        return libsais64_bwt_aux_run_32bit_omp(t, u, fs, freq, r, i, 1)
            .expect("n/r <= INT32_MAX must have 32-bit workspace");
    }

    let index = libsais64_main(t, a, LIBSAIS_FLAGS_BWT, r, Some(i), fs, freq, 1);
    if index == 0 {
        let split = usize::try_from(i[0]).expect("primary index must be non-negative");
        u[0] = t[n - 1];
        bwt_copy_8u_omp(&mut u[1..split], &a[..split - 1], i[0] - 1, 1);
        bwt_copy_8u_omp(
            &mut u[split..n],
            &a[split..n],
            SaSint::try_from(n - split).expect("fits"),
            1,
        );
    }
    index
}

/// Constructs the Burrows-Wheeler transformed string (BWT) of a given string using a libsais64 context.
///
/// - `ctx`: the libsais64 context.
/// - `t` (`[0..n-1]`): the input string.
/// - `u` (`[0..n-1]`): the output string (can alias `t`).
/// - `a` (`[0..n-1+fs]`): the temporary array.
/// - `fs`: extra space available at the end of `a` (0 should be enough for most cases).
/// - `freq` (`[0..255]`): optional output symbol frequency table.
///
/// Returns the primary index on success, -1 or -2 on error.
pub fn libsais64_bwt_ctx(
    ctx: &mut Context,
    t: &[u8],
    u: &mut [u8],
    a: &mut [SaSint],
    fs: SaSint,
    freq: Option<&mut [SaSint]>,
) -> SaSint {
    if fs < 0
        || u.len() < t.len()
        || a.len()
            < t.len()
                .saturating_add(usize::try_from(fs).unwrap_or(usize::MAX))
    {
        return -1;
    }
    if let Some(freq) = freq.as_ref() {
        if freq.len() < ALPHABET_SIZE {
            return -1;
        }
    }

    let n = t.len();
    if n <= 1 {
        if let Some(freq) = freq {
            freq[..ALPHABET_SIZE].fill(0);
            if n == 1 {
                u[0] = t[0];
                freq[t[0] as usize] += 1;
            }
        } else if n == 1 {
            u[0] = t[0];
        }
        return n as SaSint;
    }

    let mut index = libsais64_main_ctx(ctx, t, a, LIBSAIS_FLAGS_BWT, 0, None, fs, freq);
    if index >= 0 {
        index += 1;
        let split = usize::try_from(index).expect("index must be non-negative");
        u[0] = t[n - 1];
        bwt_copy_8u_omp(
            &mut u[1..split],
            &a[..split - 1],
            index - 1,
            ctx.threads as SaSint,
        );
        bwt_copy_8u_omp(
            &mut u[split..n],
            &a[split..n],
            SaSint::try_from(n - split).expect("fits"),
            ctx.threads as SaSint,
        );
    }
    index
}

/// Constructs the BWT of a given string with auxiliary indexes using a libsais64 context.
///
/// - `ctx`: the libsais64 context.
/// - `t` (`[0..n-1]`): the input string.
/// - `u` (`[0..n-1]`): the output string (can alias `t`).
/// - `a` (`[0..n-1+fs]`): the temporary array.
/// - `fs`: extra space available at the end of `a` (0 should be enough for most cases).
/// - `freq` (`[0..255]`): optional output symbol frequency table.
/// - `r`: sampling rate for the auxiliary indexes (must be a power of two).
/// - `i` (`[0..(n-1)/r]`): output auxiliary indexes.
///
/// Returns 0 on success, -1 or -2 on error.
pub fn libsais64_bwt_aux_ctx(
    ctx: &mut Context,
    t: &[u8],
    u: &mut [u8],
    a: &mut [SaSint],
    fs: SaSint,
    freq: Option<&mut [SaSint]>,
    r: SaSint,
    i: &mut [SaSint],
) -> SaSint {
    let n = t.len();
    if fs < 0
        || r < 2
        || (r & (r - 1)) != 0
        || u.len() < n
        || a.len() < n.saturating_add(usize::try_from(fs).unwrap_or(usize::MAX))
    {
        return -1;
    }
    if let Some(freq) = freq.as_ref() {
        if freq.len() < ALPHABET_SIZE {
            return -1;
        }
    }
    let sample_count = if n == 0 {
        1
    } else {
        usize::try_from((SaSint::try_from(n).expect("input length must fit SaSint") - 1) / r)
            .expect("sample count must be non-negative")
            + 1
    };
    if i.len() < sample_count {
        return -1;
    }

    if n <= 1 {
        if let Some(freq) = freq {
            freq[..ALPHABET_SIZE].fill(0);
            if n == 1 {
                u[0] = t[0];
                freq[t[0] as usize] += 1;
            }
        } else if n == 1 {
            u[0] = t[0];
        }
        i[0] = n as SaSint;
        return 0;
    }

    let index = libsais64_main_ctx(ctx, t, a, LIBSAIS_FLAGS_BWT, r, Some(i), fs, freq);
    if index == 0 {
        let split = usize::try_from(i[0]).expect("primary index must be non-negative");
        u[0] = t[n - 1];
        bwt_copy_8u_omp(
            &mut u[1..split],
            &a[..split - 1],
            i[0] - 1,
            ctx.threads as SaSint,
        );
        bwt_copy_8u_omp(
            &mut u[split..n],
            &a[split..n],
            SaSint::try_from(n - split).expect("fits"),
            ctx.threads as SaSint,
        );
    }
    index
}

/// Creates the libsais64 context for parallel operations using OpenMP-style threading.
///
/// In multi-threaded environments, use one context per thread for parallel executions.
///
/// - `threads`: number of worker threads (can be 0 for the implementation default).
///
/// Returns the context, or `None` on allocation failure.
pub fn create_ctx_omp(threads: SaSint) -> Option<Context> {
    if threads < 0 {
        return None;
    }

    create_ctx_main(normalize_omp_threads(threads))
}

fn libsais64_new_32bit_fs(n: usize, fs: SaSint) -> Option<i32> {
    if n > i32::MAX as usize {
        return None;
    }

    let n = n as SaSint;
    let int32_max = i32::MAX as SaSint;
    let expanded_space = fs as i128 + fs as i128 + n as i128 + n as i128;
    let new_fs = if expanded_space <= int32_max as i128 {
        fs + fs + n
    } else {
        int32_max - n
    };

    i32::try_from(new_fs).ok()
}

fn libsais64_run_32bit_omp(
    t: &[u8],
    sa: &mut [SaSint],
    fs: SaSint,
    freq: Option<&mut [SaSint]>,
    threads: SaSint,
    gsa: bool,
) -> Option<SaSint> {
    let new_fs = libsais64_new_32bit_fs(t.len(), fs)?;
    let mut sa32 = vec![0i32; t.len() + usize::try_from(new_fs).expect("new_fs is non-negative")];

    let index = if let Some(freq) = freq {
        let mut freq32 = vec![0i32; ALPHABET_SIZE];
        let index = if gsa {
            crate::libsais_gsa_omp(t, &mut sa32, new_fs, Some(&mut freq32), threads as i32)
        } else {
            crate::libsais_omp(t, &mut sa32, new_fs, Some(&mut freq32), threads as i32)
        };
        if index >= 0 {
            for (dst, src) in freq.iter_mut().zip(freq32.iter()) {
                *dst = SaSint::from(*src);
            }
        }
        index
    } else if gsa {
        crate::libsais_gsa_omp(t, &mut sa32, new_fs, None, threads as i32)
    } else {
        crate::libsais_omp(t, &mut sa32, new_fs, None, threads as i32)
    };

    if index >= 0 {
        for (dst, src) in sa.iter_mut().zip(sa32.iter()).take(t.len()) {
            *dst = SaSint::from(*src as u32);
        }
    }

    Some(SaSint::from(index))
}

fn copy_freq32_to_64(freq: &mut [SaSint], freq32: &[i32]) {
    for (dst, src) in freq.iter_mut().zip(freq32.iter()).take(ALPHABET_SIZE) {
        *dst = SaSint::from(*src);
    }
}

fn libsais64_bwt_run_32bit_omp(
    t: &[u8],
    u: &mut [u8],
    fs: SaSint,
    freq: Option<&mut [SaSint]>,
    threads: SaSint,
) -> Option<SaSint> {
    let new_fs = libsais64_new_32bit_fs(t.len(), fs)?;
    let mut a32 = vec![0i32; t.len() + usize::try_from(new_fs).expect("new_fs is non-negative")];

    let index = if let Some(freq) = freq {
        let mut freq32 = vec![0i32; ALPHABET_SIZE];
        let index =
            crate::libsais_bwt_omp(t, u, &mut a32, new_fs, Some(&mut freq32), threads as i32);
        if index >= 0 {
            copy_freq32_to_64(freq, &freq32);
        }
        index
    } else {
        crate::libsais_bwt_omp(t, u, &mut a32, new_fs, None, threads as i32)
    };

    Some(SaSint::from(index))
}

fn libsais64_bwt_aux_run_32bit_omp(
    t: &[u8],
    u: &mut [u8],
    fs: SaSint,
    freq: Option<&mut [SaSint]>,
    r: SaSint,
    i: &mut [SaSint],
    threads: SaSint,
) -> Option<SaSint> {
    if r > i32::MAX as SaSint {
        return None;
    }

    let new_fs = libsais64_new_32bit_fs(t.len(), fs)?;
    let mut a32 = vec![0i32; t.len() + usize::try_from(new_fs).expect("new_fs is non-negative")];
    let sample_count = if t.is_empty() {
        1
    } else {
        (t.len() - 1) / usize::try_from(r).expect("r must be positive") + 1
    };
    let mut i32_out = vec![0i32; sample_count];

    let index = if let Some(freq) = freq {
        let mut freq32 = vec![0i32; ALPHABET_SIZE];
        let index = crate::libsais_bwt_aux_omp(
            t,
            u,
            &mut a32,
            new_fs,
            Some(&mut freq32),
            r as i32,
            &mut i32_out,
            threads as i32,
        );
        if index >= 0 {
            copy_freq32_to_64(freq, &freq32);
        }
        index
    } else {
        crate::libsais_bwt_aux_omp(
            t,
            u,
            &mut a32,
            new_fs,
            None,
            r as i32,
            &mut i32_out,
            threads as i32,
        )
    };

    if index >= 0 {
        for (dst, src) in i.iter_mut().zip(i32_out.iter()).take(sample_count) {
            *dst = SaSint::from(*src);
        }
    }

    Some(SaSint::from(index))
}

/// Constructs the suffix array of a given string in parallel using OpenMP.
///
/// # Arguments
/// - `t` (`[0..n-1]`): The input string.
/// - `sa` (`[0..n-1+fs]`): The output array of suffixes.
/// - `fs`: The extra space available at the end of SA array (0 should be enough for most cases).
/// - `freq` (`[0..255]`): The output symbol frequency table (can be `None`).
/// - `threads`: The number of OpenMP threads to use (can be 0 for OpenMP default).
///
/// # Returns
/// 0 if no error occurred, -1 or -2 otherwise.
pub fn libsais64_omp(
    t: &[u8],
    sa: &mut [SaSint],
    fs: SaSint,
    freq: Option<&mut [SaSint]>,
    threads: SaSint,
) -> SaSint {
    if threads < 0 {
        return -1;
    }
    if fs < 0
        || sa.len()
            < t.len()
                .saturating_add(usize::try_from(fs).unwrap_or(usize::MAX))
    {
        return -1;
    }
    if let Some(freq) = freq.as_ref() {
        if freq.len() < ALPHABET_SIZE {
            return -1;
        }
    }
    let n = t.len();
    if n <= 1 {
        if let Some(freq) = freq {
            freq[..ALPHABET_SIZE].fill(0);
            if n == 1 {
                sa[0] = 0;
                freq[t[0] as usize] += 1;
            }
        } else if n == 1 {
            sa[0] = 0;
        }
        return 0;
    }

    let threads = normalize_omp_threads(threads);
    if n <= i32::MAX as usize {
        return libsais64_run_32bit_omp(t, sa, fs, freq, threads, false)
            .expect("n <= INT32_MAX must have 32-bit workspace");
    }

    libsais64_main(t, sa, LIBSAIS_FLAGS_NONE, 0, None, fs, freq, threads)
}

/// Constructs the generalized suffix array (GSA) of given string set in parallel using OpenMP.
///
/// # Arguments
/// - `t` (`[0..n-1]`): The input string set using 0 as separators (`t[n-1]` must be 0).
/// - `sa` (`[0..n-1+fs]`): The output array of suffixes.
/// - `fs`: The extra space available at the end of SA array (0 should be enough for most cases).
/// - `freq` (`[0..255]`): The output symbol frequency table (can be `None`).
/// - `threads`: The number of OpenMP threads to use (can be 0 for OpenMP default).
///
/// # Returns
/// 0 if no error occurred, -1 or -2 otherwise.
pub fn libsais64_gsa_omp(
    t: &[u8],
    sa: &mut [SaSint],
    fs: SaSint,
    freq: Option<&mut [SaSint]>,
    threads: SaSint,
) -> SaSint {
    if threads < 0
        || t.last().copied().unwrap_or(0) != 0
        || fs < 0
        || sa.len()
            < t.len()
                .saturating_add(usize::try_from(fs).unwrap_or(usize::MAX))
    {
        return -1;
    }
    if let Some(freq) = freq.as_ref() {
        if freq.len() < ALPHABET_SIZE {
            return -1;
        }
    }
    let n = t.len();
    if n <= 1 {
        if let Some(freq) = freq {
            freq[..ALPHABET_SIZE].fill(0);
            if n == 1 {
                sa[0] = 0;
                freq[t[0] as usize] += 1;
            }
        } else if n == 1 {
            sa[0] = 0;
        }
        return 0;
    }

    let threads = normalize_omp_threads(threads);
    if n <= i32::MAX as usize {
        return libsais64_run_32bit_omp(t, sa, fs, freq, threads, true)
            .expect("n <= INT32_MAX must have 32-bit workspace");
    }

    libsais64_main(t, sa, LIBSAIS_FLAGS_GSA, 0, None, fs, freq, threads)
}

/// Alias for `libsais64_long_omp`. See its documentation.
pub fn libsais64_int_omp(
    t: &mut [SaSint],
    sa: &mut [SaSint],
    k: SaSint,
    fs: SaSint,
    threads: SaSint,
) -> SaSint {
    if threads < 0 {
        return -1;
    }
    if fs < 0
        || sa.len()
            < t.len()
                .saturating_add(usize::try_from(fs).unwrap_or(usize::MAX))
    {
        return -1;
    }
    if t.len() <= 1 {
        if t.len() == 1 {
            sa[0] = 0;
        }
        return 0;
    }

    libsais64_main_int(t, sa, k, fs, normalize_omp_threads(threads))
}

/// Constructs the suffix array of a given integer array in parallel using OpenMP.
///
/// Note, during construction input array will be modified, but restored at the end if no errors occurred.
///
/// # Arguments
/// - `t` (`[0..n-1]`): The input integer array.
/// - `sa` (`[0..n-1+fs]`): The output array of suffixes.
/// - `k`: The alphabet size of the input integer array.
/// - `fs`: Extra space available at the end of SA array (can be 0, but 4k or better 6k is recommended for optimal performance).
/// - `threads`: The number of OpenMP threads to use (can be 0 for OpenMP default).
///
/// # Returns
/// 0 if no error occurred, -1 or -2 otherwise.
pub fn libsais64_long_omp(
    t: &mut [SaSint],
    sa: &mut [SaSint],
    k: SaSint,
    fs: SaSint,
    threads: SaSint,
) -> SaSint {
    libsais64_int_omp(t, sa, k, fs, threads)
}

/// Constructs the burrows-wheeler transformed string (BWT) of a given string in parallel using OpenMP.
///
/// # Arguments
/// - `t` (`[0..n-1]`): The input string.
/// - `u` (`[0..n-1]`): The output string (can be `t`).
/// - `a` (`[0..n-1+fs]`): The temporary array.
/// - `fs`: The extra space available at the end of A array (0 should be enough for most cases).
/// - `freq` (`[0..255]`): The output symbol frequency table (can be `None`).
/// - `threads`: The number of OpenMP threads to use (can be 0 for OpenMP default).
///
/// # Returns
/// The primary index if no error occurred, -1 or -2 otherwise.
pub fn libsais64_bwt_omp(
    t: &[u8],
    u: &mut [u8],
    a: &mut [SaSint],
    fs: SaSint,
    freq: Option<&mut [SaSint]>,
    threads: SaSint,
) -> SaSint {
    let n = t.len();
    if threads < 0
        || fs < 0
        || u.len() < n
        || a.len() < n.saturating_add(usize::try_from(fs).unwrap_or(usize::MAX))
        || freq.as_ref().is_some_and(|freq| freq.len() < ALPHABET_SIZE)
    {
        return -1;
    }

    if n <= 1 {
        if let Some(freq) = freq {
            freq[..ALPHABET_SIZE].fill(0);
            if n == 1 {
                u[0] = t[0];
                freq[t[0] as usize] += 1;
            }
        } else if n == 1 {
            u[0] = t[0];
        }
        return n as SaSint;
    }

    let threads = normalize_omp_threads(threads);
    if n <= i32::MAX as usize {
        return libsais64_bwt_run_32bit_omp(t, u, fs, freq, threads)
            .expect("n <= INT32_MAX must have 32-bit workspace");
    }

    let mut index = libsais64_main(t, a, LIBSAIS_FLAGS_BWT, 0, None, fs, freq, threads);
    if index >= 0 {
        index += 1;
        let index_usize = usize::try_from(index).expect("index must be non-negative");
        u[0] = t[n - 1];
        bwt_copy_8u_omp(
            &mut u[1..index_usize],
            &a[..index_usize - 1],
            index - 1,
            threads,
        );
        bwt_copy_8u_omp(
            &mut u[index_usize..n],
            &a[index_usize..n],
            SaSint::try_from(n - index_usize).expect("fits"),
            threads,
        );
    }
    index
}

/// Constructs the burrows-wheeler transformed string (BWT) of a given string with auxiliary indexes in parallel using OpenMP.
///
/// # Arguments
/// - `t` (`[0..n-1]`): The input string.
/// - `u` (`[0..n-1]`): The output string (can be `t`).
/// - `a` (`[0..n-1+fs]`): The temporary array.
/// - `fs`: The extra space available at the end of A array (0 should be enough for most cases).
/// - `freq` (`[0..255]`): The output symbol frequency table (can be `None`).
/// - `r`: The sampling rate for auxiliary indexes (must be power of 2).
/// - `i` (`[0..(n-1)/r]`): The output auxiliary indexes.
/// - `threads`: The number of OpenMP threads to use (can be 0 for OpenMP default).
///
/// # Returns
/// 0 if no error occurred, -1 or -2 otherwise.
pub fn libsais64_bwt_aux_omp(
    t: &[u8],
    u: &mut [u8],
    a: &mut [SaSint],
    fs: SaSint,
    freq: Option<&mut [SaSint]>,
    r: SaSint,
    i: &mut [SaSint],
    threads: SaSint,
) -> SaSint {
    let n = t.len();
    if threads < 0
        || fs < 0
        || r < 2
        || (r & (r - 1)) != 0
        || u.len() < n
        || a.len() < n.saturating_add(usize::try_from(fs).unwrap_or(usize::MAX))
    {
        return -1;
    }
    if let Some(freq) = freq.as_ref() {
        if freq.len() < ALPHABET_SIZE {
            return -1;
        }
    }
    let sample_count = if n == 0 {
        1
    } else {
        usize::try_from((SaSint::try_from(n).expect("input length must fit SaSint") - 1) / r)
            .expect("sample count must be non-negative")
            + 1
    };
    if i.len() < sample_count {
        return -1;
    }
    if n <= 1 {
        if let Some(freq) = freq {
            freq[..ALPHABET_SIZE].fill(0);
            if n == 1 {
                u[0] = t[0];
                freq[t[0] as usize] += 1;
            }
        } else if n == 1 {
            u[0] = t[0];
        }
        i[0] = n as SaSint;
        return 0;
    }

    let threads = normalize_omp_threads(threads);
    if n <= i32::MAX as usize && r <= i32::MAX as SaSint {
        return libsais64_bwt_aux_run_32bit_omp(t, u, fs, freq, r, i, threads)
            .expect("n/r <= INT32_MAX must have 32-bit workspace");
    }

    let index = libsais64_main(t, a, LIBSAIS_FLAGS_BWT, r, Some(i), fs, freq, threads);
    if index == 0 {
        let split = usize::try_from(i[0]).expect("primary index must be non-negative");
        u[0] = t[n - 1];
        bwt_copy_8u_omp(&mut u[1..split], &a[..split - 1], i[0] - 1, threads);
        bwt_copy_8u_omp(
            &mut u[split..n],
            &a[split..n],
            SaSint::try_from(n - split).expect("fits"),
            threads,
        );
    }
    index
}

/// Internal helper: compute phi.
#[doc(hidden)]
pub fn compute_phi(
    sa: &[SaSint],
    plcp: &mut [SaSint],
    n: SaSint,
    omp_block_start: FastSint,
    omp_block_size: FastSint,
) {
    let start = usize::try_from(omp_block_start).expect("omp_block_start must be non-negative");
    let size = usize::try_from(omp_block_size).expect("omp_block_size must be non-negative");
    let end = start + size;
    let n_usize = usize::try_from(n).expect("n must be non-negative");
    let mut i = start;
    let mut k = if omp_block_start > 0 {
        sa[start - 1]
    } else {
        n
    };

    let fast_end = omp_block_start + omp_block_size - 64 - 3;
    while (i as FastSint) < fast_end {
        plcp[usize::try_from(sa[i]).expect("suffix index must be non-negative")] = k;
        k = sa[i];
        plcp[usize::try_from(sa[i + 1]).expect("suffix index must be non-negative")] = k;
        k = sa[i + 1];
        plcp[usize::try_from(sa[i + 2]).expect("suffix index must be non-negative")] = k;
        k = sa[i + 2];
        plcp[usize::try_from(sa[i + 3]).expect("suffix index must be non-negative")] = k;
        k = sa[i + 3];
        i += 4;
    }

    while i < end.min(n_usize) {
        plcp[usize::try_from(sa[i]).expect("suffix index must be non-negative")] = k;
        k = sa[i];
        i += 1;
    }
}

/// Internal helper: compute phi (OpenMP variant).
#[doc(hidden)]
pub fn compute_phi_omp(sa: &[SaSint], plcp: &mut [SaSint], n: SaSint, threads: SaSint) {
    if threads == 1 || n < 65_536 {
        compute_phi(sa, plcp, n, 0, n as FastSint);
        return;
    }

    let threads_usize = usize::try_from(threads).expect("threads must be non-negative");
    let block_stride = ((n as FastSint) / (threads as FastSint)) & !15;
    let plcp_addr = plcp.as_mut_ptr() as usize;
    let n_usize = usize::try_from(n).expect("n must be non-negative");

    run_rayon_with_threads(threads_usize, || {
        (0..threads_usize).into_par_iter().for_each(|thread| {
            let block_start = thread as FastSint * block_stride;
            let block_size = if thread + 1 < threads_usize {
                block_stride
            } else {
                n as FastSint - block_start
            };
            let start = usize::try_from(block_start).expect("omp_block_start must be non-negative");
            let size = usize::try_from(block_size).expect("omp_block_size must be non-negative");
            let end = start + size;
            let mut i = start;
            let mut k = if block_start > 0 { sa[start - 1] } else { n };
            let plcp_ptr = plcp_addr as *mut SaSint;

            let fast_end = block_start + block_size - 64 - 3;
            while (i as FastSint) < fast_end {
                unsafe {
                    // SA is a suffix-array permutation, so each thread writes a disjoint PLCP slot.
                    *plcp_ptr
                        .add(usize::try_from(sa[i]).expect("suffix index must be non-negative")) =
                        k;
                    k = sa[i];
                    *plcp_ptr.add(
                        usize::try_from(sa[i + 1]).expect("suffix index must be non-negative"),
                    ) = k;
                    k = sa[i + 1];
                    *plcp_ptr.add(
                        usize::try_from(sa[i + 2]).expect("suffix index must be non-negative"),
                    ) = k;
                    k = sa[i + 2];
                    *plcp_ptr.add(
                        usize::try_from(sa[i + 3]).expect("suffix index must be non-negative"),
                    ) = k;
                    k = sa[i + 3];
                }
                i += 4;
            }

            while i < end.min(n_usize) {
                unsafe {
                    // SA is a suffix-array permutation, so each thread writes a disjoint PLCP slot.
                    *plcp_ptr
                        .add(usize::try_from(sa[i]).expect("suffix index must be non-negative")) =
                        k;
                }
                k = sa[i];
                i += 1;
            }
        });
    });
}

/// Internal helper: compute plcp.
#[doc(hidden)]
pub fn compute_plcp(
    t: &[u8],
    plcp: &mut [SaSint],
    n: FastSint,
    omp_block_start: FastSint,
    omp_block_size: FastSint,
) {
    let start = usize::try_from(omp_block_start).expect("omp_block_start must be non-negative");
    let size = usize::try_from(omp_block_size).expect("omp_block_size must be non-negative");
    let end = start + size;
    let n_usize = usize::try_from(n).expect("n must be non-negative");
    let mut l = 0usize;

    for i in start..end.min(n_usize) {
        let k = usize::try_from(plcp[i]).expect("phi entry must be non-negative");
        let m = n_usize - i.max(k);
        while l < m && t[i + l] == t[k + l] {
            l += 1;
        }
        plcp[i] = SaSint::try_from(l).expect("LCP length must fit SaSint");
        l = l.saturating_sub(1);
    }
}

/// Internal helper: compute plcp (OpenMP variant).
#[doc(hidden)]
pub fn compute_plcp_omp(t: &[u8], plcp: &mut [SaSint], n: SaSint, threads: SaSint) {
    if threads == 1 || n < 65_536 {
        compute_plcp(t, plcp, n as FastSint, 0, n as FastSint);
        return;
    }

    let n_usize = usize::try_from(n).expect("n must be non-negative");
    let threads_usize = usize::try_from(threads).expect("threads must be non-negative");
    let chunk_size = ((n_usize / threads_usize) & !15usize).max(16);
    run_rayon_with_threads(threads_usize, || {
        plcp[..n_usize]
            .par_chunks_mut(chunk_size)
            .enumerate()
            .for_each(|(chunk_index, chunk)| {
                let start = chunk_index * chunk_size;
                let mut l = 0usize;
                for (offset, value) in chunk.iter_mut().enumerate() {
                    let i = start + offset;
                    let k = usize::try_from(*value).expect("phi entry must be non-negative");
                    let m = n_usize - i.max(k);
                    while l < m && t[i + l] == t[k + l] {
                        l += 1;
                    }
                    *value = SaSint::try_from(l).expect("LCP length must fit SaSint");
                    l = l.saturating_sub(1);
                }
            });
    });
}

/// Internal helper: compute plcp gsa.
#[doc(hidden)]
pub fn compute_plcp_gsa(
    t: &[u8],
    plcp: &mut [SaSint],
    omp_block_start: FastSint,
    omp_block_size: FastSint,
) {
    let start = usize::try_from(omp_block_start).expect("omp_block_start must be non-negative");
    let size = usize::try_from(omp_block_size).expect("omp_block_size must be non-negative");
    let end = start + size;
    let mut l = 0usize;

    for i in start..end.min(t.len()) {
        let k = usize::try_from(plcp[i]).expect("phi entry must be non-negative");
        while t[i + l] > 0 && t[i + l] == t[k + l] {
            l += 1;
        }
        plcp[i] = SaSint::try_from(l).expect("LCP length must fit SaSint");
        l = l.saturating_sub(1);
    }
}

/// Internal helper: compute plcp gsa (OpenMP variant).
#[doc(hidden)]
pub fn compute_plcp_gsa_omp(t: &[u8], plcp: &mut [SaSint], n: SaSint, threads: SaSint) {
    if threads == 1 || n < 65_536 {
        compute_plcp_gsa(t, plcp, 0, n as FastSint);
        return;
    }

    let n_usize = usize::try_from(n).expect("n must be non-negative");
    let threads_usize = usize::try_from(threads).expect("threads must be non-negative");
    let chunk_size = ((n_usize / threads_usize) & !15usize).max(16);
    run_rayon_with_threads(threads_usize, || {
        plcp[..n_usize]
            .par_chunks_mut(chunk_size)
            .enumerate()
            .for_each(|(chunk_index, chunk)| {
                let start = chunk_index * chunk_size;
                let mut l = 0usize;
                for (offset, value) in chunk.iter_mut().enumerate() {
                    let i = start + offset;
                    let k = usize::try_from(*value).expect("phi entry must be non-negative");
                    while t[i + l] > 0 && t[i + l] == t[k + l] {
                        l += 1;
                    }
                    *value = SaSint::try_from(l).expect("LCP length must fit SaSint");
                    l = l.saturating_sub(1);
                }
            });
    });
}

/// Internal helper: compute plcp int.
#[doc(hidden)]
pub fn compute_plcp_int(
    t: &[SaSint],
    plcp: &mut [SaSint],
    n: FastSint,
    omp_block_start: FastSint,
    omp_block_size: FastSint,
) {
    let start = usize::try_from(omp_block_start).expect("omp_block_start must be non-negative");
    let size = usize::try_from(omp_block_size).expect("omp_block_size must be non-negative");
    let end = start + size;
    let n_usize = usize::try_from(n).expect("n must be non-negative");
    let mut l = 0usize;

    for i in start..end.min(n_usize) {
        let k = usize::try_from(plcp[i]).expect("phi entry must be non-negative");
        let m = n_usize - i.max(k);
        while l < m && t[i + l] == t[k + l] {
            l += 1;
        }
        plcp[i] = SaSint::try_from(l).expect("LCP length must fit SaSint");
        l = l.saturating_sub(1);
    }
}

/// Internal helper: compute plcp int (OpenMP variant).
#[doc(hidden)]
pub fn compute_plcp_int_omp(t: &[SaSint], plcp: &mut [SaSint], n: SaSint, threads: SaSint) {
    if threads == 1 || n < 65_536 {
        compute_plcp_int(t, plcp, n as FastSint, 0, n as FastSint);
        return;
    }

    let n_usize = usize::try_from(n).expect("n must be non-negative");
    let threads_usize = usize::try_from(threads).expect("threads must be non-negative");
    let chunk_size = ((n_usize / threads_usize) & !15usize).max(16);
    run_rayon_with_threads(threads_usize, || {
        plcp[..n_usize]
            .par_chunks_mut(chunk_size)
            .enumerate()
            .for_each(|(chunk_index, chunk)| {
                let start = chunk_index * chunk_size;
                let mut l = 0usize;
                for (offset, value) in chunk.iter_mut().enumerate() {
                    let i = start + offset;
                    let k = usize::try_from(*value).expect("phi entry must be non-negative");
                    let m = n_usize - i.max(k);
                    while l < m && t[i + l] == t[k + l] {
                        l += 1;
                    }
                    *value = SaSint::try_from(l).expect("LCP length must fit SaSint");
                    l = l.saturating_sub(1);
                }
            });
    });
}

/// Internal helper: compute lcp.
#[doc(hidden)]
pub fn compute_lcp(
    plcp: &[SaSint],
    sa: &[SaSint],
    lcp: &mut [SaSint],
    omp_block_start: FastSint,
    omp_block_size: FastSint,
) {
    let start = usize::try_from(omp_block_start).expect("omp_block_start must be non-negative");
    let size = usize::try_from(omp_block_size).expect("omp_block_size must be non-negative");
    let end = start + size;

    for i in start..end.min(sa.len()) {
        lcp[i] = plcp[usize::try_from(sa[i]).expect("suffix index must be non-negative")];
    }
}

/// Internal helper: compute lcp (OpenMP variant).
#[doc(hidden)]
pub fn compute_lcp_omp(
    plcp: &[SaSint],
    sa: &[SaSint],
    lcp: &mut [SaSint],
    n: SaSint,
    threads: SaSint,
) {
    if threads == 1 || n < 65_536 {
        compute_lcp(plcp, sa, lcp, 0, n as FastSint);
        return;
    }

    let n_usize = usize::try_from(n).expect("n must be non-negative");
    assert!(plcp.len() >= n_usize);
    assert!(sa.len() >= n_usize);
    assert!(lcp.len() >= n_usize);
    let threads_usize = usize::try_from(threads).expect("threads must be non-negative");
    let chunk_size = ((n_usize / threads_usize) & !15usize).max(16);
    let plcp_ptr = plcp.as_ptr() as usize;
    let sa_ptr = sa.as_ptr() as usize;
    run_rayon_with_threads(threads_usize, || {
        lcp[..n_usize]
            .par_chunks_mut(chunk_size)
            .enumerate()
            .for_each(|(chunk_index, chunk)| {
                let start = chunk_index * chunk_size;
                let dst_ptr = chunk.as_mut_ptr();
                let sa_ptr = sa_ptr as *const SaSint;
                let plcp_ptr = plcp_ptr as *const SaSint;
                for offset in 0..chunk.len() {
                    let i = start + offset;
                    let suffix = unsafe { *sa_ptr.add(i) };
                    let suffix =
                        usize::try_from(suffix).expect("suffix index must be non-negative");
                    assert!(suffix < plcp.len());
                    unsafe {
                        *dst_ptr.add(offset) = *plcp_ptr.add(suffix);
                    }
                }
            });
    });
}

/// Constructs the permuted longest common prefix array (PLCP) of a given string and a suffix array.
///
/// # Arguments
/// - `t` (`[0..n-1]`): The input string.
/// - `sa` (`[0..n-1]`): The input suffix array.
/// - `plcp` (`[0..n-1]`): The output permuted longest common prefix array.
///
/// # Returns
/// 0 if no error occurred, -1 otherwise.
pub fn libsais64_plcp(t: &[u8], sa: &[SaSint], plcp: &mut [SaSint]) -> SaSint {
    if sa.len() != t.len() || plcp.len() != t.len() {
        return -1;
    }
    if !suffix_entries_in_bounds(sa, t.len()) {
        return -1;
    }
    if t.len() <= 1 {
        if t.len() == 1 {
            plcp[0] = 0;
        }
        return 0;
    }

    let n = SaSint::try_from(t.len()).expect("input length must fit SaSint");
    compute_phi_omp(sa, plcp, n, 1);
    compute_plcp_omp(t, plcp, n, 1);
    0
}

/// Constructs the permuted longest common prefix array (PLCP) of a given string set and a generalized suffix array (GSA).
///
/// # Arguments
/// - `t` (`[0..n-1]`): The input string set using 0 as separators (`t[n-1]` must be 0).
/// - `sa` (`[0..n-1]`): The input generalized suffix array.
/// - `plcp` (`[0..n-1]`): The output permuted longest common prefix array.
///
/// # Returns
/// 0 if no error occurred, -1 otherwise.
pub fn libsais64_plcp_gsa(t: &[u8], sa: &[SaSint], plcp: &mut [SaSint]) -> SaSint {
    if t.last().copied().unwrap_or(0) != 0 {
        return -1;
    }
    if sa.len() != t.len() || plcp.len() != t.len() {
        return -1;
    }
    if !suffix_entries_in_bounds(sa, t.len()) {
        return -1;
    }
    if t.len() <= 1 {
        if t.len() == 1 {
            plcp[0] = 0;
        }
        return 0;
    }

    let n = SaSint::try_from(t.len()).expect("input length must fit SaSint");
    compute_phi_omp(sa, plcp, n, 1);
    compute_plcp_gsa_omp(t, plcp, n, 1);
    0
}

/// Constructs the PLCP of a given integer array and suffix array.
///
/// - `t` (`[0..n-1]`): the input integer array.
/// - `sa` (`[0..n-1]`): the input suffix array.
/// - `plcp` (`[0..n-1]`): the output permuted longest common prefix array.
///
/// Returns 0 on success, -1 on error.
pub fn libsais64_plcp_int(t: &[SaSint], sa: &[SaSint], plcp: &mut [SaSint]) -> SaSint {
    if sa.len() != t.len() || plcp.len() != t.len() {
        return -1;
    }
    if !suffix_entries_in_bounds(sa, t.len()) {
        return -1;
    }
    if t.len() <= 1 {
        if t.len() == 1 {
            plcp[0] = 0;
        }
        return 0;
    }

    let n = SaSint::try_from(t.len()).expect("input length must fit SaSint");
    compute_phi_omp(sa, plcp, n, 1);
    compute_plcp_int_omp(t, plcp, n, 1);
    0
}

/// Constructs the longest common prefix array (LCP) of a given permuted longest common prefix array (PLCP) and a suffix array.
///
/// # Arguments
/// - `plcp` (`[0..n-1]`): The input permuted longest common prefix array.
/// - `sa` (`[0..n-1]`): The input suffix array or generalized suffix array (GSA).
/// - `lcp` (`[0..n-1]`): The output longest common prefix array (can be `sa`).
///
/// # Returns
/// 0 if no error occurred, -1 otherwise.
pub fn libsais64_lcp(plcp: &[SaSint], sa: &[SaSint], lcp: &mut [SaSint]) -> SaSint {
    if plcp.len() != sa.len() || lcp.len() != sa.len() {
        return -1;
    }
    if !suffix_entries_in_bounds(sa, plcp.len()) {
        return -1;
    }
    if sa.len() <= 1 {
        if sa.len() == 1 {
            lcp[0] = plcp[usize::try_from(sa[0]).expect("suffix index must be non-negative")];
        }
        return 0;
    }

    compute_lcp_omp(
        plcp,
        sa,
        lcp,
        SaSint::try_from(sa.len()).expect("suffix array length must fit SaSint"),
        1,
    );
    0
}

/// Constructs the permuted longest common prefix array (PLCP) of a given string and a suffix array in parallel using OpenMP.
///
/// # Arguments
/// - `t` (`[0..n-1]`): The input string.
/// - `sa` (`[0..n-1]`): The input suffix array.
/// - `plcp` (`[0..n-1]`): The output permuted longest common prefix array.
/// - `threads`: The number of OpenMP threads to use (can be 0 for OpenMP default).
///
/// # Returns
/// 0 if no error occurred, -1 otherwise.
pub fn libsais64_plcp_omp(t: &[u8], sa: &[SaSint], plcp: &mut [SaSint], threads: SaSint) -> SaSint {
    if threads < 0 {
        return -1;
    }
    if sa.len() != t.len() || plcp.len() != t.len() {
        return -1;
    }
    if !suffix_entries_in_bounds(sa, t.len()) {
        return -1;
    }
    if t.len() <= 1 {
        if t.len() == 1 {
            plcp[0] = 0;
        }
        return 0;
    }

    let n = SaSint::try_from(t.len()).expect("input length must fit SaSint");
    let threads = normalize_omp_threads(threads);
    compute_phi_omp(sa, plcp, n, threads);
    compute_plcp_omp(t, plcp, n, threads);
    0
}

/// Constructs the permuted longest common prefix array (PLCP) of a given string set and a generalized suffix array (GSA) in parallel using OpenMP.
///
/// # Arguments
/// - `t` (`[0..n-1]`): The input string set using 0 as separators (`t[n-1]` must be 0).
/// - `sa` (`[0..n-1]`): The input generalized suffix array.
/// - `plcp` (`[0..n-1]`): The output permuted longest common prefix array.
/// - `threads`: The number of OpenMP threads to use (can be 0 for OpenMP default).
///
/// # Returns
/// 0 if no error occurred, -1 otherwise.
pub fn libsais64_plcp_gsa_omp(
    t: &[u8],
    sa: &[SaSint],
    plcp: &mut [SaSint],
    threads: SaSint,
) -> SaSint {
    if threads < 0 || t.last().copied().unwrap_or(0) != 0 {
        return -1;
    }
    if sa.len() != t.len() || plcp.len() != t.len() {
        return -1;
    }
    if !suffix_entries_in_bounds(sa, t.len()) {
        return -1;
    }
    if t.len() <= 1 {
        if t.len() == 1 {
            plcp[0] = 0;
        }
        return 0;
    }

    let n = SaSint::try_from(t.len()).expect("input length must fit SaSint");
    let threads = normalize_omp_threads(threads);
    compute_phi_omp(sa, plcp, n, threads);
    compute_plcp_gsa_omp(t, plcp, n, threads);
    0
}

/// Constructs the PLCP of a given integer array and suffix array in parallel using OpenMP-style threading.
///
/// - `t` (`[0..n-1]`): the input integer array.
/// - `sa` (`[0..n-1]`): the input suffix array.
/// - `plcp` (`[0..n-1]`): the output permuted longest common prefix array.
/// - `threads`: number of worker threads (can be 0 for the implementation default).
///
/// Returns 0 on success, -1 on error.
pub fn libsais64_plcp_int_omp(
    t: &[SaSint],
    sa: &[SaSint],
    plcp: &mut [SaSint],
    threads: SaSint,
) -> SaSint {
    if threads < 0 {
        return -1;
    }
    if sa.len() != t.len() || plcp.len() != t.len() {
        return -1;
    }
    if !suffix_entries_in_bounds(sa, t.len()) {
        return -1;
    }
    if t.len() <= 1 {
        if t.len() == 1 {
            plcp[0] = 0;
        }
        return 0;
    }

    let n = SaSint::try_from(t.len()).expect("input length must fit SaSint");
    let threads = normalize_omp_threads(threads);
    compute_phi_omp(sa, plcp, n, threads);
    compute_plcp_int_omp(t, plcp, n, threads);
    0
}

/// Constructs the longest common prefix array (LCP) of a given permuted longest common prefix array (PLCP) and a suffix array in parallel using OpenMP.
///
/// # Arguments
/// - `plcp` (`[0..n-1]`): The input permuted longest common prefix array.
/// - `sa` (`[0..n-1]`): The input suffix array or generalized suffix array (GSA).
/// - `lcp` (`[0..n-1]`): The output longest common prefix array (can be `sa`).
/// - `threads`: The number of OpenMP threads to use (can be 0 for OpenMP default).
///
/// # Returns
/// 0 if no error occurred, -1 otherwise.
pub fn libsais64_lcp_omp(
    plcp: &[SaSint],
    sa: &[SaSint],
    lcp: &mut [SaSint],
    threads: SaSint,
) -> SaSint {
    if threads < 0 {
        return -1;
    }
    if plcp.len() != sa.len() || lcp.len() != sa.len() {
        return -1;
    }
    if !suffix_entries_in_bounds(sa, plcp.len()) {
        return -1;
    }
    if sa.len() <= 1 {
        if sa.len() == 1 {
            lcp[0] = plcp[usize::try_from(sa[0]).expect("suffix index must be non-negative")];
        }
        return 0;
    }

    compute_lcp_omp(
        plcp,
        sa,
        lcp,
        SaSint::try_from(sa.len()).expect("suffix array length must fit SaSint"),
        normalize_omp_threads(threads),
    );
    0
}

fn suffix_entries_in_bounds(sa: &[SaSint], len: usize) -> bool {
    sa.iter()
        .all(|&value| usize::try_from(value).is_ok_and(|index| index < len))
}

/// Internal helper: unbwt compute histogram.
#[doc(hidden)]
pub fn unbwt_compute_histogram(t: &[u8], n: FastSint, count: &mut [SaUint]) {
    let n = usize::try_from(n).expect("n must be non-negative");
    assert!(count.len() >= ALPHABET_SIZE);
    for &byte in &t[..n] {
        count[byte as usize] += 1;
    }
}

/// Internal helper: unbwt transpose bucket2.
#[doc(hidden)]
pub fn unbwt_transpose_bucket2(bucket2: &mut [SaUint]) {
    assert!(bucket2.len() >= ALPHABET_SIZE * ALPHABET_SIZE);
    for x in 0..ALPHABET_SIZE {
        for y in x + 1..ALPHABET_SIZE {
            bucket2.swap((y << 8) + x, (x << 8) + y);
        }
    }
}

/// Internal helper: unbwt compute bigram histogram single.
#[doc(hidden)]
pub fn unbwt_compute_bigram_histogram_single(
    t: &[u8],
    bucket1: &mut [SaUint],
    bucket2: &mut [SaUint],
    index: FastUint,
) {
    let mut sum = 1usize;
    for c in 0..ALPHABET_SIZE {
        let prev = sum;
        sum += bucket1[c] as usize;
        bucket1[c] = prev as SaUint;
        if prev != sum {
            let bucket2_p = &mut bucket2[c << 8..(c + 1) << 8];

            let hi = sum.min(index);
            if hi > prev {
                unbwt_compute_histogram(&t[prev..], (hi - prev) as FastSint, bucket2_p);
            }

            let lo = prev.max(index + 1);
            if sum > lo {
                unbwt_compute_histogram(&t[lo - 1..], (sum - lo) as FastSint, bucket2_p);
            }
        }
    }

    unbwt_transpose_bucket2(bucket2);
}

/// Internal helper: unbwt calculate fastbits.
#[doc(hidden)]
pub fn unbwt_calculate_fastbits(
    bucket2: &mut [SaUint],
    fastbits: &mut [u16],
    lastc: FastUint,
    shift: FastUint,
) {
    let mut v = 0usize;
    let mut w = 0usize;
    let mut sum = 1usize;

    for c in 0..ALPHABET_SIZE {
        if c == lastc {
            sum += 1;
        }

        for _d in 0..ALPHABET_SIZE {
            let prev = sum;
            sum += bucket2[w] as usize;
            bucket2[w] = prev as SaUint;
            if prev != sum {
                while v <= ((sum - 1) >> shift) {
                    fastbits[v] = w as u16;
                    v += 1;
                }
            }
            w += 1;
        }
    }
}

/// Internal helper: unbwt calculate bi psi.
#[doc(hidden)]
pub fn unbwt_calculate_bi_psi(
    t: &[u8],
    p: &mut [SaUint],
    bucket1: &mut [SaUint],
    bucket2: &mut [SaUint],
    index: FastUint,
    omp_block_start: FastSint,
    omp_block_end: FastSint,
) {
    let mut i = usize::try_from(omp_block_start).expect("omp_block_start must be non-negative");
    let mut j = index;
    let block_end = usize::try_from(omp_block_end).expect("omp_block_end must be non-negative");
    if block_end < j {
        j = block_end;
    }
    while i < j {
        let c = t[i] as usize;
        let pidx = bucket1[c] as usize;
        bucket1[c] += 1;
        let tidx = index.wrapping_sub(pidx) as i64;
        if tidx != 0 {
            let src = pidx.wrapping_add((tidx >> 63) as usize);
            let w = ((t[src] as usize) << 8) + c;
            let dst = bucket2[w] as usize;
            p[dst] = i as SaUint;
            bucket2[w] += 1;
        }
        i += 1;
    }

    let mut i = index;
    if usize::try_from(omp_block_start).expect("omp_block_start must be non-negative") > i {
        i = usize::try_from(omp_block_start).expect("omp_block_start must be non-negative");
    }
    i += 1;
    while i <= block_end {
        let c = t[i - 1] as usize;
        let pidx = bucket1[c] as usize;
        bucket1[c] += 1;
        let tidx = index.wrapping_sub(pidx) as i64;
        if tidx != 0 {
            let src = pidx.wrapping_add((tidx >> 63) as usize);
            let w = ((t[src] as usize) << 8) + c;
            let dst = bucket2[w] as usize;
            p[dst] = i as SaUint;
            bucket2[w] += 1;
        }
        i += 1;
    }
}

/// Internal helper: unbwt calculate biPSI.
#[doc(hidden)]
#[allow(dead_code, non_snake_case)]
pub fn unbwt_calculate_biPSI(
    t: &[u8],
    p: &mut [SaUint],
    bucket1: &mut [SaUint],
    bucket2: &mut [SaUint],
    index: FastUint,
    omp_block_start: FastSint,
    omp_block_end: FastSint,
) {
    unbwt_calculate_bi_psi(
        t,
        p,
        bucket1,
        bucket2,
        index,
        omp_block_start,
        omp_block_end,
    );
}

/// Internal helper: unbwt init single.
#[doc(hidden)]
pub fn unbwt_init_single(
    t: &[u8],
    p: &mut [SaUint],
    n: SaSint,
    freq: Option<&[SaSint]>,
    i: &[SaUint],
    bucket2: &mut [SaUint],
    fastbits: &mut [u16],
) {
    let mut bucket1 = vec![0u64; ALPHABET_SIZE];
    let index = i[0] as usize;
    let lastc = t[0] as usize;
    let mut shift = 0usize;
    while (usize::try_from(n).expect("n must be non-negative") >> shift)
        > (1usize << UNBWT_FASTBITS)
    {
        shift += 1;
    }

    if let Some(freq) = freq {
        for c in 0..ALPHABET_SIZE {
            bucket1[c] = freq[c] as SaUint;
        }
    } else {
        unbwt_compute_histogram(t, n as FastSint, &mut bucket1);
    }

    bucket2.fill(0);
    unbwt_compute_bigram_histogram_single(t, &mut bucket1, bucket2, index);
    unbwt_calculate_fastbits(bucket2, fastbits, lastc, shift);
    unbwt_calculate_bi_psi(t, p, &mut bucket1, bucket2, index, 0, n as FastSint);
}

/// Internal helper: unbwt compute bigram histogram parallel.
#[doc(hidden)]
pub fn unbwt_compute_bigram_histogram_parallel(
    t: &[u8],
    index: FastUint,
    bucket1: &mut [SaUint],
    bucket2: &mut [SaUint],
    omp_block_start: FastSint,
    omp_block_size: FastSint,
) {
    let start = usize::try_from(omp_block_start).expect("omp_block_start must be non-negative");
    let end = start + usize::try_from(omp_block_size).expect("omp_block_size must be non-negative");
    for &c_u8 in &t[start..end] {
        let c = c_u8 as usize;
        let p = bucket1[c] as usize;
        bucket1[c] += 1;
        let tidx = index.wrapping_sub(p) as i64;
        if tidx != 0 {
            let src = p.wrapping_add((tidx >> 63) as usize);
            let w = ((t[src] as usize) << 8) + c;
            bucket2[w] += 1;
        }
    }
}

/// Internal helper: unbwt init parallel.
#[doc(hidden)]
pub fn unbwt_init_parallel(
    t: &[u8],
    p: &mut [SaUint],
    n: SaSint,
    freq: Option<&[SaSint]>,
    i: &[SaUint],
    bucket2: &mut [SaUint],
    fastbits: &mut [u16],
    buckets: Option<&mut [SaUint]>,
    threads: SaSint,
) {
    let num_threads = usize::try_from(threads.max(1)).expect("threads must be non-negative");
    if num_threads <= 1 || usize::try_from(n).expect("n must be non-negative") < 65_536 {
        unbwt_init_single(t, p, n, freq, i, bucket2, fastbits);
        return;
    }

    let buckets = match buckets {
        Some(buckets) => buckets,
        None => {
            unbwt_init_single(t, p, n, freq, i, bucket2, fastbits);
            return;
        }
    };

    let segment_len = ALPHABET_SIZE + ALPHABET_SIZE * ALPHABET_SIZE;
    assert!(buckets.len() >= num_threads * segment_len);

    let index = i[0] as usize;
    let lastc = t[0] as usize;
    let mut shift = 0usize;
    while (usize::try_from(n).expect("n must be non-negative") >> shift)
        > (1usize << UNBWT_FASTBITS)
    {
        shift += 1;
    }

    let mut bucket1 = vec![0u64; ALPHABET_SIZE];
    bucket2.fill(0);

    let n_fast = n as FastSint;
    let block_stride = (n_fast / num_threads as FastSint) & (-16);
    let mut block_starts = vec![0usize; num_threads];
    let mut block_sizes = vec![0usize; num_threads];

    for thread in 0..num_threads {
        let start = usize::try_from(thread as FastSint * block_stride)
            .expect("block start must be non-negative");
        let size = if thread + 1 < num_threads {
            usize::try_from(block_stride).expect("block stride must be non-negative")
        } else {
            usize::try_from(n_fast - thread as FastSint * block_stride)
                .expect("block size must be non-negative")
        };
        block_starts[thread] = start;
        block_sizes[thread] = size;

        let segment = &mut buckets[thread * segment_len..(thread + 1) * segment_len];
        let (bucket1_local, _) = segment.split_at_mut(ALPHABET_SIZE);
        bucket1_local.fill(0);
        unbwt_compute_histogram(&t[start..], size as FastSint, bucket1_local);
    }

    for thread in 0..num_threads {
        let segment = &mut buckets[thread * segment_len..(thread + 1) * segment_len];
        let (bucket1_temp, _) = segment.split_at_mut(ALPHABET_SIZE);
        for c in 0..ALPHABET_SIZE {
            let a = bucket1[c];
            let b = bucket1_temp[c];
            bucket1[c] = a + b;
            bucket1_temp[c] = a;
        }
    }

    let mut sum = 1usize;
    for c in 0..ALPHABET_SIZE {
        let prev = sum;
        sum += bucket1[c] as usize;
        bucket1[c] = prev as SaUint;
    }

    for thread in 0..num_threads {
        let start = block_starts[thread];
        let size = block_sizes[thread];
        let segment = &mut buckets[thread * segment_len..(thread + 1) * segment_len];
        let (bucket1_local, bucket2_local) = segment.split_at_mut(ALPHABET_SIZE);
        for c in 0..ALPHABET_SIZE {
            bucket1_local[c] += bucket1[c];
        }
        bucket2_local.fill(0);
        unbwt_compute_bigram_histogram_parallel(
            t,
            index,
            bucket1_local,
            bucket2_local,
            start as FastSint,
            size as FastSint,
        );
    }

    for thread in 0..num_threads {
        let segment = &mut buckets[thread * segment_len..(thread + 1) * segment_len];
        let (_, bucket2_temp) = segment.split_at_mut(ALPHABET_SIZE);
        for c in 0..ALPHABET_SIZE * ALPHABET_SIZE {
            let a = bucket2[c];
            let b = bucket2_temp[c];
            bucket2[c] = a + b;
            bucket2_temp[c] = a;
        }
    }

    unbwt_calculate_fastbits(bucket2, fastbits, lastc, shift);

    for thread in (1..num_threads).rev() {
        let src_start = (thread - 1) * segment_len;
        let dst_start = thread * segment_len;
        let (head, tail) = buckets.split_at_mut(dst_start);
        let src = &head[src_start..src_start + ALPHABET_SIZE];
        let dst = &mut tail[..ALPHABET_SIZE];
        dst.copy_from_slice(src);
    }
    buckets[..ALPHABET_SIZE].copy_from_slice(&bucket1);

    for thread in 0..num_threads {
        let start = block_starts[thread];
        let size = block_sizes[thread];
        let segment = &mut buckets[thread * segment_len..(thread + 1) * segment_len];
        let (bucket1_local, bucket2_local) = segment.split_at_mut(ALPHABET_SIZE);
        for c in 0..ALPHABET_SIZE * ALPHABET_SIZE {
            bucket2_local[c] += bucket2[c];
        }
        unbwt_calculate_bi_psi(
            t,
            p,
            bucket1_local,
            bucket2_local,
            index,
            start as FastSint,
            (start + size) as FastSint,
        );
    }

    let last_segment = &buckets[(num_threads - 1) * segment_len..num_threads * segment_len];
    let (_, last_bucket2) = last_segment.split_at(ALPHABET_SIZE);
    bucket2.copy_from_slice(last_bucket2);
}

fn bswap16(value: u16) -> u16 {
    value.swap_bytes()
}

fn unbwt_resolve_symbol(bucket2: &[SaUint], fastbits: &[u16], shift: FastUint, p: SaUint) -> u16 {
    let mut c = fastbits[(p as usize) >> shift];
    while bucket2[c as usize] <= p {
        c += 1;
    }
    c
}

/// Internal helper: unbwt decode 1.
#[doc(hidden)]
pub fn unbwt_decode_1(
    u: &mut [u8],
    p: &[SaUint],
    bucket2: &[SaUint],
    fastbits: &[u16],
    shift: FastUint,
    i0: &mut FastUint,
    k: FastUint,
) {
    let words = &mut u[..2 * k];
    let mut p0 = *i0 as SaUint;

    for i in 0..k {
        let c0 = unbwt_resolve_symbol(bucket2, fastbits, shift, p0);
        p0 = p[p0 as usize];
        let bytes = bswap16(c0).to_ne_bytes();
        words[2 * i] = bytes[0];
        words[2 * i + 1] = bytes[1];
    }

    *i0 = p0 as FastUint;
}

/// Internal helper: unbwt decode 2.
#[doc(hidden)]
pub fn unbwt_decode_2(
    u: &mut [u8],
    p: &[SaUint],
    bucket2: &[SaUint],
    fastbits: &[u16],
    shift: FastUint,
    r: FastUint,
    i0: &mut FastUint,
    i1: &mut FastUint,
    k: FastUint,
) {
    let width = 2 * k;
    unbwt_decode_1(&mut u[0..width], p, bucket2, fastbits, shift, i0, k);
    unbwt_decode_1(&mut u[r..r + width], p, bucket2, fastbits, shift, i1, k);
}

/// Internal helper: unbwt decode 3.
#[doc(hidden)]
pub fn unbwt_decode_3(
    u: &mut [u8],
    p: &[SaUint],
    bucket2: &[SaUint],
    fastbits: &[u16],
    shift: FastUint,
    r: FastUint,
    i0: &mut FastUint,
    i1: &mut FastUint,
    i2: &mut FastUint,
    k: FastUint,
) {
    let width = 2 * k;
    unbwt_decode_1(&mut u[0..width], p, bucket2, fastbits, shift, i0, k);
    unbwt_decode_1(&mut u[r..r + width], p, bucket2, fastbits, shift, i1, k);
    unbwt_decode_1(
        &mut u[2 * r..2 * r + width],
        p,
        bucket2,
        fastbits,
        shift,
        i2,
        k,
    );
}

/// Internal helper: unbwt decode 4.
#[doc(hidden)]
pub fn unbwt_decode_4(
    u: &mut [u8],
    p: &[SaUint],
    bucket2: &[SaUint],
    fastbits: &[u16],
    shift: FastUint,
    r: FastUint,
    i0: &mut FastUint,
    i1: &mut FastUint,
    i2: &mut FastUint,
    i3: &mut FastUint,
    k: FastUint,
) {
    let width = 2 * k;
    unbwt_decode_1(&mut u[0..width], p, bucket2, fastbits, shift, i0, k);
    unbwt_decode_1(&mut u[r..r + width], p, bucket2, fastbits, shift, i1, k);
    unbwt_decode_1(
        &mut u[2 * r..2 * r + width],
        p,
        bucket2,
        fastbits,
        shift,
        i2,
        k,
    );
    unbwt_decode_1(
        &mut u[3 * r..3 * r + width],
        p,
        bucket2,
        fastbits,
        shift,
        i3,
        k,
    );
}

/// Internal helper: unbwt decode 5.
#[doc(hidden)]
pub fn unbwt_decode_5(
    u: &mut [u8],
    p: &[SaUint],
    bucket2: &[SaUint],
    fastbits: &[u16],
    shift: FastUint,
    r: FastUint,
    i0: &mut FastUint,
    i1: &mut FastUint,
    i2: &mut FastUint,
    i3: &mut FastUint,
    i4: &mut FastUint,
    k: FastUint,
) {
    let width = 2 * k;
    unbwt_decode_1(&mut u[0..width], p, bucket2, fastbits, shift, i0, k);
    unbwt_decode_1(&mut u[r..r + width], p, bucket2, fastbits, shift, i1, k);
    unbwt_decode_1(
        &mut u[2 * r..2 * r + width],
        p,
        bucket2,
        fastbits,
        shift,
        i2,
        k,
    );
    unbwt_decode_1(
        &mut u[3 * r..3 * r + width],
        p,
        bucket2,
        fastbits,
        shift,
        i3,
        k,
    );
    unbwt_decode_1(
        &mut u[4 * r..4 * r + width],
        p,
        bucket2,
        fastbits,
        shift,
        i4,
        k,
    );
}

/// Internal helper: unbwt decode 6.
#[doc(hidden)]
pub fn unbwt_decode_6(
    u: &mut [u8],
    p: &[SaUint],
    bucket2: &[SaUint],
    fastbits: &[u16],
    shift: FastUint,
    r: FastUint,
    i0: &mut FastUint,
    i1: &mut FastUint,
    i2: &mut FastUint,
    i3: &mut FastUint,
    i4: &mut FastUint,
    i5: &mut FastUint,
    k: FastUint,
) {
    let width = 2 * k;
    unbwt_decode_1(&mut u[0..width], p, bucket2, fastbits, shift, i0, k);
    unbwt_decode_1(&mut u[r..r + width], p, bucket2, fastbits, shift, i1, k);
    unbwt_decode_1(
        &mut u[2 * r..2 * r + width],
        p,
        bucket2,
        fastbits,
        shift,
        i2,
        k,
    );
    unbwt_decode_1(
        &mut u[3 * r..3 * r + width],
        p,
        bucket2,
        fastbits,
        shift,
        i3,
        k,
    );
    unbwt_decode_1(
        &mut u[4 * r..4 * r + width],
        p,
        bucket2,
        fastbits,
        shift,
        i4,
        k,
    );
    unbwt_decode_1(
        &mut u[5 * r..5 * r + width],
        p,
        bucket2,
        fastbits,
        shift,
        i5,
        k,
    );
}

/// Internal helper: unbwt decode 7.
#[doc(hidden)]
pub fn unbwt_decode_7(
    u: &mut [u8],
    p: &[SaUint],
    bucket2: &[SaUint],
    fastbits: &[u16],
    shift: FastUint,
    r: FastUint,
    i0: &mut FastUint,
    i1: &mut FastUint,
    i2: &mut FastUint,
    i3: &mut FastUint,
    i4: &mut FastUint,
    i5: &mut FastUint,
    i6: &mut FastUint,
    k: FastUint,
) {
    let width = 2 * k;
    unbwt_decode_1(&mut u[0..width], p, bucket2, fastbits, shift, i0, k);
    unbwt_decode_1(&mut u[r..r + width], p, bucket2, fastbits, shift, i1, k);
    unbwt_decode_1(
        &mut u[2 * r..2 * r + width],
        p,
        bucket2,
        fastbits,
        shift,
        i2,
        k,
    );
    unbwt_decode_1(
        &mut u[3 * r..3 * r + width],
        p,
        bucket2,
        fastbits,
        shift,
        i3,
        k,
    );
    unbwt_decode_1(
        &mut u[4 * r..4 * r + width],
        p,
        bucket2,
        fastbits,
        shift,
        i4,
        k,
    );
    unbwt_decode_1(
        &mut u[5 * r..5 * r + width],
        p,
        bucket2,
        fastbits,
        shift,
        i5,
        k,
    );
    unbwt_decode_1(
        &mut u[6 * r..6 * r + width],
        p,
        bucket2,
        fastbits,
        shift,
        i6,
        k,
    );
}

/// Internal helper: unbwt decode 8.
#[doc(hidden)]
pub fn unbwt_decode_8(
    u: &mut [u8],
    p: &[SaUint],
    bucket2: &[SaUint],
    fastbits: &[u16],
    shift: FastUint,
    r: FastUint,
    i0: &mut FastUint,
    i1: &mut FastUint,
    i2: &mut FastUint,
    i3: &mut FastUint,
    i4: &mut FastUint,
    i5: &mut FastUint,
    i6: &mut FastUint,
    i7: &mut FastUint,
    k: FastUint,
) {
    let width = 2 * k;
    unbwt_decode_1(&mut u[0..width], p, bucket2, fastbits, shift, i0, k);
    unbwt_decode_1(&mut u[r..r + width], p, bucket2, fastbits, shift, i1, k);
    unbwt_decode_1(
        &mut u[2 * r..2 * r + width],
        p,
        bucket2,
        fastbits,
        shift,
        i2,
        k,
    );
    unbwt_decode_1(
        &mut u[3 * r..3 * r + width],
        p,
        bucket2,
        fastbits,
        shift,
        i3,
        k,
    );
    unbwt_decode_1(
        &mut u[4 * r..4 * r + width],
        p,
        bucket2,
        fastbits,
        shift,
        i4,
        k,
    );
    unbwt_decode_1(
        &mut u[5 * r..5 * r + width],
        p,
        bucket2,
        fastbits,
        shift,
        i5,
        k,
    );
    unbwt_decode_1(
        &mut u[6 * r..6 * r + width],
        p,
        bucket2,
        fastbits,
        shift,
        i6,
        k,
    );
    unbwt_decode_1(
        &mut u[7 * r..7 * r + width],
        p,
        bucket2,
        fastbits,
        shift,
        i7,
        k,
    );
}

/// Internal helper: unbwt decode.
#[doc(hidden)]
pub fn unbwt_decode(
    u: &mut [u8],
    p: &[SaUint],
    n: SaSint,
    r: SaSint,
    i: &[SaUint],
    bucket2: &[SaUint],
    fastbits: &[u16],
    mut blocks: FastSint,
    remainder: FastUint,
) {
    let mut shift = 0usize;
    while (usize::try_from(n).expect("n must be non-negative") >> shift)
        > (1usize << UNBWT_FASTBITS)
    {
        shift += 1;
    }
    let mut offset = 0usize;
    let mut i_index = 0usize;
    let r_usize = usize::try_from(r).expect("r must be non-negative");

    while blocks > 8 {
        let mut i0 = i[i_index] as FastUint;
        let mut i1 = i[i_index + 1] as FastUint;
        let mut i2 = i[i_index + 2] as FastUint;
        let mut i3 = i[i_index + 3] as FastUint;
        let mut i4 = i[i_index + 4] as FastUint;
        let mut i5 = i[i_index + 5] as FastUint;
        let mut i6 = i[i_index + 6] as FastUint;
        let mut i7 = i[i_index + 7] as FastUint;
        unbwt_decode_8(
            &mut u[offset..],
            p,
            bucket2,
            fastbits,
            shift,
            r_usize,
            &mut i0,
            &mut i1,
            &mut i2,
            &mut i3,
            &mut i4,
            &mut i5,
            &mut i6,
            &mut i7,
            r_usize >> 1,
        );
        i_index += 8;
        blocks -= 8;
        offset += 8 * r_usize;
    }

    match blocks {
        1 => {
            let mut i0 = i[i_index] as FastUint;
            unbwt_decode_1(
                &mut u[offset..],
                p,
                bucket2,
                fastbits,
                shift,
                &mut i0,
                remainder >> 1,
            );
        }
        2 => {
            let mut i0 = i[i_index] as FastUint;
            let mut i1 = i[i_index + 1] as FastUint;
            unbwt_decode_2(
                &mut u[offset..],
                p,
                bucket2,
                fastbits,
                shift,
                r_usize,
                &mut i0,
                &mut i1,
                remainder >> 1,
            );
            unbwt_decode_1(
                &mut u[offset + 2 * (remainder >> 1)..],
                p,
                bucket2,
                fastbits,
                shift,
                &mut i0,
                (r_usize >> 1) - (remainder >> 1),
            );
        }
        3 => {
            let mut i0 = i[i_index] as FastUint;
            let mut i1 = i[i_index + 1] as FastUint;
            let mut i2 = i[i_index + 2] as FastUint;
            unbwt_decode_3(
                &mut u[offset..],
                p,
                bucket2,
                fastbits,
                shift,
                r_usize,
                &mut i0,
                &mut i1,
                &mut i2,
                remainder >> 1,
            );
            unbwt_decode_2(
                &mut u[offset + 2 * (remainder >> 1)..],
                p,
                bucket2,
                fastbits,
                shift,
                r_usize,
                &mut i0,
                &mut i1,
                (r_usize >> 1) - (remainder >> 1),
            );
        }
        4 => {
            let mut i0 = i[i_index] as FastUint;
            let mut i1 = i[i_index + 1] as FastUint;
            let mut i2 = i[i_index + 2] as FastUint;
            let mut i3 = i[i_index + 3] as FastUint;
            unbwt_decode_4(
                &mut u[offset..],
                p,
                bucket2,
                fastbits,
                shift,
                r_usize,
                &mut i0,
                &mut i1,
                &mut i2,
                &mut i3,
                remainder >> 1,
            );
            unbwt_decode_3(
                &mut u[offset + 2 * (remainder >> 1)..],
                p,
                bucket2,
                fastbits,
                shift,
                r_usize,
                &mut i0,
                &mut i1,
                &mut i2,
                (r_usize >> 1) - (remainder >> 1),
            );
        }
        5 => {
            let mut i0 = i[i_index] as FastUint;
            let mut i1 = i[i_index + 1] as FastUint;
            let mut i2 = i[i_index + 2] as FastUint;
            let mut i3 = i[i_index + 3] as FastUint;
            let mut i4 = i[i_index + 4] as FastUint;
            unbwt_decode_5(
                &mut u[offset..],
                p,
                bucket2,
                fastbits,
                shift,
                r_usize,
                &mut i0,
                &mut i1,
                &mut i2,
                &mut i3,
                &mut i4,
                remainder >> 1,
            );
            unbwt_decode_4(
                &mut u[offset + 2 * (remainder >> 1)..],
                p,
                bucket2,
                fastbits,
                shift,
                r_usize,
                &mut i0,
                &mut i1,
                &mut i2,
                &mut i3,
                (r_usize >> 1) - (remainder >> 1),
            );
        }
        6 => {
            let mut i0 = i[i_index] as FastUint;
            let mut i1 = i[i_index + 1] as FastUint;
            let mut i2 = i[i_index + 2] as FastUint;
            let mut i3 = i[i_index + 3] as FastUint;
            let mut i4 = i[i_index + 4] as FastUint;
            let mut i5 = i[i_index + 5] as FastUint;
            unbwt_decode_6(
                &mut u[offset..],
                p,
                bucket2,
                fastbits,
                shift,
                r_usize,
                &mut i0,
                &mut i1,
                &mut i2,
                &mut i3,
                &mut i4,
                &mut i5,
                remainder >> 1,
            );
            unbwt_decode_5(
                &mut u[offset + 2 * (remainder >> 1)..],
                p,
                bucket2,
                fastbits,
                shift,
                r_usize,
                &mut i0,
                &mut i1,
                &mut i2,
                &mut i3,
                &mut i4,
                (r_usize >> 1) - (remainder >> 1),
            );
        }
        7 => {
            let mut i0 = i[i_index] as FastUint;
            let mut i1 = i[i_index + 1] as FastUint;
            let mut i2 = i[i_index + 2] as FastUint;
            let mut i3 = i[i_index + 3] as FastUint;
            let mut i4 = i[i_index + 4] as FastUint;
            let mut i5 = i[i_index + 5] as FastUint;
            let mut i6 = i[i_index + 6] as FastUint;
            unbwt_decode_7(
                &mut u[offset..],
                p,
                bucket2,
                fastbits,
                shift,
                r_usize,
                &mut i0,
                &mut i1,
                &mut i2,
                &mut i3,
                &mut i4,
                &mut i5,
                &mut i6,
                remainder >> 1,
            );
            unbwt_decode_6(
                &mut u[offset + 2 * (remainder >> 1)..],
                p,
                bucket2,
                fastbits,
                shift,
                r_usize,
                &mut i0,
                &mut i1,
                &mut i2,
                &mut i3,
                &mut i4,
                &mut i5,
                (r_usize >> 1) - (remainder >> 1),
            );
        }
        8 => {
            let mut i0 = i[i_index] as FastUint;
            let mut i1 = i[i_index + 1] as FastUint;
            let mut i2 = i[i_index + 2] as FastUint;
            let mut i3 = i[i_index + 3] as FastUint;
            let mut i4 = i[i_index + 4] as FastUint;
            let mut i5 = i[i_index + 5] as FastUint;
            let mut i6 = i[i_index + 6] as FastUint;
            let mut i7 = i[i_index + 7] as FastUint;
            unbwt_decode_8(
                &mut u[offset..],
                p,
                bucket2,
                fastbits,
                shift,
                r_usize,
                &mut i0,
                &mut i1,
                &mut i2,
                &mut i3,
                &mut i4,
                &mut i5,
                &mut i6,
                &mut i7,
                remainder >> 1,
            );
            unbwt_decode_7(
                &mut u[offset + 2 * (remainder >> 1)..],
                p,
                bucket2,
                fastbits,
                shift,
                r_usize,
                &mut i0,
                &mut i1,
                &mut i2,
                &mut i3,
                &mut i4,
                &mut i5,
                &mut i6,
                (r_usize >> 1) - (remainder >> 1),
            );
        }
        _ => {}
    }
}

/// Internal helper: unbwt decode (OpenMP variant).
#[doc(hidden)]
pub fn unbwt_decode_omp(
    t: &[u8],
    u: &mut [u8],
    p: &[SaUint],
    n: SaSint,
    r: SaSint,
    i: &[SaUint],
    bucket2: &[SaUint],
    fastbits: &[u16],
    threads: SaSint,
) {
    let lastc = t[0];
    let blocks = 1 + ((n as FastSint - 1) / r as FastSint);
    let remainder = usize::try_from(n).expect("n must be non-negative")
        - usize::try_from(r).expect("r must be non-negative")
            * (usize::try_from(blocks).expect("blocks") - 1);
    let max_threads = usize::try_from(blocks.min(threads.max(1) as FastSint))
        .expect("thread count must fit usize");
    let block_stride = usize::try_from(blocks).expect("blocks must be non-negative") / max_threads;
    let block_remainder =
        usize::try_from(blocks).expect("blocks must be non-negative") % max_threads;
    let r_usize = usize::try_from(r).expect("r must be non-negative");

    let u_ptr = SyncMutPtr::new(u);
    run_rayon_with_threads(max_threads, || {
        (0..max_threads).into_par_iter().for_each(|thread| {
            let block_size = block_stride + usize::from(thread < block_remainder);
            let block_start = block_stride * thread + thread.min(block_remainder);
            let u = unsafe { u_ptr.as_slice() };
            unbwt_decode(
                &mut u[r_usize * block_start..],
                p,
                n,
                r,
                &i[block_start..],
                bucket2,
                fastbits,
                block_size as FastSint,
                if thread + 1 < max_threads {
                    r_usize
                } else {
                    remainder
                },
            );
        });
    });
    u[usize::try_from(n).expect("n must be non-negative") - 1] = lastc;
}

/// Internal helper: unbwt core.
#[doc(hidden)]
pub fn unbwt_core(
    t: &[u8],
    u: &mut [u8],
    p: &mut [SaUint],
    n: SaSint,
    freq: Option<&[SaSint]>,
    r: SaSint,
    i: &[SaUint],
    bucket2: &mut [SaUint],
    fastbits: &mut [u16],
    buckets: Option<&mut [SaUint]>,
    threads: SaSint,
) -> SaSint {
    if threads > 1 && n >= 262_144 {
        unbwt_init_parallel(t, p, n, freq, i, bucket2, fastbits, buckets, threads);
    } else {
        unbwt_init_single(t, p, n, freq, i, bucket2, fastbits);
    }

    unbwt_decode_omp(t, u, p, n, r, i, bucket2, fastbits, threads);
    0
}

/// Internal helper: unbwt main.
#[doc(hidden)]
pub fn unbwt_main(
    t: &[u8],
    u: &mut [u8],
    p: &mut [SaUint],
    n: SaSint,
    freq: Option<&[SaSint]>,
    r: SaSint,
    i: &[SaUint],
    threads: SaSint,
) -> SaSint {
    let mut shift = 0usize;
    while (usize::try_from(n).expect("n must be non-negative") >> shift)
        > (1usize << UNBWT_FASTBITS)
    {
        shift += 1;
    }

    let mut bucket2 = vec![0u64; ALPHABET_SIZE * ALPHABET_SIZE];
    let mut fastbits =
        vec![0u16; 1 + (usize::try_from(n).expect("n must be non-negative") >> shift)];
    let mut buckets = if threads > 1 && n >= 262_144 {
        Some(vec![
            0u64;
            usize::try_from(threads)
                .expect("threads must be non-negative")
                * (ALPHABET_SIZE + ALPHABET_SIZE * ALPHABET_SIZE)
        ])
    } else {
        None
    };

    unbwt_core(
        t,
        u,
        p,
        n,
        freq,
        r,
        i,
        &mut bucket2,
        &mut fastbits,
        buckets.as_deref_mut(),
        threads,
    )
}

/// Internal helper: unbwt main ctx.
#[doc(hidden)]
pub fn unbwt_main_ctx(
    ctx: &mut UnbwtContext,
    t: &[u8],
    u: &mut [u8],
    p: &mut [SaUint],
    n: SaSint,
    freq: Option<&[SaSint]>,
    r: SaSint,
    i: &[SaUint],
) -> SaSint {
    if ctx.threads <= 0 {
        return -2;
    }
    let mut shift = 0usize;
    while (usize::try_from(n).expect("n must be non-negative") >> shift)
        > (1usize << UNBWT_FASTBITS)
    {
        shift += 1;
    }
    let required_fastbits = 1 + (usize::try_from(n).expect("n must be non-negative") >> shift);
    if ctx.bucket2.len() < ALPHABET_SIZE * ALPHABET_SIZE
        || ctx.fastbits.len() < required_fastbits
        || (ctx.threads > 1 && ctx.buckets.is_none())
    {
        return -2;
    }

    unbwt_core(
        t,
        u,
        p,
        n,
        freq,
        r,
        i,
        &mut ctx.bucket2,
        &mut ctx.fastbits,
        ctx.buckets.as_deref_mut(),
        ctx.threads as SaSint,
    )
}

/// Constructs the original string from a given burrows-wheeler transformed string (BWT) with primary index.
///
/// # Arguments
/// - `t` (`[0..n-1]`): The input string.
/// - `u` (`[0..n-1]`): The output string (can be `t`).
/// - `a` (`[0..n]`): The temporary array (NOTE, temporary array must be n + 1 size).
/// - `freq` (`[0..255]`): The input symbol frequency table (can be `None`).
/// - `i`: The primary index.
///
/// # Returns
/// 0 if no error occurred, -1 or -2 otherwise.
pub fn libsais64_unbwt(
    t: &[u8],
    u: &mut [u8],
    a: &mut [SaSint],
    freq: Option<&[SaSint]>,
    i: SaSint,
) -> SaSint {
    libsais64_unbwt_aux(
        t,
        u,
        a,
        freq,
        SaSint::try_from(t.len()).expect("input length must fit SaSint"),
        &[i],
    )
}

/// Reconstructs the original string from a given BWT and primary index using a libsais64 reverse-BWT context.
///
/// - `ctx`: the libsais64 reverse-BWT context.
/// - `t` (`[0..n-1]`): the input string.
/// - `u` (`[0..n-1]`): the output string (can alias `t`).
/// - `a` (`[0..n]`): the temporary array (must have length `n + 1`).
/// - `freq` (`[0..255]`): optional input symbol frequency table.
/// - `i`: the primary index.
///
/// Returns 0 on success, -1 or -2 on error.
pub fn libsais64_unbwt_ctx(
    ctx: &mut UnbwtContext,
    t: &[u8],
    u: &mut [u8],
    a: &mut [SaSint],
    freq: Option<&[SaSint]>,
    i: SaSint,
) -> SaSint {
    libsais64_unbwt_aux_ctx(
        ctx,
        t,
        u,
        a,
        freq,
        SaSint::try_from(t.len()).expect("input length must fit SaSint"),
        &[i],
    )
}

/// Constructs the original string from a given burrows-wheeler transformed string (BWT) with auxiliary indexes.
///
/// # Arguments
/// - `t` (`[0..n-1]`): The input string.
/// - `u` (`[0..n-1]`): The output string (can be `t`).
/// - `a` (`[0..n]`): The temporary array (NOTE, temporary array must be n + 1 size).
/// - `freq` (`[0..255]`): The input symbol frequency table (can be `None`).
/// - `r`: The sampling rate for auxiliary indexes (must be power of 2).
/// - `i` (`[0..(n-1)/r]`): The input auxiliary indexes.
///
/// # Returns
/// 0 if no error occurred, -1 or -2 otherwise.
pub fn libsais64_unbwt_aux(
    t: &[u8],
    u: &mut [u8],
    a: &mut [SaSint],
    freq: Option<&[SaSint]>,
    r: SaSint,
    i: &[SaSint],
) -> SaSint {
    let t_len = t.len();
    let n = SaSint::try_from(t_len).expect("input length must fit SaSint");
    if u.len() < t_len
        || a.len() < t_len
        || freq.is_some_and(|freq| freq.len() < ALPHABET_SIZE)
        || (r != n && (r < 2 || (r & (r - 1)) != 0))
    {
        return -1;
    }
    let sample_count = if n == 0 {
        1
    } else {
        ((n - 1) / r + 1) as usize
    };
    if i.len() < sample_count {
        return -1;
    }

    if n <= 1 {
        if i[0] != n {
            return -1;
        }
        if n == 1 {
            u[0] = t[0];
        }
        return 0;
    }

    for t in 0..sample_count {
        let sample = i[t];
        if sample <= 0 || sample > n {
            return -1;
        }
    }

    let i_uint: Vec<SaUint> = i
        .iter()
        .take(sample_count)
        .map(|&sample| SaUint::try_from(sample).expect("sample was validated positive"))
        .collect();
    let mut p = vec![0u64; t_len + 1];
    let result = unbwt_main(t, u, &mut p, n, freq, r, &i_uint, 1);
    for t in 0..t_len {
        a[t] = p[t] as SaSint;
    }
    result
}

/// Reconstructs the original string from a given BWT with auxiliary indexes using a libsais64 reverse-BWT context.
///
/// - `ctx`: the libsais64 reverse-BWT context.
/// - `t` (`[0..n-1]`): the input string.
/// - `u` (`[0..n-1]`): the output string (can alias `t`).
/// - `a` (`[0..n]`): the temporary array (must have length `n + 1`).
/// - `freq` (`[0..255]`): optional input symbol frequency table.
/// - `r`: sampling rate for the auxiliary indexes (must be a power of two).
/// - `i` (`[0..(n-1)/r]`): input auxiliary indexes.
///
/// Returns 0 on success, -1 or -2 on error.
pub fn libsais64_unbwt_aux_ctx(
    ctx: &mut UnbwtContext,
    t: &[u8],
    u: &mut [u8],
    a: &mut [SaSint],
    freq: Option<&[SaSint]>,
    r: SaSint,
    i: &[SaSint],
) -> SaSint {
    let t_len = t.len();
    let n = SaSint::try_from(t_len).expect("input length must fit SaSint");
    if u.len() < t_len
        || a.len() < t_len
        || freq.is_some_and(|freq| freq.len() < ALPHABET_SIZE)
        || (r != n && (r < 2 || (r & (r - 1)) != 0))
    {
        return -1;
    }
    let sample_count = if n == 0 {
        1
    } else {
        ((n - 1) / r + 1) as usize
    };
    if i.len() < sample_count {
        return -1;
    }

    if n <= 1 {
        if i[0] != n {
            return -1;
        }
        if n == 1 {
            u[0] = t[0];
        }
        return 0;
    }

    for t in 0..sample_count {
        let sample = i[t];
        if sample <= 0 || sample > n {
            return -1;
        }
    }

    let i_uint: Vec<SaUint> = i
        .iter()
        .take(sample_count)
        .map(|&sample| SaUint::try_from(sample).expect("sample was validated positive"))
        .collect();
    let mut p = vec![0u64; t_len + 1];
    let result = unbwt_main_ctx(ctx, t, u, &mut p, n, freq, r, &i_uint);
    for t in 0..t_len {
        a[t] = p[t] as SaSint;
    }
    result
}

/// Creates the libsais64 reverse-BWT context for parallel `libsais64_unbwt_*` operations using OpenMP-style threading.
///
/// In multi-threaded environments, use one context per thread for parallel executions.
///
/// - `threads`: number of worker threads (can be 0 for the implementation default).
///
/// Returns the context, or `None` on allocation failure.
pub fn unbwt_create_ctx_omp(threads: SaSint) -> Option<UnbwtContext> {
    if threads < 0 {
        return None;
    }
    unbwt_create_ctx_main(normalize_omp_threads(threads))
}

/// Constructs the original string from a given burrows-wheeler transformed string (BWT) with primary index in parallel using OpenMP.
///
/// # Arguments
/// - `t` (`[0..n-1]`): The input string.
/// - `u` (`[0..n-1]`): The output string (can be `t`).
/// - `a` (`[0..n]`): The temporary array (NOTE, temporary array must be n + 1 size).
/// - `freq` (`[0..255]`): The input symbol frequency table (can be `None`).
/// - `i`: The primary index.
/// - `threads`: The number of OpenMP threads to use (can be 0 for OpenMP default).
///
/// # Returns
/// 0 if no error occurred, -1 or -2 otherwise.
pub fn libsais64_unbwt_omp(
    t: &[u8],
    u: &mut [u8],
    a: &mut [SaSint],
    freq: Option<&[SaSint]>,
    i: SaSint,
    threads: SaSint,
) -> SaSint {
    libsais64_unbwt_aux_omp(
        t,
        u,
        a,
        freq,
        SaSint::try_from(t.len()).expect("input length must fit SaSint"),
        &[i],
        threads,
    )
}

/// Constructs the original string from a given burrows-wheeler transformed string (BWT) with auxiliary indexes in parallel using OpenMP.
///
/// # Arguments
/// - `t` (`[0..n-1]`): The input string.
/// - `u` (`[0..n-1]`): The output string (can be `t`).
/// - `a` (`[0..n]`): The temporary array (NOTE, temporary array must be n + 1 size).
/// - `freq` (`[0..255]`): The input symbol frequency table (can be `None`).
/// - `r`: The sampling rate for auxiliary indexes (must be power of 2).
/// - `i` (`[0..(n-1)/r]`): The input auxiliary indexes.
/// - `threads`: The number of OpenMP threads to use (can be 0 for OpenMP default).
///
/// # Returns
/// 0 if no error occurred, -1 or -2 otherwise.
pub fn libsais64_unbwt_aux_omp(
    t: &[u8],
    u: &mut [u8],
    a: &mut [SaSint],
    freq: Option<&[SaSint]>,
    r: SaSint,
    i: &[SaSint],
    threads: SaSint,
) -> SaSint {
    let t_len = t.len();
    let n = SaSint::try_from(t_len).expect("input length must fit SaSint");
    if threads < 0
        || u.len() < t_len
        || a.len() < t_len
        || freq.is_some_and(|freq| freq.len() < ALPHABET_SIZE)
        || (r != n && (r < 2 || (r & (r - 1)) != 0))
    {
        return -1;
    }
    let sample_count = if n == 0 {
        1
    } else {
        ((n - 1) / r + 1) as usize
    };
    if i.len() < sample_count {
        return -1;
    }

    if n <= 1 {
        if i[0] != n {
            return -1;
        }
        if n == 1 {
            u[0] = t[0];
        }
        return 0;
    }

    for sample in i.iter().take(sample_count) {
        let sample = *sample;
        if sample <= 0 || sample > n {
            return -1;
        }
    }

    let threads = if threads > 0 { threads } else { 1 };
    let i_uint: Vec<SaUint> = i
        .iter()
        .take(sample_count)
        .map(|&sample| SaUint::try_from(sample).expect("sample was validated positive"))
        .collect();
    let mut p = vec![0u64; t_len + 1];
    let result = unbwt_main(t, u, &mut p, n, freq, r, &i_uint, threads);
    for idx in 0..t_len {
        a[idx] = p[idx] as SaSint;
    }
    result
}

/// Internal helper: bwt copy 8u.
#[doc(hidden)]
pub fn bwt_copy_8u(u: &mut [u8], a: &[SaSint], n: SaSint) {
    if n <= 0 {
        return;
    }

    let n_usize = usize::try_from(n).expect("n must be non-negative");
    for i in 0..n_usize {
        u[i] = a[i] as u8;
    }
}

/// Internal helper: bwt copy 8u (OpenMP variant).
#[doc(hidden)]
pub fn bwt_copy_8u_omp(u: &mut [u8], a: &[SaSint], n: SaSint, threads: SaSint) {
    if threads == 1 || n < 65_536 {
        bwt_copy_8u(u, a, n);
        return;
    }

    let n_usize = usize::try_from(n).expect("n must be non-negative");
    assert!(u.len() >= n_usize);
    assert!(a.len() >= n_usize);
    let threads_usize = usize::try_from(threads).expect("threads must be non-negative");
    let chunk_size = ((n_usize / threads_usize) & !15usize).max(16);
    let a_ptr = a.as_ptr() as usize;
    run_rayon_with_threads(threads_usize, || {
        u[..n_usize]
            .par_chunks_mut(chunk_size)
            .enumerate()
            .for_each(|(chunk_index, chunk)| {
                let start = chunk_index * chunk_size;
                let dst_ptr = chunk.as_mut_ptr();
                let src_ptr = unsafe { (a_ptr as *const SaSint).add(start) };
                for offset in 0..chunk.len() {
                    unsafe {
                        *dst_ptr.add(offset) = *src_ptr.add(offset) as u8;
                    }
                }
            });
    });
}

/// Internal helper: accumulate counts s32 2.
#[doc(hidden)]
pub fn accumulate_counts_s32_2(bucket00: &mut [SaSint], bucket01: &[SaSint]) {
    assert_eq!(bucket00.len(), bucket01.len());
    for (dst, src) in bucket00.iter_mut().zip(bucket01.iter()) {
        *dst += *src;
    }
}

/// Internal helper: accumulate counts s32 3.
#[doc(hidden)]
pub fn accumulate_counts_s32_3(bucket00: &mut [SaSint], bucket01: &[SaSint], bucket02: &[SaSint]) {
    assert_eq!(bucket00.len(), bucket01.len());
    assert_eq!(bucket00.len(), bucket02.len());
    for ((dst, src1), src2) in bucket00
        .iter_mut()
        .zip(bucket01.iter())
        .zip(bucket02.iter())
    {
        *dst += *src1 + *src2;
    }
}

/// Internal helper: accumulate counts s32 4.
#[doc(hidden)]
pub fn accumulate_counts_s32_4(
    bucket00: &mut [SaSint],
    bucket01: &[SaSint],
    bucket02: &[SaSint],
    bucket03: &[SaSint],
) {
    assert_eq!(bucket00.len(), bucket01.len());
    assert_eq!(bucket00.len(), bucket02.len());
    assert_eq!(bucket00.len(), bucket03.len());
    for (((dst, src1), src2), src3) in bucket00
        .iter_mut()
        .zip(bucket01.iter())
        .zip(bucket02.iter())
        .zip(bucket03.iter())
    {
        *dst += *src1 + *src2 + *src3;
    }
}

/// Internal helper: accumulate counts s32 5.
#[doc(hidden)]
pub fn accumulate_counts_s32_5(
    bucket00: &mut [SaSint],
    bucket01: &[SaSint],
    bucket02: &[SaSint],
    bucket03: &[SaSint],
    bucket04: &[SaSint],
) {
    assert_eq!(bucket00.len(), bucket01.len());
    assert_eq!(bucket00.len(), bucket02.len());
    assert_eq!(bucket00.len(), bucket03.len());
    assert_eq!(bucket00.len(), bucket04.len());
    for ((((dst, src1), src2), src3), src4) in bucket00
        .iter_mut()
        .zip(bucket01.iter())
        .zip(bucket02.iter())
        .zip(bucket03.iter())
        .zip(bucket04.iter())
    {
        *dst += *src1 + *src2 + *src3 + *src4;
    }
}

/// Internal helper: accumulate counts s32 6.
#[doc(hidden)]
pub fn accumulate_counts_s32_6(
    bucket00: &mut [SaSint],
    bucket01: &[SaSint],
    bucket02: &[SaSint],
    bucket03: &[SaSint],
    bucket04: &[SaSint],
    bucket05: &[SaSint],
) {
    assert_eq!(bucket00.len(), bucket01.len());
    assert_eq!(bucket00.len(), bucket02.len());
    assert_eq!(bucket00.len(), bucket03.len());
    assert_eq!(bucket00.len(), bucket04.len());
    assert_eq!(bucket00.len(), bucket05.len());
    for (((((dst, src1), src2), src3), src4), src5) in bucket00
        .iter_mut()
        .zip(bucket01.iter())
        .zip(bucket02.iter())
        .zip(bucket03.iter())
        .zip(bucket04.iter())
        .zip(bucket05.iter())
    {
        *dst += *src1 + *src2 + *src3 + *src4 + *src5;
    }
}

/// Internal helper: accumulate counts s32 7.
#[doc(hidden)]
pub fn accumulate_counts_s32_7(
    bucket00: &mut [SaSint],
    bucket01: &[SaSint],
    bucket02: &[SaSint],
    bucket03: &[SaSint],
    bucket04: &[SaSint],
    bucket05: &[SaSint],
    bucket06: &[SaSint],
) {
    assert_eq!(bucket00.len(), bucket01.len());
    assert_eq!(bucket00.len(), bucket02.len());
    assert_eq!(bucket00.len(), bucket03.len());
    assert_eq!(bucket00.len(), bucket04.len());
    assert_eq!(bucket00.len(), bucket05.len());
    assert_eq!(bucket00.len(), bucket06.len());
    for ((((((dst, src1), src2), src3), src4), src5), src6) in bucket00
        .iter_mut()
        .zip(bucket01.iter())
        .zip(bucket02.iter())
        .zip(bucket03.iter())
        .zip(bucket04.iter())
        .zip(bucket05.iter())
        .zip(bucket06.iter())
    {
        *dst += *src1 + *src2 + *src3 + *src4 + *src5 + *src6;
    }
}

/// Internal helper: accumulate counts s32 8.
#[doc(hidden)]
pub fn accumulate_counts_s32_8(
    bucket00: &mut [SaSint],
    bucket01: &[SaSint],
    bucket02: &[SaSint],
    bucket03: &[SaSint],
    bucket04: &[SaSint],
    bucket05: &[SaSint],
    bucket06: &[SaSint],
    bucket07: &[SaSint],
) {
    assert_eq!(bucket00.len(), bucket01.len());
    assert_eq!(bucket00.len(), bucket02.len());
    assert_eq!(bucket00.len(), bucket03.len());
    assert_eq!(bucket00.len(), bucket04.len());
    assert_eq!(bucket00.len(), bucket05.len());
    assert_eq!(bucket00.len(), bucket06.len());
    assert_eq!(bucket00.len(), bucket07.len());
    for (((((((dst, src1), src2), src3), src4), src5), src6), src7) in bucket00
        .iter_mut()
        .zip(bucket01.iter())
        .zip(bucket02.iter())
        .zip(bucket03.iter())
        .zip(bucket04.iter())
        .zip(bucket05.iter())
        .zip(bucket06.iter())
        .zip(bucket07.iter())
    {
        *dst += *src1 + *src2 + *src3 + *src4 + *src5 + *src6 + *src7;
    }
}

/// Internal helper: accumulate counts s32 9.
#[doc(hidden)]
pub fn accumulate_counts_s32_9(
    bucket00: &mut [SaSint],
    bucket01: &[SaSint],
    bucket02: &[SaSint],
    bucket03: &[SaSint],
    bucket04: &[SaSint],
    bucket05: &[SaSint],
    bucket06: &[SaSint],
    bucket07: &[SaSint],
    bucket08: &[SaSint],
) {
    assert_eq!(bucket00.len(), bucket01.len());
    assert_eq!(bucket00.len(), bucket02.len());
    assert_eq!(bucket00.len(), bucket03.len());
    assert_eq!(bucket00.len(), bucket04.len());
    assert_eq!(bucket00.len(), bucket05.len());
    assert_eq!(bucket00.len(), bucket06.len());
    assert_eq!(bucket00.len(), bucket07.len());
    assert_eq!(bucket00.len(), bucket08.len());
    for ((((((((dst, src1), src2), src3), src4), src5), src6), src7), src8) in bucket00
        .iter_mut()
        .zip(bucket01.iter())
        .zip(bucket02.iter())
        .zip(bucket03.iter())
        .zip(bucket04.iter())
        .zip(bucket05.iter())
        .zip(bucket06.iter())
        .zip(bucket07.iter())
        .zip(bucket08.iter())
    {
        *dst += *src1 + *src2 + *src3 + *src4 + *src5 + *src6 + *src7 + *src8;
    }
}

/// Internal helper: accumulate counts s32.
#[doc(hidden)]
pub fn accumulate_counts_s32(
    buckets: &mut [SaSint],
    bucket_size: FastSint,
    bucket_stride: FastSint,
    mut num_buckets: FastSint,
) {
    if num_buckets <= 1 {
        return;
    }

    let bucket_size = usize::try_from(bucket_size).expect("bucket_size must be non-negative");
    let bucket_stride = usize::try_from(bucket_stride).expect("bucket_stride must be non-negative");
    let num_buckets_usize = usize::try_from(num_buckets).expect("num_buckets must be non-negative");
    assert!(buckets.len() >= bucket_size + (num_buckets_usize - 1) * bucket_stride);
    let bucket00_start = (num_buckets_usize - 1) * bucket_stride;

    while num_buckets >= 9 {
        let start = bucket00_start
            - usize::try_from(num_buckets - 9).expect("non-negative") * bucket_stride;
        accumulate_counts_at(buckets, start, bucket_size, bucket_stride, 9);
        num_buckets -= 8;
    }

    match num_buckets {
        1 => {}
        2..=8 => accumulate_counts_at(
            buckets,
            bucket00_start,
            bucket_size,
            bucket_stride,
            usize::try_from(num_buckets).expect("non-negative"),
        ),
        _ => {}
    }
}

fn block_slice<T>(slice: &[T], block_start: FastSint, block_size: FastSint) -> &[T] {
    let start = usize::try_from(block_start).expect("block_start must be non-negative");
    let len = usize::try_from(block_size).expect("block_size must be non-negative");
    &slice[start..start + len]
}

#[allow(dead_code)]
struct SharedMutArray<'a> {
    ptr: *mut SaSint,
    len: usize,
    _marker: PhantomData<&'a mut [SaSint]>,
}

#[allow(dead_code)]
impl<'a> SharedMutArray<'a> {
    fn new(slice: &'a mut [SaSint]) -> Self {
        Self {
            ptr: slice.as_mut_ptr(),
            len: slice.len(),
            _marker: PhantomData,
        }
    }

    fn len(&self) -> usize {
        self.len
    }

    fn slice_mut(&mut self, start: usize, len: usize) -> &mut [SaSint] {
        assert!(start <= self.len);
        assert!(len <= self.len - start);
        unsafe {
            // The recursive driver aliases multiple logical views into one SA backing store.
            // This helper centralizes that checked projection so the driver can be translated
            // without pretending those regions are independent Rust slices.
            std::slice::from_raw_parts_mut(self.ptr.add(start), len)
        }
    }
}

fn accumulate_counts_at(
    buckets: &mut [SaSint],
    bucket00_start: usize,
    bucket_size: usize,
    bucket_stride: usize,
    count: usize,
) {
    assert!((2..=9).contains(&count));
    assert!(bucket00_start >= (count - 1) * bucket_stride);

    let dst_end = bucket00_start + bucket_size;
    let mut sums = vec![0; bucket_size];

    for i in 0..count {
        let start = bucket00_start - i * bucket_stride;
        let end = start + bucket_size;
        for (sum, value) in sums.iter_mut().zip(buckets[start..end].iter()) {
            *sum += *value;
        }
    }

    buckets[bucket00_start..dst_end].copy_from_slice(&sums);
}

/// Internal helper: thread state size.
#[doc(hidden)]
pub fn thread_state_size() -> usize {
    mem::size_of::<ThreadState>()
}

#[cfg(all(test, feature = "upstream-c"))]
mod tests {
    use super::*;

    unsafe extern "C" {
        fn probe_public_libsais64(t: *const u8, sa: *mut SaSint, n: SaSint, fs: SaSint) -> SaSint;
        fn probe_public_libsais64_freq(
            t: *const u8,
            sa: *mut SaSint,
            n: SaSint,
            fs: SaSint,
            freq: *mut SaSint,
        ) -> SaSint;
        fn probe_public_libsais64_gsa(
            t: *const u8,
            sa: *mut SaSint,
            n: SaSint,
            fs: SaSint,
        ) -> SaSint;
        fn probe_public_libsais64_gsa_freq(
            t: *const u8,
            sa: *mut SaSint,
            n: SaSint,
            fs: SaSint,
            freq: *mut SaSint,
        ) -> SaSint;
        fn probe_public_libsais64_long(
            t: *mut SaSint,
            sa: *mut SaSint,
            n: SaSint,
            k: SaSint,
            fs: SaSint,
        ) -> SaSint;
        fn probe_public_libsais64_bwt(
            t: *const u8,
            u: *mut u8,
            a: *mut SaSint,
            n: SaSint,
            fs: SaSint,
        ) -> SaSint;
        fn probe_public_libsais64_bwt_freq(
            t: *const u8,
            u: *mut u8,
            a: *mut SaSint,
            n: SaSint,
            fs: SaSint,
            freq: *mut SaSint,
        ) -> SaSint;
        fn probe_public_libsais64_bwt_aux(
            t: *const u8,
            u: *mut u8,
            a: *mut SaSint,
            n: SaSint,
            fs: SaSint,
            r: SaSint,
            i: *mut SaSint,
        ) -> SaSint;
        fn probe_public_libsais64_bwt_aux_freq(
            t: *const u8,
            u: *mut u8,
            a: *mut SaSint,
            n: SaSint,
            fs: SaSint,
            freq: *mut SaSint,
            r: SaSint,
            i: *mut SaSint,
        ) -> SaSint;
        fn probe_public_libsais64_unbwt(
            t: *const u8,
            u: *mut u8,
            a: *mut SaSint,
            n: SaSint,
            i: SaSint,
        ) -> SaSint;
        fn probe_public_libsais64_unbwt_freq(
            t: *const u8,
            u: *mut u8,
            a: *mut SaSint,
            n: SaSint,
            freq: *const SaSint,
            i: SaSint,
        ) -> SaSint;
        fn probe_public_libsais64_unbwt_aux(
            t: *const u8,
            u: *mut u8,
            a: *mut SaSint,
            n: SaSint,
            r: SaSint,
            i: *const SaSint,
        ) -> SaSint;
        fn probe_public_libsais64_unbwt_aux_freq(
            t: *const u8,
            u: *mut u8,
            a: *mut SaSint,
            n: SaSint,
            freq: *const SaSint,
            r: SaSint,
            i: *const SaSint,
        ) -> SaSint;
        fn probe_public_libsais64_plcp(
            t: *const u8,
            sa: *const SaSint,
            plcp: *mut SaSint,
            n: SaSint,
        ) -> SaSint;
        fn probe_public_libsais64_plcp_gsa(
            t: *const u8,
            sa: *const SaSint,
            plcp: *mut SaSint,
            n: SaSint,
        ) -> SaSint;
        fn probe_public_libsais64_lcp(
            plcp: *const SaSint,
            sa: *const SaSint,
            lcp: *mut SaSint,
            n: SaSint,
        ) -> SaSint;
        fn probe_libsais64_renumber_lms_suffixes_8u(
            sa: *mut SaSint,
            m: SaSint,
            name: SaSint,
            omp_block_start: FastSint,
            omp_block_size: FastSint,
        ) -> SaSint;
        fn probe_libsais64_gather_marked_lms_suffixes(
            sa: *mut SaSint,
            m: SaSint,
            l: FastSint,
            omp_block_start: FastSint,
            omp_block_size: FastSint,
        ) -> FastSint;
        fn probe_libsais64_renumber_and_gather_lms_suffixes_omp(
            sa: *mut SaSint,
            n: SaSint,
            m: SaSint,
            fs: SaSint,
            threads: SaSint,
        ) -> SaSint;
        fn probe_libsais64_renumber_distinct_lms_suffixes_32s_4k(
            sa: *mut SaSint,
            m: SaSint,
            name: SaSint,
            omp_block_start: FastSint,
            omp_block_size: FastSint,
        ) -> SaSint;
        fn probe_libsais64_renumber_and_mark_distinct_lms_suffixes_32s_4k_omp(
            sa: *mut SaSint,
            n: SaSint,
            m: SaSint,
            threads: SaSint,
        ) -> SaSint;
        fn probe_libsais64_renumber_unique_and_nonunique_lms_suffixes_32s(
            t: *mut SaSint,
            sa: *mut SaSint,
            m: SaSint,
            f: SaSint,
            omp_block_start: FastSint,
            omp_block_size: FastSint,
        ) -> SaSint;
        fn probe_libsais64_renumber_unique_and_nonunique_lms_suffixes_32s_omp(
            t: *mut SaSint,
            sa: *mut SaSint,
            m: SaSint,
            threads: SaSint,
        ) -> SaSint;
    }

    #[test]
    fn libsais64_align_up_matches_power_of_two_alignment() {
        assert_eq!(align_up(0, 4096), 0);
        assert_eq!(align_up(1, 4096), 4096);
        assert_eq!(align_up(4095, 4096), 4096);
        assert_eq!(align_up(4096, 4096), 4096);
        assert_eq!(align_up(4097, 4096), 8192);
        assert_eq!(align_up(65, 64), 128);
    }

    #[test]
    fn libsais64_shared_mut_array_projects_mutable_spans_from_one_backing_buffer() {
        let mut backing = vec![1, 2, 3, 4, 5, 6];
        let len;
        {
            let mut shared = SharedMutArray::new(&mut backing);
            shared.slice_mut(1, 3).copy_from_slice(&[20, 30, 40]);
            shared.slice_mut(4, 2).copy_from_slice(&[50, 60]);
            len = shared.len();
        }
        assert_eq!(backing, vec![1, 20, 30, 40, 50, 60]);
        assert_eq!(len, 6);
    }

    #[test]
    fn libsais64_create_ctx_main_matches_single_thread_layout() {
        let ctx = create_ctx_main(1).expect("context");
        assert_eq!(ctx.buckets.len(), 8 * ALPHABET_SIZE);
        assert_eq!(ctx.threads, 1);
        assert!(ctx.thread_state.is_none());
    }

    #[test]
    fn libsais64_create_ctx_main_allocates_thread_state_for_multi_threaded_mode() {
        let ctx = create_ctx_main(3).expect("context");
        let states = ctx.thread_state.expect("thread state");
        assert_eq!(states.len(), 3);
        assert!(states
            .iter()
            .all(|state| state.buckets.len() == 4 * ALPHABET_SIZE));
        assert!(states
            .iter()
            .all(|state| state.cache.len() == LIBSAIS_PER_THREAD_CACHE_SIZE));
    }

    #[test]
    fn libsais64_create_ctx_wraps_single_thread_main_context() {
        let ctx = create_ctx().expect("context");
        assert_eq!(ctx.threads, 1);
        assert_eq!(ctx.buckets.len(), 8 * ALPHABET_SIZE);
        assert!(ctx.thread_state.is_none());
    }

    #[test]
    fn libsais64_free_ctx_accepts_context_value() {
        let ctx = create_ctx().expect("context");
        free_ctx(ctx);
    }

    #[test]
    fn libsais64_unbwt_create_ctx_main_allocates_expected_buffers() {
        let ctx = unbwt_create_ctx_main(3).expect("context");
        assert_eq!(ctx.bucket2.len(), ALPHABET_SIZE * ALPHABET_SIZE);
        assert_eq!(ctx.fastbits.len(), 1 + (1 << UNBWT_FASTBITS));
        assert_eq!(
            ctx.buckets.as_ref().expect("parallel buckets").len(),
            3 * (ALPHABET_SIZE + ALPHABET_SIZE * ALPHABET_SIZE)
        );
        assert_eq!(ctx.threads, 3);
    }

    #[test]
    fn libsais64_unbwt_compute_histogram_counts_bytes() {
        let t = b"banana";
        let mut count = vec![0u64; ALPHABET_SIZE];
        unbwt_compute_histogram(t, t.len() as FastSint, &mut count);
        assert_eq!(count[b'a' as usize], 3);
        assert_eq!(count[b'b' as usize], 1);
        assert_eq!(count[b'n' as usize], 2);
    }

    #[test]
    fn libsais64_unbwt_transpose_bucket2_swaps_matrix_entries() {
        let mut bucket2 = vec![0u64; ALPHABET_SIZE * ALPHABET_SIZE];
        bucket2[(2 << 8) + 1] = 7;
        bucket2[(1 << 8) + 2] = 9;
        unbwt_transpose_bucket2(&mut bucket2);
        assert_eq!(bucket2[(1 << 8) + 2], 7);
        assert_eq!(bucket2[(2 << 8) + 1], 9);
    }

    #[test]
    fn libsais64_unbwt_init_single_builds_monotone_fastbits_and_writes_psi() {
        let t = b"annb\x00aa";
        let mut p = vec![0u64; t.len() + 1];
        let mut bucket2 = vec![0u64; ALPHABET_SIZE * ALPHABET_SIZE];
        let mut fastbits = vec![0u16; 1 + (1 << UNBWT_FASTBITS)];
        let i = vec![4u64];

        unbwt_init_single(
            t,
            &mut p,
            t.len() as SaSint,
            None,
            &i,
            &mut bucket2,
            &mut fastbits,
        );

        assert!(fastbits
            .iter()
            .all(|&value| usize::from(value) < ALPHABET_SIZE * ALPHABET_SIZE));
        assert!(fastbits.iter().any(|&value| value != 0));
        assert!(p.iter().any(|&value| value != 0));
    }

    #[test]
    fn libsais64_unbwt_init_parallel_currently_matches_single_initializer() {
        let t = b"annb\x00aa";
        let mut p_single = vec![0u64; t.len() + 1];
        let mut p_parallel = vec![0u64; t.len() + 1];
        let mut bucket2_single = vec![0u64; ALPHABET_SIZE * ALPHABET_SIZE];
        let mut bucket2_parallel = vec![0u64; ALPHABET_SIZE * ALPHABET_SIZE];
        let mut fastbits_single = vec![0u16; 1 + (1 << UNBWT_FASTBITS)];
        let mut fastbits_parallel = vec![0u16; 1 + (1 << UNBWT_FASTBITS)];
        let i = vec![4u64];
        let mut scratch = vec![0u64; 2 * (ALPHABET_SIZE + ALPHABET_SIZE * ALPHABET_SIZE)];

        unbwt_init_single(
            t,
            &mut p_single,
            t.len() as SaSint,
            None,
            &i,
            &mut bucket2_single,
            &mut fastbits_single,
        );
        unbwt_init_parallel(
            t,
            &mut p_parallel,
            t.len() as SaSint,
            None,
            &i,
            &mut bucket2_parallel,
            &mut fastbits_parallel,
            Some(&mut scratch),
            2,
        );

        assert_eq!(p_parallel, p_single);
        assert_eq!(bucket2_parallel, bucket2_single);
        assert_eq!(fastbits_parallel, fastbits_single);
    }

    #[test]
    fn libsais64_unbwt_decode_1_writes_big_endian_symbol_words() {
        let mut u = vec![0u8; 4];
        let p = vec![1u64, 0u64];
        let mut bucket2 = vec![0u64; ALPHABET_SIZE * ALPHABET_SIZE];
        bucket2[0x1234] = 0;
        bucket2[0x1235] = 2;
        let mut fastbits = vec![0u16; 1 + (1 << UNBWT_FASTBITS)];
        fastbits[0] = 0x1234;
        let mut i0 = 0usize;

        unbwt_decode_1(&mut u, &p, &bucket2, &fastbits, 0, &mut i0, 2);

        assert_eq!(u, vec![0x12, 0x35, 0x12, 0x35]);
        assert_eq!(i0, 0);
    }

    #[test]
    fn libsais64_unbwt_decode_dispatches_two_block_tail_shape() {
        let mut u = vec![0u8; 8];
        let p = vec![1u64, 0u64];
        let mut bucket2 = vec![0u64; ALPHABET_SIZE * ALPHABET_SIZE];
        bucket2[0x1234] = 0;
        bucket2[0x1235] = 2;
        let mut fastbits = vec![0u16; 1 + (1 << UNBWT_FASTBITS)];
        fastbits[0] = 0x1234;
        let i = vec![0u64, 0u64];

        unbwt_decode(&mut u, &p, 4, 2, &i, &bucket2, &fastbits, 2, 2);

        assert_eq!(u, vec![0x12, 0x35, 0x12, 0x35, 0x00, 0x00, 0x00, 0x00]);
    }

    fn brute_force_suffix_array_u8(t: &[u8]) -> Vec<SaSint> {
        let mut sa: Vec<SaSint> = (0..t.len())
            .map(|index| SaSint::try_from(index).expect("index must fit SaSint"))
            .collect();
        sa.sort_by(|&lhs, &rhs| {
            t[usize::try_from(lhs).expect("non-negative")..]
                .cmp(&t[usize::try_from(rhs).expect("non-negative")..])
        });
        sa
    }

    fn brute_force_plcp_u8(t: &[u8], sa: &[SaSint]) -> Vec<SaSint> {
        let mut rank = vec![0usize; t.len()];
        for (i, &suffix) in sa.iter().enumerate() {
            rank[usize::try_from(suffix).expect("suffix index must be non-negative")] = i;
        }

        let mut plcp = vec![0; t.len()];
        for i in 0..t.len() {
            let r = rank[i];
            let prev = if r == 0 {
                t.len()
            } else {
                usize::try_from(sa[r - 1]).expect("suffix index must be non-negative")
            };
            if prev == t.len() {
                plcp[i] = 0;
                continue;
            }

            let mut l = 0usize;
            while i + l < t.len() && prev + l < t.len() && t[i + l] == t[prev + l] {
                l += 1;
            }
            plcp[i] = l as SaSint;
        }
        plcp
    }

    fn brute_force_lcp_from_sa_u8(t: &[u8], sa: &[SaSint]) -> Vec<SaSint> {
        let mut lcp = vec![0; sa.len()];
        for i in 0..sa.len() {
            let lhs = usize::try_from(sa[i]).expect("suffix index must be non-negative");
            let rhs = if i == 0 {
                sa.len()
            } else {
                usize::try_from(sa[i - 1]).expect("suffix index must be non-negative")
            };
            if rhs == sa.len() {
                lcp[i] = 0;
                continue;
            }

            let mut l = 0usize;
            while lhs + l < t.len() && rhs + l < t.len() && t[lhs + l] == t[rhs + l] {
                l += 1;
            }
            lcp[i] = l as SaSint;
        }
        lcp
    }

    fn make_libsais64_recursive_main_32s_text(repeats: usize) -> Vec<SaSint> {
        let motif = [9, 4, 9, 2, 9, 4, 9, 1];
        let mut t = Vec::with_capacity(repeats * motif.len() + 1);
        for _ in 0..repeats {
            t.extend_from_slice(&motif);
        }
        t.push(0);
        t
    }

    fn make_libsais64_large_main_32s_stress_text(len: usize, alphabet: SaSint) -> Vec<SaSint> {
        let mut state: u32 = 0x1357_9bdf;
        let mut t = Vec::with_capacity(len + 1);

        for i in 0..len {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let mut value = ((state >> 16) % (alphabet as u32 - 1)) as SaSint + 1;

            if i % 17 < 8 {
                value = ((i / 17) as SaSint % 11) + 1;
            }
            if i % 29 < 10 {
                value = (((i / 29) as SaSint * 3) % 19) + 1;
            }
            if i % 64 >= 48 {
                value = t[i - 48];
            }

            t.push(value);
        }

        t.push(0);
        t
    }

    fn assert_libsais64_main_32s_entry_matches_public_c_long(
        t: Vec<SaSint>,
        k: SaSint,
        fs: SaSint,
        compare_full_sa: bool,
    ) {
        let n = t.len() as SaSint;
        let n_usize = t.len();
        let threads = 1;
        let extra = usize::try_from(fs).expect("fs must be non-negative");

        let mut c_t = t.clone();
        let mut c_sa = vec![0; t.len() + extra];
        let c_result =
            unsafe { probe_public_libsais64_long(c_t.as_mut_ptr(), c_sa.as_mut_ptr(), n, k, fs) };

        let mut rust_t = t;
        let mut rust_sa = vec![0; rust_t.len() + extra];
        let mut thread_state = alloc_thread_state(threads).expect("thread state");
        let rust_result = libsais64_main_32s_entry(
            &mut rust_t,
            &mut rust_sa,
            n,
            k,
            fs,
            threads,
            &mut thread_state,
        );

        assert_eq!(rust_result, c_result);
        assert_eq!(rust_t, c_t);
        if compare_full_sa {
            assert_eq!(rust_sa, c_sa);
        } else {
            assert_eq!(&rust_sa[..n_usize], &c_sa[..n_usize]);
        }
    }

    fn assert_libsais64_main_32s_entry_matches_public_c_long_for_branch(k: SaSint) {
        assert_libsais64_main_32s_entry_matches_public_c_long(
            vec![17, 3, 17, 9, 5, 9, 2, 11, 2, 7, 1, 7, 0],
            k,
            0,
            true,
        );
    }

    #[test]
    fn libsais64_matches_bruteforce_suffix_array_for_small_text() {
        let t = b"banana";
        let mut sa = vec![0; t.len()];
        let mut freq = vec![0; ALPHABET_SIZE];

        let result = libsais64(t, &mut sa, 0, Some(&mut freq));

        assert_eq!(result, 0);
        assert_eq!(sa, brute_force_suffix_array_u8(t));
        assert_eq!(freq[b'a' as usize], 3);
        assert_eq!(freq[b'b' as usize], 1);
        assert_eq!(freq[b'n' as usize], 2);
    }

    #[test]
    fn libsais64_int_matches_bruteforce_suffix_array_for_small_integer_text() {
        let mut t = vec![2, 1, 3, 1, 0];
        let expected = {
            let mut sa: Vec<SaSint> = (0..t.len())
                .map(|index| SaSint::try_from(index).expect("index must fit SaSint"))
                .collect();
            sa.sort_by(|&lhs, &rhs| {
                t[usize::try_from(lhs).expect("non-negative")..]
                    .cmp(&t[usize::try_from(rhs).expect("non-negative")..])
            });
            sa
        };
        let mut sa = vec![0; t.len()];

        let result = libsais64_int(&mut t, &mut sa, 4, 0);

        assert_eq!(result, 0);
        assert_eq!(sa, expected);
    }

    #[test]
    fn libsais64_plcp_matches_bruteforce_for_small_text() {
        let t = b"banana";
        let sa = brute_force_suffix_array_u8(t);
        let expected = brute_force_plcp_u8(t, &sa);
        let mut plcp = vec![0; t.len()];

        let result = libsais64_plcp(t, &sa, &mut plcp);

        assert_eq!(result, 0);
        assert_eq!(plcp, expected);
    }

    #[test]
    fn libsais64_plcp_gsa_stops_at_separator() {
        let t = b"ab\0b\0";
        let sa = brute_force_suffix_array_u8(t);
        let mut plcp = vec![0; t.len()];

        let result = libsais64_plcp_gsa(t, &sa, &mut plcp);

        assert_eq!(result, 0);
        assert_eq!(plcp[2], 0);
        assert_eq!(plcp[4], 0);
    }

    #[test]
    fn libsais64_lcp_matches_bruteforce_for_small_text() {
        let t = b"banana";
        let sa = brute_force_suffix_array_u8(t);
        let plcp = brute_force_plcp_u8(t, &sa);
        let expected = brute_force_lcp_from_sa_u8(t, &sa);
        let mut lcp = vec![0; t.len()];

        let result = libsais64_lcp(&plcp, &sa, &mut lcp);

        assert_eq!(result, 0);
        assert_eq!(lcp, expected);
    }

    #[test]
    fn libsais64_unbwt_init_parallel_uses_block_partition_for_large_inputs() {
        let n = 70_003usize;
        let t: Vec<u8> = (0..n)
            .map(|i| i.wrapping_mul(37).wrapping_add(i >> 3) as u8)
            .collect();
        let i = [12_345u64];

        let mut single_p = vec![0u64; n + 1];
        let mut threaded_p = vec![0u64; n + 1];
        let mut single_bucket2 = vec![0u64; ALPHABET_SIZE * ALPHABET_SIZE];
        let mut threaded_bucket2 = vec![0u64; ALPHABET_SIZE * ALPHABET_SIZE];
        let mut single_fastbits = vec![0u16; 1 + (1 << UNBWT_FASTBITS)];
        let mut threaded_fastbits = vec![0u16; 1 + (1 << UNBWT_FASTBITS)];
        let mut buckets = vec![0u64; 4 * (ALPHABET_SIZE + ALPHABET_SIZE * ALPHABET_SIZE)];

        unbwt_init_single(
            &t,
            &mut single_p,
            n as SaSint,
            None,
            &i,
            &mut single_bucket2,
            &mut single_fastbits,
        );
        unbwt_init_parallel(
            &t,
            &mut threaded_p,
            n as SaSint,
            None,
            &i,
            &mut threaded_bucket2,
            &mut threaded_fastbits,
            Some(&mut buckets),
            4,
        );

        assert_eq!(threaded_p, single_p);
        assert_eq!(threaded_bucket2, single_bucket2);
        assert_eq!(threaded_fastbits, single_fastbits);
    }

    #[test]
    fn libsais64_radix_sort_lms_suffixes_8u_places_suffixes_by_bucket() {
        let t = vec![1_u8, 0, 1, 0];
        let mut sa = vec![9, 9, 9, 9, 0, 1, 2, 3];
        let mut induction_bucket = vec![0; 2 * ALPHABET_SIZE];
        induction_bucket[buckets_index2(0, 0)] = 2;
        induction_bucket[buckets_index2(1, 0)] = 4;
        radix_sort_lms_suffixes_8u(&t, &mut sa, &mut induction_bucket, 4, 4);
        assert_eq!(&sa[..4], &[1, 3, 0, 2]);
    }

    #[test]
    fn libsais64_radix_sort_lms_suffixes_8u_omp_wraps_sequential_version() {
        let t = vec![9_u8, 1, 0, 1, 0];
        let mut sa = vec![9, 9, 9, 9, 9, 1, 2, 3, 4];
        let mut buckets = vec![0; 6 * ALPHABET_SIZE];
        buckets[4 * ALPHABET_SIZE + buckets_index2(0, 0)] = 2;
        buckets[4 * ALPHABET_SIZE + buckets_index2(1, 0)] = 4;
        let mut thread_state = alloc_thread_state(2).unwrap();
        radix_sort_lms_suffixes_8u_omp(&t, &mut sa, 9, 5, 0, &mut buckets, 2, &mut thread_state);
        assert_eq!(&sa[..4], &[2, 4, 1, 3]);
    }

    #[test]
    fn libsais64_radix_sort_lms_suffixes_32s_6k_places_suffixes_by_bucket() {
        let t = vec![1, 0, 1, 0];
        let mut sa = vec![9, 9, 9, 9, 0, 1, 2, 3];
        let mut induction_bucket = vec![2, 4];
        radix_sort_lms_suffixes_32s_6k(&t, &mut sa, &mut induction_bucket, 4, 4);
        assert_eq!(&sa[..4], &[1, 3, 0, 2]);
    }

    #[test]
    fn libsais64_radix_sort_lms_suffixes_32s_2k_places_suffixes_by_bucket() {
        let t = vec![1, 0, 1, 0];
        let mut sa = vec![9, 9, 9, 9, 0, 1, 2, 3];
        let mut induction_bucket = vec![2, 0, 4, 0];
        radix_sort_lms_suffixes_32s_2k(&t, &mut sa, &mut induction_bucket, 4, 4);
        assert_eq!(&sa[..4], &[1, 3, 0, 2]);
    }

    #[test]
    fn libsais64_radix_sort_lms_suffixes_32s_6k_omp_wraps_sequential_version() {
        let t = vec![9, 1, 0, 1, 0];
        let mut sa = vec![9, 9, 9, 9, 9, 1, 2, 3, 4];
        let mut induction_bucket = vec![2, 4];
        let mut thread_state = alloc_thread_state(2).unwrap();
        radix_sort_lms_suffixes_32s_6k_omp(
            &t,
            &mut sa,
            9,
            5,
            &mut induction_bucket,
            2,
            &mut thread_state,
        );
        assert_eq!(&sa[..4], &[2, 4, 1, 3]);
    }

    #[test]
    fn libsais64_radix_sort_lms_suffixes_32s_2k_omp_wraps_sequential_version() {
        let t = vec![9, 1, 0, 1, 0];
        let mut sa = vec![9, 9, 9, 9, 9, 1, 2, 3, 4];
        let mut induction_bucket = vec![2, 0, 4, 0];
        let mut thread_state = alloc_thread_state(2).unwrap();
        radix_sort_lms_suffixes_32s_2k_omp(
            &t,
            &mut sa,
            9,
            5,
            &mut induction_bucket,
            2,
            &mut thread_state,
        );
        assert_eq!(&sa[..4], &[2, 4, 1, 3]);
    }

    #[test]
    fn libsais64_radix_sort_lms_suffixes_32s_block_omp_runs_cache_pipeline() {
        let t = vec![9, 1, 0, 1, 0];
        let mut sa_6k = vec![9, 9, 9, 9, 9, 1, 2, 3, 4];
        let mut bucket_6k = vec![2, 4];
        let mut cache = vec![ThreadCache::default(); 9];
        radix_sort_lms_suffixes_32s_6k_block_omp(
            &t,
            &mut sa_6k,
            &mut bucket_6k,
            &mut cache,
            5,
            4,
            2,
        );
        assert_eq!(&sa_6k[..4], &[2, 4, 1, 3]);

        let mut sa_2k = vec![9, 9, 9, 9, 9, 1, 2, 3, 4];
        let mut bucket_2k = vec![2, 0, 4, 0];
        cache.fill(ThreadCache::default());
        radix_sort_lms_suffixes_32s_2k_block_omp(
            &t,
            &mut sa_2k,
            &mut bucket_2k,
            &mut cache,
            5,
            4,
            2,
        );
        assert_eq!(&sa_2k[..4], &[2, 4, 1, 3]);
    }

    #[test]
    fn libsais64_radix_sort_lms_suffixes_8u_omp_uses_thread_state_for_large_inputs() {
        let m = 65_600usize;
        let n = 2 * m + 16;
        let start = n - m + 1;
        let t: Vec<u8> = (0..n).map(|i| (i % 4) as u8).collect();
        let suffixes: Vec<SaSint> = (0..m - 1).map(|i| i as SaSint).collect();

        let mut buckets = vec![0; 6 * ALPHABET_SIZE];
        for &suffix in &suffixes {
            buckets[4 * ALPHABET_SIZE + buckets_index2(t[suffix as usize] as usize, 0)] += 1;
        }
        let mut sum = 0;
        for symbol in 0..ALPHABET_SIZE {
            let bucket = 4 * ALPHABET_SIZE + buckets_index2(symbol, 0);
            sum += buckets[bucket];
            buckets[bucket] = sum;
        }

        let mut sa_single = vec![0; n];
        sa_single[start..start + suffixes.len()].copy_from_slice(&suffixes);
        let mut sa_threaded = sa_single.clone();
        let mut buckets_single = buckets.clone();
        let mut buckets_threaded = buckets;
        let mut thread_state = alloc_thread_state(4).unwrap();
        thread_state[3].m = m as FastSint;

        radix_sort_lms_suffixes_8u_omp(
            &t,
            &mut sa_single,
            n as SaSint,
            m as SaSint,
            0,
            &mut buckets_single,
            1,
            &mut [],
        );
        radix_sort_lms_suffixes_8u_omp(
            &t,
            &mut sa_threaded,
            n as SaSint,
            m as SaSint,
            0,
            &mut buckets_threaded,
            4,
            &mut thread_state,
        );

        assert_eq!(sa_threaded, sa_single);
    }

    #[test]
    fn libsais64_radix_sort_lms_suffixes_32s_omp_uses_block_pipeline_for_large_inputs() {
        let m = 65_600usize;
        let n = 2 * m + 16;
        let start = n - m + 1;
        let t: Vec<SaSint> = (0..n).map(|i| (i % 4) as SaSint).collect();
        let suffixes: Vec<SaSint> = (0..m - 1).map(|i| i as SaSint).collect();

        let mut bucket_ends = vec![0; 4];
        for &suffix in &suffixes {
            bucket_ends[t[suffix as usize] as usize] += 1;
        }
        let mut sum = 0;
        for bucket in &mut bucket_ends {
            sum += *bucket;
            *bucket = sum;
        }

        let mut sa_single = vec![0; n];
        sa_single[start..start + suffixes.len()].copy_from_slice(&suffixes);
        let mut sa_threaded = sa_single.clone();
        let mut bucket_single = bucket_ends.clone();
        let mut bucket_threaded = bucket_ends.clone();
        let mut thread_state = alloc_thread_state(4).unwrap();

        radix_sort_lms_suffixes_32s_6k_omp(
            &t,
            &mut sa_single,
            n as SaSint,
            m as SaSint,
            &mut bucket_single,
            1,
            &mut [],
        );
        radix_sort_lms_suffixes_32s_6k_omp(
            &t,
            &mut sa_threaded,
            n as SaSint,
            m as SaSint,
            &mut bucket_threaded,
            4,
            &mut thread_state,
        );
        assert_eq!(sa_threaded, sa_single);
        assert_eq!(bucket_threaded, bucket_single);

        let mut bucket_2k = vec![0; 8];
        for (symbol, &end) in bucket_ends.iter().enumerate() {
            bucket_2k[buckets_index2(symbol, 0)] = end;
        }
        let mut sa_single = vec![0; n];
        sa_single[start..start + suffixes.len()].copy_from_slice(&suffixes);
        let mut sa_threaded = sa_single.clone();
        let mut bucket_single = bucket_2k.clone();
        let mut bucket_threaded = bucket_2k;

        radix_sort_lms_suffixes_32s_2k_omp(
            &t,
            &mut sa_single,
            n as SaSint,
            m as SaSint,
            &mut bucket_single,
            1,
            &mut [],
        );
        radix_sort_lms_suffixes_32s_2k_omp(
            &t,
            &mut sa_threaded,
            n as SaSint,
            m as SaSint,
            &mut bucket_threaded,
            4,
            &mut thread_state,
        );
        assert_eq!(sa_threaded, sa_single);
        assert_eq!(bucket_threaded, bucket_single);
    }

    #[test]
    fn libsais64_radix_sort_lms_suffixes_32s_1k_collects_lms_suffixes() {
        let t = vec![2, 1, 3, 1, 0];
        let mut sa = vec![0; t.len()];
        let mut buckets = vec![0, 2, 4, 5];
        let m = radix_sort_lms_suffixes_32s_1k(&t, &mut sa, t.len() as SaSint, &mut buckets);
        assert!(m >= 0);
    }

    #[test]
    fn libsais64_radix_sort_set_markers_32s_6k_marks_target_suffixes() {
        let mut sa = vec![0; 6];
        let induction_bucket = vec![1, 3, 5];
        radix_sort_set_markers_32s_6k(&mut sa, &induction_bucket, 0, 3);
        assert_eq!(sa[1], SAINT_MIN);
        assert_eq!(sa[3], SAINT_MIN);
        assert_eq!(sa[5], SAINT_MIN);
    }

    #[test]
    fn libsais64_radix_sort_set_markers_32s_4k_marks_target_suffixes() {
        let mut sa = vec![0; 6];
        let induction_bucket = vec![1, 0, 3, 0, 5, 0];
        radix_sort_set_markers_32s_4k(&mut sa, &induction_bucket, 0, 3);
        assert_eq!(sa[1], SUFFIX_GROUP_MARKER);
        assert_eq!(sa[3], SUFFIX_GROUP_MARKER);
        assert_eq!(sa[5], SUFFIX_GROUP_MARKER);
    }

    #[test]
    fn libsais64_radix_sort_set_markers_32s_6k_omp_wraps_sequential_version() {
        let mut sa = vec![0; 6];
        let induction_bucket = vec![1, 3, 5];
        radix_sort_set_markers_32s_6k_omp(&mut sa, 4, &induction_bucket, 2);
        assert_eq!(sa[1], SAINT_MIN);
        assert_eq!(sa[3], SAINT_MIN);
        assert_eq!(sa[5], SAINT_MIN);
    }

    #[test]
    fn libsais64_radix_sort_set_markers_32s_4k_omp_wraps_sequential_version() {
        let mut sa = vec![0; 6];
        let induction_bucket = vec![1, 0, 3, 0, 5, 0];
        radix_sort_set_markers_32s_4k_omp(&mut sa, 4, &induction_bucket, 2);
        assert_eq!(sa[1], SUFFIX_GROUP_MARKER);
        assert_eq!(sa[3], SUFFIX_GROUP_MARKER);
        assert_eq!(sa[5], SUFFIX_GROUP_MARKER);
    }

    #[test]
    fn libsais64_radix_sort_set_markers_32s_omp_partitions_large_inputs() {
        let k = 65_600usize;
        let induction_bucket_6k: Vec<SaSint> = (0..k).map(|i| i as SaSint).collect();
        let mut sa_single = vec![0; k];
        let mut sa_threaded = vec![0; k];
        radix_sort_set_markers_32s_6k_omp(&mut sa_single, k as SaSint, &induction_bucket_6k, 1);
        radix_sort_set_markers_32s_6k_omp(&mut sa_threaded, k as SaSint, &induction_bucket_6k, 4);
        assert_eq!(sa_threaded, sa_single);

        let mut induction_bucket_4k = vec![0; 2 * k];
        for i in 0..k {
            induction_bucket_4k[buckets_index2(i, 0)] = i as SaSint;
        }
        let mut sa_single = vec![0; k];
        let mut sa_threaded = vec![0; k];
        radix_sort_set_markers_32s_4k_omp(&mut sa_single, k as SaSint, &induction_bucket_4k, 1);
        radix_sort_set_markers_32s_4k_omp(&mut sa_threaded, k as SaSint, &induction_bucket_4k, 4);
        assert_eq!(sa_threaded, sa_single);
    }

    #[test]
    fn libsais64_partial_sorting_scan_left_to_right_8u_emits_induced_suffixes() {
        let t = vec![2_u8, 1, 3, 1, 0];
        let mut sa = vec![2 | SAINT_MIN, 4, 0, 0, 0, 0];
        let mut buckets = vec![0; 6 * ALPHABET_SIZE];
        buckets[4 * ALPHABET_SIZE + buckets_index2(1, 0)] = 2;
        let d = partial_sorting_scan_left_to_right_8u(&t, &mut sa, &mut buckets, 0, 0, 2);
        assert!(d >= 0);
        assert!(sa.iter().any(|&v| v != 0));
    }

    #[test]
    fn libsais64_partial_sorting_scan_left_to_right_8u_omp_wraps_sequential_version() {
        let t = vec![2_u8, 1, 3, 1, 0];
        let mut sa = vec![0; 8];
        let mut buckets = vec![0; 6 * ALPHABET_SIZE];
        buckets[4 * ALPHABET_SIZE + buckets_index2(0, 0)] = 1;
        let mut thread_state = alloc_thread_state(2).unwrap();
        let d = partial_sorting_scan_left_to_right_8u_omp(
            &t,
            &mut sa,
            5,
            4,
            &mut buckets,
            0,
            0,
            2,
            &mut thread_state,
        );
        assert!(d >= 1);
    }

    #[test]
    fn libsais64_partial_sorting_scan_left_to_right_32s_6k_emits_induced_suffixes() {
        let t = vec![2, 1, 3, 1, 0];
        let mut sa = vec![2 | SAINT_MIN, 4, 0, 0, 0, 0];
        let mut buckets = vec![0; 4 * 4];
        buckets[buckets_index4(1, 0)] = 2;
        let d = partial_sorting_scan_left_to_right_32s_6k(&t, &mut sa, &mut buckets, 0, 0, 2);
        assert!(d >= 0);
        assert!(sa.iter().any(|&v| v != 0));
    }

    #[test]
    fn libsais64_partial_sorting_scan_left_to_right_32s_4k_emits_induced_suffixes() {
        let t = vec![2, 1, 3, 1, 0];
        let k = 4usize;
        let mut sa = vec![2 | SUFFIX_GROUP_MARKER, 4, 0, 0, 0, 0];
        let mut buckets = vec![0; 4 * k];
        buckets[2 * k + 1] = 2;
        let d = partial_sorting_scan_left_to_right_32s_4k(
            &t,
            &mut sa,
            k as SaSint,
            &mut buckets,
            0,
            0,
            2,
        );
        assert!(d >= 0);
        assert!(sa.iter().any(|&v| v != 0));
    }

    #[test]
    fn libsais64_partial_sorting_scan_left_to_right_32s_1k_emits_induced_suffixes() {
        let t = vec![2, 1, 3, 1, 0];
        let mut sa = vec![2, 4, 0, 0, 0, 0];
        let mut buckets = vec![0; 4];
        buckets[1] = 2;
        partial_sorting_scan_left_to_right_32s_1k(&t, &mut sa, &mut buckets, 0, 2);
        assert!(sa.iter().any(|&v| v != 0));
    }

    #[test]
    fn libsais64_partial_sorting_scan_left_to_right_32s_6k_omp_wraps_sequential_version() {
        let t = vec![2, 1, 3, 1, 0];
        let mut sa = vec![0; 8];
        let mut buckets = vec![0; 4 * 4];
        let mut thread_state = alloc_thread_state(2).unwrap();
        let d = partial_sorting_scan_left_to_right_32s_6k_omp(
            &t,
            &mut sa,
            5,
            &mut buckets,
            0,
            0,
            2,
            &mut thread_state,
        );
        assert!(d >= 1);
    }

    #[test]
    fn libsais64_partial_sorting_scan_left_to_right_32s_4k_omp_wraps_sequential_version() {
        let t = vec![2, 1, 3, 1, 0];
        let k = 4usize;
        let mut sa = vec![0; 8];
        let mut buckets = vec![0; 4 * k];
        let mut thread_state = alloc_thread_state(2).unwrap();
        let d = partial_sorting_scan_left_to_right_32s_4k_omp(
            &t,
            &mut sa,
            5,
            k as SaSint,
            &mut buckets,
            0,
            2,
            &mut thread_state,
        );
        assert!(d >= 1);
    }

    #[test]
    fn libsais64_partial_sorting_scan_left_to_right_32s_1k_omp_wraps_sequential_version() {
        let t = vec![2, 1, 3, 1, 0];
        let mut sa = vec![0; 8];
        let mut buckets = vec![0; 4];
        let mut thread_state = alloc_thread_state(2).unwrap();
        partial_sorting_scan_left_to_right_32s_1k_omp(
            &t,
            &mut sa,
            5,
            &mut buckets,
            2,
            &mut thread_state,
        );
        assert!(sa.iter().any(|&v| v != 0));
    }

    #[test]
    fn libsais64_partial_sorting_scan_left_to_right_32s_6k_block_gather_records_bucket_symbols() {
        let t = vec![3, 1, 2, 0];
        let mut sa = vec![2 | SAINT_MIN, 0, 0, 0];
        let mut cache = vec![ThreadCache::default(); 1];

        partial_sorting_scan_left_to_right_32s_6k_block_gather(&t, &mut sa, &mut cache, 0, 1);

        assert_eq!(cache[0].index, 2 | SAINT_MIN);
        assert_eq!(cache[0].symbol, buckets_index4(1, 1) as SaSint);
    }

    #[test]
    fn libsais64_partial_sorting_scan_left_to_right_32s_1k_block_gather_zeroes_positive_entries() {
        let t = vec![3, 1, 2, 0];
        let mut sa = vec![2, 0, 0, 0];
        let mut cache = vec![ThreadCache::default(); 1];

        partial_sorting_scan_left_to_right_32s_1k_block_gather(&t, &mut sa, &mut cache, 0, 1);

        assert_eq!(cache[0].symbol, 1);
        assert_eq!(cache[0].index, 1);
        assert_eq!(sa[0], 0);
    }

    #[test]
    fn libsais64_partial_sorting_scan_left_to_right_32s_1k_block_omp_uses_relative_cache() {
        let block_start = 20_000usize;
        let block_size = 16_384usize;
        let n = block_start + block_size + 8;
        let t = vec![1; n];
        let suffixes: Vec<SaSint> = (2..2 + block_size).map(|i| i as SaSint).collect();

        let mut sa_single = vec![0; n];
        sa_single[block_start..block_start + block_size].copy_from_slice(&suffixes);
        let mut sa_threaded = sa_single.clone();
        let mut bucket_single = vec![0, 0];
        let mut bucket_threaded = bucket_single.clone();
        let mut cache = vec![ThreadCache::default(); 4 * LIBSAIS_PER_THREAD_CACHE_SIZE];

        partial_sorting_scan_left_to_right_32s_1k(
            &t,
            &mut sa_single,
            &mut bucket_single,
            block_start as FastSint,
            block_size as FastSint,
        );
        partial_sorting_scan_left_to_right_32s_1k_block_omp(
            &t,
            &mut sa_threaded,
            &mut bucket_threaded,
            &mut cache,
            block_start as FastSint,
            block_size as FastSint,
            4,
        );

        assert_eq!(sa_threaded, sa_single);
        assert_eq!(bucket_threaded, bucket_single);
    }

    #[test]
    fn libsais64_partial_sorting_scan_left_to_right_8u_block_prepare_records_cache_and_counts() {
        let t = vec![2_u8, 1, 3, 1, 0];
        let sa = vec![2 | SAINT_MIN, 4, 0, 0, 0, 0];
        let k = 4;
        let mut buckets = vec![0; 4 * k];
        let mut cache = vec![ThreadCache::default(); 8];
        let mut state = ThreadState::new();
        let (position, count) = partial_sorting_scan_left_to_right_8u_block_prepare(
            &t,
            &sa,
            k as SaSint,
            &mut buckets,
            &mut cache,
            0,
            2,
        );
        state.position = position;
        state.count = count;
        assert!(state.count >= 1);
        assert!(cache
            .iter()
            .take(state.count as usize)
            .any(|entry| entry.symbol >= 0));
    }

    #[test]
    fn libsais64_partial_sorting_scan_left_to_right_8u_block_place_writes_induced_values() {
        let mut sa = vec![0; 8];
        let mut buckets = vec![0; 8];
        buckets[0] = 0;
        buckets[1] = 1;
        let cache = vec![
            ThreadCache {
                index: 3 | SAINT_MIN,
                symbol: 0,
            },
            ThreadCache {
                index: 5,
                symbol: 1,
            },
        ];
        partial_sorting_scan_left_to_right_8u_block_place(&mut sa, &mut buckets, 2, &cache, 2, 0);
        assert!(sa[0] != 0 || sa[1] != 0);
    }

    #[test]
    fn libsais64_partial_sorting_scan_left_to_right_8u_block_omp_wraps_sequential_version() {
        let t = vec![2_u8, 1, 3, 1, 0];
        let mut sa = vec![2 | SAINT_MIN, 4, 0, 0, 0, 0];
        let mut buckets = vec![0; 6 * ALPHABET_SIZE];
        let mut thread_state = alloc_thread_state(2).unwrap();
        let d = partial_sorting_scan_left_to_right_8u_block_omp(
            &t,
            &mut sa,
            4,
            &mut buckets,
            0,
            0,
            2,
            2,
            &mut thread_state,
        );
        assert!(d >= 0);
    }

    #[test]
    fn libsais64_partial_sorting_shift_buckets_32s_6k_moves_temp_bucket_view_into_main_slots() {
        let k = 3usize;
        let mut buckets = vec![0; 6 * k];
        buckets[4 * k] = 10;
        buckets[4 * k + 1] = 11;
        buckets[4 * k + 2] = 12;
        buckets[4 * k + 3] = 13;
        partial_sorting_shift_buckets_32s_6k(k as SaSint, &mut buckets);
        assert_eq!(buckets[0], 10);
        assert_eq!(buckets[1], 11);
        assert_eq!(buckets[4], 12);
        assert_eq!(buckets[5], 13);
    }

    #[test]
    fn libsais64_partial_sorting_scan_right_to_left_8u_emits_induced_suffixes() {
        let t = vec![0_u8, 1, 2, 1, 0];
        let mut sa = vec![0, 0, 4 | SAINT_MIN];
        let mut buckets = vec![0; 4 * ALPHABET_SIZE];
        buckets[buckets_index2(1, 1)] = 2;

        let d = partial_sorting_scan_right_to_left_8u(&t, &mut sa, &mut buckets, 0, 2, 1);

        assert_eq!(d, 1);
        assert_eq!(sa[1], 3 | SAINT_MIN);
        assert_eq!(buckets[buckets_index2(1, 1)], 1);
        assert_eq!(buckets[2 * ALPHABET_SIZE + buckets_index2(1, 1)], 1);
    }

    #[test]
    fn libsais64_partial_gsa_scan_right_to_left_8u_skips_separator_bucket() {
        let t = vec![1_u8, 0, 0];
        let mut sa = vec![0, 2 | SAINT_MIN];
        let mut buckets = vec![0; 4 * ALPHABET_SIZE];
        buckets[buckets_index2(0, 1)] = 2;

        let d = partial_gsa_scan_right_to_left_8u(&t, &mut sa, &mut buckets, 0, 1, 1);

        assert_eq!(d, 1);
        assert_eq!(sa, vec![0, 2 | SAINT_MIN]);
        assert_eq!(buckets[buckets_index2(0, 1)], 2);
    }

    #[test]
    fn libsais64_partial_sorting_scan_right_to_left_32s_6k_emits_induced_suffixes() {
        let t = vec![0, 1, 2, 1, 0];
        let mut sa = vec![0, 0, 4 | SAINT_MIN];
        let mut buckets = vec![0; 4 * 3];
        buckets[buckets_index4(1, 1)] = 2;

        let d = partial_sorting_scan_right_to_left_32s_6k(&t, &mut sa, &mut buckets, 0, 2, 1);

        assert_eq!(d, 1);
        assert_eq!(sa[1], 3 | SAINT_MIN);
        assert_eq!(buckets[buckets_index4(1, 1)], 1);
        assert_eq!(buckets[buckets_index4(1, 1) + 2], 1);
    }

    #[test]
    fn libsais64_partial_sorting_scan_right_to_left_32s_1k_omp_wraps_sequential_version() {
        let t = vec![0, 1, 2, 1, 0];
        let mut sa = vec![0, 0, 4];
        let mut buckets = vec![0; 3];
        buckets[1] = 2;
        let mut thread_state = alloc_thread_state(2).unwrap();

        partial_sorting_scan_right_to_left_32s_1k_omp(
            &t,
            &mut sa,
            3,
            &mut buckets,
            2,
            &mut thread_state,
        );

        assert_eq!(sa[1], 3 | SAINT_MIN);
        assert_eq!(buckets[1], 1);
    }

    #[test]
    fn libsais64_partial_sorting_scan_right_to_left_32s_6k_block_gather_records_symbols() {
        let t = vec![0, 1, 2, 1, 0];
        let sa = vec![0, 4 | SAINT_MIN, 0];
        let mut cache = vec![ThreadCache::default(); sa.len()];

        partial_sorting_scan_right_to_left_32s_6k_block_gather(&t, &sa, &mut cache, 1, 1);

        assert_eq!(cache[0].index, 4 | SAINT_MIN);
        assert_eq!(cache[0].symbol, buckets_index4(1, 1) as SaSint);
    }

    #[test]
    fn libsais64_partial_sorting_scan_right_to_left_32s_4k_block_gather_zeroes_positive_entries() {
        let t = vec![0, 1, 2, 1, 0];
        let mut sa = vec![0, 4 | SUFFIX_GROUP_MARKER, 0];
        let mut cache = vec![ThreadCache::default(); sa.len()];

        partial_sorting_scan_right_to_left_32s_4k_block_gather(&t, &mut sa, &mut cache, 1, 1);

        assert_eq!(sa[1], 0);
        assert_eq!(cache[0].index, 4 | SUFFIX_GROUP_MARKER);
        assert_eq!(cache[0].symbol, buckets_index2(1, 1) as SaSint);
    }

    #[test]
    fn libsais64_partial_sorting_scan_right_to_left_32s_1k_block_gather_stores_preinduced_entries()
    {
        let t = vec![0, 1, 2, 1, 0];
        let mut sa = vec![0, 4, 0];
        let mut cache = vec![ThreadCache::default(); sa.len()];

        partial_sorting_scan_right_to_left_32s_1k_block_gather(&t, &mut sa, &mut cache, 1, 1);

        assert_eq!(sa[1], 0);
        assert_eq!(cache[0].index, 3 | SAINT_MIN);
        assert_eq!(cache[0].symbol, 1);
    }

    #[test]
    fn libsais64_partial_sorting_scan_right_to_left_32s_6k_block_sort_updates_bucket_and_marker_state(
    ) {
        let t = vec![0, 1, 2, 1, 0];
        let mut cache = vec![ThreadCache::default(); 3];
        cache[0].index = 4 | SAINT_MIN;
        cache[0].symbol = buckets_index4(1, 1) as SaSint;
        let mut buckets = vec![0; 4 * 3];
        buckets[buckets_index4(1, 1)] = 2;

        let d = partial_sorting_scan_right_to_left_32s_6k_block_sort(
            &t,
            &mut buckets,
            0,
            &mut cache,
            1,
            1,
        );

        assert_eq!(d, 1);
        assert_eq!(cache[0].index, 3 | SAINT_MIN);
        assert_eq!(buckets[buckets_index4(1, 1)], 1);
        assert_eq!(buckets[buckets_index4(1, 1) + 2], 1);
    }

    #[test]
    fn libsais64_partial_sorting_scan_right_to_left_32s_1k_block_omp_places_cached_suffixes() {
        let t = vec![0, 1, 2, 1, 0];
        let mut sa = vec![0, 4, 0];
        let mut buckets = vec![0; 3];
        buckets[1] = 2;
        let mut cache = vec![ThreadCache::default(); sa.len()];

        partial_sorting_scan_right_to_left_32s_1k_block_omp(
            &t,
            &mut sa,
            &mut buckets,
            &mut cache,
            1,
            1,
            2,
        );

        assert_eq!(sa[1], 3 | SAINT_MIN);
        assert_eq!(buckets[1], 1);
    }

    #[test]
    fn libsais64_partial_sorting_scan_right_to_left_32s_1k_block_omp_uses_relative_cache() {
        let block_start = 20_000usize;
        let block_size = 16_384usize;
        let n = block_start + block_size + 8;
        let t = vec![1; n];
        let suffixes: Vec<SaSint> = (2..2 + block_size).map(|i| i as SaSint).collect();

        let mut sa_single = vec![0; n];
        sa_single[block_start..block_start + block_size].copy_from_slice(&suffixes);
        let mut sa_threaded = sa_single.clone();
        let mut bucket_single = vec![0, block_size as SaSint];
        let mut bucket_threaded = bucket_single.clone();
        let mut cache = vec![ThreadCache::default(); 4 * LIBSAIS_PER_THREAD_CACHE_SIZE];

        partial_sorting_scan_right_to_left_32s_1k(
            &t,
            &mut sa_single,
            &mut bucket_single,
            block_start as FastSint,
            block_size as FastSint,
        );
        partial_sorting_scan_right_to_left_32s_1k_block_omp(
            &t,
            &mut sa_threaded,
            &mut bucket_threaded,
            &mut cache,
            block_start as FastSint,
            block_size as FastSint,
            4,
        );

        assert_eq!(sa_threaded, sa_single);
        assert_eq!(bucket_threaded, bucket_single);
    }

    #[test]
    fn libsais64_partial_sorting_gather_lms_suffixes_32s_4k_compacts_negative_marked_entries() {
        let mut sa = vec![1 | SUFFIX_GROUP_MARKER, -3, 5 | SUFFIX_GROUP_MARKER, -7];
        let n = sa.len() as FastSint;

        let l = partial_sorting_gather_lms_suffixes_32s_4k(&mut sa, 0, n);

        assert_eq!(l, 2);
        assert_eq!(sa[0], (SAINT_MIN | SUFFIX_GROUP_MARKER) - 3);
        assert_eq!(sa[1], (SAINT_MIN | SUFFIX_GROUP_MARKER) - 7);
    }

    #[test]
    fn libsais64_partial_sorting_gather_lms_suffixes_32s_1k_compacts_negative_marked_entries() {
        let mut sa = vec![1, -3, 5, -7];
        let n = sa.len() as FastSint;

        let l = partial_sorting_gather_lms_suffixes_32s_1k(&mut sa, 0, n);

        assert_eq!(l, 2);
        assert_eq!(sa[0], SAINT_MAX - 2);
        assert_eq!(sa[1], SAINT_MAX - 6);
    }

    #[test]
    fn libsais64_partial_sorting_gather_lms_suffixes_32s_4k_omp_wraps_sequential_version() {
        let mut sa = vec![1 | SUFFIX_GROUP_MARKER, -3, 5 | SUFFIX_GROUP_MARKER, -7];
        let mut thread_state = alloc_thread_state(2).unwrap();

        partial_sorting_gather_lms_suffixes_32s_4k_omp(&mut sa, 4, 2, &mut thread_state);

        assert_eq!(sa[0], (SAINT_MIN | SUFFIX_GROUP_MARKER) - 3);
        assert_eq!(sa[1], (SAINT_MIN | SUFFIX_GROUP_MARKER) - 7);
    }

    #[test]
    fn libsais64_partial_sorting_gather_lms_suffixes_32s_1k_omp_wraps_sequential_version() {
        let mut sa = vec![1, -3, 5, -7];
        let mut thread_state = alloc_thread_state(2).unwrap();

        partial_sorting_gather_lms_suffixes_32s_1k_omp(&mut sa, 4, 2, &mut thread_state);

        assert_eq!(sa[0], SAINT_MAX - 2);
        assert_eq!(sa[1], SAINT_MAX - 6);
    }

    #[test]
    fn libsais64_partial_sorting_gather_lms_suffixes_32s_omp_uses_block_partition() {
        let n = 65_600usize;
        let input_4k: Vec<SaSint> = (0..n)
            .map(|i| {
                let value = (i as SaSint) | SUFFIX_GROUP_MARKER;
                if i % 5 == 0 {
                    value | SAINT_MIN
                } else {
                    value
                }
            })
            .collect();
        let count_4k = input_4k.iter().filter(|&&value| value < 0).count();

        let mut single = input_4k.clone();
        let mut threaded = input_4k;
        let mut thread_state = alloc_thread_state(4).unwrap();
        partial_sorting_gather_lms_suffixes_32s_4k_omp(&mut single, n as SaSint, 1, &mut []);
        partial_sorting_gather_lms_suffixes_32s_4k_omp(
            &mut threaded,
            n as SaSint,
            4,
            &mut thread_state,
        );
        assert_eq!(&threaded[..count_4k], &single[..count_4k]);

        let input_1k: Vec<SaSint> = (0..n)
            .map(|i| {
                let value = i as SaSint;
                if i % 7 == 0 {
                    value | SAINT_MIN
                } else {
                    value
                }
            })
            .collect();
        let count_1k = input_1k.iter().filter(|&&value| value < 0).count();

        let mut single = input_1k.clone();
        let mut threaded = input_1k;
        partial_sorting_gather_lms_suffixes_32s_1k_omp(&mut single, n as SaSint, 1, &mut []);
        partial_sorting_gather_lms_suffixes_32s_1k_omp(
            &mut threaded,
            n as SaSint,
            4,
            &mut thread_state,
        );
        assert_eq!(&threaded[..count_1k], &single[..count_1k]);
    }

    #[test]
    fn libsais64_partial_sorting_shift_markers_8u_omp_toggles_segment_markers() {
        let mut sa = vec![1 | SAINT_MIN, 2 | SAINT_MIN, 3, 4 | SAINT_MIN, 5];
        let mut buckets = vec![0; 6 * ALPHABET_SIZE];
        buckets[4 * ALPHABET_SIZE + buckets_index2(1, 0)] = 5;
        buckets[buckets_index2(0, 0)] = 0;
        let len = sa.len() as SaSint;
        partial_sorting_shift_markers_8u_omp(&mut sa, len, &buckets, 1);
        assert!(sa.iter().any(|&v| (v & SAINT_MIN) == 0));
    }

    #[test]
    fn libsais64_partial_sorting_shift_markers_32s_6k_omp_toggles_segment_markers() {
        let mut sa = vec![1 | SAINT_MIN, 2 | SAINT_MIN, 3, 4 | SAINT_MIN, 5];
        let k = 3usize;
        let mut buckets = vec![0; 6 * k];
        buckets[buckets_index4(1, 0)] = 5;
        buckets[4 * k + buckets_index2(0, 0)] = 0;
        partial_sorting_shift_markers_32s_6k_omp(&mut sa, k as SaSint, &buckets, 1);
        assert!(sa.iter().any(|&v| (v & SAINT_MIN) == 0));
    }

    #[test]
    fn libsais64_partial_sorting_shift_markers_32s_4k_toggles_group_markers() {
        let mut sa = vec![
            1 | SUFFIX_GROUP_MARKER,
            2 | SUFFIX_GROUP_MARKER,
            3,
            4 | SUFFIX_GROUP_MARKER,
        ];
        let len = sa.len() as SaSint;
        partial_sorting_shift_markers_32s_4k(&mut sa, len);
        assert!(sa.iter().any(|&v| (v & SUFFIX_GROUP_MARKER) == 0));
    }

    #[test]
    fn libsais64_clear_lms_suffixes_omp_zeroes_requested_bucket_ranges() {
        let mut sa = vec![5, 4, 3, 2, 1, 9];
        let n = sa.len() as SaSint;
        let bucket_start = vec![1, 4, 5];
        let bucket_end = vec![3, 5, 5];

        clear_lms_suffixes_omp(&mut sa, n, 3, &bucket_start, &bucket_end, 2);

        assert_eq!(sa, vec![5, 0, 0, 2, 0, 9]);
    }

    #[test]
    fn libsais64_final_bwt_scan_left_to_right_8u_rewrites_sa_and_induces_suffixes() {
        let t = vec![0_u8, 1, 2, 1, 0];
        let mut sa = vec![1, 0, 0];
        let mut induction_bucket = vec![0, 1, 3];

        final_bwt_scan_left_to_right_8u(&t, &mut sa, &mut induction_bucket, 0, 1);

        assert_eq!(sa[0], 0);
        assert_eq!(induction_bucket[0], 1);
    }

    #[test]
    fn libsais64_final_bwt_aux_scan_left_to_right_8u_updates_sampling_array() {
        let t = vec![0_u8, 1, 2, 1, 0];
        let mut sa = vec![1, 0, 0];
        let mut induction_bucket = vec![0, 1, 3];
        let mut i_out = vec![0; 2];

        final_bwt_aux_scan_left_to_right_8u(
            &t,
            &mut sa,
            0,
            &mut i_out,
            &mut induction_bucket,
            0,
            1,
        );

        assert_eq!(i_out[0], 1);
    }

    #[test]
    fn libsais64_final_sorting_scan_left_to_right_8u_clears_marker_and_places_suffix() {
        let t = vec![0_u8, 1, 2, 1, 0];
        let mut sa = vec![1, 0, 0];
        let mut induction_bucket = vec![0, 1, 3];

        final_sorting_scan_left_to_right_8u(&t, &mut sa, &mut induction_bucket, 0, 1);

        assert_eq!(sa[0], 0);
        assert_eq!(induction_bucket[0], 1);
    }

    #[test]
    fn libsais64_final_sorting_scan_left_to_right_32s_clears_marker_and_places_suffix() {
        let t = vec![0, 1, 2, 1, 0];
        let mut sa = vec![1, 0, 0];
        let mut induction_bucket = vec![0, 1, 3];

        final_sorting_scan_left_to_right_32s(&t, &mut sa, &mut induction_bucket, 0, 1);

        assert_eq!(sa[0], 0);
        assert_eq!(induction_bucket[0], 1);
    }

    #[test]
    fn libsais64_final_bwt_scan_left_to_right_8u_block_prepare_records_cache_and_counts() {
        let t = vec![0_u8, 1, 2, 1, 0];
        let mut sa = vec![1, 2, 0];
        let mut buckets = vec![99; ALPHABET_SIZE];
        let mut cache = vec![ThreadCache::default(); 4];

        let count = final_bwt_scan_left_to_right_8u_block_prepare(
            &t,
            &mut sa,
            ALPHABET_SIZE as SaSint,
            &mut buckets,
            &mut cache,
            0,
            2,
        );

        assert_eq!(count, 2);
        assert_eq!(sa[0] & SAINT_MAX, 0);
        assert_eq!(sa[1], 1 | SAINT_MIN);
        assert_eq!(buckets[0], 1);
        assert_eq!(buckets[1], 1);
        assert_eq!(cache[0].symbol, 0);
        assert_eq!(cache[0].index & SAINT_MAX, 0);
        assert_eq!(cache[1].symbol, 1);
        assert_eq!(cache[1].index & SAINT_MAX, 1);
    }

    #[test]
    fn libsais64_final_sorting_scan_left_to_right_8u_omp_wraps_sequential_behavior() {
        let t = vec![0_u8, 1, 2, 1, 0];
        let mut sa = vec![0; t.len()];
        let mut induction_bucket = vec![0, 1, 3];
        let mut expected_sa = sa.clone();
        let mut expected_bucket = induction_bucket.clone();

        final_sorting_scan_left_to_right_8u_omp(
            &t,
            &mut expected_sa,
            t.len() as FastSint,
            ALPHABET_SIZE as SaSint,
            &mut expected_bucket,
            1,
            &mut [],
        );

        let mut thread_state = alloc_thread_state(2).unwrap();

        final_sorting_scan_left_to_right_8u_omp(
            &t,
            &mut sa,
            t.len() as FastSint,
            ALPHABET_SIZE as SaSint,
            &mut induction_bucket,
            2,
            &mut thread_state,
        );

        assert_eq!(sa, expected_sa);
        assert_eq!(induction_bucket, expected_bucket);
    }

    #[test]
    fn libsais64_final_bwt_scan_right_to_left_8u_returns_zero_index_and_induces_suffixes() {
        let t = vec![0_u8, 1, 2, 1, 0];
        let mut sa = vec![0, 2, 0];
        let mut induction_bucket = vec![1, 2, 3];

        let index = final_bwt_scan_right_to_left_8u(&t, &mut sa, &mut induction_bucket, 0, 2);

        assert_eq!(index, 0);
        assert_eq!(sa[1], 1);
        assert_eq!(induction_bucket[1], 1);
    }

    #[test]
    fn libsais64_final_sorting_scan_right_to_left_8u_omp_matches_sequential_path() {
        let t = vec![0_u8, 1, 2, 1, 0];
        let mut sa = vec![0, 2, 0, 0];
        let mut induction_bucket = vec![1, 2, 3];
        let mut expected_sa = sa.clone();
        let mut expected_bucket = induction_bucket.clone();

        final_sorting_scan_right_to_left_8u_omp(
            &t,
            &mut expected_sa,
            0,
            2,
            ALPHABET_SIZE as SaSint,
            &mut expected_bucket,
            1,
            &mut [],
        );

        let mut thread_state = alloc_thread_state(2).unwrap();
        final_sorting_scan_right_to_left_8u_omp(
            &t,
            &mut sa,
            0,
            2,
            ALPHABET_SIZE as SaSint,
            &mut induction_bucket,
            2,
            &mut thread_state,
        );

        assert_eq!(sa, expected_sa);
        assert_eq!(induction_bucket, expected_bucket);
    }

    #[test]
    fn libsais64_induce_final_order_8u_omp_non_bwt_matches_direct_final_scans() {
        let t = vec![0_u8, 1, 2, 1, 0];
        let mut sa = vec![0, 2, 0, 0, 0];
        let mut buckets = vec![0; 8 * ALPHABET_SIZE];
        buckets[6 * ALPHABET_SIZE..6 * ALPHABET_SIZE + 3].copy_from_slice(&[0, 1, 3]);
        buckets[7 * ALPHABET_SIZE..7 * ALPHABET_SIZE + 3].copy_from_slice(&[2, 4, 5]);

        let mut expected_sa = sa.clone();
        let mut expected_left = vec![0, 1, 3];
        let mut expected_right = vec![2, 4, 5];
        final_sorting_scan_left_to_right_8u_omp(
            &t,
            &mut expected_sa,
            t.len() as FastSint,
            ALPHABET_SIZE as SaSint,
            &mut expected_left,
            1,
            &mut [],
        );
        final_sorting_scan_right_to_left_8u_omp(
            &t,
            &mut expected_sa,
            0,
            t.len() as FastSint,
            ALPHABET_SIZE as SaSint,
            &mut expected_right,
            1,
            &mut [],
        );

        let mut thread_state = alloc_thread_state(2).unwrap();
        let result = induce_final_order_8u_omp(
            &t,
            &mut sa,
            t.len() as SaSint,
            ALPHABET_SIZE as SaSint,
            LIBSAIS_FLAGS_NONE,
            0,
            None,
            &mut buckets,
            2,
            &mut thread_state,
        );

        assert_eq!(result, 0);
        assert_eq!(sa, expected_sa);
        assert_eq!(
            &buckets[6 * ALPHABET_SIZE..6 * ALPHABET_SIZE + 3],
            expected_left.as_slice()
        );
        assert_eq!(
            &buckets[7 * ALPHABET_SIZE..7 * ALPHABET_SIZE + 3],
            expected_right.as_slice()
        );
    }

    #[test]
    fn libsais64_count_helpers_match_c_predicates() {
        let sa = [1, -1, 0, -3, 4, 0, -9];
        assert_eq!(
            count_negative_marked_suffixes(&sa, 0, sa.len() as FastSint),
            3
        );
        assert_eq!(count_zero_marked_suffixes(&sa, 0, sa.len() as FastSint), 2);
        assert_eq!(count_negative_marked_suffixes(&sa, 2, 3), 1);
        assert_eq!(count_zero_marked_suffixes(&sa, 2, 3), 1);
    }

    #[test]
    fn libsais64_flip_suffix_markers_omp_toggles_saint_min_bits() {
        let mut sa = vec![1, -2, 3, -4];
        flip_suffix_markers_omp(&mut sa, 4, 1);
        assert_eq!(
            sa,
            vec![1 ^ SAINT_MIN, -2 ^ SAINT_MIN, 3 ^ SAINT_MIN, -4 ^ SAINT_MIN]
        );
    }

    #[test]
    fn libsais64_place_cached_suffixes_writes_indices_to_symbol_slots() {
        let mut sa = vec![0; 8];
        let cache = vec![
            ThreadCache {
                symbol: 2,
                index: 10,
            },
            ThreadCache {
                symbol: 5,
                index: 20,
            },
            ThreadCache {
                symbol: 1,
                index: 30,
            },
        ];

        place_cached_suffixes(&mut sa, &cache, 0, cache.len() as FastSint);

        assert_eq!(sa[2], 10);
        assert_eq!(sa[5], 20);
        assert_eq!(sa[1], 30);
    }

    #[test]
    fn libsais64_compact_and_place_cached_suffixes_discards_negative_symbols() {
        let mut sa = vec![0; 8];
        let mut cache = vec![
            ThreadCache {
                symbol: 2,
                index: 10,
            },
            ThreadCache {
                symbol: -1,
                index: 99,
            },
            ThreadCache {
                symbol: 5,
                index: 20,
            },
            ThreadCache {
                symbol: -4,
                index: 77,
            },
            ThreadCache {
                symbol: 1,
                index: 30,
            },
        ];
        let cache_len = cache.len() as FastSint;

        compact_and_place_cached_suffixes(&mut sa, &mut cache, 0, cache_len);

        assert_eq!(sa[2], 10);
        assert_eq!(sa[5], 20);
        assert_eq!(sa[1], 30);
        assert_eq!(
            cache[0],
            ThreadCache {
                symbol: 2,
                index: 10
            }
        );
        assert_eq!(
            cache[1],
            ThreadCache {
                symbol: 5,
                index: 20
            }
        );
        assert_eq!(
            cache[2],
            ThreadCache {
                symbol: 1,
                index: 30
            }
        );
    }

    #[test]
    fn libsais64_gather_lms_suffixes_32s_collects_expected_suffix_starts() {
        let t = vec![2, 1, 3, 1, 0];
        let mut sa = vec![0; t.len()];
        let m = gather_lms_suffixes_32s(&t, &mut sa, t.len() as SaSint);
        assert!(m >= 0);
        assert!(sa
            .iter()
            .all(|&value| value >= 0 && value <= t.len() as SaSint));
        assert!(sa[t.len() - 1] >= 1 && sa[t.len() - 1] <= t.len() as SaSint - 1);
    }

    #[test]
    fn libsais64_gather_compacted_lms_suffixes_32s_skips_negative_marked_symbols() {
        let t = vec![2, -1, 3, 1, 0];
        let mut sa = vec![0; t.len()];
        let m = gather_compacted_lms_suffixes_32s(&t, &mut sa, t.len() as SaSint);
        assert!(m >= 0);
        assert!(sa
            .iter()
            .all(|&value| value >= 0 && value <= t.len() as SaSint));
    }

    #[test]
    fn libsais64_count_lms_suffixes_32s_2k_counts_two_bucket_categories() {
        let t = vec![2, 1, 3, 1, 0];
        let mut buckets = vec![0; 2 * 4];
        count_lms_suffixes_32s_2k(&t, t.len() as SaSint, 4, &mut buckets);
        assert_eq!(buckets.iter().sum::<SaSint>(), t.len() as SaSint);
    }

    #[test]
    fn libsais64_count_lms_suffixes_32s_4k_counts_four_bucket_categories() {
        let t = vec![2, 1, 3, 1, 0];
        let mut buckets = vec![0; 4 * 4];
        count_lms_suffixes_32s_4k(&t, t.len() as SaSint, 4, &mut buckets);
        assert_eq!(buckets.iter().sum::<SaSint>(), t.len() as SaSint);
    }

    #[test]
    fn libsais64_count_compacted_lms_suffixes_32s_2k_masks_saint_bits() {
        let t = vec![2, SAINT_MIN | 1, 3, 1, 0];
        let mut buckets = vec![0; 2 * 4];
        count_compacted_lms_suffixes_32s_2k(&t, t.len() as SaSint, 4, &mut buckets);
        assert_eq!(buckets.iter().sum::<SaSint>(), t.len() as SaSint);
    }

    #[test]
    fn libsais64_count_and_gather_lms_suffixes_8u_updates_sa_and_buckets() {
        let t = vec![2_u8, 1, 3, 1, 0];
        let mut sa = vec![0; t.len()];
        let mut buckets = vec![0; 4 * ALPHABET_SIZE];
        let m = count_and_gather_lms_suffixes_8u(
            &t,
            &mut sa,
            t.len() as SaSint,
            &mut buckets,
            0,
            t.len() as FastSint,
        );
        assert_eq!(m, 1);
        assert_eq!(sa[t.len() - 1], 1);
        assert_eq!(buckets.iter().sum::<SaSint>(), t.len() as SaSint);
    }

    #[test]
    fn libsais64_count_and_gather_lms_suffixes_8u_omp_preserves_sequential_wrapper_behavior() {
        let t = vec![2_u8, 1, 3, 1, 0];
        let mut sa = vec![0; t.len()];
        let mut buckets = vec![0; 4 * ALPHABET_SIZE];
        let mut thread_state = alloc_thread_state(2).unwrap();
        let m = count_and_gather_lms_suffixes_8u_omp(
            &t,
            &mut sa,
            t.len() as SaSint,
            &mut buckets,
            2,
            &mut thread_state,
        );
        assert_eq!(m, 1);
        assert_eq!(buckets.iter().sum::<SaSint>(), t.len() as SaSint);
    }

    #[test]
    fn libsais64_get_bucket_stride_prefers_aligned_sizes_when_space_allows() {
        assert_eq!(get_bucket_stride(8192, 1000, 2), 1024);
        assert_eq!(get_bucket_stride(256, 17, 2), 32);
        assert_eq!(get_bucket_stride(8, 17, 2), 17);
    }

    #[test]
    fn libsais64_count_suffixes_32s_counts_symbol_histogram() {
        let t = vec![2, 1, 2, 3, 1, 0, 2];
        let mut buckets = vec![0; 4];
        count_suffixes_32s(&t, t.len() as SaSint, 4, &mut buckets);
        assert_eq!(buckets, vec![1, 2, 3, 1]);
    }

    #[test]
    fn libsais64_initialize_buckets_start_and_end_8u_sets_ranges_and_freq() {
        let mut buckets = vec![0; 8 * ALPHABET_SIZE];
        buckets[buckets_index4(0, 0)] = 1;
        buckets[buckets_index4(1, 1)] = 2;
        buckets[buckets_index4(2, 3)] = 3;
        let mut freq = vec![0; ALPHABET_SIZE];
        let k = initialize_buckets_start_and_end_8u(&mut buckets, Some(&mut freq));
        assert_eq!(k, 3);
        assert_eq!(freq[0], 1);
        assert_eq!(freq[1], 2);
        assert_eq!(freq[2], 3);
        assert_eq!(buckets[6 * ALPHABET_SIZE], 0);
        assert_eq!(buckets[7 * ALPHABET_SIZE], 1);
        assert_eq!(buckets[6 * ALPHABET_SIZE + 1], 1);
        assert_eq!(buckets[7 * ALPHABET_SIZE + 1], 3);
    }

    #[test]
    fn libsais64_initialize_buckets_start_and_end_32s_6k_sets_prefix_ranges() {
        let k = 3;
        let mut buckets = vec![0; 6 * k];
        buckets[buckets_index4(0, 0)] = 1;
        buckets[buckets_index4(0, 1)] = 2;
        buckets[buckets_index4(1, 2)] = 3;
        buckets[buckets_index4(2, 3)] = 4;
        initialize_buckets_start_and_end_32s_6k(k as SaSint, &mut buckets);
        assert_eq!(&buckets[4 * k..5 * k], &[0, 3, 6]);
        assert_eq!(&buckets[5 * k..6 * k], &[3, 6, 10]);
    }

    #[test]
    fn libsais64_initialize_buckets_start_and_end_32s_4k_sets_prefix_ranges() {
        let k = 3;
        let mut buckets = vec![0; 4 * k];
        buckets[buckets_index2(0, 0)] = 1;
        buckets[buckets_index2(0, 1)] = 2;
        buckets[buckets_index2(1, 0)] = 3;
        buckets[buckets_index2(2, 1)] = 4;
        initialize_buckets_start_and_end_32s_4k(k as SaSint, &mut buckets);
        assert_eq!(&buckets[2 * k..3 * k], &[0, 3, 6]);
        assert_eq!(&buckets[3 * k..4 * k], &[3, 6, 10]);
    }

    #[test]
    fn libsais64_initialize_buckets_end_32s_2k_rewrites_first_lanes_to_end_positions() {
        let k = 3;
        let mut buckets = vec![1, 2, 3, 4, 5, 6];
        initialize_buckets_end_32s_2k(k as SaSint, &mut buckets);
        assert_eq!(buckets[0], 3);
        assert_eq!(buckets[2], 10);
        assert_eq!(buckets[4], 21);
    }

    #[test]
    fn libsais64_initialize_buckets_start_and_end_32s_2k_copies_start_positions() {
        let k = 3;
        let mut buckets = vec![3, 2, 10, 4, 21, 6];
        initialize_buckets_start_and_end_32s_2k(k as SaSint, &mut buckets);
        assert_eq!(&buckets[..k], &[3, 10, 21]);
        assert_eq!(&buckets[k..2 * k], &[0, 3, 10]);
    }

    #[test]
    fn libsais64_initialize_buckets_start_32s_1k_builds_prefix_starts() {
        let mut buckets = vec![1, 2, 3];
        initialize_buckets_start_32s_1k(3, &mut buckets);
        assert_eq!(buckets, vec![0, 1, 3]);
    }

    #[test]
    fn libsais64_initialize_buckets_end_32s_1k_builds_prefix_ends() {
        let mut buckets = vec![1, 2, 3];
        initialize_buckets_end_32s_1k(3, &mut buckets);
        assert_eq!(buckets, vec![1, 3, 6]);
    }

    #[test]
    fn libsais64_initialize_buckets_for_lms_suffixes_radix_sort_8u_returns_total_lms_slots() {
        let t = vec![2_u8, 1, 3, 1, 0];
        let mut buckets = vec![0; 6 * ALPHABET_SIZE];
        buckets[buckets_index4(0, 1)] = 1;
        buckets[buckets_index4(1, 3)] = 2;
        let sum = initialize_buckets_for_lms_suffixes_radix_sort_8u(&t, &mut buckets, 4);
        assert!(sum >= 0);
    }

    #[test]
    fn libsais64_initialize_buckets_for_lms_suffixes_radix_sort_32s_2k_rewrites_two_lane_prefixes()
    {
        let t = vec![2, 1, 3, 1, 0];
        let mut buckets = vec![0; 2 * 4];
        initialize_buckets_for_lms_suffixes_radix_sort_32s_2k(&t, 4, &mut buckets, 4);
        assert!(buckets.iter().any(|&v| v != 0));
    }

    #[test]
    fn libsais64_initialize_buckets_for_lms_suffixes_radix_sort_32s_6k_returns_total_lms_slots() {
        let t = vec![2, 1, 3, 1, 0];
        let mut buckets = vec![0; 6 * 4];
        buckets[buckets_index4(0, 1)] = 1;
        buckets[buckets_index4(1, 3)] = 2;
        let sum = initialize_buckets_for_lms_suffixes_radix_sort_32s_6k(&t, 4, &mut buckets, 4);
        assert!(sum >= 0);
    }

    #[test]
    fn libsais64_initialize_buckets_for_radix_and_partial_sorting_32s_4k_sets_start_end_views() {
        let t = vec![2, 1, 3, 1, 0];
        let k = 4usize;
        let mut buckets = vec![0; 4 * k];
        buckets[buckets_index2(0, 0)] = 1;
        buckets[buckets_index2(0, 1)] = 2;
        buckets[buckets_index2(1, 0)] = 3;
        initialize_buckets_for_radix_and_partial_sorting_32s_4k(&t, k as SaSint, &mut buckets, 4);
        assert_eq!(buckets[2 * k], 0);
        assert!(buckets[3 * k] >= buckets[2 * k]);
    }

    #[test]
    fn libsais64_initialize_buckets_for_partial_sorting_8u_sets_start_and_distinct_views() {
        let t = vec![2_u8, 1, 3, 1, 0];
        let mut buckets = vec![0; 6 * ALPHABET_SIZE];
        buckets[buckets_index4(0, 0)] = 1;
        buckets[buckets_index4(0, 2)] = 2;
        initialize_buckets_for_partial_sorting_8u(&t, &mut buckets, 4, 3);
        assert!(buckets[0] >= 4);
        assert!(buckets[1] >= 0);
        assert!(buckets[4 * ALPHABET_SIZE] >= 4);
    }

    #[test]
    fn libsais64_initialize_buckets_for_partial_sorting_32s_6k_rewrites_bucket_views() {
        let t = vec![2, 1, 3, 1, 0];
        let k = 4usize;
        let mut buckets = vec![0; 6 * k];
        buckets[buckets_index4(0, 0)] = 1;
        buckets[buckets_index4(0, 1)] = 2;
        buckets[buckets_index4(1, 2)] = 3;
        initialize_buckets_for_partial_sorting_32s_6k(&t, k as SaSint, &mut buckets, 4, 3);
        assert!(buckets[0] >= 4);
        assert!(buckets[4 * k] >= 4);
    }

    #[test]
    fn libsais64_place_lms_suffixes_interval_32s_4k_moves_suffixes_into_bucket_intervals() {
        let mut sa = vec![10, 11, 12, 13, 14];
        let k = 3usize;
        let mut buckets = vec![0; 4 * k];
        buckets[buckets_index2(0, 1)] = 0;
        buckets[buckets_index2(1, 1)] = 2;
        buckets[buckets_index2(2, 1)] = 3;
        buckets[3 * k] = 2;
        buckets[3 * k + 1] = 5;

        place_lms_suffixes_interval_32s_4k(&mut sa, 5, k as SaSint, 5, &buckets);

        assert_eq!(sa, vec![0, 0, 0, 0, 14]);
    }

    #[test]
    fn libsais64_place_lms_suffixes_interval_32s_2k_moves_suffixes_into_bucket_intervals() {
        let mut sa = vec![10, 11, 12, 13, 14];
        let mut buckets = vec![0; 2 * 3];
        buckets[buckets_index2(0, 0)] = 2;
        buckets[buckets_index2(0, 1)] = 0;
        buckets[buckets_index2(1, 0)] = 5;
        buckets[buckets_index2(1, 1)] = 2;
        buckets[buckets_index2(2, 0)] = 5;
        buckets[buckets_index2(2, 1)] = 3;

        place_lms_suffixes_interval_32s_2k(&mut sa, 5, 3, 5, &buckets);

        assert_eq!(sa, vec![0, 0, 0, 0, 14]);
    }

    #[test]
    fn libsais64_place_lms_suffixes_interval_32s_1k_places_suffixes_by_symbol_bucket() {
        let t = vec![0, 1, 1, 2, 2];
        let mut sa = vec![1, 2, 3, 4, 99];
        let buckets = vec![0, 2, 5];

        place_lms_suffixes_interval_32s_1k(&t, &mut sa, 3, 4, &buckets);

        assert_eq!(sa, vec![1, 2, 0, 3, 4]);
    }

    #[test]
    fn libsais64_accumulate_counts_helpers_match_prefix_bucket_addition() {
        let mut bucket00 = vec![4, 5, 6];
        let bucket01 = vec![1, 2, 3];
        let bucket02 = vec![7, 8, 9];
        let bucket03 = vec![10, 11, 12];
        let bucket04 = vec![13, 14, 15];
        let bucket05 = vec![16, 17, 18];
        let bucket06 = vec![19, 20, 21];
        let bucket07 = vec![22, 23, 24];
        let bucket08 = vec![25, 26, 27];

        accumulate_counts_s32_2(&mut bucket00, &bucket01);
        assert_eq!(bucket00, vec![5, 7, 9]);

        accumulate_counts_s32_3(&mut bucket00, &bucket01, &bucket02);
        assert_eq!(bucket00, vec![13, 17, 21]);

        accumulate_counts_s32_4(&mut bucket00, &bucket01, &bucket02, &bucket03);
        assert_eq!(bucket00, vec![31, 38, 45]);

        accumulate_counts_s32_5(&mut bucket00, &bucket01, &bucket02, &bucket03, &bucket04);
        assert_eq!(bucket00, vec![62, 73, 84]);

        accumulate_counts_s32_6(
            &mut bucket00,
            &bucket01,
            &bucket02,
            &bucket03,
            &bucket04,
            &bucket05,
        );
        assert_eq!(bucket00, vec![109, 125, 141]);

        accumulate_counts_s32_7(
            &mut bucket00,
            &bucket01,
            &bucket02,
            &bucket03,
            &bucket04,
            &bucket05,
            &bucket06,
        );
        assert_eq!(bucket00, vec![175, 197, 219]);

        accumulate_counts_s32_8(
            &mut bucket00,
            &bucket01,
            &bucket02,
            &bucket03,
            &bucket04,
            &bucket05,
            &bucket06,
            &bucket07,
        );
        assert_eq!(bucket00, vec![263, 292, 321]);

        accumulate_counts_s32_9(
            &mut bucket00,
            &bucket01,
            &bucket02,
            &bucket03,
            &bucket04,
            &bucket05,
            &bucket06,
            &bucket07,
            &bucket08,
        );
        assert_eq!(bucket00, vec![376, 413, 450]);
    }

    #[test]
    fn libsais64_accumulate_counts_s32_matches_dispatch_for_small_bucket_counts() {
        let mut buckets = vec![1, 2, 3, 4, 5, 6, 7, 8];
        accumulate_counts_s32(&mut buckets, 2, 2, 4);
        assert_eq!(buckets, vec![1, 2, 3, 4, 5, 6, 16, 20]);
    }

    #[test]
    fn libsais64_accumulate_counts_s32_matches_dispatch_for_nine_buckets() {
        let mut buckets = vec![
            1, 10, 2, 20, 3, 30, 4, 40, 5, 50, 6, 60, 7, 70, 8, 80, 9, 90,
        ];
        accumulate_counts_s32(&mut buckets, 2, 2, 9);
        assert_eq!(
            buckets,
            vec![1, 10, 2, 20, 3, 30, 4, 40, 5, 50, 6, 60, 7, 70, 8, 80, 45, 450]
        );
    }

    #[test]
    fn libsais64_accumulate_counts_s32_matches_chunked_nine_then_tail_behavior() {
        let mut buckets = (1..=11).collect::<Vec<SaSint>>();
        accumulate_counts_s32(&mut buckets, 1, 1, 11);
        assert_eq!(buckets, vec![1, 2, 3, 4, 5, 6, 7, 8, 45, 10, 66]);
    }

    #[test]
    fn libsais64_final_sorting_scan_left_to_right_32s_block_omp_places_cached_suffixes() {
        let t = vec![0, 1, 2, 1, 0];
        let mut sa = vec![1, 2, 0, 0];
        let mut induction_bucket = vec![0, 1, 3];
        let mut cache = vec![ThreadCache::default(); LIBSAIS_PER_THREAD_CACHE_SIZE];

        final_sorting_scan_left_to_right_32s_block_omp(
            &t,
            &mut sa,
            &mut induction_bucket,
            &mut cache,
            0,
            2,
            2,
        );

        assert_eq!(sa[0] & SAINT_MAX, 0);
        assert_eq!(sa[1] & SAINT_MAX, 1);
        assert_eq!(induction_bucket[0], 1);
        assert_eq!(induction_bucket[1], 2);
    }

    #[test]
    fn libsais64_final_sorting_scan_left_to_right_8u_block_omp_uses_thread_buckets() {
        let block_start = 20_000usize;
        let block_size = 16_384usize;
        let n = block_start + block_size + 8;
        let t = vec![1_u8; n];
        let suffixes: Vec<SaSint> = (2..2 + block_size).map(|i| i as SaSint).collect();

        let mut expected_sa = vec![0; n];
        expected_sa[block_start..block_start + block_size].copy_from_slice(&suffixes);
        let mut threaded_sa = expected_sa.clone();
        let mut expected_bucket = vec![0; ALPHABET_SIZE];
        let mut threaded_bucket = expected_bucket.clone();
        let mut thread_state = alloc_thread_state(4).unwrap();

        final_sorting_scan_left_to_right_8u(
            &t,
            &mut expected_sa,
            &mut expected_bucket,
            block_start as FastSint,
            block_size as FastSint,
        );
        final_sorting_scan_left_to_right_8u_block_omp(
            &t,
            &mut threaded_sa,
            ALPHABET_SIZE as SaSint,
            &mut threaded_bucket,
            block_start as FastSint,
            block_size as FastSint,
            4,
            &mut thread_state,
        );

        assert_eq!(threaded_sa, expected_sa);
        assert_eq!(threaded_bucket, expected_bucket);
    }

    #[test]
    fn libsais64_final_sorting_scan_right_to_left_32s_block_omp_runs_block_pipeline() {
        let t = vec![0, 1, 2, 1, 0];
        let mut sa = vec![0, 2, 0, 0];
        let mut induction_bucket = vec![1, 2, 3];
        let mut expected_sa = sa.clone();
        let mut expected_bucket = induction_bucket.clone();
        let mut cache = vec![ThreadCache::default(); LIBSAIS_PER_THREAD_CACHE_SIZE];

        final_sorting_scan_right_to_left_32s(&t, &mut expected_sa, &mut expected_bucket, 0, 2);
        final_sorting_scan_right_to_left_32s_block_omp(
            &t,
            &mut sa,
            &mut induction_bucket,
            &mut cache,
            0,
            2,
            2,
        );

        assert_eq!(sa, expected_sa);
        assert_eq!(induction_bucket, expected_bucket);
    }

    #[test]
    fn libsais64_bwt_copy_8u_copies_low_bytes_from_suffix_array_storage() {
        let a = vec![65, 255, 256, -1];
        let mut u = vec![0_u8; 4];

        bwt_copy_8u(&mut u, &a, 4);

        assert_eq!(u, vec![65, 255, 0, 255]);
    }

    #[test]
    fn libsais64_bwt_copy_8u_omp_matches_sequential_copy() {
        let a = vec![1, 2, 3, 4, 5];
        let mut u = vec![0_u8; 5];

        bwt_copy_8u_omp(&mut u, &a, 5, 4);

        assert_eq!(u, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn libsais64_conversion_helpers_use_little_endian_word_layout() {
        let s = vec![11_u32, 22, 33, 44];
        let mut d = vec![0_u64; 4];
        convert_32u_to_64u(&s, &mut d, 1, 2);
        assert_eq!(d, vec![0, 22, 33, 0]);

        let mut words = vec![5_u32, 6, 7, 8, 0, 0, 0, 0];
        convert_inplace_32u_to_64u(&mut words, 0, 4);
        assert_eq!(words, vec![5, 0, 6, 0, 7, 0, 8, 0]);
        convert_inplace_64u_to_32u(&mut words, 0, 4);
        assert_eq!(&words[..4], &[5, 6, 7, 8]);

        let mut words = vec![9_u32, 10, 11, 12, 0, 0, 0, 0];
        convert_inplace_32u_to_64u_omp(&mut words, 4, 2);
        assert_eq!(words, vec![9, 0, 10, 0, 11, 0, 12, 0]);
    }

    #[test]
    fn libsais64_32bit_workspace_sizing_matches_upstream_capacity_rules() {
        assert_eq!(libsais64_new_32bit_fs(10, 4), Some(18));
        assert_eq!(libsais64_new_32bit_fs(i32::MAX as usize - 4, 100), Some(4));
        assert_eq!(libsais64_new_32bit_fs(i32::MAX as usize + 1, 0), None);
    }

    #[test]
    fn libsais64_32bit_suffix_adapter_widens_suffix_array_and_frequency() {
        let text = b"banana";
        let fs = 4;
        let new_fs = libsais64_new_32bit_fs(text.len(), fs).expect("small workspace");
        let mut sa64 = vec![-1; text.len() + fs as usize];
        let mut freq64 = vec![-1; ALPHABET_SIZE];
        let rc64 = libsais64_run_32bit_omp(text, &mut sa64, fs, Some(&mut freq64), 2, false)
            .expect("small input uses 32-bit adapter");

        let mut sa32 = vec![-1; text.len() + new_fs as usize];
        let mut freq32 = vec![-1; ALPHABET_SIZE];
        let rc32 = crate::libsais_omp(text, &mut sa32, new_fs, Some(&mut freq32), 2);

        assert_eq!(rc64, SaSint::from(rc32));
        assert_eq!(
            &sa64[..text.len()],
            &sa32[..text.len()]
                .iter()
                .map(|&value| SaSint::from(value as u32))
                .collect::<Vec<_>>()
        );
        assert_eq!(freq64[b'a' as usize], 3);
        assert_eq!(freq64[b'b' as usize], 1);
        assert_eq!(freq64[b'n' as usize], 2);
        assert_eq!(
            freq64[..ALPHABET_SIZE],
            freq32
                .iter()
                .map(|&value| SaSint::from(value))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn libsais64_32bit_gsa_adapter_widens_suffix_array_and_frequency() {
        let text = b"ban\0ana\0";
        let fs = 2;
        let mut sa64 = vec![-1; text.len() + fs as usize];
        let mut freq64 = vec![-1; ALPHABET_SIZE];

        let rc = libsais64_run_32bit_omp(text, &mut sa64, fs, Some(&mut freq64), 2, true)
            .expect("small GSA input uses 32-bit adapter");

        let mut direct_sa = vec![0; text.len()];
        let mut direct_freq = vec![0; ALPHABET_SIZE];
        assert_eq!(
            crate::libsais_gsa(text, &mut direct_sa, 0, Some(&mut direct_freq)),
            0
        );
        assert_eq!(rc, 0);
        assert_eq!(
            sa64[..text.len()],
            direct_sa
                .iter()
                .map(|&value| SaSint::from(value as u32))
                .collect::<Vec<_>>()
        );
        assert_eq!(
            freq64[..ALPHABET_SIZE],
            direct_freq
                .iter()
                .map(|&value| SaSint::from(value))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn libsais64_32bit_bwt_adapters_widen_frequency_and_aux_samples() {
        let text = b"mississippi";
        let fs = 6;
        let r = 4;

        let mut bwt64 = vec![0; text.len()];
        let mut freq64 = vec![-1; ALPHABET_SIZE];
        let primary64 = libsais64_bwt_run_32bit_omp(text, &mut bwt64, fs, Some(&mut freq64), 2)
            .expect("small input uses 32-bit BWT adapter");

        let mut bwt32 = vec![0; text.len()];
        let mut work32 = vec![0; text.len() + fs as usize * 2 + text.len()];
        let mut freq32 = vec![-1; ALPHABET_SIZE];
        let primary32 =
            crate::libsais_bwt_omp(text, &mut bwt32, &mut work32, 23, Some(&mut freq32), 2);

        assert_eq!(primary64, SaSint::from(primary32));
        assert_eq!(bwt64, bwt32);
        assert_eq!(
            freq64[..ALPHABET_SIZE],
            freq32
                .iter()
                .map(|&value| SaSint::from(value))
                .collect::<Vec<_>>()
        );

        let mut aux_bwt64 = vec![0; text.len()];
        let mut aux64 = vec![-1; (text.len() - 1) / r as usize + 1];
        let mut aux_freq64 = vec![-1; ALPHABET_SIZE];
        let rc64 = libsais64_bwt_aux_run_32bit_omp(
            text,
            &mut aux_bwt64,
            fs,
            Some(&mut aux_freq64),
            r,
            &mut aux64,
            2,
        )
        .expect("small input uses 32-bit aux BWT adapter");

        let mut aux_bwt32 = vec![0; text.len()];
        let mut aux_work32 = vec![0; text.len() + fs as usize * 2 + text.len()];
        let mut aux32 = vec![-1; aux64.len()];
        let mut aux_freq32 = vec![-1; ALPHABET_SIZE];
        let rc32 = crate::libsais_bwt_aux_omp(
            text,
            &mut aux_bwt32,
            &mut aux_work32,
            23,
            Some(&mut aux_freq32),
            r as i32,
            &mut aux32,
            2,
        );

        assert_eq!(rc64, SaSint::from(rc32));
        assert_eq!(aux_bwt64, aux_bwt32);
        assert_eq!(
            aux64,
            aux32
                .iter()
                .map(|&value| SaSint::from(value))
                .collect::<Vec<_>>()
        );
        assert_eq!(
            aux_freq64[..ALPHABET_SIZE],
            aux_freq32
                .iter()
                .map(|&value| SaSint::from(value))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn libsais64_bwt_copy_8u_omp_uses_block_partition_for_large_inputs() {
        let n = 65_600usize;
        let a: Vec<SaSint> = (0..n).map(|i| (i * 17) as SaSint).collect();
        let mut threaded = vec![0; n];
        let mut sequential = vec![0; n];

        bwt_copy_8u_omp(&mut threaded, &a, n as SaSint, 4);
        bwt_copy_8u(&mut sequential, &a, n as SaSint);

        assert_eq!(threaded, sequential);
    }

    #[test]
    fn libsais64_flip_suffix_markers_omp_uses_block_partition_for_large_inputs() {
        let n = 65_600usize;
        let mut single: Vec<SaSint> = (0..n).map(|i| (i as SaSint) ^ SAINT_MIN).collect();
        let mut threaded = single.clone();

        flip_suffix_markers_omp(&mut single, n as SaSint, 1);
        flip_suffix_markers_omp(&mut threaded, n as SaSint, 4);

        assert_eq!(threaded, single);
    }

    #[test]
    fn libsais64_renumber_lms_suffixes_8u_writes_names_into_second_half() {
        let mut sa = vec![1 | SAINT_MIN, 3, 0, 0];

        let name = renumber_lms_suffixes_8u(&mut sa, 2, 0, 0, 2);

        assert_eq!(name, 1);
        assert_eq!(sa[2], SAINT_MIN);
        assert_eq!(sa[3], SAINT_MIN | 1);
    }

    #[test]
    fn libsais64_renumber_lms_suffixes_8u_matches_upstream_c_helper() {
        let mut sa_rust = vec![1 | SAINT_MIN, 3, 0, 0];
        let mut sa_c = sa_rust.clone();

        let rust_name = renumber_lms_suffixes_8u(&mut sa_rust, 2, 0, 0, 2);
        let c_name =
            unsafe { probe_libsais64_renumber_lms_suffixes_8u(sa_c.as_mut_ptr(), 2, 0, 0, 2) };

        assert_eq!(rust_name, c_name);
        assert_eq!(sa_rust, sa_c);
    }

    #[test]
    fn libsais64_gather_marked_lms_suffixes_moves_negative_marked_entries_to_tail() {
        let mut sa = vec![0, 0, 1 | SAINT_MIN, 3];

        let l = gather_marked_lms_suffixes(&mut sa, 2, 4, 0, 2);

        assert_eq!(l, 3);
        assert_eq!(sa[3], 1);
    }

    #[test]
    fn libsais64_gather_marked_lms_suffixes_matches_upstream_c_helper() {
        let mut sa_rust = vec![0, 0, 1 | SAINT_MIN, 3];
        let mut sa_c = sa_rust.clone();

        let rust_l = gather_marked_lms_suffixes(&mut sa_rust, 2, 4, 0, 2);
        let c_l =
            unsafe { probe_libsais64_gather_marked_lms_suffixes(sa_c.as_mut_ptr(), 2, 4, 0, 2) };

        assert_eq!(rust_l, c_l);
        assert_eq!(sa_rust, sa_c);
    }

    #[test]
    fn libsais64_renumber_lms_suffixes_8u_omp_wraps_sequential_version() {
        let mut sa = vec![1 | SAINT_MIN, 3, 0, 0];
        let mut thread_state = alloc_thread_state(2).unwrap();

        let name = renumber_lms_suffixes_8u_omp(&mut sa, 2, 2, &mut thread_state);

        assert_eq!(name, 1);
        assert_eq!(sa[2], SAINT_MIN);
    }

    #[test]
    fn libsais64_renumber_lms_suffixes_8u_omp_uses_block_partition_for_large_inputs() {
        let m = 65_600usize;
        let mut input = vec![0; 2 * m];
        for (i, slot) in input[..m].iter_mut().enumerate() {
            let suffix = (2 * i + 1) as SaSint;
            *slot = if i % 5 == 0 {
                suffix | SAINT_MIN
            } else {
                suffix
            };
        }

        let mut single = input.clone();
        let mut threaded = input;
        let mut thread_state = alloc_thread_state(4).unwrap();
        let single_name = renumber_lms_suffixes_8u(&mut single, m as SaSint, 0, 0, m as FastSint);
        let threaded_name =
            renumber_lms_suffixes_8u_omp(&mut threaded, m as SaSint, 4, &mut thread_state);

        assert_eq!(threaded_name, single_name);
        assert_eq!(threaded, single);
    }

    #[test]
    fn libsais64_gather_marked_lms_suffixes_omp_uses_block_partition_for_large_inputs() {
        let n = 131_200usize;
        let half_n = n >> 1;
        let mut input = vec![-77; n];
        for (i, slot) in input[..half_n].iter_mut().enumerate() {
            let suffix = (3 * i + 1) as SaSint;
            *slot = if i % 7 == 0 {
                suffix | SAINT_MIN
            } else {
                suffix
            };
        }
        let marked_count = input[..half_n].iter().filter(|&&value| value < 0).count();

        let mut single = input.clone();
        let mut threaded = input;
        let mut thread_state = alloc_thread_state(4).unwrap();
        let _ = gather_marked_lms_suffixes(&mut single, 0, n as FastSint, 0, half_n as FastSint);
        gather_marked_lms_suffixes_omp(&mut threaded, n as SaSint, 0, 0, 4, &mut thread_state);

        assert_eq!(&threaded[n - marked_count..], &single[n - marked_count..]);
    }

    #[test]
    fn libsais64_renumber_and_gather_lms_suffixes_omp_uses_large_input_paths() {
        let m = 65_600usize;
        let n = 2 * m;
        let mut input = vec![0; n];
        for (i, slot) in input[..m].iter_mut().enumerate() {
            let suffix = (2 * i + 1) as SaSint;
            *slot = if i % 5 == 0 {
                suffix | SAINT_MIN
            } else {
                suffix
            };
        }

        let mut single = input.clone();
        let mut threaded = input;
        let mut single_state = alloc_thread_state(1).unwrap();
        let mut threaded_state = alloc_thread_state(4).unwrap();
        let single_name = renumber_and_gather_lms_suffixes_omp(
            &mut single,
            n as SaSint,
            m as SaSint,
            0,
            1,
            &mut single_state,
        );
        let threaded_name = renumber_and_gather_lms_suffixes_omp(
            &mut threaded,
            n as SaSint,
            m as SaSint,
            0,
            4,
            &mut threaded_state,
        );

        assert_eq!(threaded_name, single_name);
        assert_eq!(threaded, single);
    }

    #[test]
    fn libsais64_renumber_and_gather_lms_suffixes_omp_gathers_when_names_are_not_distinct() {
        let mut sa = vec![1 | SAINT_MIN, 3, 0, 0];
        let mut thread_state = alloc_thread_state(2).unwrap();

        let name = renumber_and_gather_lms_suffixes_omp(&mut sa, 4, 2, 0, 2, &mut thread_state);

        assert_eq!(name, 1);
        assert_eq!(sa[3], 1);
    }

    #[test]
    fn libsais64_renumber_and_gather_lms_suffixes_omp_matches_upstream_c_helper() {
        let mut sa_rust = vec![1 | SAINT_MIN, 3, 0, 0];
        let mut sa_c = sa_rust.clone();
        let mut thread_state = alloc_thread_state(2).unwrap();

        let rust_name =
            renumber_and_gather_lms_suffixes_omp(&mut sa_rust, 4, 2, 0, 2, &mut thread_state);
        let c_name = unsafe {
            probe_libsais64_renumber_and_gather_lms_suffixes_omp(sa_c.as_mut_ptr(), 4, 2, 0, 2)
        };

        assert_eq!(rust_name, c_name);
        assert_eq!(sa_rust, sa_c);
    }

    #[test]
    fn libsais64_renumber_distinct_lms_suffixes_32s_4k_masks_sources_and_writes_second_half() {
        let mut sa = vec![1 | SAINT_MIN, 3 | SAINT_MIN, 0, 0];

        let name = renumber_distinct_lms_suffixes_32s_4k(&mut sa, 2, 1, 0, 2);

        assert_eq!(name, 3);
        assert_eq!(sa[0], 1);
        assert_eq!(sa[1], 3);
        assert_eq!(sa[2], 1);
        assert_eq!(sa[3], 2 | SAINT_MIN);
    }

    #[test]
    fn libsais64_renumber_distinct_lms_suffixes_32s_4k_matches_upstream_c_helper() {
        let mut sa_rust = vec![1 | SAINT_MIN, 3 | SAINT_MIN, 0, 0];
        let mut sa_c = sa_rust.clone();

        let rust_name = renumber_distinct_lms_suffixes_32s_4k(&mut sa_rust, 2, 1, 0, 2);
        let c_name = unsafe {
            probe_libsais64_renumber_distinct_lms_suffixes_32s_4k(sa_c.as_mut_ptr(), 2, 1, 0, 2)
        };

        assert_eq!(rust_name, c_name);
        assert_eq!(sa_rust, sa_c);
    }

    #[test]
    fn libsais64_mark_distinct_lms_suffixes_32s_propagates_previous_nonzero_marker() {
        let mut sa = vec![0, 0, SAINT_MIN | 5, 0, SAINT_MIN | 7];

        mark_distinct_lms_suffixes_32s(&mut sa, 2, 0, 3);

        assert_eq!(sa[2], 5);
        assert_eq!(sa[3], 0);
        assert_eq!(sa[4], SAINT_MIN | 7);
    }

    #[test]
    fn libsais64_clamp_lms_suffixes_length_32s_keeps_only_negative_lengths() {
        let mut sa = vec![0, 0, SAINT_MIN | 5, 7, SAINT_MIN | 3];

        clamp_lms_suffixes_length_32s(&mut sa, 2, 0, 3);

        assert_eq!(sa[2], 5);
        assert_eq!(sa[3], 0);
        assert_eq!(sa[4], 3);
    }

    #[test]
    fn libsais64_renumber_and_mark_distinct_lms_suffixes_32s_4k_omp_marks_second_half_when_names_repeat(
    ) {
        let mut sa = vec![1 | SAINT_MIN, 3 | SAINT_MIN, 0, 0];
        let mut thread_state = alloc_thread_state(2).unwrap();

        let name =
            renumber_and_mark_distinct_lms_suffixes_32s_4k_omp(&mut sa, 4, 2, 2, &mut thread_state);

        assert_eq!(name, 2);
        assert_eq!(sa[2], 1);
        assert_eq!(sa[3], SAINT_MIN | 2);
    }

    #[test]
    fn libsais64_renumber_and_mark_distinct_lms_suffixes_32s_4k_omp_matches_upstream_c_helper() {
        let mut sa_rust = vec![1 | SAINT_MIN, 3 | SAINT_MIN, 0, 0];
        let mut sa_c = sa_rust.clone();
        let mut thread_state = alloc_thread_state(2).unwrap();

        let rust_name = renumber_and_mark_distinct_lms_suffixes_32s_4k_omp(
            &mut sa_rust,
            4,
            2,
            2,
            &mut thread_state,
        );
        let c_name = unsafe {
            probe_libsais64_renumber_and_mark_distinct_lms_suffixes_32s_4k_omp(
                sa_c.as_mut_ptr(),
                4,
                2,
                2,
            )
        };

        assert_eq!(rust_name, c_name);
        assert_eq!(sa_rust, sa_c);
    }

    #[test]
    fn libsais64_renumber_and_mark_distinct_lms_suffixes_32s_1k_omp_handles_single_lms_suffix() {
        let t = vec![2, 1, 0];
        let mut sa = vec![0; t.len()];

        let name = renumber_and_mark_distinct_lms_suffixes_32s_1k_omp(&t, &mut sa, 3, 1, 1);

        assert_eq!(name, 1);
        assert_eq!(sa[1], SAINT_MIN | 1);
    }

    #[test]
    fn libsais64_main_32s_entry_matches_public_c_long_on_6k_branch() {
        assert_libsais64_main_32s_entry_matches_public_c_long_for_branch(300);
    }

    #[test]
    fn libsais64_main_32s_entry_matches_public_c_long_on_4k_branch() {
        assert_libsais64_main_32s_entry_matches_public_c_long_for_branch(400);
    }

    #[test]
    fn libsais64_main_32s_entry_matches_public_c_long_on_2k_branch() {
        assert_libsais64_main_32s_entry_matches_public_c_long_for_branch(700);
    }

    #[test]
    fn libsais64_main_32s_entry_matches_public_c_long_on_1k_branch() {
        assert_libsais64_main_32s_entry_matches_public_c_long_for_branch(1501);
    }

    #[test]
    fn libsais64_main_32s_entry_matches_public_c_long_on_recursive_repetitive_6k_case() {
        assert_libsais64_main_32s_entry_matches_public_c_long(
            make_libsais64_recursive_main_32s_text(24),
            300,
            0,
            true,
        );
    }

    #[test]
    fn libsais64_main_32s_entry_matches_public_c_long_on_recursive_repetitive_1k_case() {
        assert_libsais64_main_32s_entry_matches_public_c_long(
            make_libsais64_recursive_main_32s_text(24),
            1501,
            0,
            true,
        );
    }

    #[test]
    fn libsais64_main_32s_entry_matches_public_c_long_on_large_generated_6k_case() {
        assert_libsais64_main_32s_entry_matches_public_c_long(
            make_libsais64_large_main_32s_stress_text(1024, 300),
            300,
            0,
            true,
        );
    }

    #[test]
    fn libsais64_main_32s_entry_matches_public_c_long_on_large_generated_6k_case_with_fs() {
        assert_libsais64_main_32s_entry_matches_public_c_long(
            make_libsais64_large_main_32s_stress_text(1024, 300),
            300,
            2048,
            false,
        );
    }

    #[test]
    fn libsais64_main_32s_entry_matches_public_c_long_on_large_generated_4k_case() {
        assert_libsais64_main_32s_entry_matches_public_c_long(
            make_libsais64_large_main_32s_stress_text(1024, 400),
            400,
            0,
            true,
        );
    }

    #[test]
    fn libsais64_main_32s_entry_matches_public_c_long_on_large_generated_4k_case_with_fs() {
        assert_libsais64_main_32s_entry_matches_public_c_long(
            make_libsais64_large_main_32s_stress_text(1024, 400),
            400,
            2048,
            false,
        );
    }

    #[test]
    fn libsais64_main_32s_entry_matches_public_c_long_on_large_generated_2k_case() {
        assert_libsais64_main_32s_entry_matches_public_c_long(
            make_libsais64_large_main_32s_stress_text(1024, 700),
            700,
            0,
            true,
        );
    }

    #[test]
    fn libsais64_main_32s_entry_matches_public_c_long_on_large_generated_2k_case_with_fs() {
        assert_libsais64_main_32s_entry_matches_public_c_long(
            make_libsais64_large_main_32s_stress_text(1024, 700),
            700,
            2048,
            false,
        );
    }

    #[test]
    fn libsais64_main_32s_entry_matches_public_c_long_on_large_generated_1k_case_with_fs() {
        assert_libsais64_main_32s_entry_matches_public_c_long(
            make_libsais64_large_main_32s_stress_text(1024, 1501),
            1501,
            2048,
            false,
        );
    }

    #[test]
    fn libsais64_reconstruct_lms_suffixes_maps_indices_from_tail_interval() {
        let mut sa = vec![0, 1, 2, 7, 11, 13];

        reconstruct_lms_suffixes(&mut sa, 6, 3, 0, 3);

        assert_eq!(&sa[..3], &[7, 11, 13]);
    }

    #[test]
    fn libsais64_reconstruct_lms_suffixes_omp_wraps_sequential_version() {
        let mut sa = vec![0, 1, 2, 7, 11, 13];

        reconstruct_lms_suffixes_omp(&mut sa, 6, 3, 2);

        assert_eq!(&sa[..3], &[7, 11, 13]);
    }

    #[test]
    fn libsais64_lms_late_omp_wrappers_use_block_partitions_for_large_inputs() {
        let m = 65_600usize;
        let n = 2 * m;
        let mut input = vec![0; n];
        for (i, slot) in input[..m].iter_mut().enumerate() {
            let suffix = (2 * i + 1) as SaSint;
            *slot = if i % 5 == 0 {
                suffix | SAINT_MIN
            } else {
                suffix
            };
        }

        let mut single = input.clone();
        let mut threaded = input.clone();
        let mut thread_state = alloc_thread_state(4).unwrap();
        let single_name = renumber_lms_suffixes_8u(&mut single, m as SaSint, 0, 0, m as FastSint);
        let threaded_name =
            renumber_lms_suffixes_8u_omp(&mut threaded, m as SaSint, 4, &mut thread_state);
        assert_eq!(threaded_name, single_name);
        assert_eq!(threaded, single);

        let mut single = input.clone();
        let mut threaded = input.clone();
        let mut single_state = alloc_thread_state(1).unwrap();
        let mut threaded_state = alloc_thread_state(4).unwrap();
        let single_name = renumber_and_gather_lms_suffixes_omp(
            &mut single,
            n as SaSint,
            m as SaSint,
            0,
            1,
            &mut single_state,
        );
        let threaded_name = renumber_and_gather_lms_suffixes_omp(
            &mut threaded,
            n as SaSint,
            m as SaSint,
            0,
            4,
            &mut threaded_state,
        );
        assert_eq!(threaded_name, single_name);
        assert_eq!(threaded, single);

        let mut single = input.clone();
        let mut threaded = input;
        let marked_count = single[..m].iter().filter(|&&value| value < 0).count();
        let _ = gather_marked_lms_suffixes(&mut single, 0, n as FastSint, 0, m as FastSint);
        gather_marked_lms_suffixes_omp(&mut threaded, n as SaSint, 0, 0, 4, &mut thread_state);
        assert_eq!(&threaded[n - marked_count..], &single[n - marked_count..]);
    }

    #[test]
    fn libsais64_reconstruct_lms_suffixes_omp_uses_block_partition_for_large_inputs() {
        let m = 65_600usize;
        let n = 2 * m;
        let mut input = vec![0; n];
        for (i, slot) in input[..m].iter_mut().enumerate() {
            *slot = (m - 1 - i) as SaSint;
        }
        for (i, slot) in input[m..].iter_mut().enumerate() {
            *slot = (i * 17 + 3) as SaSint;
        }

        let mut single = input.clone();
        let mut threaded = input;
        reconstruct_lms_suffixes(&mut single, n as SaSint, m as SaSint, 0, m as FastSint);
        reconstruct_lms_suffixes_omp(&mut threaded, n as SaSint, m as SaSint, 4);

        assert_eq!(threaded, single);
    }

    #[test]
    fn libsais64_renumber_unique_and_nonunique_lms_suffixes_32s_marks_new_unique_names() {
        let mut t = vec![0, 0, 0, 0];
        let mut sa = vec![0, 2, -1, 5];

        let f = renumber_unique_and_nonunique_lms_suffixes_32s(&mut t, &mut sa, 2, 0, 0, 2);

        assert_eq!(f, 1);
        assert_eq!(t[0], SAINT_MIN);
        assert_eq!(sa[2], SAINT_MIN);
        assert_eq!(sa[3], 4);
    }

    #[test]
    fn libsais64_renumber_unique_and_nonunique_lms_suffixes_32s_matches_upstream_c_helper() {
        let mut t_rust = vec![0, 0, 0, 0];
        let mut sa_rust = vec![0, 2, -1, 5];
        let mut t_c = t_rust.clone();
        let mut sa_c = sa_rust.clone();

        let rust_f =
            renumber_unique_and_nonunique_lms_suffixes_32s(&mut t_rust, &mut sa_rust, 2, 0, 0, 2);
        let c_f = unsafe {
            probe_libsais64_renumber_unique_and_nonunique_lms_suffixes_32s(
                t_c.as_mut_ptr(),
                sa_c.as_mut_ptr(),
                2,
                0,
                0,
                2,
            )
        };

        assert_eq!(rust_f, c_f);
        assert_eq!(t_rust, t_c);
        assert_eq!(sa_rust, sa_c);
    }

    #[test]
    fn libsais64_renumber_unique_and_nonunique_lms_suffixes_32s_omp_matches_upstream_c_helper() {
        let mut t_rust = vec![0, 0, 0, 0];
        let mut sa_rust = vec![0, 2, -1, 5];
        let mut t_c = t_rust.clone();
        let mut sa_c = sa_rust.clone();
        let mut thread_state = alloc_thread_state(1).unwrap();

        let rust_f = renumber_unique_and_nonunique_lms_suffixes_32s_omp(
            &mut t_rust,
            &mut sa_rust,
            2,
            1,
            &mut thread_state,
        );
        let c_f = unsafe {
            probe_libsais64_renumber_unique_and_nonunique_lms_suffixes_32s_omp(
                t_c.as_mut_ptr(),
                sa_c.as_mut_ptr(),
                2,
                1,
            )
        };

        assert_eq!(rust_f, c_f);
        assert_eq!(t_rust, t_c);
        assert_eq!(sa_rust, sa_c);
    }

    #[test]
    fn libsais64_renumber_unique_and_nonunique_lms_suffixes_32s_omp_uses_block_partition() {
        let m = 65_600usize;
        let n = 2 * m;
        let t = vec![0; n];
        let mut sa = vec![0; n];
        for i in 0..m {
            sa[i] = (2 * i) as SaSint;
            sa[m + i] = if i % 5 == 0 {
                -((i as SaSint) + 1)
            } else {
                i as SaSint + 7
            };
        }

        let mut single_t = t.clone();
        let mut single_sa = sa.clone();
        let mut threaded_t = t;
        let mut threaded_sa = sa;
        let mut thread_state = alloc_thread_state(4).unwrap();
        let single_f = renumber_unique_and_nonunique_lms_suffixes_32s(
            &mut single_t,
            &mut single_sa,
            m as SaSint,
            0,
            0,
            m as FastSint,
        );
        let threaded_f = renumber_unique_and_nonunique_lms_suffixes_32s_omp(
            &mut threaded_t,
            &mut threaded_sa,
            m as SaSint,
            4,
            &mut thread_state,
        );

        assert_eq!(threaded_f, single_f);
        assert_eq!(threaded_t, single_t);
        assert_eq!(threaded_sa, single_sa);
    }

    #[test]
    fn libsais64_compact_unique_and_nonunique_lms_suffixes_32s_splits_unique_and_nonunique_ranges()
    {
        let mut sa = vec![0, 0, 0, 0, SAINT_MIN, 4];
        let mut l = 2;
        let mut r = 6;

        compact_unique_and_nonunique_lms_suffixes_32s(&mut sa, 2, &mut l, &mut r, 0, 2);

        assert_eq!(l, 2);
        assert_eq!(r, 6);
        assert_eq!(sa[2], 0);
        assert_eq!(sa[3] & SAINT_MAX, 0);
    }

    #[test]
    fn libsais64_compact_lms_suffixes_32s_omp_runs_renumber_then_compaction() {
        let mut t = vec![0, 0, 0, 0];
        let mut sa = vec![0, 2, -1, 5, 77, 88];
        let mut thread_state = alloc_thread_state(2).unwrap();

        let f = compact_lms_suffixes_32s_omp(&mut t, &mut sa, 4, 2, 2, 2, &mut thread_state);

        assert_eq!(f, 1);
        assert_eq!(sa[2] & SAINT_MAX, 0);
        assert_eq!(sa[5], 3);
    }

    #[test]
    fn libsais64_compact_unique_and_nonunique_lms_suffixes_32s_omp_uses_block_partition() {
        let n = 131_200usize;
        let m = 65_600usize;
        let fs = m + 32;
        let half_n = n >> 1;
        let f = m / 5;
        let mut sa = vec![0; n + fs];
        for i in 0..half_n {
            sa[m + i] = if i % 5 == 0 {
                SAINT_MIN | i as SaSint
            } else {
                i as SaSint + 1
            };
        }
        for i in 0..f {
            sa[m - f + i] = (10_000 + i) as SaSint;
        }

        let mut single = sa.clone();
        let mut threaded = sa;
        let mut single_state = alloc_thread_state(1).unwrap();
        let mut threaded_state = alloc_thread_state(4).unwrap();
        compact_unique_and_nonunique_lms_suffixes_32s_omp(
            &mut single,
            n as SaSint,
            m as SaSint,
            fs as SaSint,
            f as SaSint,
            1,
            &mut single_state,
        );
        compact_unique_and_nonunique_lms_suffixes_32s_omp(
            &mut threaded,
            n as SaSint,
            m as SaSint,
            fs as SaSint,
            f as SaSint,
            4,
            &mut threaded_state,
        );

        let unique_dst = n + fs - m;
        assert_eq!(
            &threaded[unique_dst..unique_dst + f],
            &single[unique_dst..unique_dst + f]
        );
    }

    #[test]
    fn libsais64_compact_lms_suffixes_32s_omp_uses_large_input_paths() {
        let n = 131_200usize;
        let m = 65_600usize;
        let fs = m + 32;
        let t = vec![0; n];
        let mut sa = vec![0; n + fs];
        for i in 0..m {
            sa[i] = (2 * i) as SaSint;
            sa[m + i] = if i % 5 == 0 {
                -((i as SaSint) + 1)
            } else {
                i as SaSint + 7
            };
        }

        let mut single_t = t.clone();
        let mut single_sa = sa.clone();
        let mut threaded_t = t;
        let mut threaded_sa = sa;
        let mut single_state = alloc_thread_state(1).unwrap();
        let mut threaded_state = alloc_thread_state(4).unwrap();
        let single_f = compact_lms_suffixes_32s_omp(
            &mut single_t,
            &mut single_sa,
            n as SaSint,
            m as SaSint,
            fs as SaSint,
            1,
            &mut single_state,
        );
        let threaded_f = compact_lms_suffixes_32s_omp(
            &mut threaded_t,
            &mut threaded_sa,
            n as SaSint,
            m as SaSint,
            fs as SaSint,
            4,
            &mut threaded_state,
        );

        assert_eq!(threaded_f, single_f);
        assert_eq!(threaded_t, single_t);
        let unique_dst = n + fs - m;
        let unique_len = usize::try_from(threaded_f).expect("f must be non-negative");
        assert_eq!(
            &threaded_sa[unique_dst..unique_dst + unique_len],
            &single_sa[unique_dst..unique_dst + unique_len]
        );
    }

    #[test]
    fn libsais64_merge_unique_lms_suffixes_32s_noops_for_empty_block() {
        let mut t = vec![1, SAINT_MIN, 2, SAINT_MIN];
        let mut sa = vec![0, 0, 1, 3];
        let before_t = t.clone();
        let before_sa = sa.clone();

        merge_unique_lms_suffixes_32s(&mut t, &mut sa, 4, 1, 0, 0, 0);

        assert_eq!(t, before_t);
        assert_eq!(sa, before_sa);
    }

    #[test]
    fn libsais64_merge_unique_lms_suffixes_32s_omp_uses_block_partition_for_large_inputs() {
        let n = 65_600usize;
        let m = 1_024usize;
        let mut t = vec![1; n];
        for i in (0..n).step_by(257) {
            t[i] = SAINT_MIN | ((i % 251) as SaSint);
        }
        let f = t.iter().filter(|&&value| value < 0).count();
        let mut sa = vec![-1; n];
        let src = n - m - 1;
        for i in 0..f {
            sa[src + i] = i as SaSint;
        }

        let mut single_t = t.clone();
        let mut single_sa = sa.clone();
        let mut threaded_t = t;
        let mut threaded_sa = sa;
        let mut thread_state = alloc_thread_state(4).unwrap();
        merge_unique_lms_suffixes_32s_omp(
            &mut single_t,
            &mut single_sa,
            n as SaSint,
            m as SaSint,
            1,
            &mut [],
        );
        merge_unique_lms_suffixes_32s_omp(
            &mut threaded_t,
            &mut threaded_sa,
            n as SaSint,
            m as SaSint,
            4,
            &mut thread_state,
        );

        assert_eq!(threaded_t, single_t);
        assert_eq!(threaded_sa, single_sa);
    }

    #[test]
    fn libsais64_merge_nonunique_lms_suffixes_32s_noops_for_empty_block() {
        let mut sa = vec![0, 7, 0, 13, 11];
        let before = sa.clone();

        merge_nonunique_lms_suffixes_32s(&mut sa, 4, 1, 0, 0, 0);

        assert_eq!(sa, before);
    }

    #[test]
    fn libsais64_merge_compacted_lms_suffixes_32s_omp_preserves_input_text_and_fills_zero_slots() {
        let mut t = vec![1, 2, 3, 4];
        let mut sa = vec![0, 1, 2, 3, 4, 5];
        let before_t = t.clone();
        let mut thread_state = alloc_thread_state(2).unwrap();

        merge_compacted_lms_suffixes_32s_omp(&mut t, &mut sa, 4, 1, 1, 2, &mut thread_state);

        assert_eq!(t, before_t);
        assert_eq!(sa[0], 3);
        assert_eq!(sa[1], 1);
    }

    #[test]
    fn libsais64_merge_nonunique_lms_suffixes_32s_omp_uses_block_partition_for_large_inputs() {
        let n = 131_200usize;
        let m = 65_600usize;
        let f = 7usize;
        let mut sa = vec![1; n];
        let zero_count = (0..m).filter(|i| i % 17 == 0).count();
        for i in (0..m).step_by(17) {
            sa[i] = 0;
        }
        let src = n - m - 1 + f;
        for i in 0..zero_count {
            sa[src + i] = 10_000 + i as SaSint;
        }

        let mut single = sa.clone();
        let mut threaded = sa;
        let mut thread_state = alloc_thread_state(4).unwrap();
        merge_nonunique_lms_suffixes_32s_omp(
            &mut single,
            n as SaSint,
            m as SaSint,
            f as SaSint,
            1,
            &mut [],
        );
        merge_nonunique_lms_suffixes_32s_omp(
            &mut threaded,
            n as SaSint,
            m as SaSint,
            f as SaSint,
            4,
            &mut thread_state,
        );

        assert_eq!(threaded, single);
    }

    #[test]
    fn libsais64_merge_compacted_lms_suffixes_32s_omp_uses_block_partition_for_large_inputs() {
        let n = 131_200usize;
        let m = 65_600usize;
        let mut t = vec![1; n];
        for i in (0..n).step_by(257) {
            t[i] = SAINT_MIN | ((i % 251) as SaSint);
        }
        let f = t.iter().filter(|&&value| value < 0).count();

        let mut sa = vec![1; n];
        let zero_count = (0..m).filter(|i| i % 17 == 0).count();
        for i in (0..m).step_by(17) {
            sa[i] = 0;
        }
        let unique_src = n - m - 1;
        for i in 0..f {
            sa[unique_src + i] = i as SaSint;
        }
        for i in 0..zero_count {
            sa[unique_src + f + i] = 10_000 + i as SaSint;
        }

        let mut single_t = t.clone();
        let mut single_sa = sa.clone();
        let mut threaded_t = t;
        let mut threaded_sa = sa;
        let mut single_state = alloc_thread_state(1).unwrap();
        let mut threaded_state = alloc_thread_state(4).unwrap();
        merge_compacted_lms_suffixes_32s_omp(
            &mut single_t,
            &mut single_sa,
            n as SaSint,
            m as SaSint,
            f as SaSint,
            1,
            &mut single_state,
        );
        merge_compacted_lms_suffixes_32s_omp(
            &mut threaded_t,
            &mut threaded_sa,
            n as SaSint,
            m as SaSint,
            f as SaSint,
            4,
            &mut threaded_state,
        );

        assert_eq!(threaded_t, single_t);
        assert_eq!(threaded_sa, single_sa);
    }

    #[test]
    fn libsais64_final_bwt_left_to_right_8u_block_omp_uses_thread_buckets() {
        let block_start = 20_000usize;
        let block_size = 16_384usize;
        let n = block_start + block_size + 8;
        let t = vec![1_u8; n];
        let suffixes: Vec<SaSint> = (2..2 + block_size).map(|i| i as SaSint).collect();

        let mut expected_sa = vec![0; n];
        expected_sa[block_start..block_start + block_size].copy_from_slice(&suffixes);
        let mut threaded_sa = expected_sa.clone();
        let mut expected_bucket = vec![0; ALPHABET_SIZE];
        let mut threaded_bucket = expected_bucket.clone();
        let mut thread_state = alloc_thread_state(4).unwrap();

        final_bwt_scan_left_to_right_8u(
            &t,
            &mut expected_sa,
            &mut expected_bucket,
            block_start as FastSint,
            block_size as FastSint,
        );
        final_bwt_scan_left_to_right_8u_block_omp(
            &t,
            &mut threaded_sa,
            ALPHABET_SIZE as SaSint,
            &mut threaded_bucket,
            block_start as FastSint,
            block_size as FastSint,
            4,
            &mut thread_state,
        );

        assert_eq!(threaded_sa, expected_sa);
        assert_eq!(threaded_bucket, expected_bucket);
    }

    #[test]
    fn libsais64_final_bwt_aux_left_to_right_8u_block_omp_uses_thread_buckets() {
        let block_start = 20_000usize;
        let block_size = 16_384usize;
        let n = block_start + block_size + 8;
        let t = vec![1_u8; n];
        let suffixes: Vec<SaSint> = (2..2 + block_size).map(|i| i as SaSint).collect();

        let mut expected_sa = vec![0; n];
        expected_sa[block_start..block_start + block_size].copy_from_slice(&suffixes);
        let mut threaded_sa = expected_sa.clone();
        let mut expected_i = vec![0; n];
        let mut threaded_i = vec![0; n];
        let mut expected_bucket = vec![0; ALPHABET_SIZE];
        let mut threaded_bucket = expected_bucket.clone();
        let mut thread_state = alloc_thread_state(4).unwrap();

        final_bwt_aux_scan_left_to_right_8u(
            &t,
            &mut expected_sa,
            0,
            &mut expected_i,
            &mut expected_bucket,
            block_start as FastSint,
            block_size as FastSint,
        );
        final_bwt_aux_scan_left_to_right_8u_block_omp(
            &t,
            &mut threaded_sa,
            ALPHABET_SIZE as SaSint,
            0,
            &mut threaded_i,
            &mut threaded_bucket,
            block_start as FastSint,
            block_size as FastSint,
            4,
            &mut thread_state,
        );

        assert_eq!(threaded_sa, expected_sa);
        assert_eq!(threaded_i, expected_i);
        assert_eq!(threaded_bucket, expected_bucket);
    }

    #[test]
    fn libsais64_final_sorting_right_to_left_8u_block_omp_uses_thread_buckets() {
        let block_start = 20_000usize;
        let block_size = 16_384usize;
        let n = block_start + block_size + 8;
        let t = vec![1_u8; n];
        let suffixes: Vec<SaSint> = (2..2 + block_size).map(|i| i as SaSint).collect();

        let mut expected_sa = vec![0; n];
        expected_sa[block_start..block_start + block_size].copy_from_slice(&suffixes);
        let mut threaded_sa = expected_sa.clone();
        let mut expected_bucket = vec![0; ALPHABET_SIZE];
        expected_bucket[1] = n as SaSint;
        let mut threaded_bucket = expected_bucket.clone();
        let mut thread_state = alloc_thread_state(4).unwrap();

        final_sorting_scan_right_to_left_8u(
            &t,
            &mut expected_sa,
            &mut expected_bucket,
            block_start as FastSint,
            block_size as FastSint,
        );
        final_sorting_scan_right_to_left_8u_block_omp(
            &t,
            &mut threaded_sa,
            ALPHABET_SIZE as SaSint,
            &mut threaded_bucket,
            block_start as FastSint,
            block_size as FastSint,
            4,
            &mut thread_state,
        );

        assert_eq!(threaded_sa, expected_sa);
        assert_eq!(threaded_bucket, expected_bucket);
    }

    #[test]
    fn libsais64_final_bwt_right_to_left_8u_block_omp_uses_thread_buckets() {
        let block_start = 20_000usize;
        let block_size = 16_384usize;
        let n = block_start + block_size + 8;
        let t = vec![1_u8; n];
        let suffixes: Vec<SaSint> = (2..2 + block_size).map(|i| i as SaSint).collect();

        let mut expected_sa = vec![0; n];
        expected_sa[block_start..block_start + block_size].copy_from_slice(&suffixes);
        let mut threaded_sa = expected_sa.clone();
        let mut expected_bucket = vec![0; ALPHABET_SIZE];
        expected_bucket[1] = n as SaSint;
        let mut threaded_bucket = expected_bucket.clone();
        let mut thread_state = alloc_thread_state(4).unwrap();

        final_bwt_scan_right_to_left_8u(
            &t,
            &mut expected_sa,
            &mut expected_bucket,
            block_start as FastSint,
            block_size as FastSint,
        );
        final_bwt_scan_right_to_left_8u_block_omp(
            &t,
            &mut threaded_sa,
            ALPHABET_SIZE as SaSint,
            &mut threaded_bucket,
            block_start as FastSint,
            block_size as FastSint,
            4,
            &mut thread_state,
        );

        assert_eq!(threaded_sa, expected_sa);
        assert_eq!(threaded_bucket, expected_bucket);
    }

    #[test]
    fn libsais64_final_bwt_aux_right_to_left_8u_block_omp_uses_thread_buckets() {
        let block_start = 20_000usize;
        let block_size = 16_384usize;
        let n = block_start + block_size + 8;
        let t = vec![1_u8; n];
        let suffixes: Vec<SaSint> = (2..2 + block_size).map(|i| i as SaSint).collect();

        let mut expected_sa = vec![0; n];
        expected_sa[block_start..block_start + block_size].copy_from_slice(&suffixes);
        let mut threaded_sa = expected_sa.clone();
        let mut expected_i = vec![0; n];
        let mut threaded_i = vec![0; n];
        let mut expected_bucket = vec![0; ALPHABET_SIZE];
        expected_bucket[1] = n as SaSint;
        let mut threaded_bucket = expected_bucket.clone();
        let mut thread_state = alloc_thread_state(4).unwrap();

        final_bwt_aux_scan_right_to_left_8u(
            &t,
            &mut expected_sa,
            0,
            &mut expected_i,
            &mut expected_bucket,
            block_start as FastSint,
            block_size as FastSint,
        );
        final_bwt_aux_scan_right_to_left_8u_block_omp(
            &t,
            &mut threaded_sa,
            ALPHABET_SIZE as SaSint,
            0,
            &mut threaded_i,
            &mut threaded_bucket,
            block_start as FastSint,
            block_size as FastSint,
            4,
            &mut thread_state,
        );

        assert_eq!(threaded_sa, expected_sa);
        assert_eq!(threaded_i, expected_i);
        assert_eq!(threaded_bucket, expected_bucket);
    }

    #[test]
    fn libsais64_final_gsa_right_to_left_8u_block_omp_uses_thread_buckets() {
        let block_start = 20_000usize;
        let block_size = 16_384usize;
        let n = block_start + block_size + 8;
        let t = vec![1_u8; n];
        let suffixes: Vec<SaSint> = (2..2 + block_size).map(|i| i as SaSint).collect();

        let mut expected_sa = vec![0; n];
        expected_sa[block_start..block_start + block_size].copy_from_slice(&suffixes);
        let mut threaded_sa = expected_sa.clone();
        let mut expected_bucket = vec![0; ALPHABET_SIZE];
        expected_bucket[1] = n as SaSint;
        let mut threaded_bucket = expected_bucket.clone();
        let mut thread_state = alloc_thread_state(4).unwrap();

        final_gsa_scan_right_to_left_8u(
            &t,
            &mut expected_sa,
            &mut expected_bucket,
            block_start as FastSint,
            block_size as FastSint,
        );
        final_gsa_scan_right_to_left_8u_block_omp(
            &t,
            &mut threaded_sa,
            ALPHABET_SIZE as SaSint,
            &mut threaded_bucket,
            block_start as FastSint,
            block_size as FastSint,
            4,
            &mut thread_state,
        );

        assert_eq!(threaded_sa, expected_sa);
        assert_eq!(threaded_bucket, expected_bucket);
    }

    #[test]
    fn libsais64_count_and_gather_lms_suffixes_8u_omp_uses_block_partition_for_large_inputs() {
        let n = 65_600usize;
        let text: Vec<u8> = (0..n)
            .map(|i| 1 + ((i * 37 + i / 17) % 251) as u8)
            .collect();

        let mut sa_threaded = vec![-99; n];
        let mut sa_scalar = vec![-99; n];
        let mut buckets_threaded = vec![0; 4 * ALPHABET_SIZE];
        let mut buckets_scalar = vec![0; 4 * ALPHABET_SIZE];
        let mut thread_state = alloc_thread_state(4).unwrap();

        let m_threaded = count_and_gather_lms_suffixes_8u_omp(
            &text,
            &mut sa_threaded,
            n as SaSint,
            &mut buckets_threaded,
            4,
            &mut thread_state,
        );
        let m_scalar = count_and_gather_lms_suffixes_8u(
            &text,
            &mut sa_scalar,
            n as SaSint,
            &mut buckets_scalar,
            0,
            n as FastSint,
        );

        assert_eq!(m_threaded, m_scalar);
        assert_eq!(
            &sa_threaded[n - m_threaded as usize..],
            &sa_scalar[n - m_scalar as usize..]
        );
        assert_eq!(buckets_threaded, buckets_scalar);
    }

    #[test]
    fn libsais64_gather_lms_suffixes_8u_omp_uses_thread_state_for_large_inputs() {
        let n = 65_600usize;
        let text: Vec<u8> = (0..n)
            .map(|i| 1 + ((i * 37 + i / 17) % 251) as u8)
            .collect();
        let mut thread_state = alloc_thread_state(4).unwrap();
        let mut count_sa = vec![-99; n];
        let mut buckets = vec![0; 4 * ALPHABET_SIZE];
        let m = count_and_gather_lms_suffixes_8u_omp(
            &text,
            &mut count_sa,
            n as SaSint,
            &mut buckets,
            4,
            &mut thread_state,
        );

        let mut threaded = vec![-99; n];
        let mut scalar = vec![-99; n];
        gather_lms_suffixes_8u_omp(&text, &mut threaded, n as SaSint, 4, &mut thread_state);
        gather_lms_suffixes_8u(
            &text,
            &mut scalar,
            n as SaSint,
            n as FastSint - 1,
            0,
            n as FastSint,
        );

        assert_eq!(&threaded[n - m as usize..], &scalar[n - m as usize..]);
    }

    #[test]
    fn libsais64_count_and_gather_lms_suffixes_32s_4k_updates_counts_and_suffixes() {
        let t = vec![2, 1, 3, 1, 0];
        let mut sa = vec![0; t.len()];
        let mut buckets = vec![0; 4 * 4];
        let m = count_and_gather_lms_suffixes_32s_4k(
            &t,
            &mut sa,
            t.len() as SaSint,
            4,
            &mut buckets,
            0,
            t.len() as FastSint,
        );
        assert!(m >= 0);
        assert_eq!(buckets.iter().sum::<SaSint>(), t.len() as SaSint);
    }

    #[test]
    fn libsais64_count_and_gather_lms_suffixes_32s_2k_updates_counts_and_suffixes() {
        let t = vec![2, 1, 3, 1, 0];
        let mut sa = vec![0; t.len()];
        let mut buckets = vec![0; 2 * 4];
        let m = count_and_gather_lms_suffixes_32s_2k(
            &t,
            &mut sa,
            t.len() as SaSint,
            4,
            &mut buckets,
            0,
            t.len() as FastSint,
        );
        assert!(m >= 0);
        assert_eq!(buckets.iter().sum::<SaSint>(), t.len() as SaSint);
    }

    #[test]
    fn libsais64_count_and_gather_compacted_lms_suffixes_32s_2k_updates_counts_and_suffixes() {
        let t = vec![2, SAINT_MIN | 1, 3, 1, 0];
        let mut sa = vec![0; t.len()];
        let mut buckets = vec![0; 2 * 4];
        let m = count_and_gather_compacted_lms_suffixes_32s_2k(
            &t,
            &mut sa,
            t.len() as SaSint,
            4,
            &mut buckets,
            0,
            t.len() as FastSint,
        );
        assert!(m >= 0);
        assert_eq!(buckets.iter().sum::<SaSint>(), t.len() as SaSint);
    }

    #[test]
    fn libsais64_count_and_gather_lms_suffixes_32s_4k_nofs_omp_wraps_sequential_version() {
        let t = vec![2, 1, 3, 1, 0];
        let mut sa = vec![0; t.len()];
        let mut buckets = vec![0; 4 * 4];
        let m = count_and_gather_lms_suffixes_32s_4k_nofs_omp(
            &t,
            &mut sa,
            t.len() as SaSint,
            4,
            &mut buckets,
            2,
        );
        assert!(m >= 0);
        assert_eq!(buckets.iter().sum::<SaSint>(), t.len() as SaSint);
    }

    #[test]
    fn libsais64_count_and_gather_lms_suffixes_32s_2k_nofs_omp_wraps_sequential_version() {
        let t = vec![2, 1, 3, 1, 0];
        let mut sa = vec![0; t.len()];
        let mut buckets = vec![0; 2 * 4];
        let m = count_and_gather_lms_suffixes_32s_2k_nofs_omp(
            &t,
            &mut sa,
            t.len() as SaSint,
            4,
            &mut buckets,
            2,
        );
        assert!(m >= 0);
        assert_eq!(buckets.iter().sum::<SaSint>(), t.len() as SaSint);
    }

    #[test]
    fn libsais64_count_and_gather_compacted_lms_suffixes_32s_2k_nofs_omp_wraps_sequential_version()
    {
        let t = vec![2, SAINT_MIN | 1, 3, 1, 0];
        let mut sa = vec![0; t.len()];
        let mut buckets = vec![0; 2 * 4];
        let m = count_and_gather_compacted_lms_suffixes_32s_2k_nofs_omp(
            &t,
            &mut sa,
            t.len() as SaSint,
            4,
            &mut buckets,
            2,
        );
        assert!(m >= 0);
        assert_eq!(buckets.iter().sum::<SaSint>(), t.len() as SaSint);
    }

    #[test]
    fn libsais64_count_and_gather_lms_suffixes_32s_nofs_omp_uses_large_input_paths() {
        let n = 65_600usize;
        let k = 257usize;
        let text: Vec<SaSint> = (0..n)
            .map(|i| 1 + ((i * 37 + i / 17) % (k - 1)) as SaSint)
            .collect();

        let mut sa_threaded = vec![-99; n];
        let mut sa_scalar = vec![-99; n];
        let mut buckets_threaded = vec![0; 4 * k];
        let mut buckets_scalar = vec![0; 4 * k];
        let m_threaded = count_and_gather_lms_suffixes_32s_4k_nofs_omp(
            &text,
            &mut sa_threaded,
            n as SaSint,
            k as SaSint,
            &mut buckets_threaded,
            4,
        );
        let m_scalar = count_and_gather_lms_suffixes_32s_4k(
            &text,
            &mut sa_scalar,
            n as SaSint,
            k as SaSint,
            &mut buckets_scalar,
            0,
            n as FastSint,
        );
        assert_eq!(m_threaded, m_scalar);
        assert_eq!(
            &sa_threaded[n - m_threaded as usize..],
            &sa_scalar[n - m_scalar as usize..]
        );
        assert_eq!(buckets_threaded, buckets_scalar);

        let mut sa_threaded = vec![-99; n];
        let mut sa_scalar = vec![-99; n];
        let mut buckets_threaded = vec![0; 2 * k];
        let mut buckets_scalar = vec![0; 2 * k];
        let m_threaded = count_and_gather_lms_suffixes_32s_2k_nofs_omp(
            &text,
            &mut sa_threaded,
            n as SaSint,
            k as SaSint,
            &mut buckets_threaded,
            4,
        );
        let m_scalar = count_and_gather_lms_suffixes_32s_2k(
            &text,
            &mut sa_scalar,
            n as SaSint,
            k as SaSint,
            &mut buckets_scalar,
            0,
            n as FastSint,
        );
        assert_eq!(m_threaded, m_scalar);
        assert_eq!(
            &sa_threaded[n - m_threaded as usize..],
            &sa_scalar[n - m_scalar as usize..]
        );
        assert_eq!(buckets_threaded, buckets_scalar);
    }

    #[test]
    fn libsais64_count_and_gather_lms_suffixes_32s_fs_omp_uses_large_input_paths() {
        let n = 65_600usize;
        let k = 257usize;
        let text: Vec<SaSint> = (0..n)
            .map(|i| 1 + ((i * 37 + i / 17) % (k - 1)) as SaSint)
            .collect();
        let mut thread_state = alloc_thread_state(4).unwrap();

        let mut sa_threaded = vec![-99; n];
        let mut sa_scalar = vec![-99; n];
        let mut buckets_threaded = vec![0; 4 * k];
        let mut buckets_scalar = vec![0; 4 * k];
        let m_threaded = count_and_gather_lms_suffixes_32s_4k_fs_omp(
            &text,
            &mut sa_threaded,
            n as SaSint,
            k as SaSint,
            &mut buckets_threaded,
            0,
            4,
            &mut thread_state,
        );
        let m_scalar = count_and_gather_lms_suffixes_32s_4k(
            &text,
            &mut sa_scalar,
            n as SaSint,
            k as SaSint,
            &mut buckets_scalar,
            0,
            n as FastSint,
        );
        assert_eq!(m_threaded, m_scalar);
        assert_eq!(
            &sa_threaded[n - m_threaded as usize..],
            &sa_scalar[n - m_scalar as usize..]
        );
        assert_eq!(buckets_threaded, buckets_scalar);

        let mut sa_threaded = vec![-99; n];
        let mut sa_scalar = vec![-99; n];
        let mut buckets_threaded = vec![0; 2 * k];
        let mut buckets_scalar = vec![0; 2 * k];
        let m_threaded = count_and_gather_lms_suffixes_32s_2k_fs_omp(
            &text,
            &mut sa_threaded,
            n as SaSint,
            k as SaSint,
            &mut buckets_threaded,
            0,
            4,
            &mut thread_state,
        );
        let m_scalar = count_and_gather_lms_suffixes_32s_2k(
            &text,
            &mut sa_scalar,
            n as SaSint,
            k as SaSint,
            &mut buckets_scalar,
            0,
            n as FastSint,
        );
        assert_eq!(m_threaded, m_scalar);
        assert_eq!(
            &sa_threaded[n - m_threaded as usize..],
            &sa_scalar[n - m_scalar as usize..]
        );
        assert_eq!(buckets_threaded, buckets_scalar);
    }

    #[test]
    fn libsais64_count_and_gather_compacted_lms_suffixes_32s_nofs_omp_uses_large_input_path() {
        let n = 65_600usize;
        let k = 257usize;
        let text: Vec<SaSint> = (0..n)
            .map(|i| {
                let value = 1 + ((i * 37 + i / 17) % (k - 1)) as SaSint;
                if i % 19 == 0 {
                    value | SAINT_MIN
                } else {
                    value
                }
            })
            .collect();

        let mut sa_threaded = vec![-99; n];
        let mut sa_split = vec![-99; n];
        let mut buckets_threaded = vec![0; 2 * k];
        let mut buckets_split = vec![0; 2 * k];
        let m_threaded = count_and_gather_compacted_lms_suffixes_32s_2k_nofs_omp(
            &text,
            &mut sa_threaded,
            n as SaSint,
            k as SaSint,
            &mut buckets_threaded,
            4,
        );
        count_compacted_lms_suffixes_32s_2k(&text, n as SaSint, k as SaSint, &mut buckets_split);
        let m_split = gather_compacted_lms_suffixes_32s(&text, &mut sa_split, n as SaSint);

        assert_eq!(m_threaded, m_split);
        assert_eq!(
            &sa_threaded[n - m_threaded as usize..],
            &sa_split[n - m_split as usize..]
        );
        assert_eq!(buckets_threaded, buckets_split);
    }

    #[test]
    fn libsais64_count_and_gather_compacted_lms_suffixes_32s_fs_omp_uses_large_input_path() {
        let n = 65_600usize;
        let k = 257usize;
        let text: Vec<SaSint> = (0..n)
            .map(|i| {
                let value = 1 + ((i * 37 + i / 17) % (k - 1)) as SaSint;
                if i % 19 == 0 {
                    value | SAINT_MIN
                } else {
                    value
                }
            })
            .collect();

        let mut sa_threaded = vec![-99; 2 * n];
        let mut sa_scalar = vec![-99; n];
        let mut buckets_threaded = vec![0; 2 * k];
        let mut buckets_scalar = vec![0; 2 * k];
        let mut thread_state = alloc_thread_state(4).unwrap();
        count_and_gather_compacted_lms_suffixes_32s_2k_fs_omp(
            &text,
            &mut sa_threaded,
            n as SaSint,
            k as SaSint,
            &mut buckets_threaded,
            0,
            4,
            &mut thread_state,
        );
        let m_scalar = count_and_gather_compacted_lms_suffixes_32s_2k(
            &text,
            &mut sa_scalar,
            n as SaSint,
            k as SaSint,
            &mut buckets_scalar,
            0,
            n as FastSint,
        );

        assert_eq!(
            &sa_threaded[n - m_scalar as usize..n],
            &sa_scalar[n - m_scalar as usize..]
        );
        assert_eq!(buckets_threaded, buckets_scalar);
    }

    #[test]
    fn libsais64_plcp_lcp_omp_wrappers_match_single_thread_on_large_inputs() {
        let n = 65_600usize;
        let text: Vec<u8> = (0..n).map(|i| (1 + (i % 251)) as u8).collect();
        let sa: Vec<SaSint> = (0..n as SaSint).collect();

        let mut plcp_single = vec![0; n];
        let mut plcp_threaded = vec![0; n];
        compute_phi_omp(&sa, &mut plcp_single, n as SaSint, 1);
        compute_phi_omp(&sa, &mut plcp_threaded, n as SaSint, 4);
        assert_eq!(plcp_threaded, plcp_single);

        compute_plcp_omp(&text, &mut plcp_single, n as SaSint, 1);
        compute_plcp_omp(&text, &mut plcp_threaded, n as SaSint, 4);
        assert_eq!(plcp_threaded, plcp_single);

        let mut lcp_single = vec![0; n];
        let mut lcp_threaded = vec![0; n];
        compute_lcp_omp(&plcp_single, &sa, &mut lcp_single, n as SaSint, 1);
        compute_lcp_omp(&plcp_threaded, &sa, &mut lcp_threaded, n as SaSint, 4);
        assert_eq!(lcp_threaded, lcp_single);
    }

    fn assert_libsais64_matches_c(text: &[u8]) {
        let mut rust_sa = vec![0; text.len()];
        let mut c_sa = vec![0; text.len()];

        let rust_rc = libsais64(text, &mut rust_sa, 0, None);
        let c_rc = unsafe {
            probe_public_libsais64(text.as_ptr(), c_sa.as_mut_ptr(), text.len() as SaSint, 0)
        };

        assert_eq!(rust_rc, c_rc);
        assert_eq!(rust_sa, c_sa);
    }

    fn assert_libsais64_gsa_matches_c(text: &[u8]) {
        let mut rust_sa = vec![0; text.len()];
        let mut c_sa = vec![0; text.len()];

        let rust_rc = libsais64_gsa(text, &mut rust_sa, 0, None);
        let c_rc = unsafe {
            probe_public_libsais64_gsa(text.as_ptr(), c_sa.as_mut_ptr(), text.len() as SaSint, 0)
        };

        assert_eq!(rust_rc, c_rc);
        assert_eq!(rust_sa, c_sa);
    }

    fn assert_libsais64_long_matches_c(text: &[SaSint], k: SaSint) {
        let mut rust_t = text.to_vec();
        let mut c_t = text.to_vec();
        let mut rust_sa = vec![0; text.len()];
        let mut c_sa = vec![0; text.len()];

        let rust_rc = libsais64_long(&mut rust_t, &mut rust_sa, k, 0);
        let c_rc = unsafe {
            probe_public_libsais64_long(
                c_t.as_mut_ptr(),
                c_sa.as_mut_ptr(),
                c_t.len() as SaSint,
                k,
                0,
            )
        };

        assert_eq!(rust_rc, c_rc);
        assert_eq!(rust_t, c_t);
        assert_eq!(rust_sa, c_sa);
    }

    fn assert_libsais64_bwt_matches_c(text: &[u8]) {
        let mut rust_u = vec![0; text.len()];
        let mut rust_a = vec![0; text.len()];
        let mut c_u = vec![0; text.len()];
        let mut c_a = vec![0; text.len()];

        let rust_rc = libsais64_bwt(text, &mut rust_u, &mut rust_a, 0, None);
        let c_rc = unsafe {
            probe_public_libsais64_bwt(
                text.as_ptr(),
                c_u.as_mut_ptr(),
                c_a.as_mut_ptr(),
                text.len() as SaSint,
                0,
            )
        };

        assert_eq!(rust_rc, c_rc);
        assert_eq!(rust_u, c_u);
    }

    fn assert_libsais64_bwt_aux_matches_c(text: &[u8], r: SaSint) {
        let aux_len = if text.is_empty() {
            0
        } else {
            (text.len() - 1) / r as usize + 1
        };
        let mut rust_u = vec![0; text.len()];
        let mut rust_a = vec![0; text.len()];
        let mut rust_i = vec![0; aux_len];
        let mut c_u = vec![0; text.len()];
        let mut c_a = vec![0; text.len()];
        let mut c_i = vec![0; aux_len];

        let rust_rc = libsais64_bwt_aux(text, &mut rust_u, &mut rust_a, 0, None, r, &mut rust_i);
        let c_rc = unsafe {
            probe_public_libsais64_bwt_aux(
                text.as_ptr(),
                c_u.as_mut_ptr(),
                c_a.as_mut_ptr(),
                text.len() as SaSint,
                0,
                r,
                c_i.as_mut_ptr(),
            )
        };

        assert_eq!(rust_rc, c_rc);
        assert_eq!(rust_u, c_u);
        assert_eq!(rust_i, c_i);
    }

    fn assert_libsais64_freq_outputs_match_c(text: &[u8], gsa_text: &[u8]) {
        let mut rust_sa = vec![0; text.len()];
        let mut c_sa = vec![0; text.len()];
        let mut rust_freq = vec![-1; ALPHABET_SIZE];
        let mut c_freq = vec![-1; ALPHABET_SIZE];

        let rust_rc = libsais64(text, &mut rust_sa, 0, Some(&mut rust_freq));
        let c_rc = unsafe {
            probe_public_libsais64_freq(
                text.as_ptr(),
                c_sa.as_mut_ptr(),
                text.len() as SaSint,
                0,
                c_freq.as_mut_ptr(),
            )
        };
        assert_eq!(rust_rc, c_rc);
        assert_eq!(rust_sa, c_sa);
        assert_eq!(rust_freq, c_freq);

        let mut rust_gsa = vec![0; gsa_text.len()];
        let mut c_gsa = vec![0; gsa_text.len()];
        rust_freq.fill(-1);
        c_freq.fill(-1);
        let rust_rc = libsais64_gsa(gsa_text, &mut rust_gsa, 0, Some(&mut rust_freq));
        let c_rc = unsafe {
            probe_public_libsais64_gsa_freq(
                gsa_text.as_ptr(),
                c_gsa.as_mut_ptr(),
                gsa_text.len() as SaSint,
                0,
                c_freq.as_mut_ptr(),
            )
        };
        assert_eq!(rust_rc, c_rc);
        assert_eq!(rust_gsa, c_gsa);
        assert_eq!(rust_freq, c_freq);

        let mut rust_u = vec![0; text.len()];
        let mut rust_a = vec![0; text.len()];
        let mut c_u = vec![0; text.len()];
        let mut c_a = vec![0; text.len()];
        rust_freq.fill(-1);
        c_freq.fill(-1);
        let rust_rc = libsais64_bwt(text, &mut rust_u, &mut rust_a, 0, Some(&mut rust_freq));
        let c_rc = unsafe {
            probe_public_libsais64_bwt_freq(
                text.as_ptr(),
                c_u.as_mut_ptr(),
                c_a.as_mut_ptr(),
                text.len() as SaSint,
                0,
                c_freq.as_mut_ptr(),
            )
        };
        assert_eq!(rust_rc, c_rc);
        assert_eq!(rust_u, c_u);
        assert_eq!(rust_freq, c_freq);

        let r = 4;
        let aux_len = (text.len() - 1) / r as usize + 1;
        let mut rust_i = vec![0; aux_len];
        let mut c_i = vec![0; aux_len];
        rust_freq.fill(-1);
        c_freq.fill(-1);
        let rust_rc = libsais64_bwt_aux(
            text,
            &mut rust_u,
            &mut rust_a,
            0,
            Some(&mut rust_freq),
            r,
            &mut rust_i,
        );
        let c_rc = unsafe {
            probe_public_libsais64_bwt_aux_freq(
                text.as_ptr(),
                c_u.as_mut_ptr(),
                c_a.as_mut_ptr(),
                text.len() as SaSint,
                0,
                c_freq.as_mut_ptr(),
                r,
                c_i.as_mut_ptr(),
            )
        };
        assert_eq!(rust_rc, c_rc);
        assert_eq!(rust_u, c_u);
        assert_eq!(rust_i, c_i);
        assert_eq!(rust_freq, c_freq);
    }

    fn assert_libsais64_unbwt_matches_c(text: &[u8]) {
        let mut bwt = vec![0; text.len()];
        let mut work = vec![0; text.len()];
        let primary = libsais64_bwt(text, &mut bwt, &mut work, 0, None);
        assert!(primary >= 0);

        let mut rust_u = vec![0; text.len()];
        let mut rust_a = vec![0; text.len() + 1];
        let mut c_u = vec![0; text.len()];
        let mut c_a = vec![0; text.len() + 1];

        let rust_rc = libsais64_unbwt(&bwt, &mut rust_u, &mut rust_a, None, primary);
        let c_rc = unsafe {
            probe_public_libsais64_unbwt(
                bwt.as_ptr(),
                c_u.as_mut_ptr(),
                c_a.as_mut_ptr(),
                bwt.len() as SaSint,
                primary,
            )
        };

        assert_eq!(rust_rc, c_rc);
        assert_eq!(rust_u, c_u);
        assert_eq!(rust_u, text);
    }

    fn assert_libsais64_unbwt_aux_matches_c(text: &[u8], r: SaSint) {
        let mut bwt = vec![0; text.len()];
        let mut work = vec![0; text.len()];
        let mut aux = vec![0; (text.len() - 1) / r as usize + 1];
        let bwt_rc = libsais64_bwt_aux(text, &mut bwt, &mut work, 0, None, r, &mut aux);
        assert_eq!(bwt_rc, 0);

        let mut rust_u = vec![0; text.len()];
        let mut rust_a = vec![0; text.len() + 1];
        let mut c_u = vec![0; text.len()];
        let mut c_a = vec![0; text.len() + 1];

        let rust_rc = libsais64_unbwt_aux(&bwt, &mut rust_u, &mut rust_a, None, r, &aux);
        let c_rc = unsafe {
            probe_public_libsais64_unbwt_aux(
                bwt.as_ptr(),
                c_u.as_mut_ptr(),
                c_a.as_mut_ptr(),
                bwt.len() as SaSint,
                r,
                aux.as_ptr(),
            )
        };

        assert_eq!(rust_rc, c_rc);
        assert_eq!(rust_u, c_u);
        assert_eq!(rust_u, text);
    }

    fn assert_libsais64_unbwt_freq_matches_c(text: &[u8]) {
        let mut freq = vec![0; ALPHABET_SIZE];
        let mut bwt = vec![0; text.len()];
        let mut work = vec![0; text.len()];
        let primary = libsais64_bwt(text, &mut bwt, &mut work, 0, Some(&mut freq));
        assert!(primary >= 0);

        let mut rust_u = vec![0; text.len()];
        let mut rust_a = vec![0; text.len() + 1];
        let mut c_u = vec![0; text.len()];
        let mut c_a = vec![0; text.len() + 1];

        let rust_rc = libsais64_unbwt(&bwt, &mut rust_u, &mut rust_a, Some(&freq), primary);
        let c_rc = unsafe {
            probe_public_libsais64_unbwt_freq(
                bwt.as_ptr(),
                c_u.as_mut_ptr(),
                c_a.as_mut_ptr(),
                bwt.len() as SaSint,
                freq.as_ptr(),
                primary,
            )
        };
        assert_eq!(rust_rc, c_rc);
        assert_eq!(rust_u, c_u);
        assert_eq!(rust_u, text);

        let r = 4;
        let mut aux = vec![0; (text.len() - 1) / r as usize + 1];
        let bwt_rc = libsais64_bwt_aux(text, &mut bwt, &mut work, 0, Some(&mut freq), r, &mut aux);
        assert_eq!(bwt_rc, 0);

        rust_u.fill(0);
        rust_a.fill(0);
        c_u.fill(0);
        c_a.fill(0);
        let rust_rc = libsais64_unbwt_aux(&bwt, &mut rust_u, &mut rust_a, Some(&freq), r, &aux);
        let c_rc = unsafe {
            probe_public_libsais64_unbwt_aux_freq(
                bwt.as_ptr(),
                c_u.as_mut_ptr(),
                c_a.as_mut_ptr(),
                bwt.len() as SaSint,
                freq.as_ptr(),
                r,
                aux.as_ptr(),
            )
        };
        assert_eq!(rust_rc, c_rc);
        assert_eq!(rust_u, c_u);
        assert_eq!(rust_u, text);
    }

    fn assert_libsais64_plcp_lcp_matches_c(text: &[u8]) {
        let mut sa = vec![0; text.len()];
        let sa_rc = libsais64(text, &mut sa, 0, None);
        assert_eq!(sa_rc, 0);

        let mut rust_plcp = vec![0; text.len()];
        let mut c_plcp = vec![0; text.len()];
        let rust_plcp_rc = libsais64_plcp(text, &sa, &mut rust_plcp);
        let c_plcp_rc = unsafe {
            probe_public_libsais64_plcp(
                text.as_ptr(),
                sa.as_ptr(),
                c_plcp.as_mut_ptr(),
                text.len() as SaSint,
            )
        };
        assert_eq!(rust_plcp_rc, c_plcp_rc);
        assert_eq!(rust_plcp, c_plcp);

        let mut rust_lcp = vec![0; text.len()];
        let mut c_lcp = vec![0; text.len()];
        let rust_lcp_rc = libsais64_lcp(&rust_plcp, &sa, &mut rust_lcp);
        let c_lcp_rc = unsafe {
            probe_public_libsais64_lcp(
                c_plcp.as_ptr(),
                sa.as_ptr(),
                c_lcp.as_mut_ptr(),
                text.len() as SaSint,
            )
        };
        assert_eq!(rust_lcp_rc, c_lcp_rc);
        assert_eq!(rust_lcp, c_lcp);
    }

    fn assert_libsais64_plcp_gsa_matches_c(text: &[u8]) {
        let mut sa = vec![0; text.len()];
        assert_eq!(libsais64_gsa(text, &mut sa, 0, None), 0);

        let mut rust_plcp = vec![0; text.len()];
        let mut c_plcp = vec![0; text.len()];
        let rust_rc = libsais64_plcp_gsa(text, &sa, &mut rust_plcp);
        let c_rc = unsafe {
            probe_public_libsais64_plcp_gsa(
                text.as_ptr(),
                sa.as_ptr(),
                c_plcp.as_mut_ptr(),
                text.len() as SaSint,
            )
        };

        assert_eq!(rust_rc, c_rc);
        assert_eq!(rust_plcp, c_plcp);
    }

    fn assert_libsais64_bwt_aux_round_trips(text: &[u8], r: SaSint) {
        let mut bwt = vec![0; text.len()];
        let mut work = vec![0; text.len()];
        let mut restored = vec![0; text.len()];
        let mut aux = vec![0; (text.len() - 1) / r as usize + 1];

        let bwt_rc = libsais64_bwt_aux(text, &mut bwt, &mut work, 0, None, r, &mut aux);
        assert_eq!(bwt_rc, 0);

        let unbwt_rc = libsais64_unbwt_aux(&bwt, &mut restored, &mut work, None, r, &aux);
        assert_eq!(unbwt_rc, 0);
        assert_eq!(restored, text);
    }

    #[test]
    fn public_libsais64_matches_upstream_c() {
        for text in [
            b"".as_slice(),
            b"a",
            b"banana",
            b"mississippi",
            b"abracadabra",
            b"AAAAAAAAAAAAAAAA",
            b"zyxwvutsrqponmlk",
        ] {
            assert_libsais64_matches_c(text);
        }
    }

    #[test]
    fn public_libsais64_bwt_matches_upstream_c() {
        for text in [
            b"".as_slice(),
            b"a",
            b"banana",
            b"mississippi",
            b"abracadabra",
            b"AAAAAAAAAAAAAAAA",
            b"zyxwvutsrqponmlk",
        ] {
            assert_libsais64_bwt_matches_c(text);
        }
    }

    #[test]
    fn public_libsais64_gsa_matches_upstream_c() {
        for text in [
            b"\0".as_slice(),
            b"banana\0",
            b"ban\0ana\0",
            b"miss\0issippi\0",
            b"a\0a\0a\0",
        ] {
            assert_libsais64_gsa_matches_c(text);
        }
    }

    #[test]
    fn public_libsais64_long_matches_upstream_c() {
        for (text, k) in [
            (&[][..], 0),
            (&[0][..], 1),
            (&[1, 2, 1, 0][..], 3),
            (&[2, 1, 2, 1, 0][..], 3),
            (&[3, 3, 3, 2, 1, 0][..], 4),
        ] {
            assert_libsais64_long_matches_c(text, k);
        }
    }

    #[test]
    fn public_libsais64_plcp_lcp_matches_upstream_c() {
        for text in [
            b"".as_slice(),
            b"a",
            b"banana",
            b"mississippi",
            b"abracadabra",
            b"AAAAAAAAAAAAAAAA",
            b"zyxwvutsrqponmlk",
        ] {
            assert_libsais64_plcp_lcp_matches_c(text);
        }
    }

    #[test]
    fn public_libsais64_plcp_gsa_matches_upstream_c() {
        for text in [
            b"\0".as_slice(),
            b"banana\0",
            b"ban\0ana\0",
            b"miss\0issippi\0",
            b"a\0a\0a\0",
        ] {
            assert_libsais64_plcp_gsa_matches_c(text);
        }
    }

    #[test]
    fn libsais64_bwt_and_unbwt_round_trip_small_text() {
        let t = b"banana";
        let mut bwt = vec![0u8; t.len()];
        let mut a = vec![0; t.len()];

        let primary = libsais64_bwt(t, &mut bwt, &mut a, 0, None);
        assert!(primary > 0);

        let mut restored = vec![0u8; t.len()];
        let result = libsais64_unbwt(&bwt, &mut restored, &mut a, None, primary);

        assert_eq!(result, 0);
        assert_eq!(restored, t);
    }

    #[test]
    fn libsais64_bwt_aux_and_unbwt_aux_round_trip_small_text() {
        let t = b"mississippi";
        let mut bwt = vec![0u8; t.len()];
        let mut a = vec![0; t.len()];
        let mut samples = vec![0; 4];

        let result = libsais64_bwt_aux(t, &mut bwt, &mut a, 0, None, 4, &mut samples);
        assert_eq!(result, 0);

        let mut restored = vec![0u8; t.len()];
        let result = libsais64_unbwt_aux(&bwt, &mut restored, &mut a, None, 4, &samples);

        assert_eq!(result, 0);
        assert_eq!(restored, t);
    }

    #[test]
    fn libsais64_bwt_aux_and_unbwt_aux_omp_round_trip_small_text() {
        let t = b"mississippi";
        let mut bwt = vec![0u8; t.len()];
        let mut a = vec![0; t.len()];
        let mut samples = vec![0; 4];

        let result = libsais64_bwt_aux(t, &mut bwt, &mut a, 0, None, 4, &mut samples);
        assert_eq!(result, 0);

        let mut restored = vec![0u8; t.len()];
        let result = libsais64_unbwt_aux_omp(&bwt, &mut restored, &mut a, None, 4, &samples, 2);

        assert_eq!(result, 0);
        assert_eq!(restored, t);
    }

    #[test]
    fn libsais64_real_world_round_trip_on_upstream_readme() {
        let t = include_bytes!("../libsais/README.md");
        let mut bwt = vec![0u8; t.len()];
        let mut a = vec![0; t.len()];

        let primary = libsais64_bwt(t, &mut bwt, &mut a, 0, None);
        assert!(primary > 0);

        let mut restored = vec![0u8; t.len()];
        let result = libsais64_unbwt(&bwt, &mut restored, &mut a, None, primary);

        assert_eq!(result, 0);
        assert_eq!(restored, t);
    }

    #[test]
    fn libsais64_real_world_aux_omp_round_trip_on_upstream_c_source() {
        let t = include_bytes!("../libsais/src/libsais.c");
        let mut bwt = vec![0u8; t.len()];
        let mut a = vec![0; t.len()];
        let r = 128;
        let mut samples = vec![0; (t.len() - 1) / usize::try_from(r).expect("fits") + 1];

        let result = libsais64_bwt_aux(t, &mut bwt, &mut a, 0, None, r, &mut samples);
        assert_eq!(result, 0);

        let mut restored = vec![0u8; t.len()];
        let result = libsais64_unbwt_aux_omp(&bwt, &mut restored, &mut a, None, r, &samples, 2);

        assert_eq!(result, 0);
        assert_eq!(restored, t);
    }

    #[test]
    fn libsais64_bwt_aux_rejects_undersized_sampling_array() {
        let t = b"upstream source text";
        let mut bwt = vec![0u8; t.len()];
        let mut a = vec![0; t.len()];
        let mut samples = vec![0; 1];

        let result = libsais64_bwt_aux(t, &mut bwt, &mut a, 0, None, 2, &mut samples);

        assert_eq!(result, -1);

        let result = libsais64_bwt_aux(t, &mut bwt, &mut a, 0, None, 0, &mut samples);

        assert_eq!(result, -1);
    }

    #[test]
    fn libsais64_bwt_aux_omp_rejects_invalid_sampling_rate_without_panicking() {
        let t = b"upstream source text";
        let mut bwt = vec![0u8; t.len()];
        let mut a = vec![0; t.len()];
        let mut samples = vec![0; 4];

        let result = libsais64_bwt_aux_omp(t, &mut bwt, &mut a, 0, None, 0, &mut samples, 2);

        assert_eq!(result, -1);
    }

    #[test]
    fn public_libsais64_empty_and_singleton_inputs_follow_public_contract() {
        let mut empty_sa = Vec::new();
        let mut empty_freq = vec![-1; ALPHABET_SIZE];
        assert_eq!(libsais64(b"", &mut empty_sa, 0, Some(&mut empty_freq)), 0);
        assert!(empty_freq.iter().all(|&value| value == 0));

        empty_freq.fill(-1);
        assert_eq!(
            libsais64_omp(b"", &mut empty_sa, 0, Some(&mut empty_freq), 2),
            0
        );
        assert!(empty_freq.iter().all(|&value| value == 0));

        empty_freq.fill(-1);
        assert_eq!(
            libsais64_gsa(b"", &mut empty_sa, 0, Some(&mut empty_freq)),
            0
        );
        assert!(empty_freq.iter().all(|&value| value == 0));

        let mut empty_bwt = Vec::new();
        let mut empty_work = Vec::new();
        empty_freq.fill(-1);
        assert_eq!(
            libsais64_bwt(
                b"",
                &mut empty_bwt,
                &mut empty_work,
                0,
                Some(&mut empty_freq)
            ),
            0
        );
        assert!(empty_freq.iter().all(|&value| value == 0));

        let mut empty_aux = vec![-1];
        empty_freq.fill(-1);
        assert_eq!(
            libsais64_bwt_aux(
                b"",
                &mut empty_bwt,
                &mut empty_work,
                0,
                Some(&mut empty_freq),
                2,
                &mut empty_aux
            ),
            0
        );
        assert_eq!(empty_aux[0], 0);
        assert!(empty_freq.iter().all(|&value| value == 0));

        let text = b"z";
        let mut sa = vec![-1; 1];
        let mut freq = vec![-1; ALPHABET_SIZE];
        assert_eq!(libsais64(text, &mut sa, 0, Some(&mut freq)), 0);
        assert_eq!(sa, vec![0]);
        assert_eq!(freq[b'z' as usize], 1);
        assert_eq!(freq.iter().sum::<SaSint>(), 1);

        sa.fill(-1);
        freq.fill(-1);
        let mut ctx = create_ctx().expect("context");
        assert_eq!(
            libsais64_ctx(&mut ctx, text, &mut sa, 0, Some(&mut freq)),
            0
        );
        assert_eq!(sa, vec![0]);
        assert_eq!(freq[b'z' as usize], 1);
        assert_eq!(freq.iter().sum::<SaSint>(), 1);

        sa.fill(-1);
        freq.fill(-1);
        assert_eq!(libsais64_omp(text, &mut sa, 0, Some(&mut freq), 2), 0);
        assert_eq!(sa, vec![0]);
        assert_eq!(freq[b'z' as usize], 1);
        assert_eq!(freq.iter().sum::<SaSint>(), 1);

        let mut gsa_sa = vec![-1; 1];
        let mut gsa_freq = vec![-1; ALPHABET_SIZE];
        assert_eq!(libsais64_gsa(b"\0", &mut gsa_sa, 0, Some(&mut gsa_freq)), 0);
        assert_eq!(gsa_sa, vec![0]);
        assert_eq!(gsa_freq[0], 1);
        assert_eq!(gsa_freq.iter().sum::<SaSint>(), 1);

        let mut bwt = vec![0; 1];
        let mut work = vec![0; 1];
        freq.fill(-1);
        assert_eq!(
            libsais64_bwt(text, &mut bwt, &mut work, 0, Some(&mut freq)),
            1
        );
        assert_eq!(bwt, text);
        assert_eq!(freq[b'z' as usize], 1);
        assert_eq!(freq.iter().sum::<SaSint>(), 1);

        let mut aux = vec![-1];
        bwt.fill(0);
        work.fill(0);
        freq.fill(-1);
        assert_eq!(
            libsais64_bwt_aux(text, &mut bwt, &mut work, 0, Some(&mut freq), 2, &mut aux),
            0
        );
        assert_eq!(bwt, text);
        assert_eq!(aux[0], 1);
        assert_eq!(freq[b'z' as usize], 1);
        assert_eq!(freq.iter().sum::<SaSint>(), 1);
    }

    #[test]
    fn public_libsais64_rejects_invalid_aux_sampling_without_panicking() {
        let text = b"banana";
        let mut u = vec![0; text.len()];
        let mut a = vec![0; text.len() + 1];
        let mut aux = vec![0; 2];

        assert_eq!(
            libsais64_bwt_aux(text, &mut u, &mut a, 0, None, 0, &mut aux),
            -1
        );
        assert_eq!(
            libsais64_bwt_aux(text, &mut u, &mut a, 0, None, 3, &mut aux),
            -1
        );
        assert_eq!(libsais64_unbwt_aux(text, &mut u, &mut a, None, 0, &aux), -1);
        assert_eq!(
            libsais64_unbwt_aux_omp(text, &mut u, &mut a, None, 0, &aux, 1),
            -1
        );
    }

    #[test]
    fn libsais64_unbwt_aux_rejects_invalid_sampling_range() {
        let t = b"abc";
        let mut u = vec![0u8; t.len()];
        let mut a = vec![0; t.len()];

        let result = libsais64_unbwt_aux(t, &mut u, &mut a, None, 2, &[0, 4]);

        assert_eq!(result, -1);

        assert_eq!(libsais64_unbwt_aux(t, &mut u, &mut a, None, 0, &[1]), -1);

        let mut ctx = unbwt_create_ctx().expect("context");
        assert_eq!(
            libsais64_unbwt_aux_ctx(&mut ctx, t, &mut u, &mut a, None, 0, &[1]),
            -1
        );
        assert_eq!(
            libsais64_unbwt_aux_omp(t, &mut u, &mut a, None, 0, &[1], 2),
            -1
        );
    }

    #[test]
    fn public_libsais64_omp_rejects_undersized_suffix_arrays() {
        let text = b"banana";
        let mut short_sa = vec![0; text.len() - 1];
        let mut int_text = vec![1, 2, 1, 0];
        let mut short_int_sa = vec![0; int_text.len() - 1];

        assert_eq!(libsais64_omp(text, &mut short_sa, 0, None, 1), -1);
        assert_eq!(
            libsais64_gsa_omp(b"banana\0", &mut short_sa, 0, None, 1),
            -1
        );
        assert_eq!(
            libsais64_int_omp(&mut int_text, &mut short_int_sa, 3, 0, 1),
            -1
        );
    }

    #[test]
    #[ignore = "large real-data regression; requires local yeast FASTA fixture"]
    fn public_libsais64_omp_handles_minibwa_yeast_two_strand_index_input() {
        let path = "/data/henriksson/github/claude/star/.tmp/yeast_conformance/Saccharomyces_cerevisiae.R64-1-1.dna.toplevel.fa";
        let Ok(fasta) = std::fs::read_to_string(path) else {
            eprintln!("skipping missing fixture: {path}");
            return;
        };
        let mut forward = Vec::new();
        for line in fasta.lines() {
            if line.starts_with('>') {
                continue;
            }
            forward.extend(line.as_bytes().iter().filter_map(|&c| match c {
                b'A' | b'a' => Some(0),
                b'C' | b'c' => Some(1),
                b'G' | b'g' => Some(2),
                b'T' | b't' => Some(3),
                _ => None,
            }));
        }
        assert!(
            forward.len() > 12_000_000,
            "fixture should exercise the large-input 64-bit path"
        );

        let mut text = Vec::with_capacity(forward.len() * 2);
        text.extend_from_slice(&forward);
        text.extend(forward.iter().rev().map(|&c| 3 - c));

        const FS: SaSint = 10_000;
        let mut sa = vec![0; text.len() + FS as usize];
        assert_eq!(libsais64_omp(&text, &mut sa, FS, None, 4), 0);
    }

    #[test]
    #[ignore = "large real-data regression; requires local minibwa yeast fixture"]
    fn public_libsais64_omp_matches_plain_on_minibwa_yeast_two_strand_index_input() {
        let l2b_path =
            "/data/henriksson/github/claude/minibwa/.tmp/compare-yeast-now/ref.split.rust.l2b";
        let fasta_path =
            "/data/henriksson/github/claude/minibwa/.tmp/large-real/yeast/ref.sanitized.fa";
        let forward = if let Ok(bytes) = std::fs::read(l2b_path) {
            assert!(bytes.len() >= 64, "short l2b fixture: {l2b_path}");
            assert_eq!(&bytes[..4], b"L2B\x01", "bad l2b magic in {l2b_path}");
            let n_ctg = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
            let tot_len = u64::from_le_bytes(bytes[16..24].try_into().unwrap()) as usize;
            let n_ambi = u64::from_le_bytes(bytes[24..32].try_into().unwrap()) as usize;
            let n_mask = u64::from_le_bytes(bytes[32..40].try_into().unwrap()) as usize;
            let n_pac = u64::from_le_bytes(bytes[56..64].try_into().unwrap()) as usize;
            let pac_start = 64 + 8 * n_ctg + 16 * n_ambi + 16 * n_mask;
            assert!(
                bytes.len() >= pac_start + 8 * n_pac,
                "truncated l2b pac in {l2b_path}"
            );
            let mut pac = Vec::with_capacity(n_pac);
            for chunk in bytes[pac_start..pac_start + 8 * n_pac].chunks_exact(8) {
                pac.push(u64::from_le_bytes(chunk.try_into().unwrap()));
            }
            (0..tot_len)
                .map(|i| ((pac[i >> 5] >> ((i & 31) << 1)) & 3) as u8)
                .collect::<Vec<_>>()
        } else if let Ok(fasta) = std::fs::read_to_string(fasta_path) {
            let mut rng = 11u64;
            let mut forward = Vec::new();
            for line in fasta.lines() {
                if line.starts_with('>') {
                    continue;
                }
                forward.extend(line.bytes().map(|b| {
                    let mut c = match b {
                        b'A' | b'a' => 0,
                        b'C' | b'c' => 1,
                        b'G' | b'g' => 2,
                        b'T' | b't' | b'U' | b'u' => 3,
                        _ => {
                            rng = rng.wrapping_add(0x9e3779b97f4a7c15);
                            let mut z = rng;
                            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
                            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
                            4 | ((z ^ (z >> 31)) & 3) as u8
                        }
                    };
                    if b < b'A' || b > b'Z' {
                        c |= 1 << 3;
                    }
                    c & 3
                }));
            }
            forward
        } else {
            eprintln!("skipping missing fixtures: {l2b_path} and {fasta_path}");
            return;
        };
        assert!(
            forward.len() > 12_000_000,
            "fixture should exercise the minibwa yeast index workload"
        );

        let mut text = Vec::with_capacity(forward.len() * 2);
        text.extend_from_slice(&forward);
        text.extend(forward.iter().rev().map(|&c| 3 - c));

        const FS: SaSint = 10_000;
        let mut plain_sa = vec![0; text.len() + FS as usize + 1];
        let mut omp_sa = vec![0; text.len() + FS as usize + 1];
        assert_eq!(libsais64(&text, &mut plain_sa[1..], FS, None), 0);
        assert_eq!(libsais64_omp(&text, &mut omp_sa[1..], FS, None, 4), 0);
        plain_sa[0] = text.len() as SaSint;
        omp_sa[0] = text.len() as SaSint;
        if let Some(i) = plain_sa[..=text.len()]
            .iter()
            .zip(&omp_sa[..=text.len()])
            .position(|(plain, omp)| plain != omp)
        {
            panic!(
                "first suffix-array diff at {i}: plain={} omp={}",
                plain_sa[i], omp_sa[i]
            );
        }
    }

    #[test]
    #[ignore = "large real-data regression; requires local minibwa or STAR yeast FASTA fixture"]
    fn direct_libsais64_main_handles_minibwa_yeast_two_strand_index_input() {
        let minibwa_path =
            "/data/henriksson/github/claude/minibwa/.tmp/large-real/yeast/ref.sanitized.fa";
        let star_path = "/data/henriksson/github/claude/star/.tmp/yeast_conformance/Saccharomyces_cerevisiae.R64-1-1.dna.toplevel.fa";
        let (path, fasta) = if let Ok(fasta) = std::fs::read_to_string(minibwa_path) {
            (minibwa_path, fasta)
        } else if let Ok(fasta) = std::fs::read_to_string(star_path) {
            (star_path, fasta)
        } else {
            eprintln!("skipping missing fixtures: {minibwa_path} and {star_path}");
            return;
        };
        let mut forward = Vec::new();
        for line in fasta.lines() {
            if line.starts_with('>') {
                continue;
            }
            forward.extend(line.as_bytes().iter().filter_map(|&c| match c {
                b'A' | b'a' => Some(0),
                b'C' | b'c' => Some(1),
                b'G' | b'g' => Some(2),
                b'T' | b't' => Some(3),
                _ => None,
            }));
        }
        assert!(
            forward.len() > 12_000_000,
            "fixture {path} should exercise the minibwa yeast index workload"
        );

        let mut text = Vec::with_capacity(forward.len() * 2);
        text.extend_from_slice(&forward);
        text.extend(forward.iter().rev().map(|&c| 3 - c));

        const FS: SaSint = 10_000;
        let mut sa = vec![0; text.len() + FS as usize];
        assert_eq!(
            libsais64_main(&text, &mut sa, LIBSAIS_FLAGS_NONE, 0, None, FS, None, 1),
            0
        );
    }

    #[test]
    #[ignore = "large real-data regression; requires local yeast FASTA fixture"]
    fn public_libsais64_matches_c_on_minibwa_yeast_two_strand_index_input() {
        let path = "/data/henriksson/github/claude/star/.tmp/yeast_conformance/Saccharomyces_cerevisiae.R64-1-1.dna.toplevel.fa";
        let Ok(fasta) = std::fs::read_to_string(path) else {
            eprintln!("skipping missing fixture: {path}");
            return;
        };
        let mut forward = Vec::new();
        for line in fasta.lines() {
            if line.starts_with('>') {
                continue;
            }
            forward.extend(line.as_bytes().iter().filter_map(|&c| match c {
                b'A' | b'a' => Some(0),
                b'C' | b'c' => Some(1),
                b'G' | b'g' => Some(2),
                b'T' | b't' => Some(3),
                _ => None,
            }));
        }
        let mut text = Vec::with_capacity(forward.len() * 2);
        text.extend_from_slice(&forward);
        text.extend(forward.iter().rev().map(|&c| 3 - c));

        const FS: SaSint = 10_000;
        let mut rust_sa = vec![0; text.len() + FS as usize];
        let mut c_sa = vec![0; text.len() + FS as usize];
        let rust_rc = libsais64(&text, &mut rust_sa, FS, None);
        let c_rc = unsafe {
            probe_public_libsais64(text.as_ptr(), c_sa.as_mut_ptr(), text.len() as SaSint, FS)
        };
        assert_eq!(rust_rc, c_rc);
        if let Some(i) = rust_sa[..text.len()]
            .iter()
            .zip(&c_sa[..text.len()])
            .position(|(r, c)| r != c)
        {
            panic!(
                "first suffix-array diff at {i}: rust={} c={}",
                rust_sa[i], c_sa[i]
            );
        }
    }

    #[test]
    #[ignore = "large real-data regression; requires local yeast FASTA fixture"]
    fn public_libsais64_omp_matches_c_on_minibwa_yeast_two_strand_index_input() {
        let path = "/data/henriksson/github/claude/star/.tmp/yeast_conformance/Saccharomyces_cerevisiae.R64-1-1.dna.toplevel.fa";
        let Ok(fasta) = std::fs::read_to_string(path) else {
            eprintln!("skipping missing fixture: {path}");
            return;
        };
        let mut forward = Vec::new();
        for line in fasta.lines() {
            if line.starts_with('>') {
                continue;
            }
            forward.extend(line.as_bytes().iter().filter_map(|&c| match c {
                b'A' | b'a' => Some(0),
                b'C' | b'c' => Some(1),
                b'G' | b'g' => Some(2),
                b'T' | b't' => Some(3),
                _ => None,
            }));
        }
        let mut text = Vec::with_capacity(forward.len() * 2);
        text.extend_from_slice(&forward);
        text.extend(forward.iter().rev().map(|&c| 3 - c));

        const FS: SaSint = 10_000;
        let mut rust_sa = vec![0; text.len() + FS as usize];
        let mut c_sa = vec![0; text.len() + FS as usize];
        let rust_rc = libsais64_omp(&text, &mut rust_sa, FS, None, 4);
        let c_rc = unsafe {
            probe_public_libsais64(text.as_ptr(), c_sa.as_mut_ptr(), text.len() as SaSint, FS)
        };
        assert_eq!(rust_rc, c_rc);
        if let Some(i) = rust_sa[..text.len()]
            .iter()
            .zip(&c_sa[..text.len()])
            .position(|(r, c)| r != c)
        {
            panic!(
                "first omp suffix-array diff at {i}: rust={} c={}",
                rust_sa[i], c_sa[i]
            );
        }
    }

    #[test]
    fn public_libsais64_ctx_rejects_invalid_public_arguments() {
        let text = b"banana";
        let mut ctx = create_ctx().unwrap();
        let mut short_sa = vec![0; text.len() - 1];
        let mut full_sa = vec![0; text.len()];
        let mut short_freq = vec![0; ALPHABET_SIZE - 1];
        let mut short_u = vec![0; text.len() - 1];
        let mut full_u = vec![0; text.len()];
        let mut short_a = vec![0; text.len() - 1];
        let mut full_a = vec![0; text.len()];
        let mut aux = vec![0; 2];

        assert_eq!(libsais64_ctx(&mut ctx, text, &mut short_sa, 0, None), -1);
        assert_eq!(
            libsais64_ctx(&mut ctx, text, &mut full_sa, 0, Some(&mut short_freq)),
            -1
        );
        assert_eq!(
            libsais64_gsa_ctx(&mut ctx, b"banana", &mut full_sa, 0, None),
            -1
        );
        assert_eq!(
            libsais64_gsa_ctx(&mut ctx, b"banana\0", &mut short_sa, 0, None),
            -1
        );
        assert_eq!(
            libsais64_bwt_ctx(&mut ctx, text, &mut short_u, &mut full_a, 0, None),
            -1
        );
        assert_eq!(
            libsais64_bwt_ctx(&mut ctx, text, &mut full_u, &mut short_a, 0, None),
            -1
        );
        assert_eq!(
            libsais64_bwt_ctx(
                &mut ctx,
                text,
                &mut full_u,
                &mut full_a,
                0,
                Some(&mut short_freq)
            ),
            -1
        );
        assert_eq!(
            libsais64_bwt_aux_ctx(
                &mut ctx,
                text,
                &mut full_u,
                &mut full_a,
                0,
                None,
                0,
                &mut aux
            ),
            -1
        );
        assert_eq!(
            libsais64_bwt_aux_ctx(
                &mut ctx,
                text,
                &mut full_u,
                &mut full_a,
                0,
                None,
                3,
                &mut aux
            ),
            -1
        );
        assert_eq!(
            libsais64_bwt_aux_ctx(
                &mut ctx,
                text,
                &mut full_u,
                &mut full_a,
                0,
                None,
                4,
                &mut []
            ),
            -1
        );

        let mut missing_thread_state_ctx = Context {
            buckets: vec![0; 8 * ALPHABET_SIZE],
            thread_state: None,
            threads: 2,
        };
        assert_eq!(
            libsais64_ctx(&mut missing_thread_state_ctx, text, &mut full_sa, 0, None),
            -2
        );

        let mut zero_thread_ctx = Context {
            buckets: vec![0; 8 * ALPHABET_SIZE],
            thread_state: None,
            threads: 0,
        };
        assert_eq!(
            libsais64_ctx(&mut zero_thread_ctx, text, &mut full_sa, 0, None),
            -2
        );

        let mut short_thread_state_ctx = create_ctx_main(2).expect("context");
        short_thread_state_ctx
            .thread_state
            .as_mut()
            .expect("thread state")
            .truncate(1);
        assert_eq!(
            libsais64_ctx(&mut short_thread_state_ctx, text, &mut full_sa, 0, None),
            -2
        );
    }

    #[test]
    fn public_libsais64_unbwt_ctx_rejects_invalid_public_arguments() {
        let text = b"banana";
        let mut bwt = vec![0; text.len()];
        let mut work = vec![0; text.len()];
        let primary = libsais64_bwt(text, &mut bwt, &mut work, 0, None);
        let mut ctx = unbwt_create_ctx().unwrap();

        let mut short_u = vec![0; text.len() - 1];
        let mut full_u = vec![0; text.len()];
        let mut short_a = vec![0; text.len() - 1];
        let mut full_a = vec![0; text.len()];
        let short_freq = vec![0; ALPHABET_SIZE - 1];
        let good_aux = vec![primary, 4];

        assert_eq!(
            libsais64_unbwt_ctx(&mut ctx, &bwt, &mut short_u, &mut full_a, None, primary),
            -1
        );
        assert_eq!(
            libsais64_unbwt_ctx(&mut ctx, &bwt, &mut full_u, &mut short_a, None, primary),
            -1
        );
        assert_eq!(
            libsais64_unbwt_ctx(
                &mut ctx,
                &bwt,
                &mut full_u,
                &mut full_a,
                Some(&short_freq),
                primary
            ),
            -1
        );
        assert_eq!(
            libsais64_unbwt_ctx(&mut ctx, &bwt, &mut full_u, &mut full_a, None, 0),
            -1
        );
        assert_eq!(
            libsais64_unbwt_aux_ctx(&mut ctx, &bwt, &mut full_u, &mut full_a, None, 3, &good_aux),
            -1
        );
        assert_eq!(
            libsais64_unbwt_aux_ctx(
                &mut ctx,
                &bwt,
                &mut full_u,
                &mut full_a,
                None,
                4,
                &[primary]
            ),
            -1
        );

        let mut malformed_ctx = UnbwtContext {
            bucket2: Vec::new(),
            fastbits: Vec::new(),
            buckets: None,
            threads: 1,
        };
        assert_eq!(
            libsais64_unbwt_ctx(
                &mut malformed_ctx,
                &bwt,
                &mut full_u,
                &mut full_a,
                None,
                primary
            ),
            -2
        );

        let mut missing_parallel_buckets_ctx = UnbwtContext {
            bucket2: vec![0; ALPHABET_SIZE * ALPHABET_SIZE],
            fastbits: vec![0; 1 + (1 << UNBWT_FASTBITS)],
            buckets: None,
            threads: 2,
        };
        assert_eq!(
            libsais64_unbwt_ctx(
                &mut missing_parallel_buckets_ctx,
                &bwt,
                &mut full_u,
                &mut full_a,
                None,
                primary
            ),
            -2
        );
    }

    #[test]
    fn public_libsais64_lcp_helpers_reject_invalid_suffix_entries() {
        let text = b"banana";
        let mut plcp = vec![0; text.len()];
        let mut lcp = vec![0; text.len()];
        let int_text = vec![1, 2, 1, 0];
        let mut int_plcp = vec![0; int_text.len()];

        assert_eq!(libsais64_plcp(text, &[0, 1, -1, 3, 4, 5], &mut plcp), -1);
        assert_eq!(libsais64_plcp(text, &[0, 1, 2, 3, 4, 6], &mut plcp), -1);
        assert_eq!(libsais64_lcp(&plcp, &[0, 1, -1, 3, 4, 5], &mut lcp), -1);
        assert_eq!(libsais64_lcp(&plcp, &[0, 1, 2, 3, 4, 6], &mut lcp), -1);
        assert_eq!(
            libsais64_plcp_int(&int_text, &[0, 1, -1, 3], &mut int_plcp),
            -1
        );
        assert_eq!(
            libsais64_plcp_int_omp(&int_text, &[0, 1, 2, 4], &mut int_plcp, 1),
            -1
        );
    }

    #[test]
    fn public_libsais64_context_wrappers_match_direct_calls() {
        let text = b"banana";
        let gsa_text = b"ban\0ana\0";
        let mut ctx = create_ctx().unwrap();

        let mut direct_sa = vec![0; text.len()];
        let mut ctx_sa = vec![0; text.len()];
        assert_eq!(libsais64(text, &mut direct_sa, 0, None), 0);
        assert_eq!(libsais64_ctx(&mut ctx, text, &mut ctx_sa, 0, None), 0);
        assert_eq!(ctx_sa, direct_sa);

        let mut direct_gsa = vec![0; gsa_text.len()];
        let mut ctx_gsa = vec![0; gsa_text.len()];
        assert_eq!(libsais64_gsa(gsa_text, &mut direct_gsa, 0, None), 0);
        assert_eq!(
            libsais64_gsa_ctx(&mut ctx, gsa_text, &mut ctx_gsa, 0, None),
            0
        );
        assert_eq!(ctx_gsa, direct_gsa);

        let mut direct_bwt = vec![0; text.len()];
        let mut direct_work = vec![0; text.len()];
        let mut ctx_bwt = vec![0; text.len()];
        let mut ctx_work = vec![0; text.len()];
        assert_eq!(
            libsais64_bwt(text, &mut direct_bwt, &mut direct_work, 0, None),
            libsais64_bwt_ctx(&mut ctx, text, &mut ctx_bwt, &mut ctx_work, 0, None)
        );
        assert_eq!(ctx_bwt, direct_bwt);

        let mut direct_aux = vec![0; 2];
        let mut ctx_aux = vec![0; 2];
        assert_eq!(
            libsais64_bwt_aux(
                text,
                &mut direct_bwt,
                &mut direct_work,
                0,
                None,
                4,
                &mut direct_aux
            ),
            libsais64_bwt_aux_ctx(
                &mut ctx,
                text,
                &mut ctx_bwt,
                &mut ctx_work,
                0,
                None,
                4,
                &mut ctx_aux
            )
        );
        assert_eq!(ctx_bwt, direct_bwt);
        assert_eq!(ctx_aux, direct_aux);
    }

    #[test]
    fn libsais64_ctx_matches_plain_entry_point_for_small_text() {
        let t = b"mississippi";
        let mut sa_plain = vec![0; t.len()];
        let mut sa_ctx = vec![0; t.len()];
        let plain = libsais64(t, &mut sa_plain, 0, None);

        let mut ctx = create_ctx().expect("context");
        let with_ctx = libsais64_ctx(&mut ctx, t, &mut sa_ctx, 0, None);

        assert_eq!(plain, 0);
        assert_eq!(with_ctx, 0);
        assert_eq!(sa_ctx, sa_plain);
    }

    #[test]
    fn public_libsais64_unbwt_context_wrappers_match_direct_calls() {
        let text = b"banana";
        let mut bwt = vec![0; text.len()];
        let mut work = vec![0; text.len()];
        let primary = libsais64_bwt(text, &mut bwt, &mut work, 0, None);
        let mut ctx = unbwt_create_ctx().unwrap();

        let mut direct = vec![0; text.len()];
        let mut direct_work = vec![0; text.len() + 1];
        let mut via_ctx = vec![0; text.len()];
        let mut ctx_work = vec![0; text.len() + 1];
        assert_eq!(
            libsais64_unbwt(&bwt, &mut direct, &mut direct_work, None, primary),
            0
        );
        assert_eq!(
            libsais64_unbwt_ctx(&mut ctx, &bwt, &mut via_ctx, &mut ctx_work, None, primary),
            0
        );
        assert_eq!(via_ctx, direct);

        let mut aux = vec![0; 2];
        assert_eq!(
            libsais64_bwt_aux(text, &mut bwt, &mut work, 0, None, 4, &mut aux),
            0
        );
        assert_eq!(
            libsais64_unbwt_aux(&bwt, &mut direct, &mut direct_work, None, 4, &aux),
            0
        );
        assert_eq!(
            libsais64_unbwt_aux_ctx(&mut ctx, &bwt, &mut via_ctx, &mut ctx_work, None, 4, &aux),
            0
        );
        assert_eq!(via_ctx, direct);
    }

    #[test]
    fn public_libsais64_ctx_frequency_wrappers_match_direct_calls() {
        let text = b"banana";
        let gsa_text = b"ban\0ana\0";
        let mut ctx = create_ctx().unwrap();

        let mut direct_sa = vec![0; text.len()];
        let mut ctx_sa = vec![0; text.len()];
        let mut direct_freq = vec![-1; ALPHABET_SIZE];
        let mut ctx_freq = vec![-1; ALPHABET_SIZE];
        assert_eq!(
            libsais64(text, &mut direct_sa, 0, Some(&mut direct_freq)),
            0
        );
        assert_eq!(
            libsais64_ctx(&mut ctx, text, &mut ctx_sa, 0, Some(&mut ctx_freq)),
            0
        );
        assert_eq!(ctx_sa, direct_sa);
        assert_eq!(ctx_freq, direct_freq);

        let mut direct_gsa = vec![0; gsa_text.len()];
        let mut ctx_gsa = vec![0; gsa_text.len()];
        direct_freq.fill(-1);
        ctx_freq.fill(-1);
        assert_eq!(
            libsais64_gsa(gsa_text, &mut direct_gsa, 0, Some(&mut direct_freq)),
            0
        );
        assert_eq!(
            libsais64_gsa_ctx(&mut ctx, gsa_text, &mut ctx_gsa, 0, Some(&mut ctx_freq)),
            0
        );
        assert_eq!(ctx_gsa, direct_gsa);
        assert_eq!(ctx_freq, direct_freq);

        let mut direct_bwt = vec![0; text.len()];
        let mut direct_work = vec![0; text.len()];
        let mut ctx_bwt = vec![0; text.len()];
        let mut ctx_work = vec![0; text.len()];
        direct_freq.fill(-1);
        ctx_freq.fill(-1);
        assert_eq!(
            libsais64_bwt(
                text,
                &mut direct_bwt,
                &mut direct_work,
                0,
                Some(&mut direct_freq)
            ),
            libsais64_bwt_ctx(
                &mut ctx,
                text,
                &mut ctx_bwt,
                &mut ctx_work,
                0,
                Some(&mut ctx_freq)
            )
        );
        assert_eq!(ctx_bwt, direct_bwt);
        assert_eq!(ctx_freq, direct_freq);

        let mut direct_aux = vec![0; 2];
        let mut ctx_aux = vec![0; 2];
        direct_freq.fill(-1);
        ctx_freq.fill(-1);
        assert_eq!(
            libsais64_bwt_aux(
                text,
                &mut direct_bwt,
                &mut direct_work,
                0,
                Some(&mut direct_freq),
                4,
                &mut direct_aux
            ),
            libsais64_bwt_aux_ctx(
                &mut ctx,
                text,
                &mut ctx_bwt,
                &mut ctx_work,
                0,
                Some(&mut ctx_freq),
                4,
                &mut ctx_aux
            )
        );
        assert_eq!(ctx_bwt, direct_bwt);
        assert_eq!(ctx_aux, direct_aux);
        assert_eq!(ctx_freq, direct_freq);
    }

    #[test]
    fn public_libsais64_unbwt_ctx_frequency_wrappers_match_direct_calls() {
        let text = b"abracadabra";
        let mut freq = vec![0; ALPHABET_SIZE];
        let mut bwt = vec![0; text.len()];
        let mut work = vec![0; text.len()];
        let primary = libsais64_bwt(text, &mut bwt, &mut work, 0, Some(&mut freq));
        assert!(primary >= 0);

        let mut ctx = unbwt_create_ctx().unwrap();
        let mut direct = vec![0; text.len()];
        let mut direct_work = vec![0; text.len() + 1];
        let mut via_ctx = vec![0; text.len()];
        let mut ctx_work = vec![0; text.len() + 1];
        assert_eq!(
            libsais64_unbwt(&bwt, &mut direct, &mut direct_work, Some(&freq), primary),
            libsais64_unbwt_ctx(
                &mut ctx,
                &bwt,
                &mut via_ctx,
                &mut ctx_work,
                Some(&freq),
                primary
            )
        );
        assert_eq!(via_ctx, direct);
        assert_eq!(via_ctx, text);

        let mut aux = vec![0; (text.len() - 1) / 4 + 1];
        assert_eq!(
            libsais64_bwt_aux(text, &mut bwt, &mut work, 0, Some(&mut freq), 4, &mut aux),
            0
        );
        direct.fill(0);
        direct_work.fill(0);
        via_ctx.fill(0);
        ctx_work.fill(0);
        assert_eq!(
            libsais64_unbwt_aux(&bwt, &mut direct, &mut direct_work, Some(&freq), 4, &aux),
            libsais64_unbwt_aux_ctx(
                &mut ctx,
                &bwt,
                &mut via_ctx,
                &mut ctx_work,
                Some(&freq),
                4,
                &aux
            )
        );
        assert_eq!(via_ctx, direct);
        assert_eq!(via_ctx, text);
    }

    #[test]
    fn public_libsais64_omp_wrappers_match_direct_calls() {
        let text = b"banana";
        let gsa_text = b"ban\0ana\0";

        let mut direct_sa = vec![0; text.len()];
        let mut omp_sa = vec![0; text.len()];
        assert_eq!(libsais64(text, &mut direct_sa, 0, None), 0);
        assert_eq!(libsais64_omp(text, &mut omp_sa, 0, None, 2), 0);
        assert_eq!(omp_sa, direct_sa);
        assert_eq!(libsais64_omp(text, &mut omp_sa, 0, None, -1), -1);

        let mut direct_gsa = vec![0; gsa_text.len()];
        let mut omp_gsa = vec![0; gsa_text.len()];
        assert_eq!(libsais64_gsa(gsa_text, &mut direct_gsa, 0, None), 0);
        assert_eq!(libsais64_gsa_omp(gsa_text, &mut omp_gsa, 0, None, 2), 0);
        assert_eq!(omp_gsa, direct_gsa);
        assert_eq!(libsais64_gsa_omp(gsa_text, &mut omp_gsa, 0, None, -1), -1);

        let int_text = vec![2, 1, 3, 1, 0];
        let mut direct_int_text = int_text.clone();
        let mut omp_int_text = int_text.clone();
        let mut direct_int_sa = vec![0; int_text.len()];
        let mut omp_int_sa = vec![0; int_text.len()];
        assert_eq!(
            libsais64_int(&mut direct_int_text, &mut direct_int_sa, 4, 0),
            0
        );
        assert_eq!(
            libsais64_int_omp(&mut omp_int_text, &mut omp_int_sa, 4, 0, 2),
            0
        );
        assert_eq!(omp_int_sa, direct_int_sa);
        assert_eq!(
            libsais64_int_omp(&mut omp_int_text, &mut omp_int_sa, 4, 0, -1),
            -1
        );

        let long_text = vec![3, 1, 4, 1, 5, 0];
        let mut direct_long_text = long_text.clone();
        let mut omp_long_text = long_text.clone();
        let mut direct_long_sa = vec![0; long_text.len()];
        let mut omp_long_sa = vec![0; long_text.len()];
        assert_eq!(
            libsais64_long(&mut direct_long_text, &mut direct_long_sa, 6, 0),
            0
        );
        assert_eq!(
            libsais64_long_omp(&mut omp_long_text, &mut omp_long_sa, 6, 0, 2),
            0
        );
        assert_eq!(omp_long_sa, direct_long_sa);
        assert_eq!(
            libsais64_long_omp(&mut omp_long_text, &mut omp_long_sa, 6, 0, -1),
            -1
        );

        let mut direct_bwt = vec![0; text.len()];
        let mut direct_work = vec![0; text.len()];
        let mut omp_bwt = vec![0; text.len()];
        let mut omp_work = vec![0; text.len()];
        assert_eq!(
            libsais64_bwt(text, &mut direct_bwt, &mut direct_work, 0, None),
            libsais64_bwt_omp(text, &mut omp_bwt, &mut omp_work, 0, None, 2)
        );
        assert_eq!(omp_bwt, direct_bwt);
        assert_eq!(
            libsais64_bwt_omp(text, &mut omp_bwt, &mut omp_work, 0, None, -1),
            -1
        );

        let mut direct_aux = vec![0; 2];
        let mut omp_aux = vec![0; 2];
        assert_eq!(
            libsais64_bwt_aux(
                text,
                &mut direct_bwt,
                &mut direct_work,
                0,
                None,
                4,
                &mut direct_aux
            ),
            libsais64_bwt_aux_omp(
                text,
                &mut omp_bwt,
                &mut omp_work,
                0,
                None,
                4,
                &mut omp_aux,
                2
            )
        );
        assert_eq!(omp_bwt, direct_bwt);
        assert_eq!(omp_aux, direct_aux);
        assert_eq!(
            libsais64_bwt_aux_omp(
                text,
                &mut omp_bwt,
                &mut omp_work,
                0,
                None,
                4,
                &mut omp_aux,
                -1
            ),
            -1
        );
    }

    #[test]
    fn public_libsais64_plcp_omp_wrappers_match_direct_calls() {
        let text = b"banana";
        let mut sa = vec![0; text.len()];
        assert_eq!(libsais64(text, &mut sa, 0, None), 0);

        let mut direct_plcp = vec![0; text.len()];
        let mut omp_plcp = vec![0; text.len()];
        assert_eq!(libsais64_plcp(text, &sa, &mut direct_plcp), 0);
        assert_eq!(libsais64_plcp_omp(text, &sa, &mut omp_plcp, 2), 0);
        assert_eq!(omp_plcp, direct_plcp);
        assert_eq!(libsais64_plcp_omp(text, &sa, &mut omp_plcp, -1), -1);

        let mut direct_lcp = vec![0; text.len()];
        let mut omp_lcp = vec![0; text.len()];
        assert_eq!(libsais64_lcp(&direct_plcp, &sa, &mut direct_lcp), 0);
        assert_eq!(libsais64_lcp_omp(&direct_plcp, &sa, &mut omp_lcp, 2), 0);
        assert_eq!(omp_lcp, direct_lcp);
        assert_eq!(libsais64_lcp_omp(&direct_plcp, &sa, &mut omp_lcp, -1), -1);

        let gsa_text = b"ban\0ana\0";
        let mut gsa = vec![0; gsa_text.len()];
        assert_eq!(libsais64_gsa(gsa_text, &mut gsa, 0, None), 0);
        let mut direct_gsa_plcp = vec![0; gsa_text.len()];
        let mut omp_gsa_plcp = vec![0; gsa_text.len()];
        assert_eq!(libsais64_plcp_gsa(gsa_text, &gsa, &mut direct_gsa_plcp), 0);
        assert_eq!(
            libsais64_plcp_gsa_omp(gsa_text, &gsa, &mut omp_gsa_plcp, 2),
            0
        );
        assert_eq!(omp_gsa_plcp, direct_gsa_plcp);
        assert_eq!(
            libsais64_plcp_gsa_omp(gsa_text, &gsa, &mut omp_gsa_plcp, -1),
            -1
        );

        let int_text = vec![2, 1, 3, 1, 0];
        let mut int_text_for_sa = int_text.clone();
        let mut int_sa = vec![0; int_text.len()];
        assert_eq!(libsais64_int(&mut int_text_for_sa, &mut int_sa, 4, 0), 0);
        let mut direct_int_plcp = vec![0; int_text.len()];
        let mut omp_int_plcp = vec![0; int_text.len()];
        assert_eq!(
            libsais64_plcp_int(&int_text, &int_sa, &mut direct_int_plcp),
            0
        );
        assert_eq!(
            libsais64_plcp_int_omp(&int_text, &int_sa, &mut omp_int_plcp, 2),
            0
        );
        assert_eq!(omp_int_plcp, direct_int_plcp);
        assert_eq!(
            libsais64_plcp_int_omp(&int_text, &int_sa, &mut omp_int_plcp, -1),
            -1
        );
    }

    #[test]
    fn public_libsais64_omp_frequency_wrappers_match_direct_calls() {
        let text = b"banana";
        let gsa_text = b"ban\0ana\0";

        let mut direct_sa = vec![0; text.len()];
        let mut omp_sa = vec![0; text.len()];
        let mut direct_freq = vec![-1; ALPHABET_SIZE];
        let mut omp_freq = vec![-1; ALPHABET_SIZE];
        assert_eq!(
            libsais64(text, &mut direct_sa, 0, Some(&mut direct_freq)),
            0
        );
        assert_eq!(
            libsais64_omp(text, &mut omp_sa, 0, Some(&mut omp_freq), 2),
            0
        );
        assert_eq!(omp_sa, direct_sa);
        assert_eq!(omp_freq, direct_freq);

        let mut direct_gsa = vec![0; gsa_text.len()];
        let mut omp_gsa = vec![0; gsa_text.len()];
        direct_freq.fill(-1);
        omp_freq.fill(-1);
        assert_eq!(
            libsais64_gsa(gsa_text, &mut direct_gsa, 0, Some(&mut direct_freq)),
            0
        );
        assert_eq!(
            libsais64_gsa_omp(gsa_text, &mut omp_gsa, 0, Some(&mut omp_freq), 2),
            0
        );
        assert_eq!(omp_gsa, direct_gsa);
        assert_eq!(omp_freq, direct_freq);

        let mut direct_bwt = vec![0; text.len()];
        let mut direct_work = vec![0; text.len()];
        let mut omp_bwt = vec![0; text.len()];
        let mut omp_work = vec![0; text.len()];
        direct_freq.fill(-1);
        omp_freq.fill(-1);
        assert_eq!(
            libsais64_bwt(
                text,
                &mut direct_bwt,
                &mut direct_work,
                0,
                Some(&mut direct_freq)
            ),
            libsais64_bwt_omp(text, &mut omp_bwt, &mut omp_work, 0, Some(&mut omp_freq), 2)
        );
        assert_eq!(omp_bwt, direct_bwt);
        assert_eq!(omp_freq, direct_freq);

        let mut direct_aux = vec![0; 2];
        let mut omp_aux = vec![0; 2];
        direct_freq.fill(-1);
        omp_freq.fill(-1);
        assert_eq!(
            libsais64_bwt_aux(
                text,
                &mut direct_bwt,
                &mut direct_work,
                0,
                Some(&mut direct_freq),
                4,
                &mut direct_aux
            ),
            libsais64_bwt_aux_omp(
                text,
                &mut omp_bwt,
                &mut omp_work,
                0,
                Some(&mut omp_freq),
                4,
                &mut omp_aux,
                2
            )
        );
        assert_eq!(omp_bwt, direct_bwt);
        assert_eq!(omp_aux, direct_aux);
        assert_eq!(omp_freq, direct_freq);
    }

    #[test]
    fn public_libsais64_unbwt_omp_frequency_wrappers_match_direct_calls() {
        let text = b"abracadabra";
        let mut freq = vec![0; ALPHABET_SIZE];
        let mut bwt = vec![0; text.len()];
        let mut work = vec![0; text.len()];
        let primary = libsais64_bwt(text, &mut bwt, &mut work, 0, Some(&mut freq));
        assert!(primary >= 0);

        let mut direct = vec![0; text.len()];
        let mut direct_work = vec![0; text.len() + 1];
        let mut omp = vec![0; text.len()];
        let mut omp_work = vec![0; text.len() + 1];
        assert_eq!(
            libsais64_unbwt(&bwt, &mut direct, &mut direct_work, Some(&freq), primary),
            libsais64_unbwt_omp(&bwt, &mut omp, &mut omp_work, Some(&freq), primary, 2)
        );
        assert_eq!(omp, direct);
        assert_eq!(omp, text);

        let mut aux = vec![0; (text.len() - 1) / 4 + 1];
        assert_eq!(
            libsais64_bwt_aux(text, &mut bwt, &mut work, 0, Some(&mut freq), 4, &mut aux),
            0
        );
        direct.fill(0);
        direct_work.fill(0);
        omp.fill(0);
        omp_work.fill(0);
        assert_eq!(
            libsais64_unbwt_aux(&bwt, &mut direct, &mut direct_work, Some(&freq), 4, &aux),
            libsais64_unbwt_aux_omp(&bwt, &mut omp, &mut omp_work, Some(&freq), 4, &aux, 2)
        );
        assert_eq!(omp, direct);
        assert_eq!(omp, text);
    }

    #[test]
    fn public_libsais64_bwt_aux_matches_upstream_c() {
        for text in [
            b"banana".as_slice(),
            b"mississippi",
            b"abracadabra",
            b"AAAAAAAAAAAAAAAA",
            b"zyxwvutsrqponmlk",
        ] {
            assert_libsais64_bwt_aux_matches_c(text, 4);
        }
    }

    #[test]
    fn public_libsais64_frequency_outputs_match_upstream_c() {
        assert_libsais64_freq_outputs_match_c(b"banana", b"ban\0ana\0");
    }

    #[test]
    fn public_libsais64_unbwt_with_frequency_matches_upstream_c() {
        assert_libsais64_unbwt_freq_matches_c(b"abracadabra");
    }

    #[test]
    fn public_libsais64_unbwt_matches_upstream_c() {
        for text in [
            b"a".as_slice(),
            b"banana",
            b"mississippi",
            b"abracadabra",
            b"AAAAAAAAAAAAAAAA",
            b"zyxwvutsrqponmlk",
        ] {
            assert_libsais64_unbwt_matches_c(text);
        }
    }

    #[test]
    fn public_libsais64_unbwt_aux_matches_upstream_c() {
        for text in [
            b"banana".as_slice(),
            b"mississippi",
            b"abracadabra",
            b"AAAAAAAAAAAAAAAA",
            b"zyxwvutsrqponmlk",
        ] {
            assert_libsais64_unbwt_aux_matches_c(text, 4);
        }
    }

    #[test]
    fn public_libsais64_bwt_aux_round_trips() {
        for text in [
            b"banana".as_slice(),
            b"mississippi",
            b"abracadabra",
            b"AAAAAAAAAAAAAAAA",
            b"zyxwvutsrqponmlk",
        ] {
            assert_libsais64_bwt_aux_round_trips(text, 4);
        }
    }

    fn deterministic_large_input(seed: u64, len: usize, alphabet: u32) -> Vec<u8> {
        let mut rng = seed | 1;
        (0..len)
            .map(|_| {
                rng = rng.wrapping_add(0x9e3779b97f4a7c15);
                let mut z = rng;
                z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
                z ^= z >> 31;
                (z % u64::from(alphabet)) as u8
            })
            .collect()
    }

    /// Regression test for the OMP-to-rayon port of libsais64: with threads>1
    /// (n>=65_536 to hit the parallel branch) the SA must equal the single-thread SA.
    #[test]
    fn libsais64_omp_threads_4_matches_single_thread_on_large_input() {
        let text = deterministic_large_input(0xBADF00D, 250_000, 8);
        let mut sa_single = vec![0 as SaSint; text.len() + 1];
        let mut sa_omp = vec![0 as SaSint; text.len() + 1];
        assert_eq!(libsais64(&text, &mut sa_single, 0, None), 0);
        assert_eq!(libsais64_omp(&text, &mut sa_omp, 0, None, 4), 0);
        assert_eq!(
            sa_single, sa_omp,
            "libsais64 threads=4 SA must match threads=1 SA"
        );
    }

    /// Guard against nested-rayon-pool deadlocks for libsais64.
    #[test]
    fn libsais64_omp_does_not_deadlock_when_invoked_from_rayon_worker() {
        use rayon::prelude::*;
        let text = deterministic_large_input(0x5EED64, 100_000, 4);
        let inputs: Vec<&[u8]> = vec![&text, &text, &text, &text];
        let results: Vec<SaSint> = inputs
            .par_iter()
            .map(|t| {
                let mut sa = vec![0 as SaSint; t.len() + 1];
                libsais64_omp(t, &mut sa, 0, None, 4)
            })
            .collect();
        assert!(results.iter().all(|&rc| rc == 0));
    }
}
