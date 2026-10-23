//! Rust translation of upstream [libsais](https://github.com/IlyaGrebnov/libsais)
//! 2.10.4 by Ilya Grebnov.
//!
//! This module exposes the 16-bit alphabet suffix array, BWT, unBWT, PLCP and
//! LCP entry points (mirroring `libsais16.h`).

use std::mem;

use rayon::prelude::*;

use crate::{run_rayon_with_threads, SyncMutPtr};

pub type SaSint = i32;
pub type SaUint = u32;

pub const ALPHABET_SIZE: usize = 1usize << 16;
const SAINT_MAX: SaSint = SaSint::MAX;
const SAINT_MIN: SaSint = SaSint::MIN;
const SAINT_BIT: u32 = 32;
const SUFFIX_GROUP_BIT: u32 = SAINT_BIT - 1;
const SUFFIX_GROUP_MARKER: SaSint = 1_i32 << (SUFFIX_GROUP_BIT - 1);
const LIBSAIS_FLAGS_BWT: SaSint = 1;
const LIBSAIS_FLAGS_GSA: SaSint = 2;
const LIBSAIS_LOCAL_BUFFER_SIZE: usize = 2000;
const UNBWT_FASTBITS: usize = 17;
const PER_THREAD_CACHE_SIZE: usize = 2_097_184;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct ThreadCache {
    symbol: SaSint,
    index: SaSint,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ThreadState {
    position: SaSint,
    m: SaSint,
    last_lms_suffix: SaSint,
    count: SaSint,
    buckets: Vec<SaSint>,
    cache: Vec<ThreadCache>,
    cache_entries: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Context {
    buckets: Vec<SaSint>,
    thread_state: Option<Vec<ThreadState>>,
    threads: SaSint,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct UnbwtContext {
    bucket2: Vec<usize>,
    fastbits: Vec<u16>,
    buckets: Option<Vec<usize>>,
    threads: SaSint,
}

/// Creates the libsais16 context that allows reusing allocated memory with each libsais16 operation.
///
/// In multi-threaded environments, use one context per thread for parallel executions.
///
/// Returns the context, or `None` on allocation failure.
pub fn create_ctx() -> Option<Context> {
    create_ctx_main(1)
}

/// Creates the libsais16 context for parallel operations using OpenMP-style threading.
///
/// In multi-threaded environments, use one context per thread for parallel executions.
///
/// - `threads`: number of worker threads (can be 0 for the implementation default).
///
/// Returns the context, or `None` on allocation failure.
pub fn create_ctx_omp(threads: SaSint) -> Option<Context> {
    if threads < 0 {
        None
    } else {
        create_ctx_main(normalize_threads(threads))
    }
}

/// Destroys the libsais16 context and frees previously allocated memory.
pub fn free_ctx(_ctx: Context) {}

/// Creates the libsais16 reverse-BWT context that allows reusing allocated memory with each `libsais16_unbwt_*` operation.
///
/// In multi-threaded environments, use one context per thread for parallel executions.
///
/// Returns the context, or `None` on allocation failure.
pub fn unbwt_create_ctx() -> Option<UnbwtContext> {
    unbwt_create_ctx_main(1)
}

/// Creates the libsais16 reverse-BWT context for parallel `libsais16_unbwt_*` operations using OpenMP-style threading.
///
/// In multi-threaded environments, use one context per thread for parallel executions.
///
/// - `threads`: number of worker threads (can be 0 for the implementation default).
///
/// Returns the context, or `None` on allocation failure.
pub fn unbwt_create_ctx_omp(threads: SaSint) -> Option<UnbwtContext> {
    if threads < 0 {
        None
    } else {
        unbwt_create_ctx_main(normalize_threads(threads))
    }
}

/// Destroys the libsais16 reverse-BWT context and frees previously allocated memory.
pub fn unbwt_free_ctx(_ctx: UnbwtContext) {}

fn normalize_threads(threads: SaSint) -> SaSint {
    if threads > 0 {
        threads
    } else {
        1
    }
}

fn align_up(value: usize, alignment: usize) -> usize {
    (value + (alignment - 1)) & !(alignment - 1)
}

fn alloc_thread_state(threads: SaSint) -> Option<Vec<ThreadState>> {
    let threads = usize::try_from(threads).ok()?;
    let mut thread_state = Vec::with_capacity(threads);
    for _ in 0..threads {
        thread_state.push(ThreadState {
            position: 0,
            m: 0,
            last_lms_suffix: 0,
            count: 0,
            buckets: vec![0; 4 * ALPHABET_SIZE],
            cache: vec![ThreadCache::default(); PER_THREAD_CACHE_SIZE],
            cache_entries: PER_THREAD_CACHE_SIZE,
        });
    }
    Some(thread_state)
}

fn create_ctx_main(threads: SaSint) -> Option<Context> {
    let buckets = vec![0; 8 * ALPHABET_SIZE];
    let thread_state = if threads > 1 {
        Some(alloc_thread_state(threads)?)
    } else {
        None
    };

    Some(Context {
        buckets,
        thread_state,
        threads,
    })
}

fn unbwt_create_ctx_main(threads: SaSint) -> Option<UnbwtContext> {
    let bucket2 = vec![0; ALPHABET_SIZE];
    let fastbits = vec![0; 1 + (1 << UNBWT_FASTBITS)];
    let buckets = if threads > 1 {
        Some(vec![0; usize::try_from(threads).ok()? * ALPHABET_SIZE])
    } else {
        None
    };

    Some(UnbwtContext {
        bucket2,
        fastbits,
        buckets,
        threads,
    })
}

fn fill_freq(t: &[u16], freq: Option<&mut [SaSint]>) {
    if let Some(freq) = freq {
        freq[..ALPHABET_SIZE].fill(0);
        for &symbol in t {
            freq[symbol as usize] += 1;
        }
    }
}

fn buckets_index4(c: usize, s: usize) -> usize {
    (c << 2) + s
}

fn buckets_index2(c: usize, s: usize) -> usize {
    (c << 1) + s
}

fn place_cached_suffixes(
    sa: &mut [SaSint],
    cache: &[ThreadCache],
    block_start: SaSint,
    block_size: SaSint,
) {
    let start = usize::try_from(block_start).expect("block_start must be non-negative");
    let len = usize::try_from(block_size).expect("block_size must be non-negative");
    let entries = if cache.len() >= start + len {
        &cache[start..start + len]
    } else {
        &cache[..len]
    };

    for entry in entries {
        sa[entry.symbol as usize] = entry.index;
    }
}

fn compact_and_place_cached_suffixes(
    sa: &mut [SaSint],
    cache: &mut [ThreadCache],
    block_start: SaSint,
    block_size: SaSint,
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
    place_cached_suffixes(sa, cache, block_start, (write - read_start) as SaSint);
}

fn count_negative_marked_suffixes(
    sa: &[SaSint],
    block_start: SaSint,
    block_size: SaSint,
) -> SaSint {
    let start = block_start as usize;
    let end = start + block_size as usize;
    sa[start..end].iter().filter(|&&value| value < 0).count() as SaSint
}

fn count_zero_marked_suffixes(sa: &[SaSint], block_start: SaSint, block_size: SaSint) -> SaSint {
    let start = block_start as usize;
    let end = start + block_size as usize;
    sa[start..end].iter().filter(|&&value| value == 0).count() as SaSint
}

fn accumulate_counts_s32_n(
    buckets: &mut [SaSint],
    bucket00: usize,
    bucket_size: usize,
    bucket_stride: usize,
    num_buckets: usize,
) {
    for s in 0..bucket_size {
        let mut sum = buckets[bucket00 + s];
        for bucket in 1..num_buckets {
            sum += buckets[bucket00 - bucket * bucket_stride + s];
        }
        buckets[bucket00 + s] = sum;
    }
}

fn accumulate_counts_s32_2(
    buckets: &mut [SaSint],
    bucket00: usize,
    bucket_size: usize,
    bucket_stride: usize,
) {
    accumulate_counts_s32_n(buckets, bucket00, bucket_size, bucket_stride, 2);
}

fn accumulate_counts_s32_3(
    buckets: &mut [SaSint],
    bucket00: usize,
    bucket_size: usize,
    bucket_stride: usize,
) {
    accumulate_counts_s32_n(buckets, bucket00, bucket_size, bucket_stride, 3);
}

fn accumulate_counts_s32_4(
    buckets: &mut [SaSint],
    bucket00: usize,
    bucket_size: usize,
    bucket_stride: usize,
) {
    accumulate_counts_s32_n(buckets, bucket00, bucket_size, bucket_stride, 4);
}

fn accumulate_counts_s32_5(
    buckets: &mut [SaSint],
    bucket00: usize,
    bucket_size: usize,
    bucket_stride: usize,
) {
    accumulate_counts_s32_n(buckets, bucket00, bucket_size, bucket_stride, 5);
}

fn accumulate_counts_s32_6(
    buckets: &mut [SaSint],
    bucket00: usize,
    bucket_size: usize,
    bucket_stride: usize,
) {
    accumulate_counts_s32_n(buckets, bucket00, bucket_size, bucket_stride, 6);
}

fn accumulate_counts_s32_7(
    buckets: &mut [SaSint],
    bucket00: usize,
    bucket_size: usize,
    bucket_stride: usize,
) {
    accumulate_counts_s32_n(buckets, bucket00, bucket_size, bucket_stride, 7);
}

fn accumulate_counts_s32_8(
    buckets: &mut [SaSint],
    bucket00: usize,
    bucket_size: usize,
    bucket_stride: usize,
) {
    accumulate_counts_s32_n(buckets, bucket00, bucket_size, bucket_stride, 8);
}

fn accumulate_counts_s32_9(
    buckets: &mut [SaSint],
    bucket00: usize,
    bucket_size: usize,
    bucket_stride: usize,
) {
    accumulate_counts_s32_n(buckets, bucket00, bucket_size, bucket_stride, 9);
}

fn accumulate_counts_s32(
    buckets: &mut [SaSint],
    bucket00: usize,
    bucket_size: usize,
    bucket_stride: usize,
    mut num_buckets: usize,
) {
    while num_buckets >= 9 {
        accumulate_counts_s32_9(
            buckets,
            bucket00 - (num_buckets - 9) * bucket_stride,
            bucket_size,
            bucket_stride,
        );
        num_buckets -= 8;
    }

    match num_buckets {
        2 => accumulate_counts_s32_2(buckets, bucket00, bucket_size, bucket_stride),
        3 => accumulate_counts_s32_3(buckets, bucket00, bucket_size, bucket_stride),
        4 => accumulate_counts_s32_4(buckets, bucket00, bucket_size, bucket_stride),
        5 => accumulate_counts_s32_5(buckets, bucket00, bucket_size, bucket_stride),
        6 => accumulate_counts_s32_6(buckets, bucket00, bucket_size, bucket_stride),
        7 => accumulate_counts_s32_7(buckets, bucket00, bucket_size, bucket_stride),
        8 => accumulate_counts_s32_8(buckets, bucket00, bucket_size, bucket_stride),
        _ => {}
    }
}

fn flip_suffix_markers_omp(sa: &mut [SaSint], l: SaSint, threads: SaSint) {
    let len = usize::try_from(l).expect("l must be non-negative");
    let omp_num_threads = if threads > 1 && l >= 65_536 {
        usize::try_from(threads).expect("threads must be non-negative")
    } else {
        1
    };
    let omp_block_stride = (len / omp_num_threads) & !15usize;
    if omp_num_threads > 1 {
        let chunk_size = omp_block_stride.max(16);
        run_rayon_with_threads(omp_num_threads, || {
            sa[..len].par_chunks_mut(chunk_size).for_each(|chunk| {
                for value in chunk {
                    *value ^= SAINT_MIN;
                }
            });
        });
        return;
    }
    // single-thread fallback
    for value in &mut sa[..len] {
        *value ^= SAINT_MIN;
    }
}

fn gather_lms_suffixes_32s(t: &[SaSint], sa: &mut [SaSint], n: SaSint) -> SaSint {
    let mut i = n - 2;
    let mut m = n - 1;
    let mut f0 = 1usize;
    let mut f1: usize;
    let mut c0 = t[(n - 1) as usize] as isize;
    let mut c1: isize;

    while i >= 3 {
        c1 = t[i as usize] as isize;
        f1 = usize::from(c1 > c0 - f0 as isize);
        sa[m as usize] = i + 1;
        m -= (f1 & !f0) as SaSint;

        c0 = t[(i - 1) as usize] as isize;
        f0 = usize::from(c0 > c1 - f1 as isize);
        sa[m as usize] = i;
        m -= (f0 & !f1) as SaSint;

        c1 = t[(i - 2) as usize] as isize;
        f1 = usize::from(c1 > c0 - f0 as isize);
        sa[m as usize] = i - 1;
        m -= (f1 & !f0) as SaSint;

        c0 = t[(i - 3) as usize] as isize;
        f0 = usize::from(c0 > c1 - f1 as isize);
        sa[m as usize] = i - 2;
        m -= (f0 & !f1) as SaSint;

        i -= 4;
    }

    while i >= 0 {
        c1 = c0;
        c0 = t[i as usize] as isize;
        f1 = f0;
        f0 = usize::from(c0 > c1 - f1 as isize);
        sa[m as usize] = i + 1;
        m -= (f0 & !f1) as SaSint;
        i -= 1;
    }

    n - 1 - m
}

fn gather_compacted_lms_suffixes_32s(t: &[SaSint], sa: &mut [SaSint], n: SaSint) -> SaSint {
    let mut i = n - 2;
    let mut m = n - 1;
    let mut f0 = 1usize;
    let mut f1: usize;
    let mut c0 = t[(n - 1) as usize] as isize;
    let mut c1: isize;

    while i >= 3 {
        c1 = t[i as usize] as isize;
        f1 = usize::from(c1 > c0 - f0 as isize);
        sa[m as usize] = i + 1;
        m -= (f1 & !f0 & usize::from(c0 >= 0)) as SaSint;

        c0 = t[(i - 1) as usize] as isize;
        f0 = usize::from(c0 > c1 - f1 as isize);
        sa[m as usize] = i;
        m -= (f0 & !f1 & usize::from(c1 >= 0)) as SaSint;

        c1 = t[(i - 2) as usize] as isize;
        f1 = usize::from(c1 > c0 - f0 as isize);
        sa[m as usize] = i - 1;
        m -= (f1 & !f0 & usize::from(c0 >= 0)) as SaSint;

        c0 = t[(i - 3) as usize] as isize;
        f0 = usize::from(c0 > c1 - f1 as isize);
        sa[m as usize] = i - 2;
        m -= (f0 & !f1 & usize::from(c1 >= 0)) as SaSint;

        i -= 4;
    }

    while i >= 0 {
        c1 = c0;
        c0 = t[i as usize] as isize;
        f1 = f0;
        f0 = usize::from(c0 > c1 - f1 as isize);
        sa[m as usize] = i + 1;
        m -= (f0 & !f1 & usize::from(c1 >= 0)) as SaSint;
        i -= 1;
    }

    n - 1 - m
}

fn count_lms_suffixes_32s_4k(t: &[SaSint], n: SaSint, k: SaSint, buckets: &mut [SaSint]) {
    buckets[..4 * k as usize].fill(0);
    let mut i = n - 2;
    let mut f0 = 1usize;
    let mut f1: usize;
    let mut c0 = t[(n - 1) as usize] as isize;
    let mut c1: isize;

    while i >= 3 {
        c1 = t[i as usize] as isize;
        f1 = usize::from(c1 > c0 - f0 as isize);
        buckets[buckets_index4(c0 as usize, f0 + f0 + f1)] += 1;

        c0 = t[(i - 1) as usize] as isize;
        f0 = usize::from(c0 > c1 - f1 as isize);
        buckets[buckets_index4(c1 as usize, f1 + f1 + f0)] += 1;

        c1 = t[(i - 2) as usize] as isize;
        f1 = usize::from(c1 > c0 - f0 as isize);
        buckets[buckets_index4(c0 as usize, f0 + f0 + f1)] += 1;

        c0 = t[(i - 3) as usize] as isize;
        f0 = usize::from(c0 > c1 - f1 as isize);
        buckets[buckets_index4(c1 as usize, f1 + f1 + f0)] += 1;

        i -= 4;
    }

    while i >= 0 {
        c1 = c0;
        c0 = t[i as usize] as isize;
        f1 = f0;
        f0 = usize::from(c0 > c1 - f1 as isize);
        buckets[buckets_index4(c1 as usize, f1 + f1 + f0)] += 1;
        i -= 1;
    }

    buckets[buckets_index4(c0 as usize, f0 + f0)] += 1;
}

fn count_lms_suffixes_32s_2k(t: &[SaSint], n: SaSint, k: SaSint, buckets: &mut [SaSint]) {
    buckets[..2 * k as usize].fill(0);
    let mut i = n - 2;
    let mut f0 = 1usize;
    let mut f1: usize;
    let mut c0 = t[(n - 1) as usize] as isize;
    let mut c1: isize;

    while i >= 3 {
        c1 = t[i as usize] as isize;
        f1 = usize::from(c1 > c0 - f0 as isize);
        buckets[buckets_index2(c0 as usize, f1 & !f0)] += 1;

        c0 = t[(i - 1) as usize] as isize;
        f0 = usize::from(c0 > c1 - f1 as isize);
        buckets[buckets_index2(c1 as usize, f0 & !f1)] += 1;

        c1 = t[(i - 2) as usize] as isize;
        f1 = usize::from(c1 > c0 - f0 as isize);
        buckets[buckets_index2(c0 as usize, f1 & !f0)] += 1;

        c0 = t[(i - 3) as usize] as isize;
        f0 = usize::from(c0 > c1 - f1 as isize);
        buckets[buckets_index2(c1 as usize, f0 & !f1)] += 1;

        i -= 4;
    }

    while i >= 0 {
        c1 = c0;
        c0 = t[i as usize] as isize;
        f1 = f0;
        f0 = usize::from(c0 > c1 - f1 as isize);
        buckets[buckets_index2(c1 as usize, f0 & !f1)] += 1;
        i -= 1;
    }

    buckets[buckets_index2(c0 as usize, 0)] += 1;
}

fn count_compacted_lms_suffixes_32s_2k(t: &[SaSint], n: SaSint, k: SaSint, buckets: &mut [SaSint]) {
    buckets[..2 * k as usize].fill(0);
    let mut i = n - 2;
    let mut f0 = 1usize;
    let mut f1: usize;
    let mut c0 = t[(n - 1) as usize] as isize;
    let mut c1: isize;

    while i >= 3 {
        c1 = t[i as usize] as isize;
        f1 = usize::from(c1 > c0 - f0 as isize);
        buckets[buckets_index2((c0 as SaSint & SAINT_MAX) as usize, f1 & !f0)] += 1;

        c0 = t[(i - 1) as usize] as isize;
        f0 = usize::from(c0 > c1 - f1 as isize);
        buckets[buckets_index2((c1 as SaSint & SAINT_MAX) as usize, f0 & !f1)] += 1;

        c1 = t[(i - 2) as usize] as isize;
        f1 = usize::from(c1 > c0 - f0 as isize);
        buckets[buckets_index2((c0 as SaSint & SAINT_MAX) as usize, f1 & !f0)] += 1;

        c0 = t[(i - 3) as usize] as isize;
        f0 = usize::from(c0 > c1 - f1 as isize);
        buckets[buckets_index2((c1 as SaSint & SAINT_MAX) as usize, f0 & !f1)] += 1;

        i -= 4;
    }

    while i >= 0 {
        c1 = c0;
        c0 = t[i as usize] as isize;
        f1 = f0;
        f0 = usize::from(c0 > c1 - f1 as isize);
        buckets[buckets_index2((c1 as SaSint & SAINT_MAX) as usize, f0 & !f1)] += 1;
        i -= 1;
    }

    buckets[buckets_index2((c0 as SaSint & SAINT_MAX) as usize, 0)] += 1;
}

fn get_bucket_stride(free_space: SaSint, bucket_size: SaSint, num_buckets: SaSint) -> SaSint {
    let bucket_size_1024 = (bucket_size + 1023) & !1023;
    if free_space / (num_buckets - 1) >= bucket_size_1024 {
        return bucket_size_1024;
    }
    let bucket_size_16 = (bucket_size + 15) & !15;
    if free_space / (num_buckets - 1) >= bucket_size_16 {
        return bucket_size_16;
    }
    bucket_size
}

fn count_and_gather_lms_suffixes_32s_4k(
    t: &[SaSint],
    sa: &mut [SaSint],
    n: SaSint,
    k: SaSint,
    buckets: &mut [SaSint],
    omp_block_start: isize,
    omp_block_size: isize,
) -> SaSint {
    buckets[..4 * k as usize].fill(0);
    let mut m = omp_block_start + omp_block_size - 1;

    if omp_block_size > 0 {
        let mut j = m + 1;
        let mut c0 = t[m as usize] as isize;
        let mut c1 = -1isize;
        while j < n as isize {
            c1 = t[j as usize] as isize;
            if c1 != c0 {
                break;
            }
            j += 1;
        }

        let mut f0 = usize::from(c0 >= c1);
        let mut f1: usize;
        let mut i = m - 1;
        j = omp_block_start + 64 + 3;
        while i >= j {
            c1 = t[i as usize] as isize;
            f1 = usize::from(c1 > c0 - f0 as isize);
            sa[m as usize] = (i + 1) as SaSint;
            m -= (f1 & !f0) as isize;
            buckets[buckets_index4(c0 as usize, f0 + f0 + f1)] += 1;

            c0 = t[(i - 1) as usize] as isize;
            f0 = usize::from(c0 > c1 - f1 as isize);
            sa[m as usize] = i as SaSint;
            m -= (f0 & !f1) as isize;
            buckets[buckets_index4(c1 as usize, f1 + f1 + f0)] += 1;

            c1 = t[(i - 2) as usize] as isize;
            f1 = usize::from(c1 > c0 - f0 as isize);
            sa[m as usize] = (i - 1) as SaSint;
            m -= (f1 & !f0) as isize;
            buckets[buckets_index4(c0 as usize, f0 + f0 + f1)] += 1;

            c0 = t[(i - 3) as usize] as isize;
            f0 = usize::from(c0 > c1 - f1 as isize);
            sa[m as usize] = (i - 2) as SaSint;
            m -= (f0 & !f1) as isize;
            buckets[buckets_index4(c1 as usize, f1 + f1 + f0)] += 1;

            i -= 4;
        }

        j -= 64 + 3;
        while i >= j {
            c1 = c0;
            c0 = t[i as usize] as isize;
            f1 = f0;
            f0 = usize::from(c0 > c1 - f1 as isize);
            sa[m as usize] = (i + 1) as SaSint;
            m -= (f0 & !f1) as isize;
            buckets[buckets_index4(c1 as usize, f1 + f1 + f0)] += 1;
            i -= 1;
        }

        c1 = if i >= 0 { t[i as usize] as isize } else { -1 };
        f1 = usize::from(c1 > c0 - f0 as isize);
        sa[m as usize] = (i + 1) as SaSint;
        m -= (f1 & !f0) as isize;
        buckets[buckets_index4(c0 as usize, f0 + f0 + f1)] += 1;
    }

    (omp_block_start + omp_block_size - 1 - m) as SaSint
}

fn count_and_gather_lms_suffixes_32s_2k(
    t: &[SaSint],
    sa: &mut [SaSint],
    n: SaSint,
    k: SaSint,
    buckets: &mut [SaSint],
    omp_block_start: isize,
    omp_block_size: isize,
) -> SaSint {
    buckets[..2 * k as usize].fill(0);
    let mut m = omp_block_start + omp_block_size - 1;

    if omp_block_size > 0 {
        let mut j = m + 1;
        let mut c0 = t[m as usize] as isize;
        let mut c1 = -1isize;
        while j < n as isize {
            c1 = t[j as usize] as isize;
            if c1 != c0 {
                break;
            }
            j += 1;
        }

        let mut f0 = usize::from(c0 >= c1);
        let mut f1: usize;
        let mut i = m - 1;
        j = omp_block_start + 64 + 3;
        while i >= j {
            c1 = t[i as usize] as isize;
            f1 = usize::from(c1 > c0 - f0 as isize);
            sa[m as usize] = (i + 1) as SaSint;
            m -= (f1 & !f0) as isize;
            buckets[buckets_index2(c0 as usize, f1 & !f0)] += 1;

            c0 = t[(i - 1) as usize] as isize;
            f0 = usize::from(c0 > c1 - f1 as isize);
            sa[m as usize] = i as SaSint;
            m -= (f0 & !f1) as isize;
            buckets[buckets_index2(c1 as usize, f0 & !f1)] += 1;

            c1 = t[(i - 2) as usize] as isize;
            f1 = usize::from(c1 > c0 - f0 as isize);
            sa[m as usize] = (i - 1) as SaSint;
            m -= (f1 & !f0) as isize;
            buckets[buckets_index2(c0 as usize, f1 & !f0)] += 1;

            c0 = t[(i - 3) as usize] as isize;
            f0 = usize::from(c0 > c1 - f1 as isize);
            sa[m as usize] = (i - 2) as SaSint;
            m -= (f0 & !f1) as isize;
            buckets[buckets_index2(c1 as usize, f0 & !f1)] += 1;

            i -= 4;
        }

        j -= 64 + 3;
        while i >= j {
            c1 = c0;
            c0 = t[i as usize] as isize;
            f1 = f0;
            f0 = usize::from(c0 > c1 - f1 as isize);
            sa[m as usize] = (i + 1) as SaSint;
            m -= (f0 & !f1) as isize;
            buckets[buckets_index2(c1 as usize, f0 & !f1)] += 1;
            i -= 1;
        }

        c1 = if i >= 0 { t[i as usize] as isize } else { -1 };
        f1 = usize::from(c1 > c0 - f0 as isize);
        sa[m as usize] = (i + 1) as SaSint;
        m -= (f1 & !f0) as isize;
        buckets[buckets_index2(c0 as usize, f1 & !f0)] += 1;
    }

    (omp_block_start + omp_block_size - 1 - m) as SaSint
}

fn count_and_gather_compacted_lms_suffixes_32s_2k(
    t: &[SaSint],
    sa: &mut [SaSint],
    n: SaSint,
    k: SaSint,
    buckets: &mut [SaSint],
    omp_block_start: isize,
    omp_block_size: isize,
) -> SaSint {
    buckets[..2 * k as usize].fill(0);
    let mut m = omp_block_start + omp_block_size - 1;

    if omp_block_size > 0 {
        let mut j = m + 1;
        let mut c0 = t[m as usize] as isize;
        let mut c1 = -1isize;
        while j < n as isize {
            c1 = t[j as usize] as isize;
            if c1 != c0 {
                break;
            }
            j += 1;
        }

        let mut f0 = usize::from(c0 >= c1);
        let mut f1: usize;
        let mut i = m - 1;
        j = omp_block_start + 64 + 3;
        while i >= j {
            c1 = t[i as usize] as isize;
            f1 = usize::from(c1 > c0 - f0 as isize);
            sa[m as usize] = (i + 1) as SaSint;
            m -= (f1 & !f0 & usize::from(c0 >= 0)) as isize;
            buckets[buckets_index2((c0 as SaSint & SAINT_MAX) as usize, f1 & !f0)] += 1;

            c0 = t[(i - 1) as usize] as isize;
            f0 = usize::from(c0 > c1 - f1 as isize);
            sa[m as usize] = i as SaSint;
            m -= (f0 & !f1 & usize::from(c1 >= 0)) as isize;
            buckets[buckets_index2((c1 as SaSint & SAINT_MAX) as usize, f0 & !f1)] += 1;

            c1 = t[(i - 2) as usize] as isize;
            f1 = usize::from(c1 > c0 - f0 as isize);
            sa[m as usize] = (i - 1) as SaSint;
            m -= (f1 & !f0 & usize::from(c0 >= 0)) as isize;
            buckets[buckets_index2((c0 as SaSint & SAINT_MAX) as usize, f1 & !f0)] += 1;

            c0 = t[(i - 3) as usize] as isize;
            f0 = usize::from(c0 > c1 - f1 as isize);
            sa[m as usize] = (i - 2) as SaSint;
            m -= (f0 & !f1 & usize::from(c1 >= 0)) as isize;
            buckets[buckets_index2((c1 as SaSint & SAINT_MAX) as usize, f0 & !f1)] += 1;

            i -= 4;
        }

        j -= 64 + 3;
        while i >= j {
            c1 = c0;
            c0 = t[i as usize] as isize;
            f1 = f0;
            f0 = usize::from(c0 > c1 - f1 as isize);
            sa[m as usize] = (i + 1) as SaSint;
            m -= (f0 & !f1 & usize::from(c1 >= 0)) as isize;
            buckets[buckets_index2((c1 as SaSint & SAINT_MAX) as usize, f0 & !f1)] += 1;
            i -= 1;
        }

        c1 = if i >= 0 { t[i as usize] as isize } else { -1 };
        f1 = usize::from(c1 > c0 - f0 as isize);
        sa[m as usize] = (i + 1) as SaSint;
        m -= (f1 & !f0 & usize::from(c0 >= 0)) as isize;
        buckets[buckets_index2((c0 as SaSint & SAINT_MAX) as usize, f1 & !f0)] += 1;
    }

    (omp_block_start + omp_block_size - 1 - m) as SaSint
}

fn count_and_gather_lms_suffixes_32s_4k_fs_omp(
    t: &[SaSint],
    sa: &mut [SaSint],
    n: SaSint,
    k: SaSint,
    buckets: &mut [SaSint],
    local_buckets: SaSint,
    threads: SaSint,
    thread_state: &mut [ThreadState],
) -> SaSint {
    if threads == 1 || n < 65_536 {
        return count_and_gather_lms_suffixes_32s_4k(t, sa, n, k, buckets, 0, n as isize);
    }

    let thread_count = threads as usize;
    let n_usize = n as usize;
    let bucket_size = 4 * k as usize;
    let block_stride = (n / threads) & !15;
    let free_space = if local_buckets != 0 {
        LIBSAIS_LOCAL_BUFFER_SIZE as SaSint
    } else {
        buckets.len() as SaSint
    };
    let bucket_stride = get_bucket_stride(free_space, 4 * k, threads) as usize;
    let workspace_len = bucket_size + bucket_stride * thread_count.saturating_sub(1);
    let mut workspace = vec![0; workspace_len];

    for (thread, state) in thread_state.iter_mut().take(thread_count).enumerate() {
        let block_start = thread as SaSint * block_stride;
        let block_size = if thread + 1 < thread_count {
            block_stride
        } else {
            n - block_start
        };
        let workspace_end = workspace_len - thread * bucket_stride;
        let workspace_start = workspace_end - bucket_size;
        state.count = count_and_gather_lms_suffixes_32s_4k(
            t,
            sa,
            n,
            k,
            &mut workspace[workspace_start..workspace_end],
            block_start as isize,
            block_size as isize,
        );
        state.position = block_start + block_size;
    }

    let mut m = 0usize;
    for thread in (0..thread_count).rev() {
        let count =
            usize::try_from(thread_state[thread].count).expect("count must be non-negative");
        m += count;
        if thread + 1 != thread_count && count > 0 {
            let src_end = usize::try_from(thread_state[thread].position)
                .expect("position must be non-negative");
            let src_start = src_end - count;
            let dst_start = n_usize - m;
            sa.copy_within(src_start..src_end, dst_start);
        }
    }

    let accumulation_threads = thread_count - 1;
    let block_stride = (bucket_size / accumulation_threads) & !15usize;
    for thread in 0..accumulation_threads {
        let block_start = thread * block_stride;
        let block_size = if thread + 1 < accumulation_threads {
            block_stride
        } else {
            bucket_size - block_start
        };
        accumulate_counts_s32(
            &mut workspace,
            block_start,
            block_size,
            bucket_stride,
            accumulation_threads + 1,
        );
    }

    buckets[..bucket_size].copy_from_slice(&workspace[..bucket_size]);
    m as SaSint
}

fn count_and_gather_lms_suffixes_32s_2k_fs_omp(
    t: &[SaSint],
    sa: &mut [SaSint],
    n: SaSint,
    k: SaSint,
    buckets: &mut [SaSint],
    local_buckets: SaSint,
    threads: SaSint,
    thread_state: &mut [ThreadState],
) -> SaSint {
    if threads == 1 || n < 65_536 {
        return count_and_gather_lms_suffixes_32s_2k(t, sa, n, k, buckets, 0, n as isize);
    }

    let thread_count = threads as usize;
    let n_usize = n as usize;
    let bucket_size = 2 * k as usize;
    let block_stride = (n / threads) & !15;
    let free_space = if local_buckets != 0 {
        LIBSAIS_LOCAL_BUFFER_SIZE as SaSint
    } else {
        buckets.len() as SaSint
    };
    let bucket_stride = get_bucket_stride(free_space, 2 * k, threads) as usize;
    let workspace_len = bucket_size + bucket_stride * thread_count.saturating_sub(1);
    let mut workspace = vec![0; workspace_len];

    for (thread, state) in thread_state.iter_mut().take(thread_count).enumerate() {
        let block_start = thread as SaSint * block_stride;
        let block_size = if thread + 1 < thread_count {
            block_stride
        } else {
            n - block_start
        };
        let workspace_end = workspace_len - thread * bucket_stride;
        let workspace_start = workspace_end - bucket_size;
        state.count = count_and_gather_lms_suffixes_32s_2k(
            t,
            sa,
            n,
            k,
            &mut workspace[workspace_start..workspace_end],
            block_start as isize,
            block_size as isize,
        );
        state.position = block_start + block_size;
    }

    let mut m = 0usize;
    for thread in (0..thread_count).rev() {
        let count =
            usize::try_from(thread_state[thread].count).expect("count must be non-negative");
        m += count;
        if thread + 1 != thread_count && count > 0 {
            let src_end = usize::try_from(thread_state[thread].position)
                .expect("position must be non-negative");
            let src_start = src_end - count;
            let dst_start = n_usize - m;
            sa.copy_within(src_start..src_end, dst_start);
        }
    }

    let accumulation_threads = thread_count - 1;
    let block_stride = (bucket_size / accumulation_threads) & !15usize;
    for thread in 0..accumulation_threads {
        let block_start = thread * block_stride;
        let block_size = if thread + 1 < accumulation_threads {
            block_stride
        } else {
            bucket_size - block_start
        };
        accumulate_counts_s32(
            &mut workspace,
            block_start,
            block_size,
            bucket_stride,
            accumulation_threads + 1,
        );
    }

    buckets[..bucket_size].copy_from_slice(&workspace[..bucket_size]);
    m as SaSint
}

fn count_and_gather_compacted_lms_suffixes_32s_2k_fs_omp(
    t: &[SaSint],
    sa: &mut [SaSint],
    n: SaSint,
    k: SaSint,
    buckets: &mut [SaSint],
    _local_buckets: SaSint,
    threads: SaSint,
    thread_state: &mut [ThreadState],
) {
    if threads == 1 || n < 65_536 {
        count_and_gather_compacted_lms_suffixes_32s_2k(t, sa, n, k, buckets, 0, n as isize);
        return;
    }

    let thread_count = threads as usize;
    let n_usize = n as usize;
    let bucket_size = 2 * k as usize;
    let block_stride = (n / threads) & !15;
    let mut workspaces = vec![vec![0; bucket_size]; thread_count];
    let mut gathered_runs = vec![Vec::<SaSint>::new(); thread_count];

    for (thread, state) in thread_state.iter_mut().take(thread_count).enumerate() {
        let block_start = thread as SaSint * block_stride;
        let block_size = if thread + 1 < thread_count {
            block_stride
        } else {
            n - block_start
        };
        let mut temp_sa = vec![0; n_usize + block_size as usize];
        state.count = count_and_gather_compacted_lms_suffixes_32s_2k(
            t,
            &mut temp_sa,
            n,
            k,
            &mut workspaces[thread],
            block_start as isize,
            block_size as isize,
        );
        state.position = block_start + block_size;
        let count = usize::try_from(state.count).expect("count must be non-negative");
        let src_end =
            n_usize + usize::try_from(state.position).expect("position must be non-negative");
        let src_start = src_end - count;
        gathered_runs[thread].extend_from_slice(&temp_sa[src_start..src_end]);
    }

    let mut suffixes_before = 0usize;
    for thread in (0..thread_count).rev() {
        let count =
            usize::try_from(thread_state[thread].count).expect("count must be non-negative");
        suffixes_before += count;
        if count > 0 {
            let dst_start = n_usize - suffixes_before;
            let dst_end = dst_start + count;
            sa[dst_start..dst_end].copy_from_slice(&gathered_runs[thread]);
        }
    }

    buckets.fill(0);
    for workspace in &workspaces {
        for (dst, src) in buckets.iter_mut().zip(workspace.iter()) {
            *dst += *src;
        }
    }
}

fn count_and_gather_lms_suffixes_32s_4k_nofs_omp(
    t: &[SaSint],
    sa: &mut [SaSint],
    n: SaSint,
    k: SaSint,
    buckets: &mut [SaSint],
    threads: SaSint,
) -> SaSint {
    if threads > 1 && n >= 65_536 {
        count_lms_suffixes_32s_4k(t, n, k, buckets);
        gather_lms_suffixes_32s(t, sa, n)
    } else {
        count_and_gather_lms_suffixes_32s_4k(t, sa, n, k, buckets, 0, n as isize)
    }
}

fn count_and_gather_lms_suffixes_32s_2k_nofs_omp(
    t: &[SaSint],
    sa: &mut [SaSint],
    n: SaSint,
    k: SaSint,
    buckets: &mut [SaSint],
    threads: SaSint,
) -> SaSint {
    if threads > 1 && n >= 65_536 {
        count_lms_suffixes_32s_2k(t, n, k, buckets);
        gather_lms_suffixes_32s(t, sa, n)
    } else {
        count_and_gather_lms_suffixes_32s_2k(t, sa, n, k, buckets, 0, n as isize)
    }
}

fn count_and_gather_compacted_lms_suffixes_32s_2k_nofs_omp(
    t: &[SaSint],
    sa: &mut [SaSint],
    n: SaSint,
    k: SaSint,
    buckets: &mut [SaSint],
    threads: SaSint,
) -> SaSint {
    if threads > 1 && n >= 65_536 {
        count_compacted_lms_suffixes_32s_2k(t, n, k, buckets);
        gather_compacted_lms_suffixes_32s(t, sa, n)
    } else {
        count_and_gather_compacted_lms_suffixes_32s_2k(t, sa, n, k, buckets, 0, n as isize)
    }
}

fn count_and_gather_lms_suffixes_32s_4k_omp(
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
        LIBSAIS_LOCAL_BUFFER_SIZE as SaSint
    } else {
        buckets.len() as SaSint
    };
    let mut max_threads = (free_space / (((4 * k) + 15) & !15)).min(threads);

    if max_threads > 1 && n >= 65_536 && n / k >= 2 {
        let thread_cap = n / (16 * k);
        if max_threads > thread_cap {
            max_threads = thread_cap;
        }
        count_and_gather_lms_suffixes_32s_4k_fs_omp(
            t,
            sa,
            n,
            k,
            buckets,
            local_buckets,
            max_threads.max(2),
            thread_state,
        )
    } else if threads > 1 && n >= 65_536 {
        count_lms_suffixes_32s_4k(t, n, k, buckets);
        gather_lms_suffixes_32s(t, sa, n)
    } else {
        count_and_gather_lms_suffixes_32s_4k_nofs_omp(t, sa, n, k, buckets, threads)
    }
}

fn count_and_gather_lms_suffixes_32s_2k_omp(
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
        LIBSAIS_LOCAL_BUFFER_SIZE as SaSint
    } else {
        buckets.len() as SaSint
    };
    let mut max_threads = (free_space / (((2 * k) + 15) & !15)).min(threads);

    if max_threads > 1 && n >= 65_536 && n / k >= 2 {
        let thread_cap = n / (8 * k);
        if max_threads > thread_cap {
            max_threads = thread_cap;
        }
        count_and_gather_lms_suffixes_32s_2k_fs_omp(
            t,
            sa,
            n,
            k,
            buckets,
            local_buckets,
            max_threads.max(2),
            thread_state,
        )
    } else if threads > 1 && n >= 65_536 {
        count_lms_suffixes_32s_2k(t, n, k, buckets);
        gather_lms_suffixes_32s(t, sa, n)
    } else {
        count_and_gather_lms_suffixes_32s_2k_nofs_omp(t, sa, n, k, buckets, threads)
    }
}

fn count_suffixes_32s(t: &[SaSint], n: SaSint, k: SaSint, buckets: &mut [SaSint]) {
    buckets[..k as usize].fill(0);

    let mut i = 0usize;
    let mut j = (n as usize).saturating_sub(7);
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

fn initialize_buckets_start_and_end_32s_6k(k: SaSint, buckets: &mut [SaSint]) {
    let k = k as usize;
    let mut sum = 0;
    for j in 0..k {
        let i = buckets_index4(j, 0);
        buckets[4 * k + j] = sum;
        sum += buckets[i] + buckets[i + 1] + buckets[i + 2] + buckets[i + 3];
        buckets[5 * k + j] = sum;
    }
}

fn initialize_buckets_start_and_end_32s_4k(k: SaSint, buckets: &mut [SaSint]) {
    let k = k as usize;
    let mut sum = 0;
    for j in 0..k {
        let i = buckets_index2(j, 0);
        buckets[2 * k + j] = sum;
        sum += buckets[i] + buckets[i + 1];
        buckets[3 * k + j] = sum;
    }
}

fn initialize_buckets_end_32s_2k(k: SaSint, buckets: &mut [SaSint]) {
    let mut sum0 = 0;
    for j in 0..k as usize {
        let i = buckets_index2(j, 0);
        sum0 += buckets[i] + buckets[i + 1];
        buckets[i] = sum0;
    }
}

fn initialize_buckets_start_and_end_32s_2k(k: SaSint, buckets: &mut [SaSint]) {
    let k = k as usize;
    for j in 0..k {
        let i = buckets_index2(j, 0);
        buckets[j] = buckets[i];
    }
    buckets[k] = 0;
    buckets.copy_within(0..k - 1, k + 1);
}

fn initialize_buckets_start_32s_1k(k: SaSint, buckets: &mut [SaSint]) {
    let mut sum = 0;
    for bucket in buckets.iter_mut().take(k as usize) {
        let tmp = *bucket;
        *bucket = sum;
        sum += tmp;
    }
}

fn initialize_buckets_end_32s_1k(k: SaSint, buckets: &mut [SaSint]) {
    let mut sum = 0;
    for bucket in buckets.iter_mut().take(k as usize) {
        sum += *bucket;
        *bucket = sum;
    }
}

fn initialize_buckets_for_lms_suffixes_radix_sort_32s_2k(
    t: &[SaSint],
    k: SaSint,
    buckets: &mut [SaSint],
    first_lms_suffix: SaSint,
) {
    buckets[buckets_index2(t[first_lms_suffix as usize] as usize, 0)] += 1;
    buckets[buckets_index2(t[first_lms_suffix as usize] as usize, 1)] -= 1;

    let mut sum0 = 0;
    let mut sum1 = 0;
    for j in 0..k as usize {
        let i = buckets_index2(j, 0);
        sum0 += buckets[i] + buckets[i + 1];
        sum1 += buckets[i + 1];
        buckets[i] = sum0;
        buckets[i + 1] = sum1;
    }
}

fn initialize_buckets_for_lms_suffixes_radix_sort_32s_6k(
    t: &[SaSint],
    k: SaSint,
    buckets: &mut [SaSint],
    mut first_lms_suffix: SaSint,
) -> SaSint {
    let mut f0 = 0usize;
    let mut c0 = t[first_lms_suffix as usize] as isize;

    loop {
        first_lms_suffix -= 1;
        if first_lms_suffix < 0 {
            break;
        }
        let c1 = c0;
        c0 = t[first_lms_suffix as usize] as isize;
        let f1 = f0;
        f0 = usize::from(c0 > c1 - f1 as isize);
        buckets[buckets_index4(c1 as usize, f1 + f1 + f0)] -= 1;
    }
    buckets[buckets_index4(c0 as usize, f0 + f0)] -= 1;

    let mut sum = 0;
    for j in 0..k as usize {
        let i = buckets_index4(j, 0);
        sum += buckets[i + 1] + buckets[i + 3];
        buckets[4 * k as usize + j] = sum;
    }
    sum
}

fn initialize_buckets_for_partial_sorting_32s_6k(
    t: &[SaSint],
    k: SaSint,
    buckets: &mut [SaSint],
    first_lms_suffix: SaSint,
    left_suffixes_count: SaSint,
) {
    let k = k as usize;
    let temp_offset = 4 * k;
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
    for j in first_symbol..k {
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

fn initialize_buckets_for_radix_and_partial_sorting_32s_4k(
    t: &[SaSint],
    k: SaSint,
    buckets: &mut [SaSint],
    first_lms_suffix: SaSint,
) {
    let k = k as usize;
    buckets[buckets_index2(t[first_lms_suffix as usize] as usize, 0)] += 1;
    buckets[buckets_index2(t[first_lms_suffix as usize] as usize, 1)] -= 1;

    let mut sum0 = 0;
    let mut sum1 = 0;
    for j in 0..k {
        let i = buckets_index2(j, 0);
        buckets[2 * k + j] = sum1;
        sum0 += buckets[i + 1];
        sum1 += buckets[i] + buckets[i + 1];
        buckets[i + 1] = sum0;
        buckets[3 * k + j] = sum1;
    }
}

fn count_and_gather_compacted_lms_suffixes_32s_2k_omp(
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
        LIBSAIS_LOCAL_BUFFER_SIZE as SaSint
    } else {
        buckets.len() as SaSint
    };
    let mut max_threads = (free_space / (((2 * k) + 15) & !15)).min(threads);

    if local_buckets == 0 && max_threads > 1 && n >= 65_536 && n / k >= 2 {
        let thread_cap = n / (8 * k);
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
            max_threads.max(2),
            thread_state,
        );
    } else {
        count_and_gather_compacted_lms_suffixes_32s_2k_nofs_omp(t, sa, n, k, buckets, threads);
    }
}

fn gather_lms_suffixes_16u(
    t: &[u16],
    sa: &mut [SaSint],
    n: SaSint,
    mut m: SaSint,
    omp_block_start: SaSint,
    omp_block_size: SaSint,
) {
    if omp_block_size > 0 {
        let n = n as isize;
        let mut i: isize;
        let mut j = (omp_block_start + omp_block_size) as isize;
        let mut c0 = t[(omp_block_start + omp_block_size - 1) as usize] as isize;
        let mut c1 = -1isize;

        while j < n {
            c1 = t[j as usize] as isize;
            if c1 != c0 {
                break;
            }
            j += 1;
        }

        let mut f0 = usize::from(c0 >= c1);
        let mut f1: usize;

        i = (omp_block_start + omp_block_size - 2) as isize;
        j = (omp_block_start + 3) as isize;
        while i >= j {
            c1 = t[i as usize] as isize;
            f1 = usize::from(c1 > c0 - f0 as isize);
            sa[m as usize] = (i + 1) as SaSint;
            m -= (f1 & (1 - f0)) as SaSint;

            c0 = t[(i - 1) as usize] as isize;
            f0 = usize::from(c0 > c1 - f1 as isize);
            sa[m as usize] = i as SaSint;
            m -= (f0 & (1 - f1)) as SaSint;

            c1 = t[(i - 2) as usize] as isize;
            f1 = usize::from(c1 > c0 - f0 as isize);
            sa[m as usize] = (i - 1) as SaSint;
            m -= (f1 & (1 - f0)) as SaSint;

            c0 = t[(i - 3) as usize] as isize;
            f0 = usize::from(c0 > c1 - f1 as isize);
            sa[m as usize] = (i - 2) as SaSint;
            m -= (f0 & (1 - f1)) as SaSint;

            i -= 4;
        }

        j -= 3;
        while i >= j {
            c1 = c0;
            c0 = t[i as usize] as isize;
            f1 = f0;
            f0 = usize::from(c0 > c1 - f1 as isize);
            sa[m as usize] = (i + 1) as SaSint;
            m -= (f0 & (1 - f1)) as SaSint;
            i -= 1;
        }

        sa[m as usize] = (i + 1) as SaSint;
    }
}

fn count_and_gather_lms_suffixes_16u(
    t: &[u16],
    sa: &mut [SaSint],
    n: SaSint,
    buckets: &mut [SaSint],
    omp_block_start: SaSint,
    omp_block_size: SaSint,
) -> SaSint {
    buckets[..4 * ALPHABET_SIZE].fill(0);

    let mut m = (omp_block_start + omp_block_size - 1) as isize;

    if omp_block_size > 0 {
        let n = n as isize;
        let mut i: isize;
        let mut j = m + 1;
        let mut c0 = t[m as usize] as isize;
        let mut c1 = -1isize;

        while j < n {
            c1 = t[j as usize] as isize;
            if c1 != c0 {
                break;
            }
            j += 1;
        }

        let mut f0 = usize::from(c0 >= c1);
        let mut f1: usize;

        i = m - 1;
        j = (omp_block_start + 3) as isize;
        while i >= j {
            c1 = t[i as usize] as isize;
            f1 = usize::from(c1 > c0 - f0 as isize);
            sa[m as usize] = (i + 1) as SaSint;
            m -= (f1 & (1 - f0)) as isize;
            buckets[buckets_index4(c0 as usize, f0 + f0 + f1)] += 1;

            c0 = t[(i - 1) as usize] as isize;
            f0 = usize::from(c0 > c1 - f1 as isize);
            sa[m as usize] = i as SaSint;
            m -= (f0 & (1 - f1)) as isize;
            buckets[buckets_index4(c1 as usize, f1 + f1 + f0)] += 1;

            c1 = t[(i - 2) as usize] as isize;
            f1 = usize::from(c1 > c0 - f0 as isize);
            sa[m as usize] = (i - 1) as SaSint;
            m -= (f1 & (1 - f0)) as isize;
            buckets[buckets_index4(c0 as usize, f0 + f0 + f1)] += 1;

            c0 = t[(i - 3) as usize] as isize;
            f0 = usize::from(c0 > c1 - f1 as isize);
            sa[m as usize] = (i - 2) as SaSint;
            m -= (f0 & (1 - f1)) as isize;
            buckets[buckets_index4(c1 as usize, f1 + f1 + f0)] += 1;

            i -= 4;
        }

        j -= 3;
        while i >= j {
            c1 = c0;
            c0 = t[i as usize] as isize;
            f1 = f0;
            f0 = usize::from(c0 > c1 - f1 as isize);
            sa[m as usize] = (i + 1) as SaSint;
            m -= (f0 & (1 - f1)) as isize;
            buckets[buckets_index4(c1 as usize, f1 + f1 + f0)] += 1;
            i -= 1;
        }

        c1 = if i >= 0 { t[i as usize] as isize } else { -1 };
        f1 = usize::from(c1 > c0 - f0 as isize);
        sa[m as usize] = (i + 1) as SaSint;
        m -= (f1 & (1 - f0)) as isize;
        buckets[buckets_index4(c0 as usize, f0 + f0 + f1)] += 1;
    }

    omp_block_start + omp_block_size - 1 - m as SaSint
}

fn gather_lms_suffixes_16u_omp(
    t: &[u16],
    sa: &mut [SaSint],
    n: SaSint,
    threads: SaSint,
    thread_state: &mut [ThreadState],
) {
    if threads == 1 || n < 65_536 || thread_state.is_empty() {
        gather_lms_suffixes_16u(t, sa, n, n - 1, 0, n);
        return;
    }

    let thread_count = threads as usize;
    let block_stride = (n / threads) & !15;
    let mut suffix_counts_after = vec![0; thread_count];
    let mut m = 0;
    for thread in (0..thread_count).rev() {
        suffix_counts_after[thread] = m;
        m += thread_state[thread].m;
    }

    for thread in 0..thread_count {
        let block_start = thread as SaSint * block_stride;
        let block_size = if thread + 1 < thread_count {
            block_stride
        } else {
            n - block_start
        };
        gather_lms_suffixes_16u(
            t,
            sa,
            n,
            n - 1 - suffix_counts_after[thread],
            block_start,
            block_size,
        );
    }

    for thread in 0..thread_count {
        if thread_state[thread].m > 0 {
            sa[(n - 1 - suffix_counts_after[thread]) as usize] =
                thread_state[thread].last_lms_suffix;
        }
    }
}

fn count_and_gather_lms_suffixes_16u_omp(
    t: &[u16],
    sa: &mut [SaSint],
    n: SaSint,
    buckets: &mut [SaSint],
    threads: SaSint,
    thread_state: &mut [ThreadState],
) -> SaSint {
    if threads == 1 || n < 65_536 || thread_state.is_empty() {
        return count_and_gather_lms_suffixes_16u(t, sa, n, buckets, 0, n);
    }

    let thread_count = threads as usize;
    let block_stride = (n / threads) & !15;

    for thread in 0..thread_count {
        let block_start = thread as SaSint * block_stride;
        let block_size = if thread + 1 < thread_count {
            block_stride
        } else {
            n - block_start
        };
        let count = count_and_gather_lms_suffixes_16u(
            t,
            sa,
            n,
            &mut thread_state[thread].buckets,
            block_start,
            block_size,
        );
        thread_state[thread].m = count;
        thread_state[thread].position = block_start + block_size;
        if count > 0 {
            thread_state[thread].last_lms_suffix = sa[(block_start + block_size - 1) as usize];
        }
    }

    buckets[..4 * ALPHABET_SIZE].fill(0);
    let mut m = 0;
    for thread in (0..thread_count).rev() {
        let position = thread_state[thread].position;
        let count = thread_state[thread].m;
        m += count;
        if thread + 1 != thread_count && count > 0 {
            let src_end = position as usize;
            let src_start = src_end - count as usize;
            let dst_start = (n - m) as usize;
            sa.copy_within(src_start..src_end, dst_start);
        }
        for s in 0..4 * ALPHABET_SIZE {
            let a = buckets[s];
            let b = thread_state[thread].buckets[s];
            buckets[s] = a + b;
            thread_state[thread].buckets[s] = a;
        }
    }

    m
}

fn initialize_buckets_start_and_end_16u(
    buckets: &mut [SaSint],
    freq: Option<&mut [SaSint]>,
) -> SaSint {
    let (count_buckets, start_end) = buckets.split_at_mut(6 * ALPHABET_SIZE);
    let (bucket_start, bucket_end) = start_end.split_at_mut(ALPHABET_SIZE);

    let mut k = -1;
    let mut sum = 0;

    if let Some(freq) = freq {
        for j in 0..ALPHABET_SIZE {
            let i = buckets_index4(j, 0);
            let total = count_buckets[i]
                + count_buckets[i + buckets_index4(0, 1)]
                + count_buckets[i + buckets_index4(0, 2)]
                + count_buckets[i + buckets_index4(0, 3)];

            bucket_start[j] = sum;
            sum += total;
            bucket_end[j] = sum;
            if total > 0 {
                k = j as SaSint;
            }
            freq[j] = total;
        }
    } else {
        for j in 0..ALPHABET_SIZE {
            let i = buckets_index4(j, 0);
            let total = count_buckets[i]
                + count_buckets[i + buckets_index4(0, 1)]
                + count_buckets[i + buckets_index4(0, 2)]
                + count_buckets[i + buckets_index4(0, 3)];

            bucket_start[j] = sum;
            sum += total;
            bucket_end[j] = sum;
            if total > 0 {
                k = j as SaSint;
            }
        }
    }

    k + 1
}

fn initialize_buckets_for_lms_suffixes_radix_sort_16u(
    t: &[u16],
    buckets: &mut [SaSint],
    mut first_lms_suffix: SaSint,
) -> SaSint {
    let mut f0 = 0usize;
    let mut c0 = t[first_lms_suffix as usize] as isize;

    loop {
        first_lms_suffix -= 1;
        if first_lms_suffix < 0 {
            break;
        }

        let c1 = c0;
        c0 = t[first_lms_suffix as usize] as isize;
        let f1 = f0;
        f0 = usize::from(c0 > c1 - f1 as isize);
        buckets[buckets_index4(c1 as usize, f1 + f1 + f0)] -= 1;
    }

    buckets[buckets_index4(c0 as usize, f0 + f0)] -= 1;

    let (count_buckets, temp_bucket) = buckets.split_at_mut(4 * ALPHABET_SIZE);
    let mut sum = 0;
    for c in 0..ALPHABET_SIZE {
        let i = buckets_index4(c, 0);
        let j = buckets_index2(c, 0);
        temp_bucket[j + buckets_index2(0, 1)] = sum;
        sum += count_buckets[i + buckets_index4(0, 1)] + count_buckets[i + buckets_index4(0, 3)];
        temp_bucket[j] = sum;
    }

    sum
}

fn radix_sort_lms_suffixes_16u(
    t: &[u16],
    sa: &mut [SaSint],
    induction_bucket: &mut [SaSint],
    omp_block_start: SaSint,
    omp_block_size: SaSint,
) {
    let mut i = omp_block_start + omp_block_size - 1;
    let mut j = omp_block_start + 64 + 3;
    while i >= j {
        let p0 = sa[i as usize];
        induction_bucket[buckets_index2(t[p0 as usize] as usize, 0)] -= 1;
        sa[induction_bucket[buckets_index2(t[p0 as usize] as usize, 0)] as usize] = p0;

        let p1 = sa[(i - 1) as usize];
        induction_bucket[buckets_index2(t[p1 as usize] as usize, 0)] -= 1;
        sa[induction_bucket[buckets_index2(t[p1 as usize] as usize, 0)] as usize] = p1;

        let p2 = sa[(i - 2) as usize];
        induction_bucket[buckets_index2(t[p2 as usize] as usize, 0)] -= 1;
        sa[induction_bucket[buckets_index2(t[p2 as usize] as usize, 0)] as usize] = p2;

        let p3 = sa[(i - 3) as usize];
        induction_bucket[buckets_index2(t[p3 as usize] as usize, 0)] -= 1;
        sa[induction_bucket[buckets_index2(t[p3 as usize] as usize, 0)] as usize] = p3;

        i -= 4;
    }

    j -= 64 + 3;
    while i >= j {
        let p = sa[i as usize];
        induction_bucket[buckets_index2(t[p as usize] as usize, 0)] -= 1;
        sa[induction_bucket[buckets_index2(t[p as usize] as usize, 0)] as usize] = p;
        i -= 1;
    }
}

fn radix_sort_lms_suffixes_16u_omp(
    t: &[u16],
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
    if threads == 1 || n < 65_536 || m < 65_536 || thread_state.is_empty() {
        radix_sort_lms_suffixes_16u(t, sa, &mut buckets[4 * ALPHABET_SIZE..], n - m + 1, m - 1);
        return;
    }

    let thread_count = threads as usize;
    for thread in 0..thread_count {
        let (src_buckets, state_buckets) = (
            &buckets[4 * ALPHABET_SIZE..],
            &mut thread_state[thread].buckets,
        );
        for c in 0..ALPHABET_SIZE {
            let i = buckets_index2(c, 0);
            let j = buckets_index4(c, 1);
            state_buckets[i] = src_buckets[i] - state_buckets[j];
        }

        let mut block_start = 0;
        let mut block_size = thread_state[thread].m;
        for idx in (thread..thread_count).rev() {
            block_start += thread_state[idx].m;
        }

        if block_start == m && block_size > 0 {
            block_start -= 1;
            block_size -= 1;
        }

        radix_sort_lms_suffixes_16u(
            t,
            sa,
            &mut thread_state[thread].buckets,
            n - block_start,
            block_size,
        );
    }
}

fn radix_sort_lms_suffixes_32s_6k(
    t: &[SaSint],
    sa: &mut [SaSint],
    induction_bucket: &mut [SaSint],
    omp_block_start: SaSint,
    omp_block_size: SaSint,
) {
    let mut i = omp_block_start + omp_block_size - 1;
    let mut j = omp_block_start + 64 + 3;
    while i >= j {
        let p0 = sa[i as usize];
        induction_bucket[t[p0 as usize] as usize] -= 1;
        sa[induction_bucket[t[p0 as usize] as usize] as usize] = p0;
        let p1 = sa[(i - 1) as usize];
        induction_bucket[t[p1 as usize] as usize] -= 1;
        sa[induction_bucket[t[p1 as usize] as usize] as usize] = p1;
        let p2 = sa[(i - 2) as usize];
        induction_bucket[t[p2 as usize] as usize] -= 1;
        sa[induction_bucket[t[p2 as usize] as usize] as usize] = p2;
        let p3 = sa[(i - 3) as usize];
        induction_bucket[t[p3 as usize] as usize] -= 1;
        sa[induction_bucket[t[p3 as usize] as usize] as usize] = p3;
        i -= 4;
    }

    j -= 64 + 3;
    while i >= j {
        let p = sa[i as usize];
        induction_bucket[t[p as usize] as usize] -= 1;
        sa[induction_bucket[t[p as usize] as usize] as usize] = p;
        i -= 1;
    }
}

fn radix_sort_lms_suffixes_32s_2k(
    t: &[SaSint],
    sa: &mut [SaSint],
    induction_bucket: &mut [SaSint],
    omp_block_start: SaSint,
    omp_block_size: SaSint,
) {
    let mut i = omp_block_start + omp_block_size - 1;
    let mut j = omp_block_start + 64 + 3;
    while i >= j {
        let p0 = sa[i as usize];
        induction_bucket[buckets_index2(t[p0 as usize] as usize, 0)] -= 1;
        sa[induction_bucket[buckets_index2(t[p0 as usize] as usize, 0)] as usize] = p0;
        let p1 = sa[(i - 1) as usize];
        induction_bucket[buckets_index2(t[p1 as usize] as usize, 0)] -= 1;
        sa[induction_bucket[buckets_index2(t[p1 as usize] as usize, 0)] as usize] = p1;
        let p2 = sa[(i - 2) as usize];
        induction_bucket[buckets_index2(t[p2 as usize] as usize, 0)] -= 1;
        sa[induction_bucket[buckets_index2(t[p2 as usize] as usize, 0)] as usize] = p2;
        let p3 = sa[(i - 3) as usize];
        induction_bucket[buckets_index2(t[p3 as usize] as usize, 0)] -= 1;
        sa[induction_bucket[buckets_index2(t[p3 as usize] as usize, 0)] as usize] = p3;
        i -= 4;
    }

    j -= 64 + 3;
    while i >= j {
        let p = sa[i as usize];
        induction_bucket[buckets_index2(t[p as usize] as usize, 0)] -= 1;
        sa[induction_bucket[buckets_index2(t[p as usize] as usize, 0)] as usize] = p;
        i -= 1;
    }
}

fn radix_sort_lms_suffixes_32s_block_gather(
    t: &[SaSint],
    sa: &[SaSint],
    cache: &mut [ThreadCache],
    omp_block_start: SaSint,
    omp_block_size: SaSint,
) {
    if omp_block_size <= 0 {
        return;
    }

    let start = usize::try_from(omp_block_start).expect("omp_block_start must be non-negative");
    let size = usize::try_from(omp_block_size).expect("omp_block_size must be non-negative");
    let cache_base = if cache.len() >= start + size {
        0
    } else {
        start
    };
    let mut i = start;
    let mut j = if size > 67 { start + size - 67 } else { start };

    while i < j {
        for current in [i, i + 1, i + 2, i + 3] {
            let ci = current - cache_base;
            let index = sa[current];
            cache[ci].index = index;
            cache[ci].symbol = t[index as usize];
        }
        i += 4;
    }

    j = if size > 67 { j + 67 } else { start + size };
    while i < j {
        let ci = i - cache_base;
        let index = sa[i];
        cache[ci].index = index;
        cache[ci].symbol = t[index as usize];
        i += 1;
    }
}

fn radix_sort_lms_suffixes_32s_6k_block_sort(
    induction_bucket: &mut [SaSint],
    cache: &mut [ThreadCache],
    omp_block_start: SaSint,
    omp_block_size: SaSint,
) {
    if omp_block_size <= 0 {
        return;
    }

    let start = usize::try_from(omp_block_start).expect("omp_block_start must be non-negative");
    let size = usize::try_from(omp_block_size).expect("omp_block_size must be non-negative");
    let cache_base = if cache.len() >= start + size {
        0
    } else {
        start
    };
    let mut i = start + size - 1;
    let mut j = start + 64 + 3;

    while i >= j {
        for current in [i, i - 1, i - 2, i - 3] {
            let ci = current - cache_base;
            let v = cache[ci].symbol as usize;
            induction_bucket[v] -= 1;
            cache[ci].symbol = induction_bucket[v];
        }
        i -= 4;
    }

    j -= 64 + 3;
    while i >= j {
        let ci = i - cache_base;
        let v = cache[ci].symbol as usize;
        induction_bucket[v] -= 1;
        cache[ci].symbol = induction_bucket[v];
        if i == 0 {
            break;
        }
        i -= 1;
    }
}

fn radix_sort_lms_suffixes_32s_2k_block_sort(
    induction_bucket: &mut [SaSint],
    cache: &mut [ThreadCache],
    omp_block_start: SaSint,
    omp_block_size: SaSint,
) {
    if omp_block_size <= 0 {
        return;
    }

    let start = usize::try_from(omp_block_start).expect("omp_block_start must be non-negative");
    let size = usize::try_from(omp_block_size).expect("omp_block_size must be non-negative");
    let cache_base = if cache.len() >= start + size {
        0
    } else {
        start
    };
    let mut i = start + size - 1;
    let mut j = start + 64 + 3;

    while i >= j {
        for current in [i, i - 1, i - 2, i - 3] {
            let ci = current - cache_base;
            let v = buckets_index2(cache[ci].symbol as usize, 0);
            induction_bucket[v] -= 1;
            cache[ci].symbol = induction_bucket[v];
        }
        i -= 4;
    }

    j -= 64 + 3;
    while i >= j {
        let ci = i - cache_base;
        let v = buckets_index2(cache[ci].symbol as usize, 0);
        induction_bucket[v] -= 1;
        cache[ci].symbol = induction_bucket[v];
        if i == 0 {
            break;
        }
        i -= 1;
    }
}

fn radix_sort_lms_suffixes_32s_6k_block_omp(
    t: &[SaSint],
    sa: &mut [SaSint],
    induction_bucket: &mut [SaSint],
    cache: &mut [ThreadCache],
    block_start: SaSint,
    block_size: SaSint,
    threads: SaSint,
) {
    if threads <= 1 || block_size < 16_384 {
        radix_sort_lms_suffixes_32s_6k(t, sa, induction_bucket, block_start, block_size);
        return;
    }

    radix_sort_lms_suffixes_32s_block_gather(t, sa, cache, block_start, block_size);
    radix_sort_lms_suffixes_32s_6k_block_sort(induction_bucket, cache, block_start, block_size);
    place_cached_suffixes(sa, cache, block_start, block_size);
}

fn radix_sort_lms_suffixes_32s_2k_block_omp(
    t: &[SaSint],
    sa: &mut [SaSint],
    induction_bucket: &mut [SaSint],
    cache: &mut [ThreadCache],
    block_start: SaSint,
    block_size: SaSint,
    threads: SaSint,
) {
    if threads <= 1 || block_size < 16_384 {
        radix_sort_lms_suffixes_32s_2k(t, sa, induction_bucket, block_start, block_size);
        return;
    }

    radix_sort_lms_suffixes_32s_block_gather(t, sa, cache, block_start, block_size);
    radix_sort_lms_suffixes_32s_2k_block_sort(induction_bucket, cache, block_start, block_size);
    place_cached_suffixes(sa, cache, block_start, block_size);
}

fn radix_sort_lms_suffixes_32s_6k_omp(
    t: &[SaSint],
    sa: &mut [SaSint],
    n: SaSint,
    m: SaSint,
    induction_bucket: &mut [SaSint],
    threads: SaSint,
) {
    if threads <= 1 || m < 65_536 {
        radix_sort_lms_suffixes_32s_6k(t, sa, induction_bucket, n - m + 1, m - 1);
        return;
    }

    let threads_usize = usize::try_from(threads).expect("threads must be positive");
    let mut cache = vec![ThreadCache::default(); threads_usize * PER_THREAD_CACHE_SIZE];
    let mut block_start = 0usize;
    let m_usize = usize::try_from(m).expect("m must be non-negative");
    let n_usize = usize::try_from(n).expect("n must be non-negative");
    let last = m_usize - 1;

    while block_start < last {
        let block_end = (block_start + threads_usize * PER_THREAD_CACHE_SIZE).min(last);
        radix_sort_lms_suffixes_32s_6k_block_omp(
            t,
            sa,
            induction_bucket,
            &mut cache,
            (n_usize - block_end) as SaSint,
            (block_end - block_start) as SaSint,
            threads,
        );
        block_start = block_end;
    }
}

fn radix_sort_lms_suffixes_32s_2k_omp(
    t: &[SaSint],
    sa: &mut [SaSint],
    n: SaSint,
    m: SaSint,
    induction_bucket: &mut [SaSint],
    threads: SaSint,
) {
    if threads <= 1 || m < 65_536 {
        radix_sort_lms_suffixes_32s_2k(t, sa, induction_bucket, n - m + 1, m - 1);
        return;
    }

    let threads_usize = usize::try_from(threads).expect("threads must be positive");
    let mut cache = vec![ThreadCache::default(); threads_usize * PER_THREAD_CACHE_SIZE];
    let mut block_start = 0usize;
    let m_usize = usize::try_from(m).expect("m must be non-negative");
    let n_usize = usize::try_from(n).expect("n must be non-negative");
    let last = m_usize - 1;

    while block_start < last {
        let block_end = (block_start + threads_usize * PER_THREAD_CACHE_SIZE).min(last);
        radix_sort_lms_suffixes_32s_2k_block_omp(
            t,
            sa,
            induction_bucket,
            &mut cache,
            (n_usize - block_end) as SaSint,
            (block_end - block_start) as SaSint,
            threads,
        );
        block_start = block_end;
    }
}

fn radix_sort_lms_suffixes_32s_1k(
    t: &[SaSint],
    sa: &mut [SaSint],
    n: SaSint,
    buckets: &mut [SaSint],
) -> SaSint {
    let mut i = n - 2;
    let mut m = 0;
    let mut f0 = 1usize;
    let mut f1: usize;
    let mut c0 = t[(n - 1) as usize] as isize;
    let mut c1: isize;
    let mut c2 = 0isize;

    while i >= 64 + 3 {
        c1 = t[i as usize] as isize;
        f1 = usize::from(c1 > c0 - f0 as isize);
        if (f1 & !f0) != 0 {
            c2 = c0;
            buckets[c2 as usize] -= 1;
            sa[buckets[c2 as usize] as usize] = i + 1;
            m += 1;
        }
        c0 = t[(i - 1) as usize] as isize;
        f0 = usize::from(c0 > c1 - f1 as isize);
        if (f0 & !f1) != 0 {
            c2 = c1;
            buckets[c2 as usize] -= 1;
            sa[buckets[c2 as usize] as usize] = i;
            m += 1;
        }
        c1 = t[(i - 2) as usize] as isize;
        f1 = usize::from(c1 > c0 - f0 as isize);
        if (f1 & !f0) != 0 {
            c2 = c0;
            buckets[c2 as usize] -= 1;
            sa[buckets[c2 as usize] as usize] = i - 1;
            m += 1;
        }
        c0 = t[(i - 3) as usize] as isize;
        f0 = usize::from(c0 > c1 - f1 as isize);
        if (f0 & !f1) != 0 {
            c2 = c1;
            buckets[c2 as usize] -= 1;
            sa[buckets[c2 as usize] as usize] = i - 2;
            m += 1;
        }
        i -= 4;
    }

    while i >= 0 {
        c1 = c0;
        c0 = t[i as usize] as isize;
        f1 = f0;
        f0 = usize::from(c0 > c1 - f1 as isize);
        if (f0 & !f1) != 0 {
            c2 = c1;
            buckets[c2 as usize] -= 1;
            sa[buckets[c2 as usize] as usize] = i + 1;
            m += 1;
        }
        i -= 1;
    }

    if m > 1 {
        sa[buckets[c2 as usize] as usize] = 0;
    }

    m
}

fn radix_sort_set_markers_32s_6k(
    sa: &mut [SaSint],
    induction_bucket: &[SaSint],
    omp_block_start: SaSint,
    omp_block_size: SaSint,
) {
    let mut i = omp_block_start;
    let mut j = omp_block_start + omp_block_size - 64 - 3;

    while i < j {
        sa[induction_bucket[i as usize] as usize] |= SAINT_MIN;
        sa[induction_bucket[(i + 1) as usize] as usize] |= SAINT_MIN;
        sa[induction_bucket[(i + 2) as usize] as usize] |= SAINT_MIN;
        sa[induction_bucket[(i + 3) as usize] as usize] |= SAINT_MIN;
        i += 4;
    }

    j += 64 + 3;
    while i < j {
        sa[induction_bucket[i as usize] as usize] |= SAINT_MIN;
        i += 1;
    }
}

fn radix_sort_set_markers_32s_4k(
    sa: &mut [SaSint],
    induction_bucket: &[SaSint],
    omp_block_start: SaSint,
    omp_block_size: SaSint,
) {
    let mut i = omp_block_start;
    let mut j = omp_block_start + omp_block_size - 64 - 3;

    while i < j {
        sa[induction_bucket[buckets_index2(i as usize, 0)] as usize] |= SUFFIX_GROUP_MARKER;
        sa[induction_bucket[buckets_index2((i + 1) as usize, 0)] as usize] |= SUFFIX_GROUP_MARKER;
        sa[induction_bucket[buckets_index2((i + 2) as usize, 0)] as usize] |= SUFFIX_GROUP_MARKER;
        sa[induction_bucket[buckets_index2((i + 3) as usize, 0)] as usize] |= SUFFIX_GROUP_MARKER;
        i += 4;
    }

    j += 64 + 3;
    while i < j {
        sa[induction_bucket[buckets_index2(i as usize, 0)] as usize] |= SUFFIX_GROUP_MARKER;
        i += 1;
    }
}

fn radix_sort_set_markers_32s_6k_omp(
    sa: &mut [SaSint],
    k: SaSint,
    induction_bucket: &[SaSint],
    threads: SaSint,
) {
    if k <= 1 {
        return;
    }

    if threads <= 1 || k < 65_536 {
        radix_sort_set_markers_32s_6k(sa, induction_bucket, 0, k - 1);
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
                    radix_sort_set_markers_32s_6k(
                        sa,
                        induction_bucket,
                        start as SaSint,
                        (end - start) as SaSint,
                    );
                }
            });
        });
    }
}

fn radix_sort_set_markers_32s_4k_omp(
    sa: &mut [SaSint],
    k: SaSint,
    induction_bucket: &[SaSint],
    threads: SaSint,
) {
    if k <= 1 {
        return;
    }

    if threads <= 1 || k < 65_536 {
        radix_sort_set_markers_32s_4k(sa, induction_bucket, 0, k - 1);
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
                        start as SaSint,
                        (end - start) as SaSint,
                    );
                }
            });
        });
    }
}

fn initialize_buckets_for_partial_sorting_16u(
    t: &[u16],
    buckets: &mut [SaSint],
    first_lms_suffix: SaSint,
    left_suffixes_count: SaSint,
) {
    buckets[buckets_index4(t[first_lms_suffix as usize] as usize, 1)] += 1;

    let (front, temp_bucket) = buckets.split_at_mut(4 * ALPHABET_SIZE);
    let mut sum0 = left_suffixes_count + 1;
    let mut sum1 = 0;

    for c in 0..ALPHABET_SIZE {
        let i = buckets_index4(c, 0);
        let j = buckets_index2(c, 0);

        temp_bucket[j + buckets_index2(0, 0)] = sum0;

        sum0 += front[i + buckets_index4(0, 0)] + front[i + buckets_index4(0, 2)];
        sum1 += front[i + buckets_index4(0, 1)];

        front[j + buckets_index2(0, 0)] = sum0;
        front[j + buckets_index2(0, 1)] = sum1;
    }
}

fn partial_sorting_shift_markers_32s_6k_omp(
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
                    let mut i = buckets_ref[buckets_index4(c_usize, 0)] - 1;
                    let mut j = temp_bucket_ref[buckets_index2(c_usize - 1, 0)] + 3;
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

fn partial_sorting_shift_markers_32s_4k(sa: &mut [SaSint], n: SaSint) {
    let mut i = n - 1;
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

fn partial_sorting_shift_buckets_32s_6k(k: SaSint, buckets: &mut [SaSint]) {
    let temp_offset = 4 * k as usize;
    let mut i = buckets_index2(0, 0);

    while i <= buckets_index2(k as usize - 1, 0) {
        buckets[2 * i + buckets_index4(0, 0)] = buckets[temp_offset + i + buckets_index2(0, 0)];
        buckets[2 * i + buckets_index4(0, 1)] = buckets[temp_offset + i + buckets_index2(0, 1)];
        i += buckets_index2(1, 0);
    }
}

fn partial_sorting_scan_left_to_right_16u(
    t: &[u16],
    sa: &mut [SaSint],
    buckets: &mut [SaSint],
    mut d: SaSint,
    omp_block_start: SaSint,
    omp_block_size: SaSint,
) -> SaSint {
    let mut i = omp_block_start as isize;
    let mut j = (omp_block_start + omp_block_size - 64 - 1) as isize;
    while i < j {
        let mut p0 = sa[i as usize];
        d += SaSint::from(p0 < 0);
        p0 &= SAINT_MAX;
        let v0 = buckets_index2(
            t[(p0 - 1) as usize] as usize,
            usize::from(t[(p0 - 2) as usize] >= t[(p0 - 1) as usize]),
        );
        let mark0 = if buckets[2 * ALPHABET_SIZE + v0] != d {
            SAINT_MIN
        } else {
            0
        };
        let dst0 = buckets[4 * ALPHABET_SIZE + v0] as usize;
        sa[dst0] = (p0 - 1) | mark0;
        buckets[4 * ALPHABET_SIZE + v0] += 1;
        buckets[2 * ALPHABET_SIZE + v0] = d;

        let mut p1 = sa[(i + 1) as usize];
        d += SaSint::from(p1 < 0);
        p1 &= SAINT_MAX;
        let v1 = buckets_index2(
            t[(p1 - 1) as usize] as usize,
            usize::from(t[(p1 - 2) as usize] >= t[(p1 - 1) as usize]),
        );
        let mark1 = if buckets[2 * ALPHABET_SIZE + v1] != d {
            SAINT_MIN
        } else {
            0
        };
        let dst1 = buckets[4 * ALPHABET_SIZE + v1] as usize;
        sa[dst1] = (p1 - 1) | mark1;
        buckets[4 * ALPHABET_SIZE + v1] += 1;
        buckets[2 * ALPHABET_SIZE + v1] = d;

        i += 2;
    }

    j += 64 + 1;
    while i < j {
        let mut p = sa[i as usize];
        d += SaSint::from(p < 0);
        p &= SAINT_MAX;
        let v = buckets_index2(
            t[(p - 1) as usize] as usize,
            usize::from(t[(p - 2) as usize] >= t[(p - 1) as usize]),
        );
        let mark = if buckets[2 * ALPHABET_SIZE + v] != d {
            SAINT_MIN
        } else {
            0
        };
        let dst = buckets[4 * ALPHABET_SIZE + v] as usize;
        sa[dst] = (p - 1) | mark;
        buckets[4 * ALPHABET_SIZE + v] += 1;
        buckets[2 * ALPHABET_SIZE + v] = d;
        i += 1;
    }

    d
}

fn partial_sorting_scan_left_to_right_16u_block_prepare(
    t: &[u16],
    sa: &mut [SaSint],
    k: SaSint,
    buckets: &mut [SaSint],
    cache: &mut [ThreadCache],
    omp_block_start: SaSint,
    omp_block_size: SaSint,
    state: &mut ThreadState,
) -> SaSint {
    let width = 2 * k as usize;
    buckets[..width].fill(0);
    buckets[2 * ALPHABET_SIZE..2 * ALPHABET_SIZE + width].fill(0);

    let mut count = 0usize;
    let mut d = 1;
    for i in omp_block_start as usize..(omp_block_start + omp_block_size) as usize {
        let mut p = sa[i];
        cache[count].index = p;
        d += SaSint::from(p < 0);
        p &= SAINT_MAX;
        let v = buckets_index2(
            t[(p - 1) as usize] as usize,
            usize::from(t[(p - 2) as usize] >= t[(p - 1) as usize]),
        );
        cache[count].symbol = v as SaSint;
        buckets[v] += 1;
        buckets[2 * ALPHABET_SIZE + v] = d;
        count += 1;
    }
    state.cache_entries = count;
    d - 1
}

fn partial_sorting_scan_left_to_right_16u_block_place(
    sa: &mut [SaSint],
    buckets: &mut [SaSint],
    cache: &[ThreadCache],
    count: SaSint,
    mut d: SaSint,
) {
    for entry in cache.iter().take(count as usize) {
        let mut p = entry.index;
        d += SaSint::from(p < 0);
        p &= SAINT_MAX;
        let v = entry.symbol as usize;
        let mark = if buckets[2 * ALPHABET_SIZE + v] != d {
            SAINT_MIN
        } else {
            0
        };
        let dst = buckets[v] as usize;
        sa[dst] = (p - 1) | mark;
        buckets[v] += 1;
        buckets[2 * ALPHABET_SIZE + v] = d;
    }
}

fn partial_sorting_scan_left_to_right_16u_block_omp(
    t: &[u16],
    sa: &mut [SaSint],
    k: SaSint,
    buckets: &mut [SaSint],
    d: SaSint,
    block_start: SaSint,
    block_size: SaSint,
    threads: SaSint,
    thread_state: &mut [ThreadState],
) -> SaSint {
    let thread_count = if threads > 1 && block_size >= 64 * k.max(256) {
        usize::try_from(threads)
            .expect("threads must be non-negative")
            .min(thread_state.len())
    } else {
        1
    };
    if thread_count <= 1 {
        return partial_sorting_scan_left_to_right_16u(t, sa, buckets, d, block_start, block_size);
    }

    let bucket_width = 2 * k as usize;
    let block_stride = (block_size / thread_count as SaSint) & !15;

    for (thread, state) in thread_state.iter_mut().take(thread_count).enumerate() {
        let local_start = thread as SaSint * block_stride;
        let local_size = if thread + 1 < thread_count {
            block_stride
        } else {
            block_size - local_start
        };
        let mut local_state = ThreadState::default();
        state.position = partial_sorting_scan_left_to_right_16u_block_prepare(
            t,
            sa,
            k,
            &mut state.buckets,
            &mut state.cache,
            block_start + local_start,
            local_size,
            &mut local_state,
        );
        state.count = local_state.cache_entries as SaSint;
    }

    let mut next_d = d;
    for state in thread_state.iter_mut().take(thread_count) {
        for c in 0..bucket_width {
            let a = buckets[4 * ALPHABET_SIZE + c];
            let b = state.buckets[c];
            buckets[4 * ALPHABET_SIZE + c] = a + b;
            state.buckets[c] = a;
        }

        next_d -= 1;
        for c in 0..bucket_width {
            let a = buckets[2 * ALPHABET_SIZE + c];
            let b = state.buckets[2 * ALPHABET_SIZE + c];
            let shifted = b + next_d;
            buckets[2 * ALPHABET_SIZE + c] = if b > 0 { shifted } else { a };
            state.buckets[2 * ALPHABET_SIZE + c] = a;
        }
        next_d += 1 + state.position;
        state.position = next_d - state.position;
    }

    for state in thread_state.iter_mut().take(thread_count) {
        partial_sorting_scan_left_to_right_16u_block_place(
            sa,
            &mut state.buckets,
            &state.cache,
            state.count,
            state.position,
        );
    }

    next_d
}

fn partial_sorting_scan_left_to_right_16u_omp(
    t: &[u16],
    sa: &mut [SaSint],
    n: SaSint,
    k: SaSint,
    buckets: &mut [SaSint],
    left_suffixes_count: SaSint,
    mut d: SaSint,
    threads: SaSint,
) -> SaSint {
    let v = buckets_index2(
        t[(n - 1) as usize] as usize,
        usize::from(t[(n - 2) as usize] >= t[(n - 1) as usize]),
    );
    let dst = buckets[4 * ALPHABET_SIZE + v] as usize;
    buckets[4 * ALPHABET_SIZE + v] += 1;
    sa[dst] = (n - 1) | SAINT_MIN;
    d += 1;
    buckets[2 * ALPHABET_SIZE + v] = d;

    if threads == 1 || left_suffixes_count < 65536 {
        d = partial_sorting_scan_left_to_right_16u(t, sa, buckets, d, 0, left_suffixes_count);
    } else {
        let mut thread_state = alloc_thread_state(threads).unwrap_or_default();
        let mut block_start = 0;
        while block_start < left_suffixes_count {
            if sa[block_start as usize] == 0 {
                block_start += 1;
            } else {
                let mut block_end =
                    block_start + threads * (PER_THREAD_CACHE_SIZE as SaSint - 16 * threads);
                if block_end > left_suffixes_count {
                    block_end = left_suffixes_count;
                }
                let mut block_scan_end = block_start + 1;
                while block_scan_end < block_end && sa[block_scan_end as usize] != 0 {
                    block_scan_end += 1;
                }
                let block_size = block_scan_end - block_start;

                if block_size < 32 {
                    while block_start < block_scan_end {
                        let mut p = sa[block_start as usize];
                        d += SaSint::from(p < 0);
                        p &= SAINT_MAX;
                        let v = buckets_index2(
                            t[(p - 1) as usize] as usize,
                            usize::from(t[(p - 2) as usize] >= t[(p - 1) as usize]),
                        );
                        let dst = buckets[4 * ALPHABET_SIZE + v] as usize;
                        buckets[4 * ALPHABET_SIZE + v] += 1;
                        let mark = if buckets[2 * ALPHABET_SIZE + v] != d {
                            SAINT_MIN
                        } else {
                            0
                        };
                        sa[dst] = (p - 1) | mark;
                        buckets[2 * ALPHABET_SIZE + v] = d;
                        block_start += 1;
                    }
                } else {
                    d = partial_sorting_scan_left_to_right_16u_block_omp(
                        t,
                        sa,
                        k,
                        buckets,
                        d,
                        block_start,
                        block_size,
                        threads,
                        &mut thread_state,
                    );
                    block_start = block_scan_end;
                }
            }
        }
    }
    d
}

fn partial_sorting_scan_right_to_left_16u(
    t: &[u16],
    sa: &mut [SaSint],
    buckets: &mut [SaSint],
    mut d: SaSint,
    omp_block_start: SaSint,
    omp_block_size: SaSint,
) -> SaSint {
    let mut i = (omp_block_start + omp_block_size - 1) as isize;
    let mut j = (omp_block_start + 64 + 1) as isize;
    while i >= j {
        let mut p0 = sa[i as usize];
        d += SaSint::from(p0 < 0);
        p0 &= SAINT_MAX;
        let v0 = buckets_index2(
            t[(p0 - 1) as usize] as usize,
            usize::from(t[(p0 - 2) as usize] > t[(p0 - 1) as usize]),
        );
        let mark0 = if buckets[2 * ALPHABET_SIZE + v0] != d {
            SAINT_MIN
        } else {
            0
        };
        buckets[v0] -= 1;
        sa[buckets[v0] as usize] = (p0 - 1) | mark0;
        buckets[2 * ALPHABET_SIZE + v0] = d;

        let mut p1 = sa[(i - 1) as usize];
        d += SaSint::from(p1 < 0);
        p1 &= SAINT_MAX;
        let v1 = buckets_index2(
            t[(p1 - 1) as usize] as usize,
            usize::from(t[(p1 - 2) as usize] > t[(p1 - 1) as usize]),
        );
        let mark1 = if buckets[2 * ALPHABET_SIZE + v1] != d {
            SAINT_MIN
        } else {
            0
        };
        buckets[v1] -= 1;
        sa[buckets[v1] as usize] = (p1 - 1) | mark1;
        buckets[2 * ALPHABET_SIZE + v1] = d;

        i -= 2;
    }

    j -= 64 + 1;
    while i >= j {
        let mut p = sa[i as usize];
        d += SaSint::from(p < 0);
        p &= SAINT_MAX;
        let v = buckets_index2(
            t[(p - 1) as usize] as usize,
            usize::from(t[(p - 2) as usize] > t[(p - 1) as usize]),
        );
        let mark = if buckets[2 * ALPHABET_SIZE + v] != d {
            SAINT_MIN
        } else {
            0
        };
        buckets[v] -= 1;
        sa[buckets[v] as usize] = (p - 1) | mark;
        buckets[2 * ALPHABET_SIZE + v] = d;
        i -= 1;
    }

    d
}

fn partial_sorting_scan_right_to_left_16u_block_prepare(
    t: &[u16],
    sa: &mut [SaSint],
    k: SaSint,
    buckets: &mut [SaSint],
    cache: &mut [ThreadCache],
    omp_block_start: SaSint,
    omp_block_size: SaSint,
    state: &mut ThreadState,
) -> SaSint {
    let width = 2 * k as usize;
    buckets[..width].fill(0);
    buckets[2 * ALPHABET_SIZE..2 * ALPHABET_SIZE + width].fill(0);

    let mut count = 0usize;
    let mut d = 1;
    for i in (omp_block_start as usize..(omp_block_start + omp_block_size) as usize).rev() {
        let mut p = sa[i];
        cache[count].index = p;
        d += SaSint::from(p < 0);
        p &= SAINT_MAX;
        let v = buckets_index2(
            t[(p - 1) as usize] as usize,
            usize::from(t[(p - 2) as usize] > t[(p - 1) as usize]),
        );
        cache[count].symbol = v as SaSint;
        buckets[v] += 1;
        buckets[2 * ALPHABET_SIZE + v] = d;
        count += 1;
    }
    state.cache_entries = count;
    d - 1
}

fn partial_sorting_scan_right_to_left_16u_block_place(
    sa: &mut [SaSint],
    buckets: &mut [SaSint],
    cache: &[ThreadCache],
    count: SaSint,
    mut d: SaSint,
) {
    for entry in cache.iter().take(count as usize) {
        let mut p = entry.index;
        d += SaSint::from(p < 0);
        p &= SAINT_MAX;
        let v = entry.symbol as usize;
        let mark = if buckets[2 * ALPHABET_SIZE + v] != d {
            SAINT_MIN
        } else {
            0
        };
        buckets[v] -= 1;
        sa[buckets[v] as usize] = (p - 1) | mark;
        buckets[2 * ALPHABET_SIZE + v] = d;
    }
}

fn partial_gsa_scan_right_to_left_16u_block_place(
    sa: &mut [SaSint],
    buckets: &mut [SaSint],
    cache: &[ThreadCache],
    count: SaSint,
    mut d: SaSint,
) {
    for entry in cache.iter().take(count as usize) {
        let mut p = entry.index;
        d += SaSint::from(p < 0);
        p &= SAINT_MAX;
        let v = entry.symbol as usize;
        if v != 1 {
            let mark = if buckets[2 * ALPHABET_SIZE + v] != d {
                SAINT_MIN
            } else {
                0
            };
            buckets[v] -= 1;
            sa[buckets[v] as usize] = (p - 1) | mark;
            buckets[2 * ALPHABET_SIZE + v] = d;
        }
    }
}

fn partial_sorting_scan_right_to_left_16u_block_omp(
    t: &[u16],
    sa: &mut [SaSint],
    k: SaSint,
    buckets: &mut [SaSint],
    d: SaSint,
    block_start: SaSint,
    block_size: SaSint,
    threads: SaSint,
    thread_state: &mut [ThreadState],
) -> SaSint {
    let thread_count = if threads > 1 && block_size >= 64 * k.max(256) {
        usize::try_from(threads)
            .expect("threads must be non-negative")
            .min(thread_state.len())
    } else {
        1
    };
    if thread_count <= 1 {
        return partial_sorting_scan_right_to_left_16u(t, sa, buckets, d, block_start, block_size);
    }

    let width = 2 * k as usize;
    let distinct_offset = 2 * ALPHABET_SIZE;
    let block_stride = (block_size / thread_count as SaSint) & !15;

    for (thread, state) in thread_state.iter_mut().take(thread_count).enumerate() {
        let local_start = thread as SaSint * block_stride;
        let local_size = if thread + 1 < thread_count {
            block_stride
        } else {
            block_size - local_start
        };
        let mut local_state = ThreadState::default();
        state.position = partial_sorting_scan_right_to_left_16u_block_prepare(
            t,
            sa,
            k,
            &mut state.buckets,
            &mut state.cache,
            block_start + local_start,
            local_size,
            &mut local_state,
        );
        state.count = local_state.cache_entries as SaSint;
    }

    let mut next_d = d;
    for state in thread_state.iter_mut().take(thread_count).rev() {
        for c in 0..width {
            let a = buckets[c];
            let b = state.buckets[c];
            buckets[c] = a - b;
            state.buckets[c] = a;
        }

        next_d -= 1;
        for c in 0..width {
            let offset = distinct_offset + c;
            let a = buckets[offset];
            let b = state.buckets[offset];
            let shifted = b + next_d;
            buckets[offset] = if b > 0 { shifted } else { a };
            state.buckets[offset] = a;
        }
        next_d += 1 + state.position;
        state.position = next_d - state.position;
    }

    for state in thread_state.iter_mut().take(thread_count) {
        partial_sorting_scan_right_to_left_16u_block_place(
            sa,
            &mut state.buckets,
            &state.cache,
            state.count,
            state.position,
        );
    }

    next_d
}

fn partial_sorting_scan_right_to_left_16u_omp(
    t: &[u16],
    sa: &mut [SaSint],
    n: SaSint,
    k: SaSint,
    buckets: &mut [SaSint],
    first_lms_suffix: SaSint,
    left_suffixes_count: SaSint,
    d: SaSint,
    threads: SaSint,
) {
    let scan_start = left_suffixes_count + 1;
    let scan_end = n - first_lms_suffix;

    if threads == 1 || scan_end - scan_start < 65536 {
        partial_sorting_scan_right_to_left_16u(
            t,
            sa,
            buckets,
            d,
            scan_start,
            scan_end - scan_start,
        );
    } else {
        let mut d = d;
        let mut thread_state = alloc_thread_state(threads).unwrap_or_default();
        let mut block_start = scan_end - 1;
        while block_start >= scan_start {
            if sa[block_start as usize] == 0 {
                block_start -= 1;
            } else {
                let block_limit = threads * (PER_THREAD_CACHE_SIZE as SaSint - 16 * threads);
                let mut block_max_end = block_start - block_limit;
                if block_max_end < scan_start {
                    block_max_end = scan_start - 1;
                }
                let mut block_end = block_start - 1;
                while block_end > block_max_end && sa[block_end as usize] != 0 {
                    block_end -= 1;
                }
                let block_size = block_start - block_end;

                if block_size < 32 {
                    while block_start > block_end {
                        let mut p = sa[block_start as usize];
                        d += SaSint::from(p < 0);
                        p &= SAINT_MAX;
                        let v = buckets_index2(
                            t[(p - 1) as usize] as usize,
                            usize::from(t[(p - 2) as usize] > t[(p - 1) as usize]),
                        );
                        let mark = if buckets[2 * ALPHABET_SIZE + v] != d {
                            SAINT_MIN
                        } else {
                            0
                        };
                        buckets[v] -= 1;
                        sa[buckets[v] as usize] = (p - 1) | mark;
                        buckets[2 * ALPHABET_SIZE + v] = d;
                        block_start -= 1;
                    }
                } else {
                    d = partial_sorting_scan_right_to_left_16u_block_omp(
                        t,
                        sa,
                        k,
                        buckets,
                        d,
                        block_end + 1,
                        block_size,
                        threads,
                        &mut thread_state,
                    );
                    block_start = block_end;
                }
            }
        }
    }
}

fn partial_sorting_scan_left_to_right_32s_6k(
    t: &[SaSint],
    sa: &mut [SaSint],
    buckets: &mut [SaSint],
    mut d: SaSint,
    omp_block_start: SaSint,
    omp_block_size: SaSint,
) -> SaSint {
    let mut i = omp_block_start;
    let mut j = omp_block_start + omp_block_size - 2 * 64 - 1;

    while i < j {
        let mut p2 = sa[i as usize];
        d += SaSint::from(p2 < 0);
        p2 &= SAINT_MAX;
        let v2 = buckets_index4(
            t[(p2 - 1) as usize] as usize,
            usize::from(t[(p2 - 2) as usize] >= t[(p2 - 1) as usize]),
        );
        let pos2 = buckets[v2] as usize;
        buckets[v2] += 1;
        sa[pos2] = (p2 - 1) | (((buckets[2 + v2] != d) as SaSint) << (SAINT_BIT - 1));
        buckets[2 + v2] = d;

        let mut p3 = sa[(i + 1) as usize];
        d += SaSint::from(p3 < 0);
        p3 &= SAINT_MAX;
        let v3 = buckets_index4(
            t[(p3 - 1) as usize] as usize,
            usize::from(t[(p3 - 2) as usize] >= t[(p3 - 1) as usize]),
        );
        let pos3 = buckets[v3] as usize;
        buckets[v3] += 1;
        sa[pos3] = (p3 - 1) | (((buckets[2 + v3] != d) as SaSint) << (SAINT_BIT - 1));
        buckets[2 + v3] = d;

        i += 2;
    }

    j += 2 * 64 + 1;
    while i < j {
        let mut p = sa[i as usize];
        d += SaSint::from(p < 0);
        p &= SAINT_MAX;
        let v = buckets_index4(
            t[(p - 1) as usize] as usize,
            usize::from(t[(p - 2) as usize] >= t[(p - 1) as usize]),
        );
        let pos = buckets[v] as usize;
        buckets[v] += 1;
        sa[pos] = (p - 1) | (((buckets[2 + v] != d) as SaSint) << (SAINT_BIT - 1));
        buckets[2 + v] = d;
        i += 1;
    }

    d
}

fn partial_sorting_scan_left_to_right_32s_4k(
    t: &[SaSint],
    sa: &mut [SaSint],
    k: SaSint,
    buckets: &mut [SaSint],
    mut d: SaSint,
    omp_block_start: SaSint,
    omp_block_size: SaSint,
) -> SaSint {
    let k = k as usize;
    let mut i = omp_block_start;
    let mut j = omp_block_start + omp_block_size - 2 * 64 - 1;

    while i < j {
        let mut p0 = sa[i as usize];
        sa[i as usize] = p0 & SAINT_MAX;
        if p0 > 0 {
            sa[i as usize] = 0;
            d += p0 >> (SUFFIX_GROUP_BIT - 1);
            p0 &= !SUFFIX_GROUP_MARKER;
            let v0 = buckets_index2(
                t[(p0 - 1) as usize] as usize,
                usize::from(t[(p0 - 2) as usize] < t[(p0 - 1) as usize]),
            );
            let c0 = t[(p0 - 1) as usize] as usize;
            let pos0 = buckets[2 * k + c0] as usize;
            buckets[2 * k + c0] += 1;
            sa[pos0] = (p0 - 1)
                | ((usize::from(t[(p0 - 2) as usize] < t[(p0 - 1) as usize]) as SaSint)
                    << (SAINT_BIT - 1))
                | (((buckets[v0] != d) as SaSint) << (SUFFIX_GROUP_BIT - 1));
            buckets[v0] = d;
        }

        let mut p1 = sa[(i + 1) as usize];
        sa[(i + 1) as usize] = p1 & SAINT_MAX;
        if p1 > 0 {
            sa[(i + 1) as usize] = 0;
            d += p1 >> (SUFFIX_GROUP_BIT - 1);
            p1 &= !SUFFIX_GROUP_MARKER;
            let v1 = buckets_index2(
                t[(p1 - 1) as usize] as usize,
                usize::from(t[(p1 - 2) as usize] < t[(p1 - 1) as usize]),
            );
            let c1 = t[(p1 - 1) as usize] as usize;
            let pos1 = buckets[2 * k + c1] as usize;
            buckets[2 * k + c1] += 1;
            sa[pos1] = (p1 - 1)
                | ((usize::from(t[(p1 - 2) as usize] < t[(p1 - 1) as usize]) as SaSint)
                    << (SAINT_BIT - 1))
                | (((buckets[v1] != d) as SaSint) << (SUFFIX_GROUP_BIT - 1));
            buckets[v1] = d;
        }

        i += 2;
    }

    j += 2 * 64 + 1;
    while i < j {
        let mut p = sa[i as usize];
        sa[i as usize] = p & SAINT_MAX;
        if p > 0 {
            sa[i as usize] = 0;
            d += p >> (SUFFIX_GROUP_BIT - 1);
            p &= !SUFFIX_GROUP_MARKER;
            let v = buckets_index2(
                t[(p - 1) as usize] as usize,
                usize::from(t[(p - 2) as usize] < t[(p - 1) as usize]),
            );
            let c = t[(p - 1) as usize] as usize;
            let pos = buckets[2 * k + c] as usize;
            buckets[2 * k + c] += 1;
            sa[pos] = (p - 1)
                | ((usize::from(t[(p - 2) as usize] < t[(p - 1) as usize]) as SaSint)
                    << (SAINT_BIT - 1))
                | (((buckets[v] != d) as SaSint) << (SUFFIX_GROUP_BIT - 1));
            buckets[v] = d;
        }
        i += 1;
    }

    d
}

fn partial_sorting_scan_left_to_right_32s_1k(
    t: &[SaSint],
    sa: &mut [SaSint],
    induction_bucket: &mut [SaSint],
    omp_block_start: SaSint,
    omp_block_size: SaSint,
) {
    let mut i = omp_block_start;
    let mut j = omp_block_start + omp_block_size - 2 * 64 - 1;

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

    j += 2 * 64 + 1;
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

fn partial_sorting_scan_left_to_right_32s_6k_omp(
    t: &[SaSint],
    sa: &mut [SaSint],
    n: SaSint,
    buckets: &mut [SaSint],
    left_suffixes_count: SaSint,
    mut d: SaSint,
    threads: SaSint,
    _thread_state: &mut [ThreadState],
) -> SaSint {
    let v = buckets_index4(
        t[(n - 1) as usize] as usize,
        usize::from(t[(n - 2) as usize] >= t[(n - 1) as usize]),
    );
    let pos = buckets[v] as usize;
    buckets[v] += 1;
    sa[pos] = (n - 1) | SAINT_MIN;
    d += 1;
    buckets[2 + v] = d;

    if threads == 1 || left_suffixes_count < 65536 {
        d = partial_sorting_scan_left_to_right_32s_6k(t, sa, buckets, d, 0, left_suffixes_count);
    } else {
        let mut cache = vec![ThreadCache::default(); left_suffixes_count as usize];
        let mut block_start = 0;
        while block_start < left_suffixes_count {
            let mut block_end = block_start + threads * PER_THREAD_CACHE_SIZE as SaSint;
            if block_end > left_suffixes_count {
                block_end = left_suffixes_count;
            }
            d = partial_sorting_scan_left_to_right_32s_6k_block_omp(
                t,
                sa,
                buckets,
                d,
                &mut cache,
                block_start,
                block_end - block_start,
                threads,
            );
            block_start = block_end;
        }
    }

    d
}

fn partial_sorting_scan_left_to_right_32s_4k_omp(
    t: &[SaSint],
    sa: &mut [SaSint],
    n: SaSint,
    k: SaSint,
    buckets: &mut [SaSint],
    mut d: SaSint,
    threads: SaSint,
    _thread_state: &mut [ThreadState],
) -> SaSint {
    let k_usize = k as usize;
    let pos = buckets[2 * k_usize + t[(n - 1) as usize] as usize] as usize;
    buckets[2 * k_usize + t[(n - 1) as usize] as usize] += 1;
    sa[pos] = (n - 1)
        | ((usize::from(t[(n - 2) as usize] < t[(n - 1) as usize]) as SaSint) << (SAINT_BIT - 1))
        | SUFFIX_GROUP_MARKER;
    d += 1;
    buckets[buckets_index2(
        t[(n - 1) as usize] as usize,
        usize::from(t[(n - 2) as usize] < t[(n - 1) as usize]),
    )] = d;

    if threads == 1 || n < 65536 {
        d = partial_sorting_scan_left_to_right_32s_4k(t, sa, k, buckets, d, 0, n);
    } else {
        let mut cache = vec![ThreadCache::default(); n as usize];
        let mut block_start = 0;
        while block_start < n {
            let mut block_end = block_start + threads * PER_THREAD_CACHE_SIZE as SaSint;
            if block_end > n {
                block_end = n;
            }
            d = partial_sorting_scan_left_to_right_32s_4k_block_omp(
                t,
                sa,
                k,
                buckets,
                d,
                &mut cache,
                block_start,
                block_end - block_start,
                threads,
            );
            block_start = block_end;
        }
    }

    d
}

fn partial_sorting_scan_left_to_right_32s_1k_omp(
    t: &[SaSint],
    sa: &mut [SaSint],
    n: SaSint,
    buckets: &mut [SaSint],
    threads: SaSint,
    _thread_state: &mut [ThreadState],
) {
    let pos = buckets[t[(n - 1) as usize] as usize] as usize;
    buckets[t[(n - 1) as usize] as usize] += 1;
    sa[pos] = (n - 1)
        | ((usize::from(t[(n - 2) as usize] < t[(n - 1) as usize]) as SaSint) << (SAINT_BIT - 1));

    if threads == 1 || n < 65536 {
        partial_sorting_scan_left_to_right_32s_1k(t, sa, buckets, 0, n);
    } else {
        let mut cache = vec![ThreadCache::default(); n as usize];
        let mut block_start = 0;
        while block_start < n {
            let mut block_end = block_start + threads * PER_THREAD_CACHE_SIZE as SaSint;
            if block_end > n {
                block_end = n;
            }
            partial_sorting_scan_left_to_right_32s_1k_block_omp(
                t,
                sa,
                buckets,
                &mut cache,
                block_start,
                block_end - block_start,
                threads,
            );
            block_start = block_end;
        }
    }
}

fn partial_sorting_scan_right_to_left_32s_6k(
    t: &[SaSint],
    sa: &mut [SaSint],
    buckets: &mut [SaSint],
    mut d: SaSint,
    omp_block_start: SaSint,
    omp_block_size: SaSint,
) -> SaSint {
    if omp_block_size <= 0 {
        return d;
    }

    let mut i = omp_block_start + omp_block_size - 1;
    let mut j = omp_block_start + 2 * 64 + 1;

    while i >= j {
        let mut p2 = sa[i as usize];
        d += SaSint::from(p2 < 0);
        p2 &= SAINT_MAX;
        let v2 = buckets_index4(
            t[(p2 - 1) as usize] as usize,
            usize::from(t[(p2 - 2) as usize] > t[(p2 - 1) as usize]),
        );
        buckets[v2] -= 1;
        sa[buckets[v2] as usize] =
            (p2 - 1) | (((buckets[2 + v2] != d) as SaSint) << (SAINT_BIT - 1));
        buckets[2 + v2] = d;

        let mut p3 = sa[(i - 1) as usize];
        d += SaSint::from(p3 < 0);
        p3 &= SAINT_MAX;
        let v3 = buckets_index4(
            t[(p3 - 1) as usize] as usize,
            usize::from(t[(p3 - 2) as usize] > t[(p3 - 1) as usize]),
        );
        buckets[v3] -= 1;
        sa[buckets[v3] as usize] =
            (p3 - 1) | (((buckets[2 + v3] != d) as SaSint) << (SAINT_BIT - 1));
        buckets[2 + v3] = d;

        i -= 2;
    }

    j -= 2 * 64 + 1;
    while i >= j {
        let mut p = sa[i as usize];
        d += SaSint::from(p < 0);
        p &= SAINT_MAX;
        let v = buckets_index4(
            t[(p - 1) as usize] as usize,
            usize::from(t[(p - 2) as usize] > t[(p - 1) as usize]),
        );
        buckets[v] -= 1;
        sa[buckets[v] as usize] = (p - 1) | (((buckets[2 + v] != d) as SaSint) << (SAINT_BIT - 1));
        buckets[2 + v] = d;
        i -= 1;
    }

    d
}

fn partial_sorting_scan_right_to_left_32s_4k(
    t: &[SaSint],
    sa: &mut [SaSint],
    k: SaSint,
    buckets: &mut [SaSint],
    mut d: SaSint,
    omp_block_start: SaSint,
    omp_block_size: SaSint,
) -> SaSint {
    if omp_block_size <= 0 {
        return d;
    }

    let k = k as usize;
    let mut i = omp_block_start + omp_block_size - 1;
    let mut j = omp_block_start + 2 * 64 + 1;

    while i >= j {
        let mut p0 = sa[i as usize];
        if p0 > 0 {
            sa[i as usize] = 0;
            d += p0 >> (SUFFIX_GROUP_BIT - 1);
            p0 &= !SUFFIX_GROUP_MARKER;
            let v0 = buckets_index2(
                t[(p0 - 1) as usize] as usize,
                usize::from(t[(p0 - 2) as usize] > t[(p0 - 1) as usize]),
            );
            let c0 = t[(p0 - 1) as usize] as usize;
            buckets[3 * k + c0] -= 1;
            sa[buckets[3 * k + c0] as usize] = (p0 - 1)
                | ((usize::from(t[(p0 - 2) as usize] > t[(p0 - 1) as usize]) as SaSint)
                    << (SAINT_BIT - 1))
                | (((buckets[v0] != d) as SaSint) << (SUFFIX_GROUP_BIT - 1));
            buckets[v0] = d;
        }

        let mut p1 = sa[(i - 1) as usize];
        if p1 > 0 {
            sa[(i - 1) as usize] = 0;
            d += p1 >> (SUFFIX_GROUP_BIT - 1);
            p1 &= !SUFFIX_GROUP_MARKER;
            let v1 = buckets_index2(
                t[(p1 - 1) as usize] as usize,
                usize::from(t[(p1 - 2) as usize] > t[(p1 - 1) as usize]),
            );
            let c1 = t[(p1 - 1) as usize] as usize;
            buckets[3 * k + c1] -= 1;
            sa[buckets[3 * k + c1] as usize] = (p1 - 1)
                | ((usize::from(t[(p1 - 2) as usize] > t[(p1 - 1) as usize]) as SaSint)
                    << (SAINT_BIT - 1))
                | (((buckets[v1] != d) as SaSint) << (SUFFIX_GROUP_BIT - 1));
            buckets[v1] = d;
        }

        i -= 2;
    }

    j -= 2 * 64 + 1;
    while i >= j {
        let mut p = sa[i as usize];
        if p > 0 {
            sa[i as usize] = 0;
            d += p >> (SUFFIX_GROUP_BIT - 1);
            p &= !SUFFIX_GROUP_MARKER;
            let v = buckets_index2(
                t[(p - 1) as usize] as usize,
                usize::from(t[(p - 2) as usize] > t[(p - 1) as usize]),
            );
            let c = t[(p - 1) as usize] as usize;
            buckets[3 * k + c] -= 1;
            sa[buckets[3 * k + c] as usize] = (p - 1)
                | ((usize::from(t[(p - 2) as usize] > t[(p - 1) as usize]) as SaSint)
                    << (SAINT_BIT - 1))
                | (((buckets[v] != d) as SaSint) << (SUFFIX_GROUP_BIT - 1));
            buckets[v] = d;
        }
        i -= 1;
    }

    d
}

fn partial_sorting_scan_right_to_left_32s_1k(
    t: &[SaSint],
    sa: &mut [SaSint],
    induction_bucket: &mut [SaSint],
    omp_block_start: SaSint,
    omp_block_size: SaSint,
) {
    if omp_block_size <= 0 {
        return;
    }

    let mut i = omp_block_start + omp_block_size - 1;
    let mut j = omp_block_start + 2 * 64 + 1;

    while i >= j {
        let p0 = sa[i as usize];
        if p0 > 0 {
            sa[i as usize] = 0;
            let c0 = t[(p0 - 1) as usize] as usize;
            induction_bucket[c0] -= 1;
            sa[induction_bucket[c0] as usize] = (p0 - 1)
                | ((usize::from(t[(p0 - 2) as usize] > t[(p0 - 1) as usize]) as SaSint)
                    << (SAINT_BIT - 1));
        }

        let p1 = sa[(i - 1) as usize];
        if p1 > 0 {
            sa[(i - 1) as usize] = 0;
            let c1 = t[(p1 - 1) as usize] as usize;
            induction_bucket[c1] -= 1;
            sa[induction_bucket[c1] as usize] = (p1 - 1)
                | ((usize::from(t[(p1 - 2) as usize] > t[(p1 - 1) as usize]) as SaSint)
                    << (SAINT_BIT - 1));
        }

        i -= 2;
    }

    j -= 2 * 64 + 1;
    while i >= j {
        let p = sa[i as usize];
        if p > 0 {
            sa[i as usize] = 0;
            let c = t[(p - 1) as usize] as usize;
            induction_bucket[c] -= 1;
            sa[induction_bucket[c] as usize] = (p - 1)
                | ((usize::from(t[(p - 2) as usize] > t[(p - 1) as usize]) as SaSint)
                    << (SAINT_BIT - 1));
        }
        i -= 1;
    }
}

fn partial_sorting_scan_right_to_left_32s_6k_omp(
    t: &[SaSint],
    sa: &mut [SaSint],
    n: SaSint,
    buckets: &mut [SaSint],
    first_lms_suffix: SaSint,
    left_suffixes_count: SaSint,
    mut d: SaSint,
    threads: SaSint,
    _thread_state: &mut [ThreadState],
) -> SaSint {
    let scan_start = left_suffixes_count + 1;
    let scan_end = n - first_lms_suffix;

    if threads == 1 || scan_end - scan_start < 65536 {
        d = partial_sorting_scan_right_to_left_32s_6k(
            t,
            sa,
            buckets,
            d,
            scan_start,
            scan_end - scan_start,
        );
    } else {
        let mut cache = vec![ThreadCache::default(); (scan_end - scan_start) as usize];
        let mut block_start = scan_end;
        while block_start > scan_start {
            let block_size =
                (block_start - scan_start).min(threads * PER_THREAD_CACHE_SIZE as SaSint);
            block_start -= block_size;
            d = partial_sorting_scan_right_to_left_32s_6k_block_omp(
                t,
                sa,
                buckets,
                d,
                &mut cache,
                block_start,
                block_size,
                threads,
            );
        }
    }

    d
}

fn partial_sorting_scan_right_to_left_32s_4k_omp(
    t: &[SaSint],
    sa: &mut [SaSint],
    n: SaSint,
    k: SaSint,
    buckets: &mut [SaSint],
    mut d: SaSint,
    threads: SaSint,
    _thread_state: &mut [ThreadState],
) -> SaSint {
    if threads == 1 || n < 65536 {
        d = partial_sorting_scan_right_to_left_32s_4k(t, sa, k, buckets, d, 0, n);
    } else {
        let mut cache = vec![ThreadCache::default(); n as usize];
        let mut block_start = n;
        while block_start > 0 {
            let block_size = block_start.min(threads * PER_THREAD_CACHE_SIZE as SaSint);
            block_start -= block_size;
            d = partial_sorting_scan_right_to_left_32s_4k_block_omp(
                t,
                sa,
                k,
                buckets,
                d,
                &mut cache,
                block_start,
                block_size,
                threads,
            );
        }
    }

    d
}

fn partial_sorting_scan_right_to_left_32s_1k_omp(
    t: &[SaSint],
    sa: &mut [SaSint],
    n: SaSint,
    buckets: &mut [SaSint],
    threads: SaSint,
    _thread_state: &mut [ThreadState],
) {
    if threads == 1 || n < 65536 {
        partial_sorting_scan_right_to_left_32s_1k(t, sa, buckets, 0, n);
    } else {
        let mut cache = vec![ThreadCache::default(); n as usize];
        let mut block_start = n;
        while block_start > 0 {
            let block_size = block_start.min(threads * PER_THREAD_CACHE_SIZE as SaSint);
            block_start -= block_size;
            partial_sorting_scan_right_to_left_32s_1k_block_omp(
                t,
                sa,
                buckets,
                &mut cache,
                block_start,
                block_size,
                threads,
            );
        }
    }
}

fn partial_sorting_scan_left_to_right_32s_6k_block_gather(
    t: &[SaSint],
    sa: &mut [SaSint],
    cache: &mut [ThreadCache],
    omp_block_start: SaSint,
    omp_block_size: SaSint,
) {
    let mut i = omp_block_start;
    let mut j = omp_block_start + omp_block_size - 64 - 1;

    while i < j {
        let p0 = sa[i as usize];
        cache[i as usize].index = p0;
        let p0 = p0 & SAINT_MAX;
        cache[i as usize].symbol = if p0 != 0 {
            buckets_index4(
                t[(p0 - 1) as usize] as usize,
                usize::from(t[(p0 - 2) as usize] >= t[(p0 - 1) as usize]),
            ) as SaSint
        } else {
            0
        };

        let p1 = sa[(i + 1) as usize];
        cache[(i + 1) as usize].index = p1;
        let p1 = p1 & SAINT_MAX;
        cache[(i + 1) as usize].symbol = if p1 != 0 {
            buckets_index4(
                t[(p1 - 1) as usize] as usize,
                usize::from(t[(p1 - 2) as usize] >= t[(p1 - 1) as usize]),
            ) as SaSint
        } else {
            0
        };

        i += 2;
    }

    j += 64 + 1;
    while i < j {
        let p = sa[i as usize];
        cache[i as usize].index = p;
        let p = p & SAINT_MAX;
        cache[i as usize].symbol = if p != 0 {
            buckets_index4(
                t[(p - 1) as usize] as usize,
                usize::from(t[(p - 2) as usize] >= t[(p - 1) as usize]),
            ) as SaSint
        } else {
            0
        };
        i += 1;
    }
}

fn partial_sorting_scan_left_to_right_32s_4k_block_gather(
    t: &[SaSint],
    sa: &mut [SaSint],
    cache: &mut [ThreadCache],
    omp_block_start: SaSint,
    omp_block_size: SaSint,
) {
    let mut i = omp_block_start;
    let mut j = omp_block_start + omp_block_size - 64 - 1;

    while i < j {
        let mut symbol0 = SAINT_MIN;
        let mut p0 = sa[i as usize];
        if p0 > 0 {
            cache[i as usize].index = p0;
            p0 &= !SUFFIX_GROUP_MARKER;
            symbol0 = buckets_index2(
                t[(p0 - 1) as usize] as usize,
                usize::from(t[(p0 - 2) as usize] < t[(p0 - 1) as usize]),
            ) as SaSint;
            p0 = 0;
        }
        cache[i as usize].symbol = symbol0;
        sa[i as usize] = p0 & SAINT_MAX;

        let mut symbol1 = SAINT_MIN;
        let mut p1 = sa[(i + 1) as usize];
        if p1 > 0 {
            cache[(i + 1) as usize].index = p1;
            p1 &= !SUFFIX_GROUP_MARKER;
            symbol1 = buckets_index2(
                t[(p1 - 1) as usize] as usize,
                usize::from(t[(p1 - 2) as usize] < t[(p1 - 1) as usize]),
            ) as SaSint;
            p1 = 0;
        }
        cache[(i + 1) as usize].symbol = symbol1;
        sa[(i + 1) as usize] = p1 & SAINT_MAX;

        i += 2;
    }

    j += 64 + 1;
    while i < j {
        let mut symbol = SAINT_MIN;
        let mut p = sa[i as usize];
        if p > 0 {
            cache[i as usize].index = p;
            p &= !SUFFIX_GROUP_MARKER;
            symbol = buckets_index2(
                t[(p - 1) as usize] as usize,
                usize::from(t[(p - 2) as usize] < t[(p - 1) as usize]),
            ) as SaSint;
            p = 0;
        }
        cache[i as usize].symbol = symbol;
        sa[i as usize] = p & SAINT_MAX;
        i += 1;
    }
}

fn partial_sorting_scan_left_to_right_32s_1k_block_gather(
    t: &[SaSint],
    sa: &mut [SaSint],
    cache: &mut [ThreadCache],
    omp_block_start: SaSint,
    omp_block_size: SaSint,
) {
    let mut i = omp_block_start;
    let mut j = omp_block_start + omp_block_size - 64 - 1;

    while i < j {
        let mut symbol0 = SAINT_MIN;
        let mut p0 = sa[i as usize];
        if p0 > 0 {
            cache[i as usize].index = (p0 - 1)
                | ((usize::from(t[(p0 - 2) as usize] < t[(p0 - 1) as usize]) as SaSint)
                    << (SAINT_BIT - 1));
            symbol0 = t[(p0 - 1) as usize];
            p0 = 0;
        }
        cache[i as usize].symbol = symbol0;
        sa[i as usize] = p0 & SAINT_MAX;

        let mut symbol1 = SAINT_MIN;
        let mut p1 = sa[(i + 1) as usize];
        if p1 > 0 {
            cache[(i + 1) as usize].index = (p1 - 1)
                | ((usize::from(t[(p1 - 2) as usize] < t[(p1 - 1) as usize]) as SaSint)
                    << (SAINT_BIT - 1));
            symbol1 = t[(p1 - 1) as usize];
            p1 = 0;
        }
        cache[(i + 1) as usize].symbol = symbol1;
        sa[(i + 1) as usize] = p1 & SAINT_MAX;

        i += 2;
    }

    j += 64 + 1;
    while i < j {
        let mut symbol = SAINT_MIN;
        let mut p = sa[i as usize];
        if p > 0 {
            cache[i as usize].index = (p - 1)
                | ((usize::from(t[(p - 2) as usize] < t[(p - 1) as usize]) as SaSint)
                    << (SAINT_BIT - 1));
            symbol = t[(p - 1) as usize];
            p = 0;
        }
        cache[i as usize].symbol = symbol;
        sa[i as usize] = p & SAINT_MAX;
        i += 1;
    }
}

fn partial_sorting_scan_right_to_left_32s_6k_block_gather(
    t: &[SaSint],
    sa: &mut [SaSint],
    cache: &mut [ThreadCache],
    omp_block_start: SaSint,
    omp_block_size: SaSint,
) {
    let mut i = omp_block_start;
    let mut j = omp_block_start + omp_block_size - 64 - 1;

    while i < j {
        let p0 = sa[i as usize];
        cache[i as usize].index = p0;
        let p0 = p0 & SAINT_MAX;
        cache[i as usize].symbol = if p0 != 0 {
            buckets_index4(
                t[(p0 - 1) as usize] as usize,
                usize::from(t[(p0 - 2) as usize] > t[(p0 - 1) as usize]),
            ) as SaSint
        } else {
            0
        };

        let p1 = sa[(i + 1) as usize];
        cache[(i + 1) as usize].index = p1;
        let p1 = p1 & SAINT_MAX;
        cache[(i + 1) as usize].symbol = if p1 != 0 {
            buckets_index4(
                t[(p1 - 1) as usize] as usize,
                usize::from(t[(p1 - 2) as usize] > t[(p1 - 1) as usize]),
            ) as SaSint
        } else {
            0
        };

        i += 2;
    }

    j += 64 + 1;
    while i < j {
        let p = sa[i as usize];
        cache[i as usize].index = p;
        let p = p & SAINT_MAX;
        cache[i as usize].symbol = if p != 0 {
            buckets_index4(
                t[(p - 1) as usize] as usize,
                usize::from(t[(p - 2) as usize] > t[(p - 1) as usize]),
            ) as SaSint
        } else {
            0
        };
        i += 1;
    }
}

fn partial_sorting_scan_right_to_left_32s_4k_block_gather(
    t: &[SaSint],
    sa: &mut [SaSint],
    cache: &mut [ThreadCache],
    omp_block_start: SaSint,
    omp_block_size: SaSint,
) {
    let mut i = omp_block_start;
    let mut j = omp_block_start + omp_block_size - 64 - 1;

    while i < j {
        let mut symbol0 = SAINT_MIN;
        let mut p0 = sa[i as usize];
        if p0 > 0 {
            sa[i as usize] = 0;
            cache[i as usize].index = p0;
            p0 &= !SUFFIX_GROUP_MARKER;
            symbol0 = buckets_index2(
                t[(p0 - 1) as usize] as usize,
                usize::from(t[(p0 - 2) as usize] > t[(p0 - 1) as usize]),
            ) as SaSint;
        }
        cache[i as usize].symbol = symbol0;

        let mut symbol1 = SAINT_MIN;
        let mut p1 = sa[(i + 1) as usize];
        if p1 > 0 {
            sa[(i + 1) as usize] = 0;
            cache[(i + 1) as usize].index = p1;
            p1 &= !SUFFIX_GROUP_MARKER;
            symbol1 = buckets_index2(
                t[(p1 - 1) as usize] as usize,
                usize::from(t[(p1 - 2) as usize] > t[(p1 - 1) as usize]),
            ) as SaSint;
        }
        cache[(i + 1) as usize].symbol = symbol1;

        i += 2;
    }

    j += 64 + 1;
    while i < j {
        let mut symbol = SAINT_MIN;
        let mut p = sa[i as usize];
        if p > 0 {
            sa[i as usize] = 0;
            cache[i as usize].index = p;
            p &= !SUFFIX_GROUP_MARKER;
            symbol = buckets_index2(
                t[(p - 1) as usize] as usize,
                usize::from(t[(p - 2) as usize] > t[(p - 1) as usize]),
            ) as SaSint;
        }
        cache[i as usize].symbol = symbol;
        i += 1;
    }
}

fn partial_sorting_scan_right_to_left_32s_1k_block_gather(
    t: &[SaSint],
    sa: &mut [SaSint],
    cache: &mut [ThreadCache],
    omp_block_start: SaSint,
    omp_block_size: SaSint,
) {
    let mut i = omp_block_start;
    let mut j = omp_block_start + omp_block_size - 64 - 1;

    while i < j {
        let mut symbol0 = SAINT_MIN;
        let p0 = sa[i as usize];
        if p0 > 0 {
            sa[i as usize] = 0;
            cache[i as usize].index = (p0 - 1)
                | ((usize::from(t[(p0 - 2) as usize] > t[(p0 - 1) as usize]) as SaSint)
                    << (SAINT_BIT - 1));
            symbol0 = t[(p0 - 1) as usize];
        }
        cache[i as usize].symbol = symbol0;

        let mut symbol1 = SAINT_MIN;
        let p1 = sa[(i + 1) as usize];
        if p1 > 0 {
            sa[(i + 1) as usize] = 0;
            cache[(i + 1) as usize].index = (p1 - 1)
                | ((usize::from(t[(p1 - 2) as usize] > t[(p1 - 1) as usize]) as SaSint)
                    << (SAINT_BIT - 1));
            symbol1 = t[(p1 - 1) as usize];
        }
        cache[(i + 1) as usize].symbol = symbol1;

        i += 2;
    }

    j += 64 + 1;
    while i < j {
        let mut symbol = SAINT_MIN;
        let p = sa[i as usize];
        if p > 0 {
            sa[i as usize] = 0;
            cache[i as usize].index = (p - 1)
                | ((usize::from(t[(p - 2) as usize] > t[(p - 1) as usize]) as SaSint)
                    << (SAINT_BIT - 1));
            symbol = t[(p - 1) as usize];
        }
        cache[i as usize].symbol = symbol;
        i += 1;
    }
}

fn partial_sorting_scan_left_to_right_32s_6k_block_sort(
    t: &[SaSint],
    buckets: &mut [SaSint],
    mut d: SaSint,
    cache: &mut [ThreadCache],
    omp_block_start: SaSint,
    omp_block_size: SaSint,
) -> SaSint {
    let mut i = omp_block_start;
    let omp_block_end = omp_block_start + omp_block_size;
    let mut j = omp_block_end - 64 - 1;

    while i < j {
        let v0 = cache[i as usize].symbol as usize;
        let p0 = cache[i as usize].index;
        d += SaSint::from(p0 < 0);
        cache[i as usize].symbol = buckets[v0];
        buckets[v0] += 1;
        cache[i as usize].index =
            (p0 - 1) | (((buckets[2 + v0] != d) as SaSint) << (SAINT_BIT - 1));
        buckets[2 + v0] = d;
        if cache[i as usize].symbol < omp_block_end {
            let s = cache[i as usize].symbol as usize;
            let q = cache[i as usize].index & SAINT_MAX;
            cache[s].index = cache[i as usize].index;
            cache[s].symbol = buckets_index4(
                t[(q - 1) as usize] as usize,
                usize::from(t[(q - 2) as usize] >= t[(q - 1) as usize]),
            ) as SaSint;
        }

        let v1 = cache[(i + 1) as usize].symbol as usize;
        let p1 = cache[(i + 1) as usize].index;
        d += SaSint::from(p1 < 0);
        cache[(i + 1) as usize].symbol = buckets[v1];
        buckets[v1] += 1;
        cache[(i + 1) as usize].index =
            (p1 - 1) | (((buckets[2 + v1] != d) as SaSint) << (SAINT_BIT - 1));
        buckets[2 + v1] = d;
        if cache[(i + 1) as usize].symbol < omp_block_end {
            let s = cache[(i + 1) as usize].symbol as usize;
            let q = cache[(i + 1) as usize].index & SAINT_MAX;
            cache[s].index = cache[(i + 1) as usize].index;
            cache[s].symbol = buckets_index4(
                t[(q - 1) as usize] as usize,
                usize::from(t[(q - 2) as usize] >= t[(q - 1) as usize]),
            ) as SaSint;
        }

        i += 2;
    }

    j += 64 + 1;
    while i < j {
        let v = cache[i as usize].symbol as usize;
        let p = cache[i as usize].index;
        d += SaSint::from(p < 0);
        cache[i as usize].symbol = buckets[v];
        buckets[v] += 1;
        cache[i as usize].index = (p - 1) | (((buckets[2 + v] != d) as SaSint) << (SAINT_BIT - 1));
        buckets[2 + v] = d;
        if cache[i as usize].symbol < omp_block_end {
            let s = cache[i as usize].symbol as usize;
            let q = cache[i as usize].index & SAINT_MAX;
            cache[s].index = cache[i as usize].index;
            cache[s].symbol = buckets_index4(
                t[(q - 1) as usize] as usize,
                usize::from(t[(q - 2) as usize] >= t[(q - 1) as usize]),
            ) as SaSint;
        }
        i += 1;
    }

    d
}

fn partial_sorting_scan_left_to_right_32s_4k_block_sort(
    t: &[SaSint],
    k: SaSint,
    buckets: &mut [SaSint],
    mut d: SaSint,
    cache: &mut [ThreadCache],
    omp_block_start: SaSint,
    omp_block_size: SaSint,
) -> SaSint {
    let k = k as usize;
    let mut i = omp_block_start;
    let omp_block_end = omp_block_start + omp_block_size;
    let mut j = omp_block_end - 64 - 1;

    while i < j {
        for current in [i, i + 1] {
            let v = cache[current as usize].symbol;
            if v >= 0 {
                let p = cache[current as usize].index;
                d += p >> (SUFFIX_GROUP_BIT - 1);
                let bucket_index = (v >> 1) as usize;
                let v_usize = v as usize;
                cache[current as usize].symbol = buckets[2 * k + bucket_index];
                buckets[2 * k + bucket_index] += 1;
                cache[current as usize].index = (p - 1)
                    | ((v & 1) << (SAINT_BIT - 1))
                    | (((buckets[v_usize] != d) as SaSint) << (SUFFIX_GROUP_BIT - 1));
                buckets[v_usize] = d;
                if cache[current as usize].symbol < omp_block_end {
                    let ni = cache[current as usize].symbol as usize;
                    let mut np = cache[current as usize].index;
                    if np > 0 {
                        cache[ni].index = np;
                        np &= !SUFFIX_GROUP_MARKER;
                        cache[ni].symbol = buckets_index2(
                            t[(np - 1) as usize] as usize,
                            usize::from(t[(np - 2) as usize] < t[(np - 1) as usize]),
                        ) as SaSint;
                        np = 0;
                    }
                    cache[current as usize].index = np & SAINT_MAX;
                }
            }
        }
        i += 2;
    }

    j += 64 + 1;
    while i < j {
        let v = cache[i as usize].symbol;
        if v >= 0 {
            let p = cache[i as usize].index;
            d += p >> (SUFFIX_GROUP_BIT - 1);
            let bucket_index = (v >> 1) as usize;
            let v_usize = v as usize;
            cache[i as usize].symbol = buckets[2 * k + bucket_index];
            buckets[2 * k + bucket_index] += 1;
            cache[i as usize].index = (p - 1)
                | ((v & 1) << (SAINT_BIT - 1))
                | (((buckets[v_usize] != d) as SaSint) << (SUFFIX_GROUP_BIT - 1));
            buckets[v_usize] = d;
            if cache[i as usize].symbol < omp_block_end {
                let ni = cache[i as usize].symbol as usize;
                let mut np = cache[i as usize].index;
                if np > 0 {
                    cache[ni].index = np;
                    np &= !SUFFIX_GROUP_MARKER;
                    cache[ni].symbol = buckets_index2(
                        t[(np - 1) as usize] as usize,
                        usize::from(t[(np - 2) as usize] < t[(np - 1) as usize]),
                    ) as SaSint;
                    np = 0;
                }
                cache[i as usize].index = np & SAINT_MAX;
            }
        }
        i += 1;
    }

    d
}

fn partial_sorting_scan_left_to_right_32s_1k_block_sort(
    t: &[SaSint],
    induction_bucket: &mut [SaSint],
    cache: &mut [ThreadCache],
    omp_block_start: SaSint,
    omp_block_size: SaSint,
) {
    let mut i = omp_block_start;
    let omp_block_end = omp_block_start + omp_block_size;
    let mut j = omp_block_end - 64 - 1;

    while i < j {
        for current in [i, i + 1] {
            let v = cache[current as usize].symbol;
            if v >= 0 {
                cache[current as usize].symbol = induction_bucket[v as usize];
                induction_bucket[v as usize] += 1;
                if cache[current as usize].symbol < omp_block_end {
                    let ni = cache[current as usize].symbol as usize;
                    let mut np = cache[current as usize].index;
                    if np > 0 {
                        cache[ni].index = (np - 1)
                            | ((usize::from(t[(np - 2) as usize] < t[(np - 1) as usize])
                                as SaSint)
                                << (SAINT_BIT - 1));
                        cache[ni].symbol = t[(np - 1) as usize];
                        np = 0;
                    }
                    cache[current as usize].index = np & SAINT_MAX;
                }
            }
        }
        i += 2;
    }

    j = omp_block_end;
    while i < j {
        let v = cache[i as usize].symbol;
        if v >= 0 {
            cache[i as usize].symbol = induction_bucket[v as usize];
            induction_bucket[v as usize] += 1;
            if cache[i as usize].symbol < omp_block_end {
                let ni = cache[i as usize].symbol as usize;
                let mut np = cache[i as usize].index;
                if np > 0 {
                    cache[ni].index = (np - 1)
                        | ((usize::from(t[(np - 2) as usize] < t[(np - 1) as usize]) as SaSint)
                            << (SAINT_BIT - 1));
                    cache[ni].symbol = t[(np - 1) as usize];
                    np = 0;
                }
                cache[i as usize].index = np & SAINT_MAX;
            }
        }
        i += 1;
    }
}

fn partial_sorting_scan_right_to_left_32s_6k_block_sort(
    t: &[SaSint],
    buckets: &mut [SaSint],
    mut d: SaSint,
    cache: &mut [ThreadCache],
    omp_block_start: SaSint,
    omp_block_size: SaSint,
) -> SaSint {
    let mut i = omp_block_start + omp_block_size - 1;
    let mut j = omp_block_start + 64 + 1;

    while i >= j {
        let v0 = cache[i as usize].symbol as usize;
        let p0 = cache[i as usize].index;
        d += SaSint::from(p0 < 0);
        buckets[v0] -= 1;
        cache[i as usize].symbol = buckets[v0];
        cache[i as usize].index =
            (p0 - 1) | (((buckets[2 + v0] != d) as SaSint) << (SAINT_BIT - 1));
        buckets[2 + v0] = d;
        if cache[i as usize].symbol >= omp_block_start {
            let s = cache[i as usize].symbol as usize;
            let q = cache[i as usize].index & SAINT_MAX;
            cache[s].index = cache[i as usize].index;
            cache[s].symbol = buckets_index4(
                t[(q - 1) as usize] as usize,
                usize::from(t[(q - 2) as usize] > t[(q - 1) as usize]),
            ) as SaSint;
        }

        let v1 = cache[(i - 1) as usize].symbol as usize;
        let p1 = cache[(i - 1) as usize].index;
        d += SaSint::from(p1 < 0);
        buckets[v1] -= 1;
        cache[(i - 1) as usize].symbol = buckets[v1];
        cache[(i - 1) as usize].index =
            (p1 - 1) | (((buckets[2 + v1] != d) as SaSint) << (SAINT_BIT - 1));
        buckets[2 + v1] = d;
        if cache[(i - 1) as usize].symbol >= omp_block_start {
            let s = cache[(i - 1) as usize].symbol as usize;
            let q = cache[(i - 1) as usize].index & SAINT_MAX;
            cache[s].index = cache[(i - 1) as usize].index;
            cache[s].symbol = buckets_index4(
                t[(q - 1) as usize] as usize,
                usize::from(t[(q - 2) as usize] > t[(q - 1) as usize]),
            ) as SaSint;
        }

        i -= 2;
    }

    j -= 64 + 1;
    while i >= j {
        let v = cache[i as usize].symbol as usize;
        let p = cache[i as usize].index;
        d += SaSint::from(p < 0);
        buckets[v] -= 1;
        cache[i as usize].symbol = buckets[v];
        cache[i as usize].index = (p - 1) | (((buckets[2 + v] != d) as SaSint) << (SAINT_BIT - 1));
        buckets[2 + v] = d;
        if cache[i as usize].symbol >= omp_block_start {
            let s = cache[i as usize].symbol as usize;
            let q = cache[i as usize].index & SAINT_MAX;
            cache[s].index = cache[i as usize].index;
            cache[s].symbol = buckets_index4(
                t[(q - 1) as usize] as usize,
                usize::from(t[(q - 2) as usize] > t[(q - 1) as usize]),
            ) as SaSint;
        }
        i -= 1;
    }

    d
}

fn partial_sorting_scan_right_to_left_32s_4k_block_sort(
    t: &[SaSint],
    k: SaSint,
    buckets: &mut [SaSint],
    mut d: SaSint,
    cache: &mut [ThreadCache],
    omp_block_start: SaSint,
    omp_block_size: SaSint,
) -> SaSint {
    let k = k as usize;
    let mut i = omp_block_start + omp_block_size - 1;
    let mut j = omp_block_start + 64 + 1;

    while i >= j {
        for current in [i, i - 1] {
            let v = cache[current as usize].symbol;
            if v >= 0 {
                let p = cache[current as usize].index;
                d += p >> (SUFFIX_GROUP_BIT - 1);
                let bucket_index = (v >> 1) as usize;
                let v_usize = v as usize;
                buckets[3 * k + bucket_index] -= 1;
                cache[current as usize].symbol = buckets[3 * k + bucket_index];
                cache[current as usize].index = (p - 1)
                    | ((v & 1) << (SAINT_BIT - 1))
                    | (((buckets[v_usize] != d) as SaSint) << (SUFFIX_GROUP_BIT - 1));
                buckets[v_usize] = d;
                if cache[current as usize].symbol >= omp_block_start {
                    let ni = cache[current as usize].symbol as usize;
                    let mut np = cache[current as usize].index;
                    if np > 0 {
                        cache[current as usize].index = 0;
                        cache[ni].index = np;
                        np &= !SUFFIX_GROUP_MARKER;
                        cache[ni].symbol = buckets_index2(
                            t[(np - 1) as usize] as usize,
                            usize::from(t[(np - 2) as usize] > t[(np - 1) as usize]),
                        ) as SaSint;
                    }
                }
            }
        }
        i -= 2;
    }

    j -= 64 + 1;
    while i >= j {
        let v = cache[i as usize].symbol;
        if v >= 0 {
            let p = cache[i as usize].index;
            d += p >> (SUFFIX_GROUP_BIT - 1);
            let bucket_index = (v >> 1) as usize;
            let v_usize = v as usize;
            buckets[3 * k + bucket_index] -= 1;
            cache[i as usize].symbol = buckets[3 * k + bucket_index];
            cache[i as usize].index = (p - 1)
                | ((v & 1) << (SAINT_BIT - 1))
                | (((buckets[v_usize] != d) as SaSint) << (SUFFIX_GROUP_BIT - 1));
            buckets[v_usize] = d;
            if cache[i as usize].symbol >= omp_block_start {
                let ni = cache[i as usize].symbol as usize;
                let mut np = cache[i as usize].index;
                if np > 0 {
                    cache[i as usize].index = 0;
                    cache[ni].index = np;
                    np &= !SUFFIX_GROUP_MARKER;
                    cache[ni].symbol = buckets_index2(
                        t[(np - 1) as usize] as usize,
                        usize::from(t[(np - 2) as usize] > t[(np - 1) as usize]),
                    ) as SaSint;
                }
            }
        }
        i -= 1;
    }

    d
}

fn partial_sorting_scan_right_to_left_32s_1k_block_sort(
    t: &[SaSint],
    induction_bucket: &mut [SaSint],
    cache: &mut [ThreadCache],
    omp_block_start: SaSint,
    omp_block_size: SaSint,
) {
    let mut i = omp_block_start + omp_block_size - 1;
    let mut j = omp_block_start + 64 + 1;

    while i >= j {
        for current in [i, i - 1] {
            let v = cache[current as usize].symbol;
            if v >= 0 {
                induction_bucket[v as usize] -= 1;
                cache[current as usize].symbol = induction_bucket[v as usize];
                if cache[current as usize].symbol >= omp_block_start {
                    let ni = cache[current as usize].symbol as usize;
                    let np = cache[current as usize].index;
                    if np > 0 {
                        cache[current as usize].index = 0;
                        cache[ni].index = (np - 1)
                            | ((usize::from(t[(np - 2) as usize] > t[(np - 1) as usize])
                                as SaSint)
                                << (SAINT_BIT - 1));
                        cache[ni].symbol = t[(np - 1) as usize];
                    }
                }
            }
        }
        i -= 2;
    }

    j -= 64 + 1;
    while i >= j {
        let v = cache[i as usize].symbol;
        if v >= 0 {
            induction_bucket[v as usize] -= 1;
            cache[i as usize].symbol = induction_bucket[v as usize];
            if cache[i as usize].symbol >= omp_block_start {
                let ni = cache[i as usize].symbol as usize;
                let np = cache[i as usize].index;
                if np > 0 {
                    cache[i as usize].index = 0;
                    cache[ni].index = (np - 1)
                        | ((usize::from(t[(np - 2) as usize] > t[(np - 1) as usize]) as SaSint)
                            << (SAINT_BIT - 1));
                    cache[ni].symbol = t[(np - 1) as usize];
                }
            }
        }
        i -= 1;
    }
}

fn partial_sorting_scan_left_to_right_32s_6k_block_omp(
    t: &[SaSint],
    sa: &mut [SaSint],
    buckets: &mut [SaSint],
    d: SaSint,
    cache: &mut [ThreadCache],
    block_start: SaSint,
    block_size: SaSint,
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
                    let mut omp_block_size = if omp_thread_num + 1 < omp_num_threads {
                        omp_block_stride
                    } else {
                        block_size_usize - omp_thread_num * omp_block_stride
                    };
                    let omp_block_start = block_start_usize + omp_thread_num * omp_block_stride;
                    if omp_block_size == 0 {
                        omp_block_size = block_size_usize - (omp_block_start - block_start_usize);
                    }
                    let cache = unsafe { cache_ptr.as_slice() };
                    let sa = unsafe { sa_ptr.as_slice() };
                    partial_sorting_scan_left_to_right_32s_6k_block_gather(
                        t_ro,
                        sa,
                        &mut cache[omp_thread_num * omp_block_stride
                            ..omp_thread_num * omp_block_stride + omp_block_size],
                        omp_block_start as SaSint,
                        omp_block_size as SaSint,
                    );
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
    place_cached_suffixes(sa, &cache[..block_size_usize], 0, block_size);
    d
}

fn partial_sorting_scan_left_to_right_32s_4k_block_omp(
    t: &[SaSint],
    sa: &mut [SaSint],
    k: SaSint,
    buckets: &mut [SaSint],
    d: SaSint,
    cache: &mut [ThreadCache],
    block_start: SaSint,
    block_size: SaSint,
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
                    let mut omp_block_size = if omp_thread_num + 1 < omp_num_threads {
                        omp_block_stride
                    } else {
                        block_size_usize - omp_thread_num * omp_block_stride
                    };
                    let omp_block_start = block_start_usize + omp_thread_num * omp_block_stride;
                    if omp_block_size == 0 {
                        omp_block_size = block_size_usize - (omp_block_start - block_start_usize);
                    }
                    let cache = unsafe { cache_ptr.as_slice() };
                    let sa = unsafe { sa_ptr.as_slice() };
                    partial_sorting_scan_left_to_right_32s_4k_block_gather(
                        t_ro,
                        sa,
                        &mut cache[omp_thread_num * omp_block_stride
                            ..omp_thread_num * omp_block_stride + omp_block_size],
                        omp_block_start as SaSint,
                        omp_block_size as SaSint,
                    );
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
    for entry in cache.iter() {
        if entry.symbol >= 0 {
            sa[entry.symbol as usize] = entry.index;
        }
    }
    d
}

fn partial_sorting_scan_left_to_right_32s_1k_block_omp(
    t: &[SaSint],
    sa: &mut [SaSint],
    buckets: &mut [SaSint],
    cache: &mut [ThreadCache],
    block_start: SaSint,
    block_size: SaSint,
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
                    let mut omp_block_size = if omp_thread_num + 1 < omp_num_threads {
                        omp_block_stride
                    } else {
                        block_size_usize - omp_thread_num * omp_block_stride
                    };
                    let omp_block_start = block_start_usize + omp_thread_num * omp_block_stride;
                    if omp_block_size == 0 {
                        omp_block_size = block_size_usize - (omp_block_start - block_start_usize);
                    }
                    let cache = unsafe { cache_ptr.as_slice() };
                    let sa = unsafe { sa_ptr.as_slice() };
                    partial_sorting_scan_left_to_right_32s_1k_block_gather(
                        t_ro,
                        sa,
                        &mut cache[omp_thread_num * omp_block_stride
                            ..omp_thread_num * omp_block_stride + omp_block_size],
                        omp_block_start as SaSint,
                        omp_block_size as SaSint,
                    );
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
    compact_and_place_cached_suffixes(sa, cache, block_start, block_size);
}

fn partial_sorting_scan_right_to_left_32s_6k_block_omp(
    t: &[SaSint],
    sa: &mut [SaSint],
    buckets: &mut [SaSint],
    mut d: SaSint,
    cache: &mut [ThreadCache],
    block_start: SaSint,
    block_size: SaSint,
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
                    let mut omp_block_size = if omp_thread_num + 1 < omp_num_threads {
                        omp_block_stride
                    } else {
                        block_size_usize - omp_thread_num * omp_block_stride
                    };
                    let omp_block_start = block_start_usize + omp_thread_num * omp_block_stride;
                    if omp_block_size == 0 {
                        omp_block_size = block_size_usize - (omp_block_start - block_start_usize);
                    }
                    let cache = unsafe { cache_ptr.as_slice() };
                    let sa = unsafe { sa_ptr.as_slice() };
                    partial_sorting_scan_right_to_left_32s_6k_block_gather(
                        t_ro,
                        sa,
                        &mut cache[omp_thread_num * omp_block_stride
                            ..omp_thread_num * omp_block_stride + omp_block_size],
                        omp_block_start as SaSint,
                        omp_block_size as SaSint,
                    );
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
                    let mut omp_block_size = if omp_thread_num + 1 < omp_num_threads {
                        omp_block_stride
                    } else {
                        block_size_usize - omp_thread_num * omp_block_stride
                    };
                    let cache_start = omp_thread_num * omp_block_stride;
                    if omp_block_size == 0 {
                        omp_block_size = block_size_usize - cache_start;
                    }
                    let sa = unsafe { sa_ptr.as_slice() };
                    for entry in &cache_ro[cache_start..cache_start + omp_block_size] {
                        sa[entry.symbol as usize] = entry.index;
                    }
                });
        });
    }
    d
}

fn partial_sorting_scan_right_to_left_32s_4k_block_omp(
    t: &[SaSint],
    sa: &mut [SaSint],
    k: SaSint,
    buckets: &mut [SaSint],
    mut d: SaSint,
    cache: &mut [ThreadCache],
    block_start: SaSint,
    block_size: SaSint,
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
                    let mut omp_block_size = if omp_thread_num + 1 < omp_num_threads {
                        omp_block_stride
                    } else {
                        block_size_usize - omp_thread_num * omp_block_stride
                    };
                    let omp_block_start = block_start_usize + omp_thread_num * omp_block_stride;
                    if omp_block_size == 0 {
                        omp_block_size = block_size_usize - (omp_block_start - block_start_usize);
                    }
                    let cache = unsafe { cache_ptr.as_slice() };
                    let sa = unsafe { sa_ptr.as_slice() };
                    partial_sorting_scan_right_to_left_32s_4k_block_gather(
                        t_ro,
                        sa,
                        &mut cache[omp_thread_num * omp_block_stride
                            ..omp_thread_num * omp_block_stride + omp_block_size],
                        omp_block_start as SaSint,
                        omp_block_size as SaSint,
                    );
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
    let mut write = 0usize;
    for read in 0..block_size_usize {
        let entry = cache[read];
        if entry.symbol >= 0 {
            cache[write] = entry;
            write += 1;
        }
    }
    for entry in &cache[..write] {
        sa[entry.symbol as usize] = entry.index;
    }
    d
}

fn partial_sorting_scan_right_to_left_32s_1k_block_omp(
    t: &[SaSint],
    sa: &mut [SaSint],
    buckets: &mut [SaSint],
    cache: &mut [ThreadCache],
    block_start: SaSint,
    block_size: SaSint,
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
                    let mut omp_block_size = if omp_thread_num + 1 < omp_num_threads {
                        omp_block_stride
                    } else {
                        block_size_usize - omp_thread_num * omp_block_stride
                    };
                    let omp_block_start = block_start_usize + omp_thread_num * omp_block_stride;
                    if omp_block_size == 0 {
                        omp_block_size = block_size_usize - (omp_block_start - block_start_usize);
                    }
                    let cache = unsafe { cache_ptr.as_slice() };
                    let sa = unsafe { sa_ptr.as_slice() };
                    partial_sorting_scan_right_to_left_32s_1k_block_gather(
                        t_ro,
                        sa,
                        &mut cache[omp_thread_num * omp_block_stride
                            ..omp_thread_num * omp_block_stride + omp_block_size],
                        omp_block_start as SaSint,
                        omp_block_size as SaSint,
                    );
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
    compact_and_place_cached_suffixes(sa, cache, block_start, block_size);
}

fn partial_sorting_gather_lms_suffixes_32s_4k(
    sa: &mut [SaSint],
    omp_block_start: SaSint,
    omp_block_size: SaSint,
) -> SaSint {
    let mut i = omp_block_start;
    let mut j = omp_block_start + omp_block_size - 3;
    let mut l = omp_block_start;

    while i < j {
        let s0 = sa[i as usize] as SaUint;
        sa[l as usize] = (s0.wrapping_sub(SUFFIX_GROUP_MARKER as SaUint)
            & !(SUFFIX_GROUP_MARKER as SaUint)) as SaSint;
        l += SaSint::from((s0 as SaSint) < 0);

        let s1 = sa[(i + 1) as usize] as SaUint;
        sa[l as usize] = (s1.wrapping_sub(SUFFIX_GROUP_MARKER as SaUint)
            & !(SUFFIX_GROUP_MARKER as SaUint)) as SaSint;
        l += SaSint::from((s1 as SaSint) < 0);

        let s2 = sa[(i + 2) as usize] as SaUint;
        sa[l as usize] = (s2.wrapping_sub(SUFFIX_GROUP_MARKER as SaUint)
            & !(SUFFIX_GROUP_MARKER as SaUint)) as SaSint;
        l += SaSint::from((s2 as SaSint) < 0);

        let s3 = sa[(i + 3) as usize] as SaUint;
        sa[l as usize] = (s3.wrapping_sub(SUFFIX_GROUP_MARKER as SaUint)
            & !(SUFFIX_GROUP_MARKER as SaUint)) as SaSint;
        l += SaSint::from((s3 as SaSint) < 0);

        i += 4;
    }

    j += 3;
    while i < j {
        let s = sa[i as usize] as SaUint;
        sa[l as usize] = (s.wrapping_sub(SUFFIX_GROUP_MARKER as SaUint)
            & !(SUFFIX_GROUP_MARKER as SaUint)) as SaSint;
        l += SaSint::from((s as SaSint) < 0);
        i += 1;
    }

    l
}

fn partial_sorting_gather_lms_suffixes_32s_1k(
    sa: &mut [SaSint],
    omp_block_start: SaSint,
    omp_block_size: SaSint,
) -> SaSint {
    let mut i = omp_block_start;
    let mut j = omp_block_start + omp_block_size - 3;
    let mut l = omp_block_start;

    while i < j {
        let s0 = sa[i as usize];
        sa[l as usize] = s0 & SAINT_MAX;
        l += SaSint::from(s0 < 0);

        let s1 = sa[(i + 1) as usize];
        sa[l as usize] = s1 & SAINT_MAX;
        l += SaSint::from(s1 < 0);

        let s2 = sa[(i + 2) as usize];
        sa[l as usize] = s2 & SAINT_MAX;
        l += SaSint::from(s2 < 0);

        let s3 = sa[(i + 3) as usize];
        sa[l as usize] = s3 & SAINT_MAX;
        l += SaSint::from(s3 < 0);

        i += 4;
    }

    j += 3;
    while i < j {
        let s = sa[i as usize];
        sa[l as usize] = s & SAINT_MAX;
        l += SaSint::from(s < 0);
        i += 1;
    }

    l
}

fn partial_sorting_gather_lms_suffixes_32s_4k_omp(
    sa: &mut [SaSint],
    n: SaSint,
    threads: SaSint,
    thread_state: &mut [ThreadState],
) {
    let n_usize = usize::try_from(n).expect("n must be non-negative");
    let thread_count = if threads > 1 && n >= 65_536 {
        usize::try_from(threads)
            .expect("threads must be non-negative")
            .min(thread_state.len())
            .max(1)
    } else {
        1
    };

    if thread_count == 1 {
        let _ = partial_sorting_gather_lms_suffixes_32s_4k(sa, 0, n);
        return;
    }

    let block_stride = (n_usize / thread_count) & !15usize;
    for (thread, state) in thread_state.iter_mut().take(thread_count).enumerate() {
        let block_start = thread * block_stride;
        let block_size = if thread + 1 < thread_count {
            block_stride
        } else {
            n_usize - block_start
        };
        state.position = block_start as SaSint;
        state.count = partial_sorting_gather_lms_suffixes_32s_4k(
            sa,
            block_start as SaSint,
            block_size as SaSint,
        ) - block_start as SaSint;
    }

    let mut position = 0usize;
    for (thread, state) in thread_state.iter().take(thread_count).enumerate() {
        let count = usize::try_from(state.count).expect("count must be non-negative");
        let src = usize::try_from(state.position).expect("position must be non-negative");
        if thread > 0 && count > 0 {
            sa.copy_within(src..src + count, position);
        }
        position += count;
    }
}

fn partial_sorting_gather_lms_suffixes_32s_1k_omp(
    sa: &mut [SaSint],
    n: SaSint,
    threads: SaSint,
    thread_state: &mut [ThreadState],
) {
    let n_usize = usize::try_from(n).expect("n must be non-negative");
    let thread_count = if threads > 1 && n >= 65_536 {
        usize::try_from(threads)
            .expect("threads must be non-negative")
            .min(thread_state.len())
            .max(1)
    } else {
        1
    };

    if thread_count == 1 {
        let _ = partial_sorting_gather_lms_suffixes_32s_1k(sa, 0, n);
        return;
    }

    let block_stride = (n_usize / thread_count) & !15usize;
    for (thread, state) in thread_state.iter_mut().take(thread_count).enumerate() {
        let block_start = thread * block_stride;
        let block_size = if thread + 1 < thread_count {
            block_stride
        } else {
            n_usize - block_start
        };
        state.position = block_start as SaSint;
        state.count = partial_sorting_gather_lms_suffixes_32s_1k(
            sa,
            block_start as SaSint,
            block_size as SaSint,
        ) - block_start as SaSint;
    }

    let mut position = 0usize;
    for (thread, state) in thread_state.iter().take(thread_count).enumerate() {
        let count = usize::try_from(state.count).expect("count must be non-negative");
        let src = usize::try_from(state.position).expect("position must be non-negative");
        if thread > 0 && count > 0 {
            sa.copy_within(src..src + count, position);
        }
        position += count;
    }
}

fn partial_gsa_scan_right_to_left_16u(
    t: &[u16],
    sa: &mut [SaSint],
    buckets: &mut [SaSint],
    mut d: SaSint,
    omp_block_start: SaSint,
    omp_block_size: SaSint,
) -> SaSint {
    let mut i = (omp_block_start + omp_block_size - 1) as isize;
    let mut j = (omp_block_start + 64 + 1) as isize;
    while i >= j {
        let mut p0 = sa[i as usize];
        d += SaSint::from(p0 < 0);
        p0 &= SAINT_MAX;
        let v0 = buckets_index2(
            t[(p0 - 1) as usize] as usize,
            usize::from(t[(p0 - 2) as usize] > t[(p0 - 1) as usize]),
        );
        if v0 != 1 {
            let mark0 = if buckets[2 * ALPHABET_SIZE + v0] != d {
                SAINT_MIN
            } else {
                0
            };
            buckets[v0] -= 1;
            sa[buckets[v0] as usize] = (p0 - 1) | mark0;
            buckets[2 * ALPHABET_SIZE + v0] = d;
        }

        let mut p1 = sa[(i - 1) as usize];
        d += SaSint::from(p1 < 0);
        p1 &= SAINT_MAX;
        let v1 = buckets_index2(
            t[(p1 - 1) as usize] as usize,
            usize::from(t[(p1 - 2) as usize] > t[(p1 - 1) as usize]),
        );
        if v1 != 1 {
            let mark1 = if buckets[2 * ALPHABET_SIZE + v1] != d {
                SAINT_MIN
            } else {
                0
            };
            buckets[v1] -= 1;
            sa[buckets[v1] as usize] = (p1 - 1) | mark1;
            buckets[2 * ALPHABET_SIZE + v1] = d;
        }

        i -= 2;
    }

    j -= 64 + 1;
    while i >= j {
        let mut p = sa[i as usize];
        d += SaSint::from(p < 0);
        p &= SAINT_MAX;
        let v = buckets_index2(
            t[(p - 1) as usize] as usize,
            usize::from(t[(p - 2) as usize] > t[(p - 1) as usize]),
        );
        if v != 1 {
            let mark = if buckets[2 * ALPHABET_SIZE + v] != d {
                SAINT_MIN
            } else {
                0
            };
            buckets[v] -= 1;
            sa[buckets[v] as usize] = (p - 1) | mark;
            buckets[2 * ALPHABET_SIZE + v] = d;
        }
        i -= 1;
    }

    d
}

fn partial_gsa_scan_right_to_left_16u_block_omp(
    t: &[u16],
    sa: &mut [SaSint],
    k: SaSint,
    buckets: &mut [SaSint],
    d: SaSint,
    block_start: SaSint,
    block_size: SaSint,
    threads: SaSint,
    thread_state: &mut [ThreadState],
) -> SaSint {
    let thread_count = if threads > 1 && block_size >= 64 * k.max(256) {
        usize::try_from(threads)
            .expect("threads must be non-negative")
            .min(thread_state.len())
    } else {
        1
    };
    if thread_count <= 1 {
        return partial_gsa_scan_right_to_left_16u(t, sa, buckets, d, block_start, block_size);
    }

    let width = 2 * k as usize;
    let distinct_offset = 2 * ALPHABET_SIZE;
    let block_stride = (block_size / thread_count as SaSint) & !15;

    for (thread, state) in thread_state.iter_mut().take(thread_count).enumerate() {
        let local_start = thread as SaSint * block_stride;
        let local_size = if thread + 1 < thread_count {
            block_stride
        } else {
            block_size - local_start
        };
        let mut local_state = ThreadState::default();
        state.position = partial_sorting_scan_right_to_left_16u_block_prepare(
            t,
            sa,
            k,
            &mut state.buckets,
            &mut state.cache,
            block_start + local_start,
            local_size,
            &mut local_state,
        );
        state.count = local_state.cache_entries as SaSint;
    }

    let mut next_d = d;
    for state in thread_state.iter_mut().take(thread_count).rev() {
        for c in 0..width {
            let a = buckets[c];
            let b = state.buckets[c];
            buckets[c] = a - b;
            state.buckets[c] = a;
        }

        next_d -= 1;
        for c in 0..width {
            let offset = distinct_offset + c;
            let a = buckets[offset];
            let b = state.buckets[offset];
            let shifted = b + next_d;
            buckets[offset] = if b > 0 { shifted } else { a };
            state.buckets[offset] = a;
        }
        next_d += 1 + state.position;
        state.position = next_d - state.position;
    }

    for state in thread_state.iter_mut().take(thread_count) {
        partial_gsa_scan_right_to_left_16u_block_place(
            sa,
            &mut state.buckets,
            &state.cache,
            state.count,
            state.position,
        );
    }

    next_d
}

fn partial_gsa_scan_right_to_left_16u_omp(
    t: &[u16],
    sa: &mut [SaSint],
    n: SaSint,
    k: SaSint,
    buckets: &mut [SaSint],
    first_lms_suffix: SaSint,
    left_suffixes_count: SaSint,
    d: SaSint,
    threads: SaSint,
) {
    let scan_start = left_suffixes_count + 1;
    let scan_end = n - first_lms_suffix;

    if threads == 1 || scan_end - scan_start < 65536 {
        partial_gsa_scan_right_to_left_16u(t, sa, buckets, d, scan_start, scan_end - scan_start);
    } else {
        let mut d = d;
        let mut thread_state = alloc_thread_state(threads).unwrap_or_default();
        let mut block_start = scan_end - 1;
        while block_start >= scan_start {
            if sa[block_start as usize] == 0 {
                block_start -= 1;
            } else {
                let block_limit = threads * (PER_THREAD_CACHE_SIZE as SaSint - 16 * threads);
                let mut block_max_end = block_start - block_limit;
                if block_max_end < scan_start {
                    block_max_end = scan_start - 1;
                }
                let mut block_end = block_start - 1;
                while block_end > block_max_end && sa[block_end as usize] != 0 {
                    block_end -= 1;
                }
                let block_size = block_start - block_end;

                if block_size < 32 {
                    while block_start > block_end {
                        let mut p = sa[block_start as usize];
                        d += SaSint::from(p < 0);
                        p &= SAINT_MAX;
                        let v = buckets_index2(
                            t[(p - 1) as usize] as usize,
                            usize::from(t[(p - 2) as usize] > t[(p - 1) as usize]),
                        );
                        if v != 1 {
                            let mark = if buckets[2 * ALPHABET_SIZE + v] != d {
                                SAINT_MIN
                            } else {
                                0
                            };
                            buckets[v] -= 1;
                            sa[buckets[v] as usize] = (p - 1) | mark;
                            buckets[2 * ALPHABET_SIZE + v] = d;
                        }
                        block_start -= 1;
                    }
                } else {
                    d = partial_gsa_scan_right_to_left_16u_block_omp(
                        t,
                        sa,
                        k,
                        buckets,
                        d,
                        block_end + 1,
                        block_size,
                        threads,
                        &mut thread_state,
                    );
                    block_start = block_end;
                }
            }
        }
    }
}

fn partial_sorting_shift_markers_16u_omp(
    sa: &mut [SaSint],
    n: SaSint,
    buckets: &[SaSint],
    threads: SaSint,
) {
    let thread_count = if threads > 1 && n >= 65536 {
        usize::try_from(threads).expect("threads must be positive")
    } else {
        1
    };
    let c_step = buckets_index2(1, 0) as isize;
    let c_min = buckets_index2(1, 0) as isize;
    let c_max = buckets_index2(ALPHABET_SIZE - 1, 0) as isize;
    let sa_ptr = SyncMutPtr::new(sa);
    let buckets_ref: &[SaSint] = buckets;
    run_rayon_with_threads(thread_count, || {
        (0..thread_count).into_par_iter().for_each(|t| {
            let mut c = c_max - (t as isize * c_step);
            let sa = unsafe { sa_ptr.as_slice() };
            while c >= c_min {
                let c_usize = c as usize;
                let mut s = SAINT_MIN;
                let mut i = buckets_ref[4 * ALPHABET_SIZE + c_usize] as isize - 1;
                let mut j = buckets_ref[c_usize - buckets_index2(1, 0)] as isize + 3;
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

fn induce_partial_order_16u_omp(
    t: &[u16],
    sa: &mut [SaSint],
    n: SaSint,
    k: SaSint,
    flags: SaSint,
    buckets: &mut [SaSint],
    first_lms_suffix: SaSint,
    left_suffixes_count: SaSint,
    threads: SaSint,
) {
    buckets[2 * ALPHABET_SIZE..4 * ALPHABET_SIZE].fill(0);

    if (flags & LIBSAIS_FLAGS_GSA) != 0 {
        let marker = 4 * ALPHABET_SIZE + buckets_index2(0, 1);
        buckets[marker] = buckets[4 * ALPHABET_SIZE + buckets_index2(1, 1)] - 1;
        flip_suffix_markers_omp(sa, buckets[marker], threads);
    }

    let d = partial_sorting_scan_left_to_right_16u_omp(
        t,
        sa,
        n,
        k,
        buckets,
        left_suffixes_count,
        0,
        threads,
    );
    partial_sorting_shift_markers_16u_omp(sa, n, buckets, threads);

    if (flags & LIBSAIS_FLAGS_GSA) != 0 {
        partial_gsa_scan_right_to_left_16u_omp(
            t,
            sa,
            n,
            k,
            buckets,
            first_lms_suffix,
            left_suffixes_count,
            d,
            threads,
        );

        if t[first_lms_suffix as usize] == 0 {
            let count = (buckets[buckets_index2(1, 1)] - 1) as usize;
            sa.copy_within(0..count, 1);
            sa[0] = first_lms_suffix | SAINT_MIN;
        }

        buckets[buckets_index2(0, 1)] = 0;
    } else {
        partial_sorting_scan_right_to_left_16u_omp(
            t,
            sa,
            n,
            k,
            buckets,
            first_lms_suffix,
            left_suffixes_count,
            d,
            threads,
        );
    }
}

fn induce_partial_order_32s_6k_omp(
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
    partial_sorting_scan_right_to_left_32s_6k_omp(
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

fn induce_partial_order_32s_4k_omp(
    t: &[SaSint],
    sa: &mut [SaSint],
    n: SaSint,
    k: SaSint,
    buckets: &mut [SaSint],
    threads: SaSint,
    thread_state: &mut [ThreadState],
) {
    buckets[..2 * k as usize].fill(0);
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
    partial_sorting_scan_right_to_left_32s_4k_omp(t, sa, n, k, buckets, d, threads, thread_state);
    partial_sorting_gather_lms_suffixes_32s_4k_omp(sa, n, threads, thread_state);
}

fn induce_partial_order_32s_2k_omp(
    t: &[SaSint],
    sa: &mut [SaSint],
    n: SaSint,
    k: SaSint,
    buckets: &mut [SaSint],
    threads: SaSint,
    thread_state: &mut [ThreadState],
) {
    let k = k as usize;
    let (left, right) = buckets.split_at_mut(k);
    partial_sorting_scan_left_to_right_32s_1k_omp(t, sa, n, right, threads, thread_state);
    partial_sorting_scan_right_to_left_32s_1k_omp(t, sa, n, left, threads, thread_state);
    partial_sorting_gather_lms_suffixes_32s_1k_omp(sa, n, threads, thread_state);
}

fn induce_partial_order_32s_1k_omp(
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

fn final_sorting_scan_left_to_right_16u(
    t: &[u16],
    sa: &mut [SaSint],
    induction_bucket: &mut [SaSint],
    omp_block_start: SaSint,
    omp_block_size: SaSint,
) {
    let mut i = omp_block_start as isize;
    let mut j = (omp_block_start + omp_block_size - 64 - 1) as isize;
    while i < j {
        final_sorting_ltr_step(t, sa, induction_bucket, i as usize);
        final_sorting_ltr_step(t, sa, induction_bucket, (i + 1) as usize);
        i += 2;
    }
    j += 64 + 1;
    while i < j {
        final_sorting_ltr_step(t, sa, induction_bucket, i as usize);
        i += 1;
    }
}

fn final_sorting_scan_right_to_left_16u(
    t: &[u16],
    sa: &mut [SaSint],
    induction_bucket: &mut [SaSint],
    omp_block_start: SaSint,
    omp_block_size: SaSint,
) {
    let mut i = (omp_block_start + omp_block_size - 1) as isize;
    let mut j = (omp_block_start + 64 + 1) as isize;
    while i >= j {
        final_sorting_rtl_step(t, sa, induction_bucket, i as usize, false);
        final_sorting_rtl_step(t, sa, induction_bucket, (i - 1) as usize, false);
        i -= 2;
    }
    j -= 64 + 1;
    while i >= j {
        final_sorting_rtl_step(t, sa, induction_bucket, i as usize, false);
        i -= 1;
    }
}

fn final_sorting_scan_left_to_right_32s(
    t: &[SaSint],
    sa: &mut [SaSint],
    induction_bucket: &mut [SaSint],
    omp_block_start: SaSint,
    omp_block_size: SaSint,
) {
    let mut i = omp_block_start as isize;
    let mut j = (omp_block_start + omp_block_size - 2 * 64 - 1) as isize;
    while i < j {
        for current in [i, i + 1] {
            let current = current as usize;
            let mut p = sa[current];
            sa[current] = p ^ SAINT_MIN;
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
        }
        i += 2;
    }

    j += 2 * 64 + 1;
    while i < j {
        let current = i as usize;
        let mut p = sa[current];
        sa[current] = p ^ SAINT_MIN;
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

fn final_sorting_scan_left_to_right_32s_block_gather(
    t: &[SaSint],
    sa: &mut [SaSint],
    cache: &mut [ThreadCache],
    omp_block_start: SaSint,
    omp_block_size: SaSint,
) {
    if omp_block_size <= 0 {
        return;
    }

    let start = usize::try_from(omp_block_start).expect("omp_block_start must be non-negative");
    let size = usize::try_from(omp_block_size).expect("omp_block_size must be non-negative");
    for offset in 0..size {
        let current = start + offset;
        let mut symbol = SAINT_MIN;
        let mut p = sa[current];
        sa[current] = p ^ SAINT_MIN;
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

fn final_sorting_scan_left_to_right_32s_block_sort(
    t: &[SaSint],
    induction_bucket: &mut [SaSint],
    cache: &mut [ThreadCache],
    omp_block_start: SaSint,
    omp_block_size: SaSint,
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
            let bucket_index = v as usize;
            let target = induction_bucket[bucket_index];
            cache[offset].symbol = target;
            induction_bucket[bucket_index] += 1;
            if target >= omp_block_start && target < block_end as SaSint {
                let ni = usize::try_from(target - omp_block_start)
                    .expect("cache slot must be non-negative");
                let mut np = cache[offset].index;
                cache[offset].index = np ^ SAINT_MIN;
                if np > 0 {
                    np -= 1;
                    let np_usize = np as usize;
                    cache[ni].index = np
                        | ((usize::from(t[np_usize - usize::from(np > 0)] < t[np_usize])
                            as SaSint)
                            << (SAINT_BIT - 1));
                    cache[ni].symbol = t[np_usize];
                }
            }
        }
    }
}

fn final_sorting_scan_left_to_right_32s_block_omp(
    t: &[SaSint],
    sa: &mut [SaSint],
    buckets: &mut [SaSint],
    cache: &mut [ThreadCache],
    block_start: SaSint,
    block_size: SaSint,
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
    let omp_num_threads = threads_usize.min(block_size_usize.max(1));
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
                        omp_block_start as SaSint,
                        omp_block_size as SaSint,
                    );
                });
        });
    }
}

fn final_sorting_scan_left_to_right_32s_omp(
    t: &[SaSint],
    sa: &mut [SaSint],
    n: SaSint,
    induction_bucket: &mut [SaSint],
    threads: SaSint,
    thread_state: &mut [ThreadState],
) {
    let last = (n - 1) as usize;
    let bucket = t[last] as usize;
    let slot = induction_bucket[bucket] as usize;
    sa[slot] = (n - 1) | ((usize::from(t[last - 1] < t[last]) as SaSint) << (SAINT_BIT - 1));
    induction_bucket[bucket] += 1;

    if threads == 1 || n < 65536 || thread_state.is_empty() {
        final_sorting_scan_left_to_right_32s(t, sa, induction_bucket, 0, n);
        return;
    }

    let threads_usize = usize::try_from(threads)
        .expect("threads must be non-negative")
        .max(1);
    let block_span = threads_usize * PER_THREAD_CACHE_SIZE;
    let mut cache = vec![ThreadCache::default(); block_span];
    let mut block_start = 0;
    while block_start < n {
        let block_end = (block_start + block_span as SaSint).min(n);
        final_sorting_scan_left_to_right_32s_block_omp(
            t,
            sa,
            induction_bucket,
            &mut cache,
            block_start,
            block_end - block_start,
            threads,
        );
        block_start = block_end;
    }
}

fn final_sorting_scan_right_to_left_32s(
    t: &[SaSint],
    sa: &mut [SaSint],
    induction_bucket: &mut [SaSint],
    omp_block_start: SaSint,
    omp_block_size: SaSint,
) {
    let mut i = (omp_block_start + omp_block_size - 1) as isize;
    let mut j = (omp_block_start + 2 * 64 + 1) as isize;
    while i >= j {
        for current in [i, i - 1] {
            let current = current as usize;
            let mut p = sa[current];
            sa[current] = p & SAINT_MAX;
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
        }
        i -= 2;
    }

    j -= 2 * 64 + 1;
    while i >= j {
        let current = i as usize;
        let mut p = sa[current];
        sa[current] = p & SAINT_MAX;
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
        i -= 1;
    }
}

fn final_sorting_scan_right_to_left_32s_block_gather(
    t: &[SaSint],
    sa: &mut [SaSint],
    cache: &mut [ThreadCache],
    omp_block_start: SaSint,
    omp_block_size: SaSint,
) {
    if omp_block_size <= 0 {
        return;
    }

    let start = usize::try_from(omp_block_start).expect("omp_block_start must be non-negative");
    let size = usize::try_from(omp_block_size).expect("omp_block_size must be non-negative");
    for offset in 0..size {
        let current = start + offset;
        let mut symbol = SAINT_MIN;
        let mut p = sa[current];
        sa[current] = p & SAINT_MAX;
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

fn final_sorting_scan_right_to_left_32s_block_sort(
    t: &[SaSint],
    induction_bucket: &mut [SaSint],
    cache: &mut [ThreadCache],
    omp_block_start: SaSint,
    omp_block_size: SaSint,
) {
    if omp_block_size <= 0 {
        return;
    }

    let size = usize::try_from(omp_block_size).expect("omp_block_size must be non-negative");
    let block_end = omp_block_start + omp_block_size;
    let mut offset = size;

    while offset > 0 {
        offset -= 1;
        let v = cache[offset].symbol;
        if v >= 0 {
            let bucket_index = v as usize;
            induction_bucket[bucket_index] -= 1;
            let target = induction_bucket[bucket_index];
            cache[offset].symbol = target;
            if target >= omp_block_start && target < block_end {
                let ni = usize::try_from(target - omp_block_start)
                    .expect("cache slot must be non-negative");
                let mut np = cache[offset].index;
                cache[offset].index = np & SAINT_MAX;
                if np > 0 {
                    np -= 1;
                    let np_usize = np as usize;
                    cache[ni].index = np
                        | ((usize::from(t[np_usize - usize::from(np > 0)] > t[np_usize])
                            as SaSint)
                            << (SAINT_BIT - 1));
                    cache[ni].symbol = t[np_usize];
                }
            }
        }
    }
}

fn final_sorting_scan_right_to_left_32s_block_omp(
    t: &[SaSint],
    sa: &mut [SaSint],
    buckets: &mut [SaSint],
    cache: &mut [ThreadCache],
    block_start: SaSint,
    block_size: SaSint,
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
    let omp_num_threads = threads_usize.min(block_size_usize.max(1));
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
                        omp_block_start as SaSint,
                        omp_block_size as SaSint,
                    );
                });
        });
    }
}

fn final_sorting_scan_right_to_left_32s_omp(
    t: &[SaSint],
    sa: &mut [SaSint],
    n: SaSint,
    induction_bucket: &mut [SaSint],
    threads: SaSint,
    thread_state: &mut [ThreadState],
) {
    if threads == 1 || n < 65536 || thread_state.is_empty() {
        final_sorting_scan_right_to_left_32s(t, sa, induction_bucket, 0, n);
        return;
    }

    let threads_usize = usize::try_from(threads)
        .expect("threads must be non-negative")
        .max(1);
    let block_span = threads_usize * PER_THREAD_CACHE_SIZE;
    let mut cache = vec![ThreadCache::default(); block_span];
    let mut block_start = n - 1;
    while block_start >= 0 {
        let block_end = (block_start - block_span as SaSint).max(-1);
        final_sorting_scan_right_to_left_32s_block_omp(
            t,
            sa,
            induction_bucket,
            &mut cache,
            block_end + 1,
            block_start - block_end,
            threads,
        );
        block_start = block_end;
    }
}

fn induce_final_order_32s_6k(
    t: &[SaSint],
    sa: &mut [SaSint],
    n: SaSint,
    k: SaSint,
    buckets: &mut [SaSint],
    threads: SaSint,
    thread_state: &mut [ThreadState],
) {
    let k = k as usize;
    final_sorting_scan_left_to_right_32s_omp(
        t,
        sa,
        n,
        &mut buckets[4 * k..5 * k],
        threads,
        thread_state,
    );
    final_sorting_scan_right_to_left_32s_omp(
        t,
        sa,
        n,
        &mut buckets[5 * k..6 * k],
        threads,
        thread_state,
    );
}

fn induce_final_order_32s_4k(
    t: &[SaSint],
    sa: &mut [SaSint],
    n: SaSint,
    k: SaSint,
    buckets: &mut [SaSint],
    threads: SaSint,
    thread_state: &mut [ThreadState],
) {
    let k = k as usize;
    final_sorting_scan_left_to_right_32s_omp(
        t,
        sa,
        n,
        &mut buckets[2 * k..3 * k],
        threads,
        thread_state,
    );
    final_sorting_scan_right_to_left_32s_omp(
        t,
        sa,
        n,
        &mut buckets[3 * k..4 * k],
        threads,
        thread_state,
    );
}

fn induce_final_order_32s_2k(
    t: &[SaSint],
    sa: &mut [SaSint],
    n: SaSint,
    k: SaSint,
    buckets: &mut [SaSint],
    threads: SaSint,
    thread_state: &mut [ThreadState],
) {
    let k = k as usize;
    final_sorting_scan_left_to_right_32s_omp(
        t,
        sa,
        n,
        &mut buckets[k..2 * k],
        threads,
        thread_state,
    );
    final_sorting_scan_right_to_left_32s_omp(t, sa, n, &mut buckets[..k], threads, thread_state);
}

fn induce_final_order_32s_1k(
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

fn clear_lms_suffixes_omp(
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
                        let start = bucket_start_ref[c] as usize;
                        let end = bucket_end_ref[c] as usize;
                        sa[start..end].fill(0);
                    }
                    c += thread_count;
                }
            });
        });
    }
}

fn final_gsa_scan_right_to_left_16u(
    t: &[u16],
    sa: &mut [SaSint],
    induction_bucket: &mut [SaSint],
    omp_block_start: SaSint,
    omp_block_size: SaSint,
) {
    let mut i = (omp_block_start + omp_block_size - 1) as isize;
    let mut j = (omp_block_start + 64 + 1) as isize;
    while i >= j {
        final_sorting_rtl_step(t, sa, induction_bucket, i as usize, true);
        final_sorting_rtl_step(t, sa, induction_bucket, (i - 1) as usize, true);
        i -= 2;
    }
    j -= 64 + 1;
    while i >= j {
        final_sorting_rtl_step(t, sa, induction_bucket, i as usize, true);
        i -= 1;
    }
}

fn final_sorting_ltr_step(
    t: &[u16],
    sa: &mut [SaSint],
    induction_bucket: &mut [SaSint],
    index: usize,
) {
    let mut p = sa[index];
    sa[index] = p ^ SAINT_MIN;
    if p > 0 {
        p -= 1;
        let c = t[p as usize] as usize;
        let mark = if t[(p - SaSint::from(p > 0)) as usize] < t[p as usize] {
            SAINT_MIN
        } else {
            0
        };
        let dst = induction_bucket[c] as usize;
        sa[dst] = p | mark;
        induction_bucket[c] += 1;
    }
}

fn final_sorting_rtl_step(
    t: &[u16],
    sa: &mut [SaSint],
    induction_bucket: &mut [SaSint],
    index: usize,
    gsa: bool,
) {
    let mut p = sa[index];
    sa[index] = p & SAINT_MAX;
    if p > 0 && (!gsa || t[(p - 1) as usize] > 0) {
        p -= 1;
        let c = t[p as usize] as usize;
        let mark = if t[(p - SaSint::from(p > 0)) as usize] > t[p as usize] {
            SAINT_MIN
        } else {
            0
        };
        induction_bucket[c] -= 1;
        sa[induction_bucket[c] as usize] = p | mark;
    }
}

fn final_bwt_scan_left_to_right_16u(
    t: &[u16],
    sa: &mut [SaSint],
    induction_bucket: &mut [SaSint],
    omp_block_start: SaSint,
    omp_block_size: SaSint,
) {
    let mut i = omp_block_start as isize;
    let mut j = (omp_block_start + omp_block_size - 64 - 1) as isize;
    while i < j {
        final_bwt_ltr_step(t, sa, induction_bucket, i as usize);
        final_bwt_ltr_step(t, sa, induction_bucket, (i + 1) as usize);
        i += 2;
    }
    j += 64 + 1;
    while i < j {
        final_bwt_ltr_step(t, sa, induction_bucket, i as usize);
        i += 1;
    }
}

fn final_bwt_scan_right_to_left_16u(
    t: &[u16],
    sa: &mut [SaSint],
    induction_bucket: &mut [SaSint],
    omp_block_start: SaSint,
    omp_block_size: SaSint,
) -> SaSint {
    let mut index = -1;
    let mut i = (omp_block_start + omp_block_size - 1) as isize;
    let mut j = (omp_block_start + 64 + 1) as isize;
    while i >= j {
        final_bwt_rtl_step(t, sa, induction_bucket, i as usize, &mut index);
        final_bwt_rtl_step(t, sa, induction_bucket, (i - 1) as usize, &mut index);
        i -= 2;
    }
    j -= 64 + 1;
    while i >= j {
        final_bwt_rtl_step(t, sa, induction_bucket, i as usize, &mut index);
        i -= 1;
    }
    index
}

fn final_bwt_aux_scan_left_to_right_16u(
    t: &[u16],
    sa: &mut [SaSint],
    rm: SaSint,
    i_sample: &mut [SaSint],
    induction_bucket: &mut [SaSint],
    omp_block_start: SaSint,
    omp_block_size: SaSint,
) {
    let mut i = omp_block_start as isize;
    let mut j = (omp_block_start + omp_block_size - 64 - 1) as isize;
    while i < j {
        final_bwt_aux_ltr_step(t, sa, rm, i_sample, induction_bucket, i as usize);
        final_bwt_aux_ltr_step(t, sa, rm, i_sample, induction_bucket, (i + 1) as usize);
        i += 2;
    }
    j += 64 + 1;
    while i < j {
        final_bwt_aux_ltr_step(t, sa, rm, i_sample, induction_bucket, i as usize);
        i += 1;
    }
}

fn final_bwt_aux_scan_right_to_left_16u(
    t: &[u16],
    sa: &mut [SaSint],
    rm: SaSint,
    i_sample: &mut [SaSint],
    induction_bucket: &mut [SaSint],
    omp_block_start: SaSint,
    omp_block_size: SaSint,
) {
    let mut i = (omp_block_start + omp_block_size - 1) as isize;
    let mut j = (omp_block_start + 64 + 1) as isize;
    while i >= j {
        final_bwt_aux_rtl_step(t, sa, rm, i_sample, induction_bucket, i as usize);
        final_bwt_aux_rtl_step(t, sa, rm, i_sample, induction_bucket, (i - 1) as usize);
        i -= 2;
    }
    j -= 64 + 1;
    while i >= j {
        final_bwt_aux_rtl_step(t, sa, rm, i_sample, induction_bucket, i as usize);
        i -= 1;
    }
}

fn renumber_lms_suffixes_16u(
    sa: &mut [SaSint],
    m: SaSint,
    mut name: SaSint,
    omp_block_start: SaSint,
    omp_block_size: SaSint,
) -> SaSint {
    let mut i = omp_block_start as isize;
    let mut j = (omp_block_start + omp_block_size - 64 - 3) as isize;
    while i < j {
        let p0 = sa[i as usize];
        sa[m as usize + ((p0 & SAINT_MAX) >> 1) as usize] = name | SAINT_MIN;
        name += SaSint::from(p0 < 0);

        let p1 = sa[(i + 1) as usize];
        sa[m as usize + ((p1 & SAINT_MAX) >> 1) as usize] = name | SAINT_MIN;
        name += SaSint::from(p1 < 0);

        let p2 = sa[(i + 2) as usize];
        sa[m as usize + ((p2 & SAINT_MAX) >> 1) as usize] = name | SAINT_MIN;
        name += SaSint::from(p2 < 0);

        let p3 = sa[(i + 3) as usize];
        sa[m as usize + ((p3 & SAINT_MAX) >> 1) as usize] = name | SAINT_MIN;
        name += SaSint::from(p3 < 0);

        i += 4;
    }

    j += 64 + 3;
    while i < j {
        let p = sa[i as usize];
        sa[m as usize + ((p & SAINT_MAX) >> 1) as usize] = name | SAINT_MIN;
        name += SaSint::from(p < 0);
        i += 1;
    }

    name
}

fn renumber_lms_suffixes_16u_omp(
    sa: &mut [SaSint],
    m: SaSint,
    threads: SaSint,
    thread_state: &mut [ThreadState],
) -> SaSint {
    if threads == 1 || m < 65_536 || thread_state.is_empty() {
        return renumber_lms_suffixes_16u(sa, m, 0, 0, m);
    }

    let thread_count = usize::try_from(threads)
        .expect("threads must be non-negative")
        .min(thread_state.len());
    let block_stride = (m / thread_count as SaSint) & !15;

    for (thread, state) in thread_state.iter_mut().take(thread_count).enumerate() {
        let block_start = thread as SaSint * block_stride;
        let block_size = if thread + 1 < thread_count {
            block_stride
        } else {
            m - block_start
        };
        state.count = count_negative_marked_suffixes(sa, block_start, block_size);
    }

    let mut name = 0;
    for thread in 0..thread_count {
        let block_start = thread as SaSint * block_stride;
        let block_size = if thread + 1 < thread_count {
            block_stride
        } else {
            m - block_start
        };
        renumber_lms_suffixes_16u(sa, m, name, block_start, block_size);
        name += thread_state[thread].count;
    }

    name
}

fn gather_marked_lms_suffixes(
    sa: &mut [SaSint],
    m: SaSint,
    mut l: isize,
    omp_block_start: isize,
    omp_block_size: isize,
) -> isize {
    if omp_block_size <= 0 {
        return l;
    }

    l -= 1;
    let mut i = m as isize + omp_block_start + omp_block_size - 1;
    let mut j = m as isize + omp_block_start + 3;
    while i >= j {
        let s0 = sa[i as usize];
        sa[l as usize] = s0 & SAINT_MAX;
        l -= isize::from(s0 < 0);

        let s1 = sa[(i - 1) as usize];
        sa[l as usize] = s1 & SAINT_MAX;
        l -= isize::from(s1 < 0);

        let s2 = sa[(i - 2) as usize];
        sa[l as usize] = s2 & SAINT_MAX;
        l -= isize::from(s2 < 0);

        let s3 = sa[(i - 3) as usize];
        sa[l as usize] = s3 & SAINT_MAX;
        l -= isize::from(s3 < 0);

        i -= 4;
    }

    j -= 3;
    while i >= j {
        let s = sa[i as usize];
        sa[l as usize] = s & SAINT_MAX;
        l -= isize::from(s < 0);
        i -= 1;
    }

    l + 1
}

fn gather_marked_lms_suffixes_omp(
    sa: &mut [SaSint],
    n: SaSint,
    m: SaSint,
    fs: SaSint,
    threads: SaSint,
    thread_state: &mut [ThreadState],
) {
    let half_n = n >> 1;
    if threads == 1 || n < 131_072 || thread_state.is_empty() {
        let _ = gather_marked_lms_suffixes(sa, m, (n + fs) as isize, 0, half_n as isize);
        return;
    }

    let thread_count = usize::try_from(threads)
        .expect("threads must be non-negative")
        .min(thread_state.len());
    let block_stride = (half_n / thread_count as SaSint) & !15;

    for (thread, state) in thread_state.iter_mut().take(thread_count).enumerate() {
        let block_start = thread as SaSint * block_stride;
        let block_size = if thread + 1 < thread_count {
            block_stride
        } else {
            half_n - block_start
        };
        let local_end = if thread + 1 < thread_count {
            m + block_start + block_size
        } else {
            n + fs
        } as isize;
        let gathered_position =
            gather_marked_lms_suffixes(sa, m, local_end, block_start as isize, block_size as isize);
        state.position = gathered_position as SaSint;
        state.count = (local_end - gathered_position) as SaSint;
    }

    let mut position = (n + fs) as isize;
    for thread in (0..thread_count).rev() {
        let count =
            usize::try_from(thread_state[thread].count).expect("count must be non-negative");
        position -= thread_state[thread].count as isize;
        if thread + 1 != thread_count && count > 0 {
            let src = usize::try_from(thread_state[thread].position)
                .expect("position must be non-negative");
            let dst = position as usize;
            sa.copy_within(src..src + count, dst);
        }
    }
}

fn renumber_and_gather_lms_suffixes_omp(
    sa: &mut [SaSint],
    n: SaSint,
    m: SaSint,
    fs: SaSint,
    threads: SaSint,
    thread_state: &mut [ThreadState],
) -> SaSint {
    let m_usize = m as usize;
    let half_n = (n >> 1) as usize;
    sa[m_usize..m_usize + half_n].fill(0);

    let name = renumber_lms_suffixes_16u_omp(sa, m, threads, thread_state);
    if name < m {
        gather_marked_lms_suffixes_omp(sa, n, m, fs, threads, thread_state);
    } else {
        for item in &mut sa[..m_usize] {
            *item &= SAINT_MAX;
        }
    }

    name
}

fn reconstruct_lms_suffixes(
    sa: &mut [SaSint],
    n: SaSint,
    m: SaSint,
    omp_block_start: isize,
    omp_block_size: isize,
) {
    if omp_block_size <= 0 {
        return;
    }

    let base = (n - m) as usize;
    let mut i = omp_block_start;
    let mut j = omp_block_start + omp_block_size - 64 - 3;
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

    j += 64 + 3;
    while i < j {
        let iu = i as usize;
        let s = sa[iu] as usize;
        sa[iu] = sa[base + s];
        i += 1;
    }
}

fn reconstruct_lms_suffixes_omp(sa: &mut [SaSint], n: SaSint, m: SaSint, threads: SaSint) {
    if threads == 1 || m < 65_536 {
        reconstruct_lms_suffixes(sa, n, m, 0, m as isize);
        return;
    }

    let thread_count = threads as usize;
    let block_stride = (m / threads) & !15;
    for thread in 0..thread_count {
        let block_start = thread as SaSint * block_stride;
        let block_size = if thread + 1 < thread_count {
            block_stride
        } else {
            m - block_start
        };
        reconstruct_lms_suffixes(sa, n, m, block_start as isize, block_size as isize);
    }
}

fn renumber_distinct_lms_suffixes_32s_4k(
    sa: &mut [SaSint],
    m: SaSint,
    mut name: SaSint,
    omp_block_start: isize,
    omp_block_size: isize,
) -> SaSint {
    if omp_block_size <= 0 {
        return name;
    }

    let m_usize = m as usize;
    let start = omp_block_start as usize;
    let size = omp_block_size as usize;
    let (sa_head, sam) = sa.split_at_mut(m_usize);
    let mut i = start;
    let mut j = start + size.saturating_sub(64 + 3);
    let mut p3 = 0;

    while i < j {
        let p0 = sa_head[i];
        sa_head[i] = p0 & SAINT_MAX;
        sam[(sa_head[i] >> 1) as usize] = name | (p0 & p3 & SAINT_MIN);
        name += SaSint::from(p0 < 0);

        let p1 = sa_head[i + 1];
        sa_head[i + 1] = p1 & SAINT_MAX;
        sam[(sa_head[i + 1] >> 1) as usize] = name | (p1 & p0 & SAINT_MIN);
        name += SaSint::from(p1 < 0);

        let p2 = sa_head[i + 2];
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
        let p2 = p3;
        p3 = sa_head[i];
        sa_head[i] = p3 & SAINT_MAX;
        sam[(sa_head[i] >> 1) as usize] = name | (p3 & p2 & SAINT_MIN);
        name += SaSint::from(p3 < 0);
        i += 1;
    }

    name
}

fn mark_distinct_lms_suffixes_32s(
    sa: &mut [SaSint],
    m: SaSint,
    omp_block_start: isize,
    omp_block_size: isize,
) {
    if omp_block_size <= 0 {
        return;
    }

    let mut i = m as usize + omp_block_start as usize;
    let mut j = i + (omp_block_size as usize).saturating_sub(3);
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

    j = m as usize + omp_block_start as usize + omp_block_size as usize;
    while i < j {
        let p2 = p3;
        p3 = sa[i];
        sa[i] = p3 & (p2 | SAINT_MAX);
        p3 = if p3 == 0 { p2 } else { p3 };
        i += 1;
    }
}

fn clamp_lms_suffixes_length_32s(
    sa: &mut [SaSint],
    m: SaSint,
    omp_block_start: isize,
    omp_block_size: isize,
) {
    if omp_block_size <= 0 {
        return;
    }

    let mut i = m as usize + omp_block_start as usize;
    let mut j = i + (omp_block_size as usize).saturating_sub(3);
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

    j = m as usize + omp_block_start as usize + omp_block_size as usize;
    while i < j {
        let s = sa[i];
        sa[i] = if s < 0 { s } else { 0 } & SAINT_MAX;
        i += 1;
    }
}

fn renumber_distinct_lms_suffixes_32s_4k_omp(
    sa: &mut [SaSint],
    m: SaSint,
    threads: SaSint,
    thread_state: &mut [ThreadState],
) -> SaSint {
    if threads == 1 || m < 65_536 || thread_state.is_empty() {
        return renumber_distinct_lms_suffixes_32s_4k(sa, m, 1, 0, m as isize) - 1;
    }

    let thread_count = usize::try_from(threads)
        .expect("threads must be non-negative")
        .min(thread_state.len());
    let block_stride = (m / thread_count as SaSint) & !15;

    for (thread, state) in thread_state.iter_mut().take(thread_count).enumerate() {
        let block_start = thread as SaSint * block_stride;
        let block_size = if thread + 1 < thread_count {
            block_stride
        } else {
            m - block_start
        };
        state.count = count_negative_marked_suffixes(sa, block_start, block_size);
    }

    let mut count = 1;
    for thread in 0..thread_count {
        let block_start = thread as SaSint * block_stride;
        let block_size = if thread + 1 < thread_count {
            block_stride
        } else {
            m - block_start
        };
        renumber_distinct_lms_suffixes_32s_4k(
            sa,
            m,
            count,
            block_start as isize,
            block_size as isize,
        );
        count += thread_state[thread].count;
    }

    count - 1
}

fn mark_distinct_lms_suffixes_32s_omp(sa: &mut [SaSint], n: SaSint, m: SaSint, threads: SaSint) {
    let half_n = n >> 1;
    if threads == 1 || n < 131_072 {
        mark_distinct_lms_suffixes_32s(sa, m, 0, half_n as isize);
        return;
    }

    let thread_count = threads as usize;
    let block_stride = (half_n / threads) & !15;
    for thread in 0..thread_count {
        let block_start = thread as SaSint * block_stride;
        let block_size = if thread + 1 < thread_count {
            block_stride
        } else {
            half_n - block_start
        };
        mark_distinct_lms_suffixes_32s(sa, m, block_start as isize, block_size as isize);
    }
}

fn clamp_lms_suffixes_length_32s_omp(sa: &mut [SaSint], n: SaSint, m: SaSint, threads: SaSint) {
    let half_n = n >> 1;
    if threads == 1 || n < 131_072 {
        clamp_lms_suffixes_length_32s(sa, m, 0, half_n as isize);
        return;
    }

    let thread_count = threads as usize;
    let block_stride = (half_n / threads) & !15;
    for thread in 0..thread_count {
        let block_start = thread as SaSint * block_stride;
        let block_size = if thread + 1 < thread_count {
            block_stride
        } else {
            half_n - block_start
        };
        clamp_lms_suffixes_length_32s(sa, m, block_start as isize, block_size as isize);
    }
}

fn renumber_and_mark_distinct_lms_suffixes_32s_4k_omp(
    sa: &mut [SaSint],
    n: SaSint,
    m: SaSint,
    threads: SaSint,
    thread_state: &mut [ThreadState],
) -> SaSint {
    let m_usize = m as usize;
    let half_n = (n >> 1) as usize;
    sa[m_usize..m_usize + half_n].fill(0);

    let name = renumber_distinct_lms_suffixes_32s_4k_omp(sa, m, threads, thread_state);
    if name < m {
        mark_distinct_lms_suffixes_32s_omp(sa, n, m, threads);
    }

    name
}

fn renumber_and_mark_distinct_lms_suffixes_32s_1k_omp(
    t: &[SaSint],
    sa: &mut [SaSint],
    n: SaSint,
    m: SaSint,
    threads: SaSint,
) -> SaSint {
    let m_usize = m as usize;
    let n_usize = n as usize;

    gather_lms_suffixes_32s(t, sa, n);
    sa[m_usize..n_usize - m_usize].fill(0);

    let mut i = n - m;
    let mut j = n - 1 - 64 - 3;
    while i < j {
        let s0 = (sa[i as usize] as SaUint >> 1) as usize;
        let s1 = (sa[(i + 1) as usize] as SaUint >> 1) as usize;
        let s2 = (sa[(i + 2) as usize] as SaUint >> 1) as usize;
        let s3 = (sa[(i + 3) as usize] as SaUint >> 1) as usize;
        sa[m_usize + s0] = sa[(i + 1) as usize] - sa[i as usize] + 1 + SAINT_MIN;
        sa[m_usize + s1] = sa[(i + 2) as usize] - sa[(i + 1) as usize] + 1 + SAINT_MIN;
        sa[m_usize + s2] = sa[(i + 3) as usize] - sa[(i + 2) as usize] + 1 + SAINT_MIN;
        sa[m_usize + s3] = sa[(i + 4) as usize] - sa[(i + 3) as usize] + 1 + SAINT_MIN;
        i += 4;
    }

    j += 64 + 3;
    while i < j {
        let s = (sa[i as usize] as SaUint >> 1) as usize;
        sa[m_usize + s] = sa[(i + 1) as usize] - sa[i as usize] + 1 + SAINT_MIN;
        i += 1;
    }

    let tail = (sa[n_usize - 1] as SaUint >> 1) as usize;
    sa[m_usize + tail] = 1 + SAINT_MIN;

    clamp_lms_suffixes_length_32s_omp(sa, n, m, threads);

    let mut name = 1;
    if m_usize > 0 {
        let mut i = 1usize;
        let mut j = m_usize.saturating_sub(64 + 1);
        let mut p = sa[0] as usize;
        let mut plen = sa[m_usize + (p >> 1)];
        let mut pdiff = SAINT_MIN;

        while i < j {
            let q = sa[i] as usize;
            let qlen = sa[m_usize + (q >> 1)];
            let mut qdiff = SAINT_MIN;
            if plen == qlen {
                let mut l = 0;
                while l < qlen as usize {
                    if t[p + l] != t[q + l] {
                        break;
                    }
                    l += 1;
                }
                qdiff = ((l as SaSint) - qlen) & SAINT_MIN;
            }
            sa[m_usize + (p >> 1)] = name | (pdiff & qdiff);
            name += SaSint::from(qdiff < 0);

            p = sa[i + 1] as usize;
            plen = sa[m_usize + (p >> 1)];
            pdiff = SAINT_MIN;
            if qlen == plen {
                let mut l = 0;
                while l < plen as usize {
                    if t[q + l] != t[p + l] {
                        break;
                    }
                    l += 1;
                }
                pdiff = ((l as SaSint) - plen) & SAINT_MIN;
            }
            sa[m_usize + (q >> 1)] = name | (qdiff & pdiff);
            name += SaSint::from(pdiff < 0);
            i += 2;
        }

        j = m_usize;
        while i < j {
            let q = sa[i] as usize;
            let qlen = sa[m_usize + (q >> 1)];
            let mut qdiff = SAINT_MIN;
            if plen == qlen {
                let mut l = 0;
                while l < plen as usize {
                    if t[p + l] != t[q + l] {
                        break;
                    }
                    l += 1;
                }
                qdiff = ((l as SaSint) - plen) & SAINT_MIN;
            }
            sa[m_usize + (p >> 1)] = name | (pdiff & qdiff);
            name += SaSint::from(qdiff < 0);
            p = q;
            plen = qlen;
            pdiff = qdiff;
            i += 1;
        }

        sa[m_usize + (p >> 1)] = name | pdiff;
        name += 1;
    }

    if name <= m {
        mark_distinct_lms_suffixes_32s_omp(sa, n, m, threads);
    }

    name - 1
}

fn renumber_unique_and_nonunique_lms_suffixes_32s(
    t: &mut [SaSint],
    sa: &mut [SaSint],
    m: SaSint,
    mut f: SaSint,
    omp_block_start: isize,
    omp_block_size: isize,
) -> SaSint {
    if omp_block_size <= 0 {
        return f;
    }

    let m_usize = m as usize;
    let (sa_head, sam) = sa.split_at_mut(m_usize);
    let mut i = omp_block_start;
    let mut j = omp_block_start + omp_block_size - 128 - 3;
    while i < j {
        for offset in 0..4 {
            let idx = (i + offset) as usize;
            let p = sa_head[idx] as SaUint;
            let mut s = sam[(p >> 1) as usize];
            if s < 0 {
                t[p as usize] |= SAINT_MIN;
                f += 1;
                s = i as SaSint + offset as SaSint + SAINT_MIN + f;
            }
            sam[(p >> 1) as usize] = s - f;
        }
        i += 4;
    }

    j += 128 + 3;
    while i < j {
        let p = sa_head[i as usize] as SaUint;
        let mut s = sam[(p >> 1) as usize];
        if s < 0 {
            t[p as usize] |= SAINT_MIN;
            f += 1;
            s = i as SaSint + SAINT_MIN + f;
        }
        sam[(p >> 1) as usize] = s - f;
        i += 1;
    }

    f
}

fn compact_unique_and_nonunique_lms_suffixes_32s(
    sa: &mut [SaSint],
    m: SaSint,
    pl: &mut isize,
    pr: &mut isize,
    omp_block_start: isize,
    omp_block_size: isize,
) {
    if omp_block_size <= 0 {
        return;
    }

    let m_usize = m as usize;
    let source: Vec<SaSint> = sa
        [m_usize + omp_block_start as usize..m_usize + (omp_block_start + omp_block_size) as usize]
        .to_vec();
    let mut l = *pl - 1;
    let mut r = *pr - 1;

    for &p in source.iter().rev() {
        sa[l as usize] = p & SAINT_MAX;
        l -= isize::from(p < 0);

        sa[r as usize] = p.wrapping_sub(1);
        r -= isize::from(p > 0);
    }

    *pl = l + 1;
    *pr = r + 1;
}

fn count_unique_suffixes(
    sa: &[SaSint],
    m: SaSint,
    omp_block_start: isize,
    omp_block_size: isize,
) -> SaSint {
    let base = m as usize;
    let start = omp_block_start as usize;
    let end = start + omp_block_size as usize;
    let mut count = 0;
    for i in start..end {
        count += SaSint::from(sa[base + ((sa[i] as SaUint) >> 1) as usize] < 0);
    }
    count
}

fn renumber_unique_and_nonunique_lms_suffixes_32s_omp(
    t: &mut [SaSint],
    sa: &mut [SaSint],
    m: SaSint,
    threads: SaSint,
) -> SaSint {
    if threads == 1 || m < 65_536 {
        return renumber_unique_and_nonunique_lms_suffixes_32s(t, sa, m, 0, 0, m as isize);
    }

    let thread_count = threads as usize;
    let block_stride = (m / threads) & !15;
    let mut counts = vec![0; thread_count];

    for thread in 0..thread_count {
        let block_start = thread as SaSint * block_stride;
        let block_size = if thread + 1 < thread_count {
            block_stride
        } else {
            m - block_start
        };
        counts[thread] = count_unique_suffixes(sa, m, block_start as isize, block_size as isize);
    }

    let mut f = 0;
    for thread in 0..thread_count {
        let block_start = thread as SaSint * block_stride;
        let block_size = if thread + 1 < thread_count {
            block_stride
        } else {
            m - block_start
        };
        renumber_unique_and_nonunique_lms_suffixes_32s(
            t,
            sa,
            m,
            f,
            block_start as isize,
            block_size as isize,
        );
        f += counts[thread];
    }

    f
}

fn compact_unique_and_nonunique_lms_suffixes_32s_omp(
    sa: &mut [SaSint],
    n: SaSint,
    m: SaSint,
    fs: SaSint,
    f: SaSint,
    threads: SaSint,
) {
    let half_n = n >> 1;
    if threads == 1 || n < 131_072 || m >= fs {
        let mut l = m as isize;
        let mut r = (n + fs) as isize;
        compact_unique_and_nonunique_lms_suffixes_32s(sa, m, &mut l, &mut r, 0, half_n as isize);
    } else {
        let thread_count = threads as usize;
        let block_stride = (half_n / threads) & !15;
        let mut positions = vec![0isize; thread_count];
        let mut counts = vec![0isize; thread_count];

        for thread in 0..thread_count {
            let block_start = thread as SaSint * block_stride;
            let block_size = if thread + 1 < thread_count {
                block_stride
            } else {
                half_n - block_start
            };
            let mut position = (m + half_n + block_start + block_size) as isize;
            let mut count = (m + block_start + block_size) as isize;
            compact_unique_and_nonunique_lms_suffixes_32s(
                sa,
                m,
                &mut position,
                &mut count,
                block_start as isize,
                block_size as isize,
            );
            positions[thread] = position;
            counts[thread] = count;
        }

        let mut position = m as isize;
        for thread in (0..thread_count).rev() {
            let block_end = if thread + 1 < thread_count {
                block_stride * (thread as SaSint + 1)
            } else {
                half_n
            };
            let count = (m + half_n + block_end) as isize - positions[thread];
            if count > 0 {
                position -= count;
                let src = positions[thread] as usize;
                let dst = position as usize;
                sa.copy_within(src..src + count as usize, dst);
            }
        }

        let mut position = (n + fs) as isize;
        for thread in (0..thread_count).rev() {
            let block_end = if thread + 1 < thread_count {
                block_stride * (thread as SaSint + 1)
            } else {
                half_n
            };
            let count = (m + block_end) as isize - counts[thread];
            if count > 0 {
                position -= count;
                let src = counts[thread] as usize;
                let dst = position as usize;
                sa.copy_within(src..src + count as usize, dst);
            }
        }
    }

    let dst = (n + fs - m) as usize;
    let src = (m - f) as usize;
    sa.copy_within(src..src + f as usize, dst);
}

fn compact_lms_suffixes_32s_omp(
    t: &mut [SaSint],
    sa: &mut [SaSint],
    n: SaSint,
    m: SaSint,
    fs: SaSint,
    threads: SaSint,
) -> SaSint {
    let f = renumber_unique_and_nonunique_lms_suffixes_32s_omp(t, sa, m, threads);
    compact_unique_and_nonunique_lms_suffixes_32s_omp(sa, n, m, fs, f, threads);
    f
}

fn merge_unique_lms_suffixes_32s(
    t: &mut [SaSint],
    sa: &mut [SaSint],
    n: SaSint,
    m: SaSint,
    l: isize,
    omp_block_start: isize,
    omp_block_size: isize,
) {
    let mut src_index = (n as isize - m as isize - 1 + l) as usize;
    let mut tmp = sa[src_index] as isize;
    src_index += 1;

    let mut i = omp_block_start;
    let mut j = omp_block_start + omp_block_size - 6;
    while i < j {
        let iu = i as usize;

        let c0 = t[iu];
        if c0 < 0 {
            t[iu] = c0 & SAINT_MAX;
            sa[tmp as usize] = i as SaSint;
            i += 1;
            tmp = sa[src_index] as isize;
            src_index += 1;
        }

        let c1 = t[(i + 1) as usize];
        if c1 < 0 {
            t[(i + 1) as usize] = c1 & SAINT_MAX;
            sa[tmp as usize] = i as SaSint + 1;
            i += 1;
            tmp = sa[src_index] as isize;
            src_index += 1;
        }

        let c2 = t[(i + 2) as usize];
        if c2 < 0 {
            t[(i + 2) as usize] = c2 & SAINT_MAX;
            sa[tmp as usize] = i as SaSint + 2;
            i += 1;
            tmp = sa[src_index] as isize;
            src_index += 1;
        }

        let c3 = t[(i + 3) as usize];
        if c3 < 0 {
            t[(i + 3) as usize] = c3 & SAINT_MAX;
            sa[tmp as usize] = i as SaSint + 3;
            i += 1;
            tmp = sa[src_index] as isize;
            src_index += 1;
        }

        i += 4;
    }

    j += 6;
    while i < j {
        let c = t[i as usize];
        if c < 0 {
            t[i as usize] = c & SAINT_MAX;
            sa[tmp as usize] = i as SaSint;
            i += 1;
            tmp = sa[src_index] as isize;
            src_index += 1;
        }
        i += 1;
    }
}

fn merge_nonunique_lms_suffixes_32s(
    sa: &mut [SaSint],
    n: SaSint,
    m: SaSint,
    l: isize,
    omp_block_start: isize,
    omp_block_size: isize,
) {
    let mut src_index = (n as isize - m as isize - 1 + l) as usize;
    let mut tmp = sa[src_index];
    src_index += 1;

    let mut i = omp_block_start;
    let mut j = omp_block_start + omp_block_size - 3;
    while i < j {
        if sa[i as usize] == 0 {
            sa[i as usize] = tmp;
            tmp = sa[src_index];
            src_index += 1;
        }
        if sa[(i + 1) as usize] == 0 {
            sa[(i + 1) as usize] = tmp;
            tmp = sa[src_index];
            src_index += 1;
        }
        if sa[(i + 2) as usize] == 0 {
            sa[(i + 2) as usize] = tmp;
            tmp = sa[src_index];
            src_index += 1;
        }
        if sa[(i + 3) as usize] == 0 {
            sa[(i + 3) as usize] = tmp;
            tmp = sa[src_index];
            src_index += 1;
        }
        i += 4;
    }

    j += 3;
    while i < j {
        if sa[i as usize] == 0 {
            sa[i as usize] = tmp;
            tmp = sa[src_index];
            src_index += 1;
        }
        i += 1;
    }
}

fn merge_unique_lms_suffixes_32s_omp(
    t: &mut [SaSint],
    sa: &mut [SaSint],
    n: SaSint,
    m: SaSint,
    threads: SaSint,
) {
    if threads == 1 || n < 65_536 {
        merge_unique_lms_suffixes_32s(t, sa, n, m, 0, 0, n as isize);
        return;
    }

    let thread_count = threads as usize;
    let block_stride = (n / threads) & !15;
    let mut counts = vec![0; thread_count];

    for thread in 0..thread_count {
        let block_start = thread as SaSint * block_stride;
        let block_size = if thread + 1 < thread_count {
            block_stride
        } else {
            n - block_start
        };
        counts[thread] = count_negative_marked_suffixes(t, block_start, block_size);
    }

    let mut count = 0;
    for thread in 0..thread_count {
        let block_start = thread as SaSint * block_stride;
        let block_size = if thread + 1 < thread_count {
            block_stride
        } else {
            n - block_start
        };
        merge_unique_lms_suffixes_32s(
            t,
            sa,
            n,
            m,
            count as isize,
            block_start as isize,
            block_size as isize,
        );
        count += counts[thread];
    }
}

fn merge_nonunique_lms_suffixes_32s_omp(
    sa: &mut [SaSint],
    n: SaSint,
    m: SaSint,
    f: SaSint,
    threads: SaSint,
) {
    if threads == 1 || m < 65_536 {
        merge_nonunique_lms_suffixes_32s(sa, n, m, f as isize, 0, m as isize);
        return;
    }

    let thread_count = threads as usize;
    let block_stride = (m / threads) & !15;
    let mut counts = vec![0; thread_count];

    for thread in 0..thread_count {
        let block_start = thread as SaSint * block_stride;
        let block_size = if thread + 1 < thread_count {
            block_stride
        } else {
            m - block_start
        };
        counts[thread] = count_zero_marked_suffixes(sa, block_start, block_size);
    }

    let mut count = f;
    for thread in 0..thread_count {
        let block_start = thread as SaSint * block_stride;
        let block_size = if thread + 1 < thread_count {
            block_stride
        } else {
            m - block_start
        };
        merge_nonunique_lms_suffixes_32s(
            sa,
            n,
            m,
            count as isize,
            block_start as isize,
            block_size as isize,
        );
        count += counts[thread];
    }
}

fn merge_compacted_lms_suffixes_32s_omp(
    t: &mut [SaSint],
    sa: &mut [SaSint],
    n: SaSint,
    m: SaSint,
    f: SaSint,
    threads: SaSint,
) {
    merge_unique_lms_suffixes_32s_omp(t, sa, n, m, threads);
    merge_nonunique_lms_suffixes_32s_omp(sa, n, m, f, threads);
}

fn reconstruct_compacted_lms_suffixes_32s_2k_omp(
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
        let dst = (n - m - 1) as usize;
        let src = (n + fs - m) as usize;
        sa.copy_within(src..src + f as usize, dst);

        count_and_gather_compacted_lms_suffixes_32s_2k_omp(
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

        let dst = (n - m - 1 + f) as usize;
        sa.copy_within(0..(m - f) as usize, dst);
        sa[..m as usize].fill(0);

        merge_compacted_lms_suffixes_32s_omp(t, sa, n, m, f, threads);
    } else {
        count_and_gather_lms_suffixes_32s_2k(t, sa, n, k, buckets, 0, n as isize);
        reconstruct_lms_suffixes_omp(sa, n, m, threads);
    }
}

fn reconstruct_compacted_lms_suffixes_32s_1k_omp(
    t: &mut [SaSint],
    sa: &mut [SaSint],
    n: SaSint,
    m: SaSint,
    fs: SaSint,
    f: SaSint,
    threads: SaSint,
) {
    if f > 0 {
        let dst = (n - m - 1) as usize;
        let src = (n + fs - m) as usize;
        sa.copy_within(src..src + f as usize, dst);

        gather_compacted_lms_suffixes_32s(t, sa, n);
        reconstruct_lms_suffixes_omp(sa, n, m - f, threads);

        let dst = (n - m - 1 + f) as usize;
        sa.copy_within(0..(m - f) as usize, dst);
        sa[..m as usize].fill(0);

        merge_compacted_lms_suffixes_32s_omp(t, sa, n, m, f, threads);
    } else {
        gather_lms_suffixes_32s(t, sa, n);
        reconstruct_lms_suffixes_omp(sa, n, m, threads);
    }
}

fn place_lms_suffixes_interval_16u(
    sa: &mut [SaSint],
    n: SaSint,
    mut m: SaSint,
    flags: SaSint,
    buckets: &mut [SaSint],
) {
    if (flags & LIBSAIS_FLAGS_GSA) != 0 {
        buckets[7 * ALPHABET_SIZE] -= 1;
    }

    let mut j = n as isize;
    let mut c = ALPHABET_SIZE as isize - 2;
    while c >= 0 {
        let ci = c as usize;
        let l =
            buckets[buckets_index2(ci, 1) + buckets_index2(1, 0)] - buckets[buckets_index2(ci, 1)];
        if l > 0 {
            let i = buckets[7 * ALPHABET_SIZE + ci] as isize;
            if j - i > 0 {
                sa[i as usize..j as usize].fill(0);
            }

            m -= l;
            j = i - l as isize;
            let src = m as usize;
            let dst = j as usize;
            sa.copy_within(src..src + l as usize, dst);
        }
        c -= 1;
    }

    sa[..j as usize].fill(0);

    if (flags & LIBSAIS_FLAGS_GSA) != 0 {
        buckets[7 * ALPHABET_SIZE] += 1;
    }
}

fn place_lms_suffixes_interval_32s_4k(
    sa: &mut [SaSint],
    n: SaSint,
    k: SaSint,
    mut m: SaSint,
    buckets: &[SaSint],
) {
    let bucket_end = &buckets[3 * k as usize..4 * k as usize];
    let mut j = n as usize;
    let mut c = k - 2;
    while c >= 0 {
        let cu = c as usize;
        let l =
            buckets[buckets_index2(cu, 1) + buckets_index2(1, 0)] - buckets[buckets_index2(cu, 1)];
        if l > 0 {
            let i = bucket_end[cu] as usize;
            if j > i {
                sa[i..j].fill(0);
            }

            m -= l;
            let dst = i - l as usize;
            sa.copy_within(m as usize..m as usize + l as usize, dst);
            j = dst;
        }
        c -= 1;
    }

    sa[..j].fill(0);
}

fn place_lms_suffixes_interval_32s_2k(
    sa: &mut [SaSint],
    n: SaSint,
    k: SaSint,
    mut m: SaSint,
    buckets: &[SaSint],
) {
    let mut j = n as usize;
    if k > 1 {
        let mut c = buckets_index2(k as usize - 2, 0) as isize;
        while c >= buckets_index2(0, 0) as isize {
            let cu = c as usize;
            let l = buckets[cu + buckets_index2(1, 1)] - buckets[cu + buckets_index2(0, 1)];
            if l > 0 {
                let i = buckets[cu] as usize;
                if j > i {
                    sa[i..j].fill(0);
                }

                m -= l;
                let dst = i - l as usize;
                sa.copy_within(m as usize..m as usize + l as usize, dst);
                j = dst;
            }
            c -= buckets_index2(1, 0) as isize;
        }
    }

    sa[..j].fill(0);
}

fn place_lms_suffixes_interval_32s_1k(
    t: &[SaSint],
    sa: &mut [SaSint],
    k: SaSint,
    m: SaSint,
    buckets: &[SaSint],
) {
    let mut c = k - 1;
    let mut l = buckets[c as usize] as usize;

    let mut i = m - 1;
    while i >= 0 {
        let p = sa[i as usize] as usize;
        if t[p] != c {
            c = t[p];
            let bucket_pos = buckets[c as usize] as usize;
            if l > bucket_pos {
                sa[bucket_pos..l].fill(0);
            }
            l = bucket_pos;
        }
        l -= 1;
        sa[l] = p as SaSint;
        i -= 1;
    }

    sa[..l].fill(0);
}

fn place_lms_suffixes_histogram_32s_6k(
    sa: &mut [SaSint],
    n: SaSint,
    k: SaSint,
    mut m: SaSint,
    buckets: &[SaSint],
) {
    let bucket_end = &buckets[5 * k as usize..6 * k as usize];
    let mut j = n as usize;
    let mut c = k - 2;
    while c >= 0 {
        let l = buckets[buckets_index4(c as usize, 1)] as usize;
        if l > 0 {
            let i = bucket_end[c as usize] as usize;
            if j > i {
                sa[i..j].fill(0);
            }
            let dst = i - l;
            m -= l as SaSint;
            sa.copy_within(m as usize..m as usize + l, dst);
            j = dst;
        }
        c -= 1;
    }
    sa[..j].fill(0);
}

fn place_lms_suffixes_histogram_32s_4k(
    sa: &mut [SaSint],
    n: SaSint,
    k: SaSint,
    mut m: SaSint,
    buckets: &[SaSint],
) {
    let bucket_end = &buckets[3 * k as usize..4 * k as usize];
    let mut j = n as usize;
    let mut c = k - 2;
    while c >= 0 {
        let l = buckets[buckets_index2(c as usize, 1)] as usize;
        if l > 0 {
            let i = bucket_end[c as usize] as usize;
            if j > i {
                sa[i..j].fill(0);
            }
            let dst = i - l;
            m -= l as SaSint;
            sa.copy_within(m as usize..m as usize + l, dst);
            j = dst;
        }
        c -= 1;
    }
    sa[..j].fill(0);
}

fn place_lms_suffixes_histogram_32s_2k(
    sa: &mut [SaSint],
    n: SaSint,
    k: SaSint,
    mut m: SaSint,
    buckets: &[SaSint],
) {
    let mut j = n as usize;
    if k > 1 {
        let mut c = buckets_index2(k as usize - 2, 0) as isize;
        while c >= buckets_index2(0, 0) as isize {
            let cu = c as usize;
            let l = buckets[cu + buckets_index2(0, 1)] as usize;
            if l > 0 {
                let i = buckets[cu] as usize;
                if j > i {
                    sa[i..j].fill(0);
                }
                let dst = i - l;
                m -= l as SaSint;
                sa.copy_within(m as usize..m as usize + l, dst);
                j = dst;
            }
            c -= buckets_index2(1, 0) as isize;
        }
    }
    sa[..j].fill(0);
}

fn final_bwt_scan_left_to_right_16u_block_prepare(
    t: &[u16],
    sa: &mut [SaSint],
    k: SaSint,
    buckets: &mut [SaSint],
    cache: &mut [ThreadCache],
    omp_block_start: SaSint,
    omp_block_size: SaSint,
) -> SaSint {
    buckets[..k as usize].fill(0);
    let mut count = 0usize;
    for i in omp_block_start as usize..(omp_block_start + omp_block_size) as usize {
        let mut p = sa[i];
        sa[i] = p & SAINT_MAX;
        if p > 0 {
            p -= 1;
            let c = t[p as usize] as usize;
            sa[i] = c as SaSint | SAINT_MIN;
            buckets[c] += 1;
            cache[count].symbol = c as SaSint;
            cache[count].index = p
                | ((usize::from(t[(p - SaSint::from(p > 0)) as usize] < t[p as usize]) as SaSint)
                    << (SAINT_BIT - 1));
            count += 1;
        }
    }
    count as SaSint
}

fn final_sorting_scan_left_to_right_16u_block_prepare(
    t: &[u16],
    sa: &mut [SaSint],
    k: SaSint,
    buckets: &mut [SaSint],
    cache: &mut [ThreadCache],
    omp_block_start: SaSint,
    omp_block_size: SaSint,
) -> SaSint {
    buckets[..k as usize].fill(0);
    let mut count = 0usize;
    for i in omp_block_start as usize..(omp_block_start + omp_block_size) as usize {
        let mut p = sa[i];
        sa[i] = p ^ SAINT_MIN;
        if p > 0 {
            p -= 1;
            let c = t[p as usize] as usize;
            buckets[c] += 1;
            cache[count].symbol = c as SaSint;
            cache[count].index = p
                | ((usize::from(t[(p - SaSint::from(p > 0)) as usize] < t[p as usize]) as SaSint)
                    << (SAINT_BIT - 1));
            count += 1;
        }
    }
    count as SaSint
}

fn final_order_scan_left_to_right_16u_block_place(
    sa: &mut [SaSint],
    buckets: &mut [SaSint],
    cache: &[ThreadCache],
    count: SaSint,
) {
    for entry in cache.iter().take(count as usize) {
        let c = entry.symbol as usize;
        let dst = buckets[c] as usize;
        sa[dst] = entry.index;
        buckets[c] += 1;
    }
}

fn final_bwt_aux_scan_left_to_right_16u_block_place(
    sa: &mut [SaSint],
    rm: SaSint,
    i_sample: &mut [SaSint],
    buckets: &mut [SaSint],
    cache: &[ThreadCache],
    count: SaSint,
) {
    for entry in cache.iter().take(count as usize) {
        let c = entry.symbol as usize;
        let dst = buckets[c] as usize;
        sa[dst] = entry.index;
        buckets[c] += 1;
        let p = entry.index & SAINT_MAX;
        if (p & rm) == 0 {
            i_sample[(p / (rm + 1)) as usize] = buckets[c];
        }
    }
}

fn final_bwt_scan_left_to_right_16u_block_omp(
    t: &[u16],
    sa: &mut [SaSint],
    k: SaSint,
    induction_bucket: &mut [SaSint],
    block_start: SaSint,
    block_size: SaSint,
    threads: SaSint,
    thread_state: &mut [ThreadState],
) {
    let thread_count = if threads > 1 && block_size >= 64 * k.max(256) {
        usize::try_from(threads)
            .expect("threads must be non-negative")
            .min(thread_state.len())
    } else {
        1
    };
    if thread_count <= 1 {
        final_bwt_scan_left_to_right_16u(t, sa, induction_bucket, block_start, block_size);
        return;
    }

    let k_usize = usize::try_from(k).expect("k must be non-negative");
    let block_stride = (block_size / thread_count as SaSint) & !15;

    for (thread, state) in thread_state.iter_mut().take(thread_count).enumerate() {
        let local_start = thread as SaSint * block_stride;
        let local_size = if thread + 1 < thread_count {
            block_stride
        } else {
            block_size - local_start
        };
        state.count = final_bwt_scan_left_to_right_16u_block_prepare(
            t,
            sa,
            k,
            &mut state.buckets[..k_usize],
            &mut state.cache,
            block_start + local_start,
            local_size,
        );
    }

    for state in thread_state.iter_mut().take(thread_count) {
        for c in 0..k_usize {
            let a = induction_bucket[c];
            let b = state.buckets[c];
            induction_bucket[c] = a + b;
            state.buckets[c] = a;
        }
    }

    for state in thread_state.iter_mut().take(thread_count) {
        final_order_scan_left_to_right_16u_block_place(
            sa,
            &mut state.buckets[..k_usize],
            &state.cache,
            state.count,
        );
    }
}

fn final_bwt_aux_scan_left_to_right_16u_block_omp(
    t: &[u16],
    sa: &mut [SaSint],
    k: SaSint,
    rm: SaSint,
    i_sample: &mut [SaSint],
    induction_bucket: &mut [SaSint],
    block_start: SaSint,
    block_size: SaSint,
    threads: SaSint,
    thread_state: &mut [ThreadState],
) {
    let thread_count = if threads > 1 && block_size >= 64 * k.max(256) {
        usize::try_from(threads)
            .expect("threads must be non-negative")
            .min(thread_state.len())
    } else {
        1
    };
    if thread_count <= 1 {
        final_bwt_aux_scan_left_to_right_16u(
            t,
            sa,
            rm,
            i_sample,
            induction_bucket,
            block_start,
            block_size,
        );
        return;
    }

    let k_usize = usize::try_from(k).expect("k must be non-negative");
    let block_stride = (block_size / thread_count as SaSint) & !15;

    for (thread, state) in thread_state.iter_mut().take(thread_count).enumerate() {
        let local_start = thread as SaSint * block_stride;
        let local_size = if thread + 1 < thread_count {
            block_stride
        } else {
            block_size - local_start
        };
        state.count = final_bwt_scan_left_to_right_16u_block_prepare(
            t,
            sa,
            k,
            &mut state.buckets[..k_usize],
            &mut state.cache,
            block_start + local_start,
            local_size,
        );
    }

    for state in thread_state.iter_mut().take(thread_count) {
        for c in 0..k_usize {
            let a = induction_bucket[c];
            let b = state.buckets[c];
            induction_bucket[c] = a + b;
            state.buckets[c] = a;
        }
    }

    for state in thread_state.iter_mut().take(thread_count) {
        final_bwt_aux_scan_left_to_right_16u_block_place(
            sa,
            rm,
            i_sample,
            &mut state.buckets[..k_usize],
            &state.cache,
            state.count,
        );
    }
}

fn final_sorting_scan_left_to_right_16u_block_omp(
    t: &[u16],
    sa: &mut [SaSint],
    k: SaSint,
    induction_bucket: &mut [SaSint],
    block_start: SaSint,
    block_size: SaSint,
    threads: SaSint,
    thread_state: &mut [ThreadState],
) {
    let thread_count = if threads > 1 && block_size >= 64 * k.max(256) {
        usize::try_from(threads)
            .expect("threads must be non-negative")
            .min(thread_state.len())
    } else {
        1
    };
    if thread_count <= 1 {
        final_sorting_scan_left_to_right_16u(t, sa, induction_bucket, block_start, block_size);
        return;
    }

    let k_usize = usize::try_from(k).expect("k must be non-negative");
    let block_stride = (block_size / thread_count as SaSint) & !15;

    for (thread, state) in thread_state.iter_mut().take(thread_count).enumerate() {
        let local_start = thread as SaSint * block_stride;
        let local_size = if thread + 1 < thread_count {
            block_stride
        } else {
            block_size - local_start
        };
        state.count = final_sorting_scan_left_to_right_16u_block_prepare(
            t,
            sa,
            k,
            &mut state.buckets[..k_usize],
            &mut state.cache,
            block_start + local_start,
            local_size,
        );
    }

    for state in thread_state.iter_mut().take(thread_count) {
        for c in 0..k_usize {
            let a = induction_bucket[c];
            let b = state.buckets[c];
            induction_bucket[c] = a + b;
            state.buckets[c] = a;
        }
    }

    for state in thread_state.iter_mut().take(thread_count) {
        final_order_scan_left_to_right_16u_block_place(
            sa,
            &mut state.buckets[..k_usize],
            &state.cache,
            state.count,
        );
    }
}

fn final_bwt_scan_left_to_right_16u_omp(
    t: &[u16],
    sa: &mut [SaSint],
    n: SaSint,
    k: SaSint,
    induction_bucket: &mut [SaSint],
    threads: SaSint,
) {
    let c = t[(n - 1) as usize] as usize;
    let dst = induction_bucket[c] as usize;
    induction_bucket[c] += 1;
    let mark = if t[(n - 2) as usize] < t[(n - 1) as usize] {
        SAINT_MIN
    } else {
        0
    };
    sa[dst] = (n - 1) | mark;

    if threads == 1 || n < 65536 {
        final_bwt_scan_left_to_right_16u(t, sa, induction_bucket, 0, n);
    } else {
        let mut thread_state = alloc_thread_state(threads).unwrap_or_default();
        let mut block_start = 0;
        while block_start < n {
            if sa[block_start as usize] == 0 {
                block_start += 1;
            } else {
                let mut block_end =
                    block_start + threads * (PER_THREAD_CACHE_SIZE as SaSint - 16 * threads);
                if block_end > n {
                    block_end = n;
                }
                let mut block_scan_end = block_start + 1;
                while block_scan_end < block_end && sa[block_scan_end as usize] != 0 {
                    block_scan_end += 1;
                }
                let block_size = block_scan_end - block_start;
                if block_size < 32 {
                    while block_start < block_scan_end {
                        let mut p = sa[block_start as usize];
                        sa[block_start as usize] = p & SAINT_MAX;
                        if p > 0 {
                            p -= 1;
                            let c = t[p as usize] as usize;
                            sa[block_start as usize] = c as SaSint | SAINT_MIN;
                            let dst = induction_bucket[c] as usize;
                            induction_bucket[c] += 1;
                            let mark = if t[(p - SaSint::from(p > 0)) as usize] < t[p as usize] {
                                SAINT_MIN
                            } else {
                                0
                            };
                            sa[dst] = p | mark;
                        }
                        block_start += 1;
                    }
                } else {
                    final_bwt_scan_left_to_right_16u_block_omp(
                        t,
                        sa,
                        k,
                        induction_bucket,
                        block_start,
                        block_size,
                        threads,
                        &mut thread_state,
                    );
                    block_start = block_scan_end;
                }
            }
        }
    }
}

fn final_bwt_aux_scan_left_to_right_16u_omp(
    t: &[u16],
    sa: &mut [SaSint],
    n: SaSint,
    k: SaSint,
    rm: SaSint,
    i_sample: &mut [SaSint],
    induction_bucket: &mut [SaSint],
    threads: SaSint,
) {
    let c = t[(n - 1) as usize] as usize;
    let dst = induction_bucket[c] as usize;
    induction_bucket[c] += 1;
    let mark = if t[(n - 2) as usize] < t[(n - 1) as usize] {
        SAINT_MIN
    } else {
        0
    };
    sa[dst] = (n - 1) | mark;

    if ((n - 1) & rm) == 0 {
        i_sample[((n - 1) / (rm + 1)) as usize] = induction_bucket[c];
    }

    if threads == 1 || n < 65536 {
        final_bwt_aux_scan_left_to_right_16u(t, sa, rm, i_sample, induction_bucket, 0, n);
    } else {
        let mut thread_state = alloc_thread_state(threads).unwrap_or_default();
        let mut block_start = 0;
        while block_start < n {
            if sa[block_start as usize] == 0 {
                block_start += 1;
            } else {
                let mut block_end =
                    block_start + threads * (PER_THREAD_CACHE_SIZE as SaSint - 16 * threads);
                if block_end > n {
                    block_end = n;
                }
                let mut block_scan_end = block_start + 1;
                while block_scan_end < block_end && sa[block_scan_end as usize] != 0 {
                    block_scan_end += 1;
                }
                let block_size = block_scan_end - block_start;
                if block_size < 32 {
                    while block_start < block_scan_end {
                        let mut p = sa[block_start as usize];
                        sa[block_start as usize] = p & SAINT_MAX;
                        if p > 0 {
                            p -= 1;
                            let c = t[p as usize] as usize;
                            sa[block_start as usize] = c as SaSint | SAINT_MIN;
                            let dst = induction_bucket[c] as usize;
                            induction_bucket[c] += 1;
                            let mark = if t[(p - SaSint::from(p > 0)) as usize] < t[p as usize] {
                                SAINT_MIN
                            } else {
                                0
                            };
                            sa[dst] = p | mark;
                            if (p & rm) == 0 {
                                i_sample[(p / (rm + 1)) as usize] = induction_bucket[c];
                            }
                        }
                        block_start += 1;
                    }
                } else {
                    final_bwt_aux_scan_left_to_right_16u_block_omp(
                        t,
                        sa,
                        k,
                        rm,
                        i_sample,
                        induction_bucket,
                        block_start,
                        block_size,
                        threads,
                        &mut thread_state,
                    );
                    block_start = block_scan_end;
                }
            }
        }
    }
}

fn final_sorting_scan_left_to_right_16u_omp(
    t: &[u16],
    sa: &mut [SaSint],
    n: SaSint,
    k: SaSint,
    induction_bucket: &mut [SaSint],
    threads: SaSint,
) {
    let c = t[(n - 1) as usize] as usize;
    let dst = induction_bucket[c] as usize;
    induction_bucket[c] += 1;
    let mark = if t[(n - 2) as usize] < t[(n - 1) as usize] {
        SAINT_MIN
    } else {
        0
    };
    sa[dst] = (n - 1) | mark;

    if threads == 1 || n < 65536 {
        final_sorting_scan_left_to_right_16u(t, sa, induction_bucket, 0, n);
    } else {
        let mut thread_state = alloc_thread_state(threads).unwrap_or_default();
        let mut block_start = 0;
        while block_start < n {
            if sa[block_start as usize] == 0 {
                block_start += 1;
            } else {
                let mut block_end =
                    block_start + threads * (PER_THREAD_CACHE_SIZE as SaSint - 16 * threads);
                if block_end > n {
                    block_end = n;
                }
                let mut block_scan_end = block_start + 1;
                while block_scan_end < block_end && sa[block_scan_end as usize] != 0 {
                    block_scan_end += 1;
                }
                let block_size = block_scan_end - block_start;
                if block_size < 32 {
                    while block_start < block_scan_end {
                        let mut p = sa[block_start as usize];
                        sa[block_start as usize] = p ^ SAINT_MIN;
                        if p > 0 {
                            p -= 1;
                            let c = t[p as usize] as usize;
                            let dst = induction_bucket[c] as usize;
                            induction_bucket[c] += 1;
                            let mark = if t[(p - SaSint::from(p > 0)) as usize] < t[p as usize] {
                                SAINT_MIN
                            } else {
                                0
                            };
                            sa[dst] = p | mark;
                        }
                        block_start += 1;
                    }
                } else {
                    final_sorting_scan_left_to_right_16u_block_omp(
                        t,
                        sa,
                        k,
                        induction_bucket,
                        block_start,
                        block_size,
                        threads,
                        &mut thread_state,
                    );
                    block_start = block_scan_end;
                }
            }
        }
    }
}

fn final_bwt_scan_right_to_left_16u_block_prepare(
    t: &[u16],
    sa: &mut [SaSint],
    k: SaSint,
    buckets: &mut [SaSint],
    cache: &mut [ThreadCache],
    omp_block_start: SaSint,
    omp_block_size: SaSint,
) -> SaSint {
    buckets[..k as usize].fill(0);
    let mut count = 0usize;
    for i in (omp_block_start as usize..(omp_block_start + omp_block_size) as usize).rev() {
        let mut p = sa[i];
        sa[i] = p & SAINT_MAX;
        if p > 0 {
            p -= 1;
            let c0 = t[(p - SaSint::from(p > 0)) as usize];
            let c1 = t[p as usize];
            sa[i] = c1 as SaSint;
            buckets[c1 as usize] += 1;
            cache[count].symbol = c1 as SaSint;
            cache[count].index = if c0 <= c1 {
                p
            } else {
                c0 as SaSint | SAINT_MIN
            };
            count += 1;
        }
    }
    count as SaSint
}

fn final_bwt_aux_scan_right_to_left_16u_block_prepare(
    t: &[u16],
    sa: &mut [SaSint],
    k: SaSint,
    buckets: &mut [SaSint],
    cache: &mut [ThreadCache],
    omp_block_start: SaSint,
    omp_block_size: SaSint,
) -> SaSint {
    buckets[..k as usize].fill(0);
    let mut count = 0usize;
    for i in (omp_block_start as usize..(omp_block_start + omp_block_size) as usize).rev() {
        let mut p = sa[i];
        sa[i] = p & SAINT_MAX;
        if p > 0 {
            p -= 1;
            let c0 = t[(p - SaSint::from(p > 0)) as usize];
            let c1 = t[p as usize];
            sa[i] = c1 as SaSint;
            buckets[c1 as usize] += 1;
            cache[count].symbol = c1 as SaSint;
            cache[count].index = if c0 <= c1 {
                p
            } else {
                c0 as SaSint | SAINT_MIN
            };
            cache[count + 1].index = p;
            count += 2;
        }
    }
    count as SaSint
}

fn final_sorting_scan_right_to_left_16u_block_prepare(
    t: &[u16],
    sa: &mut [SaSint],
    k: SaSint,
    buckets: &mut [SaSint],
    cache: &mut [ThreadCache],
    omp_block_start: SaSint,
    omp_block_size: SaSint,
) -> SaSint {
    buckets[..k as usize].fill(0);
    let mut count = 0usize;
    for i in (omp_block_start as usize..(omp_block_start + omp_block_size) as usize).rev() {
        let mut p = sa[i];
        sa[i] = p & SAINT_MAX;
        if p > 0 {
            p -= 1;
            let c = t[p as usize] as usize;
            buckets[c] += 1;
            cache[count].symbol = c as SaSint;
            cache[count].index = p
                | ((usize::from(t[(p - SaSint::from(p > 0)) as usize] > t[p as usize]) as SaSint)
                    << (SAINT_BIT - 1));
            count += 1;
        }
    }
    count as SaSint
}

fn final_order_scan_right_to_left_16u_block_place(
    sa: &mut [SaSint],
    buckets: &mut [SaSint],
    cache: &[ThreadCache],
    count: SaSint,
) {
    for entry in cache.iter().take(count as usize) {
        let c = entry.symbol as usize;
        buckets[c] -= 1;
        sa[buckets[c] as usize] = entry.index;
    }
}

fn final_gsa_scan_right_to_left_16u_block_place(
    sa: &mut [SaSint],
    buckets: &mut [SaSint],
    cache: &[ThreadCache],
    count: SaSint,
) {
    for entry in cache.iter().take(count as usize) {
        let c = entry.symbol as usize;
        if c > 0 {
            buckets[c] -= 1;
            sa[buckets[c] as usize] = entry.index;
        }
    }
}

fn final_bwt_aux_scan_right_to_left_16u_block_place(
    sa: &mut [SaSint],
    rm: SaSint,
    i_sample: &mut [SaSint],
    buckets: &mut [SaSint],
    cache: &[ThreadCache],
    count: SaSint,
) {
    let mut i = 0usize;
    while i < count as usize {
        let c = cache[i].symbol as usize;
        buckets[c] -= 1;
        sa[buckets[c] as usize] = cache[i].index;
        let p = cache[i + 1].index;
        if (p & rm) == 0 {
            i_sample[(p / (rm + 1)) as usize] = buckets[c] + 1;
        }
        i += 2;
    }
}

fn final_bwt_scan_right_to_left_16u_block_omp(
    t: &[u16],
    sa: &mut [SaSint],
    k: SaSint,
    induction_bucket: &mut [SaSint],
    block_start: SaSint,
    block_size: SaSint,
    threads: SaSint,
    thread_state: &mut [ThreadState],
) -> SaSint {
    let thread_count = if threads > 1 && block_size >= 64 * k.max(256) {
        usize::try_from(threads)
            .expect("threads must be non-negative")
            .min(thread_state.len())
    } else {
        1
    };
    if thread_count <= 1 {
        return final_bwt_scan_right_to_left_16u(t, sa, induction_bucket, block_start, block_size);
    }

    let k_usize = usize::try_from(k).expect("k must be non-negative");
    let block_stride = (block_size / thread_count as SaSint) & !15;

    for (thread, state) in thread_state.iter_mut().take(thread_count).enumerate() {
        let local_start = thread as SaSint * block_stride;
        let local_size = if thread + 1 < thread_count {
            block_stride
        } else {
            block_size - local_start
        };
        state.count = final_bwt_scan_right_to_left_16u_block_prepare(
            t,
            sa,
            k,
            &mut state.buckets[..k_usize],
            &mut state.cache,
            block_start + local_start,
            local_size,
        );
    }

    for state in thread_state.iter_mut().take(thread_count).rev() {
        for c in 0..k_usize {
            let a = induction_bucket[c];
            let b = state.buckets[c];
            induction_bucket[c] = a - b;
            state.buckets[c] = a;
        }
    }

    for state in thread_state.iter_mut().take(thread_count) {
        final_order_scan_right_to_left_16u_block_place(
            sa,
            &mut state.buckets[..k_usize],
            &state.cache,
            state.count,
        );
    }

    -1
}

fn final_bwt_aux_scan_right_to_left_16u_block_omp(
    t: &[u16],
    sa: &mut [SaSint],
    k: SaSint,
    rm: SaSint,
    i_sample: &mut [SaSint],
    induction_bucket: &mut [SaSint],
    block_start: SaSint,
    block_size: SaSint,
    threads: SaSint,
    thread_state: &mut [ThreadState],
) {
    let thread_count = if threads > 1 && block_size >= 64 * k.max(256) {
        usize::try_from(threads)
            .expect("threads must be non-negative")
            .min(thread_state.len())
    } else {
        1
    };
    if thread_count <= 1 {
        final_bwt_aux_scan_right_to_left_16u(
            t,
            sa,
            rm,
            i_sample,
            induction_bucket,
            block_start,
            block_size,
        );
        return;
    }

    let k_usize = usize::try_from(k).expect("k must be non-negative");
    let block_stride = (block_size / thread_count as SaSint) & !15;

    for (thread, state) in thread_state.iter_mut().take(thread_count).enumerate() {
        let local_start = thread as SaSint * block_stride;
        let local_size = if thread + 1 < thread_count {
            block_stride
        } else {
            block_size - local_start
        };
        state.count = final_bwt_aux_scan_right_to_left_16u_block_prepare(
            t,
            sa,
            k,
            &mut state.buckets[..k_usize],
            &mut state.cache,
            block_start + local_start,
            local_size,
        );
    }

    for state in thread_state.iter_mut().take(thread_count).rev() {
        for c in 0..k_usize {
            let a = induction_bucket[c];
            let b = state.buckets[c];
            induction_bucket[c] = a - b;
            state.buckets[c] = a;
        }
    }

    for state in thread_state.iter_mut().take(thread_count) {
        final_bwt_aux_scan_right_to_left_16u_block_place(
            sa,
            rm,
            i_sample,
            &mut state.buckets[..k_usize],
            &state.cache,
            state.count,
        );
    }
}

fn final_sorting_scan_right_to_left_16u_block_omp(
    t: &[u16],
    sa: &mut [SaSint],
    k: SaSint,
    induction_bucket: &mut [SaSint],
    block_start: SaSint,
    block_size: SaSint,
    threads: SaSint,
    thread_state: &mut [ThreadState],
) {
    let thread_count = if threads > 1 && block_size >= 64 * k.max(256) {
        usize::try_from(threads)
            .expect("threads must be non-negative")
            .min(thread_state.len())
    } else {
        1
    };
    if thread_count <= 1 {
        final_sorting_scan_right_to_left_16u(t, sa, induction_bucket, block_start, block_size);
        return;
    }

    let k_usize = usize::try_from(k).expect("k must be non-negative");
    let block_stride = (block_size / thread_count as SaSint) & !15;

    for (thread, state) in thread_state.iter_mut().take(thread_count).enumerate() {
        let local_start = thread as SaSint * block_stride;
        let local_size = if thread + 1 < thread_count {
            block_stride
        } else {
            block_size - local_start
        };
        state.count = final_sorting_scan_right_to_left_16u_block_prepare(
            t,
            sa,
            k,
            &mut state.buckets[..k_usize],
            &mut state.cache,
            block_start + local_start,
            local_size,
        );
    }

    for state in thread_state.iter_mut().take(thread_count).rev() {
        for c in 0..k_usize {
            let a = induction_bucket[c];
            let b = state.buckets[c];
            induction_bucket[c] = a - b;
            state.buckets[c] = a;
        }
    }

    for state in thread_state.iter_mut().take(thread_count) {
        final_order_scan_right_to_left_16u_block_place(
            sa,
            &mut state.buckets[..k_usize],
            &state.cache,
            state.count,
        );
    }
}

fn final_gsa_scan_right_to_left_16u_block_omp(
    t: &[u16],
    sa: &mut [SaSint],
    k: SaSint,
    induction_bucket: &mut [SaSint],
    block_start: SaSint,
    block_size: SaSint,
    threads: SaSint,
    thread_state: &mut [ThreadState],
) {
    let thread_count = if threads > 1 && block_size >= 64 * k.max(256) {
        usize::try_from(threads)
            .expect("threads must be non-negative")
            .min(thread_state.len())
    } else {
        1
    };
    if thread_count <= 1 {
        final_gsa_scan_right_to_left_16u(t, sa, induction_bucket, block_start, block_size);
        return;
    }

    let k_usize = usize::try_from(k).expect("k must be non-negative");
    let block_stride = (block_size / thread_count as SaSint) & !15;

    for (thread, state) in thread_state.iter_mut().take(thread_count).enumerate() {
        let local_start = thread as SaSint * block_stride;
        let local_size = if thread + 1 < thread_count {
            block_stride
        } else {
            block_size - local_start
        };
        state.count = final_sorting_scan_right_to_left_16u_block_prepare(
            t,
            sa,
            k,
            &mut state.buckets[..k_usize],
            &mut state.cache,
            block_start + local_start,
            local_size,
        );
    }

    for state in thread_state.iter_mut().take(thread_count).rev() {
        for c in 0..k_usize {
            let a = induction_bucket[c];
            let b = state.buckets[c];
            induction_bucket[c] = a - b;
            state.buckets[c] = a;
        }
    }

    for state in thread_state.iter_mut().take(thread_count) {
        final_gsa_scan_right_to_left_16u_block_place(
            sa,
            &mut state.buckets[..k_usize],
            &state.cache,
            state.count,
        );
    }
}

fn final_bwt_scan_right_to_left_16u_omp(
    t: &[u16],
    sa: &mut [SaSint],
    n: SaSint,
    k: SaSint,
    induction_bucket: &mut [SaSint],
    threads: SaSint,
) -> SaSint {
    let mut index = -1;

    if threads == 1 || n < 65536 {
        index = final_bwt_scan_right_to_left_16u(t, sa, induction_bucket, 0, n);
    } else {
        let mut thread_state = alloc_thread_state(threads).unwrap_or_default();
        let mut block_start = n - 1;
        while block_start >= 0 {
            if sa[block_start as usize] == 0 {
                index = block_start;
                block_start -= 1;
            } else {
                let mut block_max_end =
                    block_start - threads * (PER_THREAD_CACHE_SIZE as SaSint - 16 * threads);
                if block_max_end < 0 {
                    block_max_end = -1;
                }
                let mut block_end = block_start - 1;
                while block_end > block_max_end && sa[block_end as usize] != 0 {
                    block_end -= 1;
                }
                let block_size = block_start - block_end;
                if block_size < 32 {
                    while block_start > block_end {
                        let mut p = sa[block_start as usize];
                        sa[block_start as usize] = p & SAINT_MAX;
                        if p > 0 {
                            p -= 1;
                            let c0 = t[(p - SaSint::from(p > 0)) as usize];
                            let c1 = t[p as usize] as usize;
                            sa[block_start as usize] = c1 as SaSint;
                            induction_bucket[c1] -= 1;
                            sa[induction_bucket[c1] as usize] = if c0 <= c1 as u16 {
                                p
                            } else {
                                c0 as SaSint | SAINT_MIN
                            };
                        }
                        block_start -= 1;
                    }
                } else {
                    final_bwt_scan_right_to_left_16u_block_omp(
                        t,
                        sa,
                        k,
                        induction_bucket,
                        block_end + 1,
                        block_size,
                        threads,
                        &mut thread_state,
                    );
                    block_start = block_end;
                }
            }
        }
    }
    index
}

fn final_bwt_aux_scan_right_to_left_16u_omp(
    t: &[u16],
    sa: &mut [SaSint],
    n: SaSint,
    k: SaSint,
    rm: SaSint,
    i_sample: &mut [SaSint],
    induction_bucket: &mut [SaSint],
    threads: SaSint,
) {
    if threads == 1 || n < 65536 {
        final_bwt_aux_scan_right_to_left_16u(t, sa, rm, i_sample, induction_bucket, 0, n);
    } else {
        let mut thread_state = alloc_thread_state(threads).unwrap_or_default();
        let mut block_start = n - 1;
        while block_start >= 0 {
            if sa[block_start as usize] == 0 {
                block_start -= 1;
            } else {
                let mut block_max_end =
                    block_start - threads * ((PER_THREAD_CACHE_SIZE as SaSint - 16 * threads) / 2);
                if block_max_end < 0 {
                    block_max_end = -1;
                }
                let mut block_end = block_start - 1;
                while block_end > block_max_end && sa[block_end as usize] != 0 {
                    block_end -= 1;
                }
                let block_size = block_start - block_end;
                if block_size < 32 {
                    while block_start > block_end {
                        let mut p = sa[block_start as usize];
                        sa[block_start as usize] = p & SAINT_MAX;
                        if p > 0 {
                            p -= 1;
                            let c0 = t[(p - SaSint::from(p > 0)) as usize];
                            let c1 = t[p as usize] as usize;
                            sa[block_start as usize] = c1 as SaSint;
                            induction_bucket[c1] -= 1;
                            sa[induction_bucket[c1] as usize] = if c0 <= c1 as u16 {
                                p
                            } else {
                                c0 as SaSint | SAINT_MIN
                            };
                            if (p & rm) == 0 {
                                i_sample[(p / (rm + 1)) as usize] = induction_bucket[c1] + 1;
                            }
                        }
                        block_start -= 1;
                    }
                } else {
                    final_bwt_aux_scan_right_to_left_16u_block_omp(
                        t,
                        sa,
                        k,
                        rm,
                        i_sample,
                        induction_bucket,
                        block_end + 1,
                        block_size,
                        threads,
                        &mut thread_state,
                    );
                    block_start = block_end;
                }
            }
        }
    }
}

fn final_sorting_scan_right_to_left_16u_omp(
    t: &[u16],
    sa: &mut [SaSint],
    omp_block_start: SaSint,
    omp_block_size: SaSint,
    k: SaSint,
    induction_bucket: &mut [SaSint],
    threads: SaSint,
) {
    if threads == 1 || omp_block_size < 65536 {
        final_sorting_scan_right_to_left_16u(
            t,
            sa,
            induction_bucket,
            omp_block_start,
            omp_block_size,
        );
    } else {
        let mut thread_state = alloc_thread_state(threads).unwrap_or_default();
        let mut block_start = omp_block_start + omp_block_size - 1;
        while block_start >= omp_block_start {
            if sa[block_start as usize] == 0 {
                block_start -= 1;
            } else {
                let mut block_max_end =
                    block_start - threads * (PER_THREAD_CACHE_SIZE as SaSint - 16 * threads);
                if block_max_end < omp_block_start {
                    block_max_end = omp_block_start - 1;
                }
                let mut block_end = block_start - 1;
                while block_end > block_max_end && sa[block_end as usize] != 0 {
                    block_end -= 1;
                }
                let block_size = block_start - block_end;
                if block_size < 32 {
                    while block_start > block_end {
                        let mut p = sa[block_start as usize];
                        sa[block_start as usize] = p & SAINT_MAX;
                        if p > 0 {
                            p -= 1;
                            let c = t[p as usize] as usize;
                            induction_bucket[c] -= 1;
                            let mark = if t[(p - SaSint::from(p > 0)) as usize] > t[p as usize] {
                                SAINT_MIN
                            } else {
                                0
                            };
                            sa[induction_bucket[c] as usize] = p | mark;
                        }
                        block_start -= 1;
                    }
                } else {
                    final_sorting_scan_right_to_left_16u_block_omp(
                        t,
                        sa,
                        k,
                        induction_bucket,
                        block_end + 1,
                        block_size,
                        threads,
                        &mut thread_state,
                    );
                    block_start = block_end;
                }
            }
        }
    }
}

fn final_gsa_scan_right_to_left_16u_omp(
    t: &[u16],
    sa: &mut [SaSint],
    omp_block_start: SaSint,
    omp_block_size: SaSint,
    k: SaSint,
    induction_bucket: &mut [SaSint],
    threads: SaSint,
) {
    if threads == 1 || omp_block_size < 65536 {
        final_gsa_scan_right_to_left_16u(t, sa, induction_bucket, omp_block_start, omp_block_size);
    } else {
        let mut thread_state = alloc_thread_state(threads).unwrap_or_default();
        let mut block_start = omp_block_start + omp_block_size - 1;
        while block_start >= omp_block_start {
            if sa[block_start as usize] == 0 {
                block_start -= 1;
            } else {
                let mut block_max_end =
                    block_start - threads * (PER_THREAD_CACHE_SIZE as SaSint - 16 * threads);
                if block_max_end < omp_block_start {
                    block_max_end = omp_block_start - 1;
                }
                let mut block_end = block_start - 1;
                while block_end > block_max_end && sa[block_end as usize] != 0 {
                    block_end -= 1;
                }
                let block_size = block_start - block_end;
                if block_size < 32 {
                    while block_start > block_end {
                        let mut p = sa[block_start as usize];
                        sa[block_start as usize] = p & SAINT_MAX;
                        if p > 0 && t[(p - 1) as usize] > 0 {
                            p -= 1;
                            let c = t[p as usize] as usize;
                            induction_bucket[c] -= 1;
                            let mark = if t[(p - SaSint::from(p > 0)) as usize] > t[p as usize] {
                                SAINT_MIN
                            } else {
                                0
                            };
                            sa[induction_bucket[c] as usize] = p | mark;
                        }
                        block_start -= 1;
                    }
                } else {
                    final_gsa_scan_right_to_left_16u_block_omp(
                        t,
                        sa,
                        k,
                        induction_bucket,
                        block_end + 1,
                        block_size,
                        threads,
                        &mut thread_state,
                    );
                    block_start = block_end;
                }
            }
        }
    }
}

fn induce_final_order_16u_omp(
    t: &[u16],
    sa: &mut [SaSint],
    n: SaSint,
    k: SaSint,
    flags: SaSint,
    r: SaSint,
    i_out: Option<&mut [SaSint]>,
    buckets: &mut [SaSint],
    threads: SaSint,
    _thread_state: &mut [ThreadState],
) -> SaSint {
    if (flags & LIBSAIS_FLAGS_BWT) == 0 {
        if (flags & LIBSAIS_FLAGS_GSA) != 0 {
            buckets[6 * ALPHABET_SIZE] = buckets[7 * ALPHABET_SIZE] - 1;
        }

        let (left_buckets, right_tail) = buckets.split_at_mut(7 * ALPHABET_SIZE);
        let bucket_start = &mut left_buckets[6 * ALPHABET_SIZE..7 * ALPHABET_SIZE];
        let bucket_end = &mut right_tail[..ALPHABET_SIZE];

        final_sorting_scan_left_to_right_16u_omp(t, sa, n, k, bucket_start, threads);
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
            final_gsa_scan_right_to_left_16u_omp(
                t,
                sa,
                bucket_end[0],
                n - bucket_end[0],
                k,
                bucket_end,
                threads,
            );
        } else {
            final_sorting_scan_right_to_left_16u_omp(t, sa, 0, n, k, bucket_end, threads);
        }

        0
    } else if let Some(i_out) = i_out {
        let (left_buckets, right_tail) = buckets.split_at_mut(7 * ALPHABET_SIZE);
        let bucket_start = &mut left_buckets[6 * ALPHABET_SIZE..7 * ALPHABET_SIZE];
        let bucket_end = &mut right_tail[..ALPHABET_SIZE];

        final_bwt_aux_scan_left_to_right_16u_omp(t, sa, n, k, r - 1, i_out, bucket_start, threads);
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
        final_bwt_aux_scan_right_to_left_16u_omp(t, sa, n, k, r - 1, i_out, bucket_end, threads);
        0
    } else {
        let (left_buckets, right_tail) = buckets.split_at_mut(7 * ALPHABET_SIZE);
        let bucket_start = &mut left_buckets[6 * ALPHABET_SIZE..7 * ALPHABET_SIZE];
        let bucket_end = &mut right_tail[..ALPHABET_SIZE];

        final_bwt_scan_left_to_right_16u_omp(t, sa, n, k, bucket_start, threads);
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
        final_bwt_scan_right_to_left_16u_omp(t, sa, n, k, bucket_end, threads)
    }
}

fn bwt_copy_16u(u: &mut [u16], a: &[SaSint], n: SaSint) {
    let mut i = 0isize;
    let mut j = n as isize - 7;
    while i < j {
        u[i as usize] = a[i as usize] as u16;
        u[(i + 1) as usize] = a[(i + 1) as usize] as u16;
        u[(i + 2) as usize] = a[(i + 2) as usize] as u16;
        u[(i + 3) as usize] = a[(i + 3) as usize] as u16;
        u[(i + 4) as usize] = a[(i + 4) as usize] as u16;
        u[(i + 5) as usize] = a[(i + 5) as usize] as u16;
        u[(i + 6) as usize] = a[(i + 6) as usize] as u16;
        u[(i + 7) as usize] = a[(i + 7) as usize] as u16;
        i += 8;
    }

    j += 7;
    while i < j {
        u[i as usize] = a[i as usize] as u16;
        i += 1;
    }
}

#[allow(dead_code)]
fn bwt_copy_16u_omp(u: &mut [u16], a: &[SaSint], n: SaSint, threads: SaSint) {
    if threads == 1 || n < 65_536 {
        bwt_copy_16u(u, a, n);
        return;
    }

    let block_stride = (n / threads) & !15;
    for thread in 0..threads {
        let block_start = thread * block_stride;
        let block_size = if thread < threads - 1 {
            block_stride
        } else {
            n - block_start
        };
        let start = block_start as usize;
        bwt_copy_16u(&mut u[start..], &a[start..], block_size);
    }
}

fn final_bwt_ltr_step(t: &[u16], sa: &mut [SaSint], induction_bucket: &mut [SaSint], index: usize) {
    let mut p = sa[index];
    sa[index] = p & SAINT_MAX;
    if p > 0 {
        p -= 1;
        let c = t[p as usize] as usize;
        sa[index] = t[p as usize] as SaSint | SAINT_MIN;
        let mark = if t[(p - SaSint::from(p > 0)) as usize] < t[p as usize] {
            SAINT_MIN
        } else {
            0
        };
        let dst = induction_bucket[c] as usize;
        sa[dst] = p | mark;
        induction_bucket[c] += 1;
    }
}

fn final_bwt_rtl_step(
    t: &[u16],
    sa: &mut [SaSint],
    induction_bucket: &mut [SaSint],
    index: usize,
    primary_index: &mut SaSint,
) {
    let mut p = sa[index];
    if p == 0 {
        *primary_index = index as SaSint;
    }
    sa[index] = p & SAINT_MAX;
    if p > 0 {
        p -= 1;
        let c0 = t[(p - SaSint::from(p > 0)) as usize];
        let c1 = t[p as usize];
        sa[index] = c1 as SaSint;
        let induced = if c0 <= c1 {
            p
        } else {
            c0 as SaSint | SAINT_MIN
        };
        induction_bucket[c1 as usize] -= 1;
        sa[induction_bucket[c1 as usize] as usize] = induced;
    }
}

fn final_bwt_aux_ltr_step(
    t: &[u16],
    sa: &mut [SaSint],
    rm: SaSint,
    i_sample: &mut [SaSint],
    induction_bucket: &mut [SaSint],
    index: usize,
) {
    let mut p = sa[index];
    sa[index] = p & SAINT_MAX;
    if p > 0 {
        p -= 1;
        let c = t[p as usize] as usize;
        sa[index] = t[p as usize] as SaSint | SAINT_MIN;
        let mark = if t[(p - SaSint::from(p > 0)) as usize] < t[p as usize] {
            SAINT_MIN
        } else {
            0
        };
        let dst = induction_bucket[c] as usize;
        sa[dst] = p | mark;
        induction_bucket[c] += 1;
        if (p & rm) == 0 {
            i_sample[(p / (rm + 1)) as usize] = induction_bucket[c];
        }
    }
}

fn final_bwt_aux_rtl_step(
    t: &[u16],
    sa: &mut [SaSint],
    rm: SaSint,
    i_sample: &mut [SaSint],
    induction_bucket: &mut [SaSint],
    index: usize,
) {
    let mut p = sa[index];
    sa[index] = p & SAINT_MAX;
    if p > 0 {
        p -= 1;
        let c0 = t[(p - SaSint::from(p > 0)) as usize];
        let c1 = t[p as usize];
        sa[index] = c1 as SaSint;
        let induced = if c0 <= c1 {
            p
        } else {
            c0 as SaSint | SAINT_MIN
        };
        induction_bucket[c1 as usize] -= 1;
        sa[induction_bucket[c1 as usize] as usize] = induced;
        if (p & rm) == 0 {
            i_sample[(p / (rm + 1)) as usize] = induction_bucket[c1 as usize] + 1;
        }
    }
}

fn main_32s_recursion(
    t_ptr: *mut SaSint,
    sa_ptr: *mut SaSint,
    sa_capacity: usize,
    n: SaSint,
    k: SaSint,
    fs: SaSint,
    threads: SaSint,
    thread_state: &mut [ThreadState],
    local_buffer: &mut [SaSint],
) -> SaSint {
    let fs = fs.min(SAINT_MAX - n);
    let local_buffer_size = SaSint::try_from(LIBSAIS_LOCAL_BUFFER_SIZE).expect("fits");
    let n_usize = usize::try_from(n).expect("n must be non-negative");
    let fs_usize = usize::try_from(fs).expect("fs must be non-negative");
    let total_len = n_usize + fs_usize;
    assert!(total_len <= sa_capacity);

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
            local_buffer.as_mut_ptr()
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
                    std::slice::from_raw_parts(t_ptr, n_usize),
                    k,
                    buckets,
                    first_lms_suffix,
                )
            };

            unsafe {
                let t = std::slice::from_raw_parts(t_ptr, n_usize);
                let sa = std::slice::from_raw_parts_mut(sa_ptr, total_len);
                let buckets = std::slice::from_raw_parts_mut(buckets_ptr, need);
                let (_, induction_bucket) = buckets.split_at_mut(4 * k_usize);
                radix_sort_lms_suffixes_32s_6k_omp(t, sa, n, m, induction_bucket, threads);
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
                        compact_lms_suffixes_32s_omp(t, sa, n, m, fs, threads)
                    }
                } else {
                    0
                };

                let new_t_start =
                    total_len - usize::try_from(m - f).expect("m - f must be non-negative");
                if main_32s_recursion(
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
                    local_buffer,
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
                    let t = std::slice::from_raw_parts(t_ptr, n_usize);
                    let buckets = std::slice::from_raw_parts_mut(buckets_ptr, need);
                    count_lms_suffixes_32s_2k(t, n, k, buckets);
                }
            }

            unsafe {
                let buckets = std::slice::from_raw_parts_mut(buckets_ptr, need);
                initialize_buckets_start_and_end_32s_4k(k, buckets);
                let sa = std::slice::from_raw_parts_mut(sa_ptr, total_len);
                place_lms_suffixes_histogram_32s_4k(sa, n, k, m, buckets);
                let t = std::slice::from_raw_parts(t_ptr, n_usize);
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
                let t = std::slice::from_raw_parts(t_ptr, n_usize);
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
            local_buffer.as_mut_ptr()
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
            let t = std::slice::from_raw_parts(t_ptr, n_usize);
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
                let t = std::slice::from_raw_parts(t_ptr, n_usize);
                let buckets = std::slice::from_raw_parts_mut(buckets_ptr, need);
                let sa = std::slice::from_raw_parts_mut(sa_ptr, total_len);
                initialize_buckets_for_radix_and_partial_sorting_32s_4k(
                    t,
                    k,
                    buckets,
                    sa[n_usize - m_usize],
                );
                let (_, induction_bucket) = buckets.split_at_mut(1);
                radix_sort_lms_suffixes_32s_2k_omp(t, sa, n, m, induction_bucket, threads);
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
                    compact_lms_suffixes_32s_omp(t, sa, n, m, fs, threads)
                };

                let new_t_start =
                    total_len - usize::try_from(m - f).expect("m - f must be non-negative");
                if main_32s_recursion(
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
                    local_buffer,
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
                    let t = std::slice::from_raw_parts(t_ptr, n_usize);
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
            let t = std::slice::from_raw_parts(t_ptr, n_usize);
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
            local_buffer.as_mut_ptr()
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
            let t = std::slice::from_raw_parts(t_ptr, n_usize);
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
                let t = std::slice::from_raw_parts(t_ptr, n_usize);
                let buckets = std::slice::from_raw_parts_mut(buckets_ptr, need);
                let sa = std::slice::from_raw_parts_mut(sa_ptr, total_len);
                initialize_buckets_for_lms_suffixes_radix_sort_32s_2k(
                    t,
                    k,
                    buckets,
                    sa[n_usize - m_usize],
                );
                let (_, induction_bucket) = buckets.split_at_mut(1);
                radix_sort_lms_suffixes_32s_2k_omp(t, sa, n, m, induction_bucket, threads);
                place_lms_suffixes_interval_32s_2k(sa, n, k, m - 1, buckets);
            }

            unsafe {
                let buckets = std::slice::from_raw_parts_mut(buckets_ptr, need);
                initialize_buckets_start_and_end_32s_2k(k, buckets);
                let t = std::slice::from_raw_parts(t_ptr, n_usize);
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
                    compact_lms_suffixes_32s_omp(t, sa, n, m, fs, threads)
                };

                let new_t_start =
                    total_len - usize::try_from(m - f).expect("m - f must be non-negative");
                if main_32s_recursion(
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
                    local_buffer,
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
                    let t = std::slice::from_raw_parts(t_ptr, n_usize);
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
            let t = std::slice::from_raw_parts(t_ptr, n_usize);
            let sa = std::slice::from_raw_parts_mut(sa_ptr, total_len);
            induce_final_order_32s_2k(t, sa, n, k, buckets, threads, thread_state);
        }

        0
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
            let t = std::slice::from_raw_parts(t_ptr, n_usize);
            let buckets = std::slice::from_raw_parts_mut(buckets_ptr, k_usize);
            count_suffixes_32s(t, n, k, buckets);
            initialize_buckets_end_32s_1k(k, buckets);
        }

        let m = unsafe {
            let t = std::slice::from_raw_parts(t_ptr, n_usize);
            let sa = std::slice::from_raw_parts_mut(sa_ptr, total_len);
            let buckets = std::slice::from_raw_parts_mut(buckets_ptr, k_usize);
            radix_sort_lms_suffixes_32s_1k(t, sa, n, buckets)
        };
        if m > 1 {
            unsafe {
                let t = std::slice::from_raw_parts(t_ptr, n_usize);
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
                    compact_lms_suffixes_32s_omp(t, sa, n, m, fs, threads)
                };

                let new_t_start =
                    total_len - usize::try_from(m - f).expect("m - f must be non-negative");
                if main_32s_recursion(
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
                    local_buffer,
                ) != 0
                {
                    return -2;
                }

                unsafe {
                    let t = std::slice::from_raw_parts_mut(t_ptr, n_usize);
                    let sa = std::slice::from_raw_parts_mut(sa_ptr, total_len);
                    reconstruct_compacted_lms_suffixes_32s_1k_omp(t, sa, n, m, fs, f, threads);
                }

                if buckets_ptr.is_null() {
                    heap_buckets = Some(vec![0; k_usize]);
                    buckets_ptr = heap_buckets.as_mut().unwrap().as_mut_ptr();
                    if buckets_ptr.is_null() {
                        return -2;
                    }
                }
            }

            unsafe {
                let t = std::slice::from_raw_parts(t_ptr, n_usize);
                let buckets = std::slice::from_raw_parts_mut(buckets_ptr, k_usize);
                count_suffixes_32s(t, n, k, buckets);
                initialize_buckets_end_32s_1k(k, buckets);
            }
            unsafe {
                let t = std::slice::from_raw_parts(t_ptr, n_usize);
                let sa = std::slice::from_raw_parts_mut(sa_ptr, total_len);
                let buckets = std::slice::from_raw_parts_mut(buckets_ptr, k_usize);
                place_lms_suffixes_interval_32s_1k(t, sa, k, m, buckets);
            }
        }

        unsafe {
            let t = std::slice::from_raw_parts(t_ptr, n_usize);
            let sa = std::slice::from_raw_parts_mut(sa_ptr, total_len);
            let buckets = std::slice::from_raw_parts_mut(buckets_ptr, k_usize);
            induce_final_order_32s_1k(t, sa, n, k, buckets, threads, thread_state);
        }

        0
    }
}

fn main_32s_entry(
    t_ptr: *mut SaSint,
    sa: &mut [SaSint],
    n: SaSint,
    k: SaSint,
    fs: SaSint,
    threads: SaSint,
    thread_state: &mut [ThreadState],
) -> SaSint {
    let mut local_buffer = [0; 2 * LIBSAIS_LOCAL_BUFFER_SIZE];
    main_32s_recursion(
        t_ptr,
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

fn main_16u(
    t: &[u16],
    sa: &mut [SaSint],
    n: SaSint,
    buckets: &mut [SaSint],
    flags: SaSint,
    r: SaSint,
    i_out: Option<&mut [SaSint]>,
    fs: SaSint,
    freq: Option<&mut [SaSint]>,
    threads: SaSint,
    thread_state: &mut [ThreadState],
) -> SaSint {
    let fs = fs.min(SAINT_MAX - n);

    let m = count_and_gather_lms_suffixes_16u_omp(t, sa, n, buckets, threads, thread_state);
    let k = initialize_buckets_start_and_end_16u(buckets, freq);

    if (flags & LIBSAIS_FLAGS_GSA) != 0 && (buckets[0] != 0 || buckets[2] != 0 || buckets[3] != 1) {
        return -1;
    }

    if m > 0 {
        let first_lms_suffix = sa[(n - m) as usize];
        let left_suffixes_count =
            initialize_buckets_for_lms_suffixes_radix_sort_16u(t, buckets, first_lms_suffix);

        if threads > 1 && n >= 65_536 {
            sa[..(n - m) as usize].fill(0);
        }
        radix_sort_lms_suffixes_16u_omp(t, sa, n, m, flags, buckets, threads, thread_state);
        if threads > 1 && n >= 65_536 {
            sa[(n - m) as usize..n as usize].fill(0);
        }

        initialize_buckets_for_partial_sorting_16u(
            t,
            buckets,
            first_lms_suffix,
            left_suffixes_count,
        );
        induce_partial_order_16u_omp(
            t,
            sa,
            n,
            k,
            flags,
            buckets,
            first_lms_suffix,
            left_suffixes_count,
            threads,
        );

        let names = renumber_and_gather_lms_suffixes_omp(sa, n, m, fs, threads, thread_state);
        if names < m {
            let recursive_t_start = (n + fs - m) as usize;
            let recursive_t_ptr = sa[recursive_t_start..].as_mut_ptr();
            if main_32s_entry(
                recursive_t_ptr,
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

            gather_lms_suffixes_16u_omp(t, sa, n, threads, thread_state);
            reconstruct_lms_suffixes_omp(sa, n, m, threads);
        }

        place_lms_suffixes_interval_16u(sa, n, m, flags, buckets);
    } else {
        sa[..n as usize].fill(0);
    }

    induce_final_order_16u_omp(t, sa, n, k, flags, r, i_out, buckets, threads, thread_state)
}

fn main_16u_alloc(
    t: &[u16],
    sa: &mut [SaSint],
    flags: SaSint,
    r: SaSint,
    i_out: Option<&mut [SaSint]>,
    fs: SaSint,
    freq: Option<&mut [SaSint]>,
    threads: SaSint,
) -> SaSint {
    if fs < 0
        || threads < 0
        || sa.len()
            < t.len()
                .saturating_add(usize::try_from(fs).unwrap_or(usize::MAX))
        || freq.as_ref().is_some_and(|freq| freq.len() < ALPHABET_SIZE)
    {
        return -1;
    }

    fill_freq(t, freq);
    if t.len() <= 1 {
        if t.len() == 1 {
            sa[0] = 0;
        }
        return if (flags & LIBSAIS_FLAGS_BWT) != 0 {
            t.len() as SaSint
        } else {
            0
        };
    }

    let mut buckets = vec![0; 8 * ALPHABET_SIZE];
    let threads = normalize_threads(threads);
    let mut thread_state = if threads > 1 {
        match alloc_thread_state(threads) {
            Some(thread_state) => thread_state,
            None => return -2,
        }
    } else {
        Vec::new()
    };

    main_16u(
        t,
        sa,
        t.len() as SaSint,
        &mut buckets,
        flags,
        r,
        i_out,
        fs,
        None,
        threads,
        &mut thread_state,
    )
}

fn main_16u_ctx(
    ctx: &mut Context,
    t: &[u16],
    sa: &mut [SaSint],
    flags: SaSint,
    r: SaSint,
    i_out: Option<&mut [SaSint]>,
    fs: SaSint,
    freq: Option<&mut [SaSint]>,
) -> SaSint {
    if fs < 0
        || sa.len()
            < t.len()
                .saturating_add(usize::try_from(fs).unwrap_or(usize::MAX))
        || freq.as_ref().is_some_and(|freq| freq.len() < ALPHABET_SIZE)
    {
        return -1;
    }

    if ctx.threads <= 0 || ctx.buckets.len() < 8 * ALPHABET_SIZE {
        return -2;
    }

    fill_freq(t, freq);
    if t.len() <= 1 {
        if t.len() == 1 {
            sa[0] = 0;
        }
        return if (flags & LIBSAIS_FLAGS_BWT) != 0 {
            t.len() as SaSint
        } else {
            0
        };
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

    main_16u(
        t,
        sa,
        t.len() as SaSint,
        &mut ctx.buckets,
        flags,
        r,
        i_out,
        fs,
        None,
        ctx.threads,
        thread_state,
    )
}

fn main_int(t: &mut [SaSint], sa: &mut [SaSint], k: SaSint, fs: SaSint, threads: SaSint) -> SaSint {
    let threads = normalize_threads(threads);
    let mut thread_state = if threads > 1 {
        match alloc_thread_state(threads) {
            Some(thread_state) => thread_state,
            None => return -2,
        }
    } else {
        Vec::new()
    };

    main_32s_entry(
        t.as_mut_ptr(),
        sa,
        t.len() as SaSint,
        k,
        fs,
        threads,
        &mut thread_state,
    )
}

/// Constructs the suffix array of a given 16-bit string.
///
/// - `t` (`[0..n-1]`): the input 16-bit string.
/// - `sa` (`[0..n-1+fs]`): the output array of suffixes.
/// - `fs`: extra space available at the end of `sa` (0 should be enough for most cases).
/// - `freq` (`[0..65535]`): optional output symbol frequency table.
///
/// Returns 0 on success, -1 or -2 on error.
pub fn libsais16(t: &[u16], sa: &mut [SaSint], fs: SaSint, freq: Option<&mut [SaSint]>) -> SaSint {
    main_16u_alloc(t, sa, 0, 0, None, fs, freq, 1)
}

/// Constructs the generalized suffix array (GSA) of a given 16-bit string set.
///
/// - `t` (`[0..n-1]`): the input 16-bit string set using 0 as separators (`t[n-1]` must be 0).
/// - `sa` (`[0..n-1+fs]`): the output array of suffixes.
/// - `fs`: extra space available at the end of `sa` (0 should be enough for most cases).
/// - `freq` (`[0..65535]`): optional output symbol frequency table.
///
/// Returns 0 on success, -1 or -2 on error.
pub fn libsais16_gsa(
    t: &[u16],
    sa: &mut [SaSint],
    fs: SaSint,
    freq: Option<&mut [SaSint]>,
) -> SaSint {
    main_16u_alloc(t, sa, LIBSAIS_FLAGS_GSA, 0, None, fs, freq, 1)
}

/// Constructs the suffix array of a given integer array.
///
/// During construction the input array is modified, but restored at the end if no error occurred.
///
/// - `t` (`[0..n-1]`): the input integer array.
/// - `sa` (`[0..n-1+fs]`): the output array of suffixes.
/// - `k`: the alphabet size of the input integer array.
/// - `fs`: extra space available at the end of `sa` (can be 0, but 4k or better 6k is recommended for optimal performance).
///
/// Returns 0 on success, -1 or -2 on error.
pub fn libsais16_int(t: &mut [SaSint], sa: &mut [SaSint], k: SaSint, fs: SaSint) -> SaSint {
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

    main_int(t, sa, k, fs, 1)
}

/// Constructs the suffix array of a given 16-bit string using a libsais16 context.
///
/// - `ctx`: the libsais16 context.
/// - `t` (`[0..n-1]`): the input 16-bit string.
/// - `sa` (`[0..n-1+fs]`): the output array of suffixes.
/// - `fs`: extra space available at the end of `sa` (0 should be enough for most cases).
/// - `freq` (`[0..65535]`): optional output symbol frequency table.
///
/// Returns 0 on success, -1 or -2 on error.
pub fn libsais16_ctx(
    ctx: &mut Context,
    t: &[u16],
    sa: &mut [SaSint],
    fs: SaSint,
    freq: Option<&mut [SaSint]>,
) -> SaSint {
    main_16u_ctx(ctx, t, sa, 0, 0, None, fs, freq)
}

/// Constructs the generalized suffix array (GSA) of a given 16-bit string set using a libsais16 context.
///
/// - `ctx`: the libsais16 context.
/// - `t` (`[0..n-1]`): the input 16-bit string set using 0 as separators (`t[n-1]` must be 0).
/// - `sa` (`[0..n-1+fs]`): the output array of suffixes.
/// - `fs`: extra space available at the end of `sa` (0 should be enough for most cases).
/// - `freq` (`[0..65535]`): optional output symbol frequency table.
///
/// Returns 0 on success, -1 or -2 on error.
pub fn libsais16_gsa_ctx(
    ctx: &mut Context,
    t: &[u16],
    sa: &mut [SaSint],
    fs: SaSint,
    freq: Option<&mut [SaSint]>,
) -> SaSint {
    main_16u_ctx(ctx, t, sa, LIBSAIS_FLAGS_GSA, 0, None, fs, freq)
}

/// Constructs the suffix array of a given 16-bit string in parallel using OpenMP-style threading.
///
/// - `t` (`[0..n-1]`): the input 16-bit string.
/// - `sa` (`[0..n-1+fs]`): the output array of suffixes.
/// - `fs`: extra space available at the end of `sa` (0 should be enough for most cases).
/// - `freq` (`[0..65535]`): optional output symbol frequency table.
/// - `threads`: number of worker threads (can be 0 for the implementation default).
///
/// Returns 0 on success, -1 or -2 on error.
pub fn libsais16_omp(
    t: &[u16],
    sa: &mut [SaSint],
    fs: SaSint,
    freq: Option<&mut [SaSint]>,
    threads: SaSint,
) -> SaSint {
    if threads < 0 {
        -1
    } else {
        main_16u_alloc(t, sa, 0, 0, None, fs, freq, threads)
    }
}

/// Constructs the generalized suffix array (GSA) of a given 16-bit string set in parallel using OpenMP-style threading.
///
/// - `t` (`[0..n-1]`): the input 16-bit string set using 0 as separators (`t[n-1]` must be 0).
/// - `sa` (`[0..n-1+fs]`): the output array of suffixes.
/// - `fs`: extra space available at the end of `sa` (0 should be enough for most cases).
/// - `freq` (`[0..65535]`): optional output symbol frequency table.
/// - `threads`: number of worker threads (can be 0 for the implementation default).
///
/// Returns 0 on success, -1 or -2 on error.
pub fn libsais16_gsa_omp(
    t: &[u16],
    sa: &mut [SaSint],
    fs: SaSint,
    freq: Option<&mut [SaSint]>,
    threads: SaSint,
) -> SaSint {
    if threads < 0 {
        -1
    } else {
        main_16u_alloc(t, sa, LIBSAIS_FLAGS_GSA, 0, None, fs, freq, threads)
    }
}

/// Constructs the suffix array of a given integer array in parallel using OpenMP-style threading.
///
/// During construction the input array is modified, but restored at the end if no error occurred.
///
/// - `t` (`[0..n-1]`): the input integer array.
/// - `sa` (`[0..n-1+fs]`): the output array of suffixes.
/// - `k`: the alphabet size of the input integer array.
/// - `fs`: extra space available at the end of `sa` (can be 0, but 4k or better 6k is recommended for optimal performance).
/// - `threads`: number of worker threads (can be 0 for the implementation default).
///
/// Returns 0 on success, -1 or -2 on error.
pub fn libsais16_int_omp(
    t: &mut [SaSint],
    sa: &mut [SaSint],
    k: SaSint,
    fs: SaSint,
    threads: SaSint,
) -> SaSint {
    if threads < 0
        || fs < 0
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

    main_int(t, sa, k, fs, threads)
}

fn build_bwt(
    t: &[u16],
    u: &mut [u16],
    a: &mut [SaSint],
    fs: SaSint,
    freq: Option<&mut [SaSint]>,
    threads: SaSint,
) -> SaSint {
    if fs < 0
        || threads < 0
        || u.len() < t.len()
        || a.len()
            < t.len()
                .saturating_add(usize::try_from(fs).unwrap_or(usize::MAX))
        || freq.as_ref().is_some_and(|freq| freq.len() < ALPHABET_SIZE)
    {
        return -1;
    }
    if t.len() <= 1 {
        fill_freq(t, freq);
        if t.len() == 1 {
            u[0] = t[0];
        }
        return t.len() as SaSint;
    }

    let n = t.len();
    let mut index = main_16u_alloc(t, a, LIBSAIS_FLAGS_BWT, 0, None, fs, freq, threads);
    if index >= 0 {
        index += 1;
        u[0] = t[n - 1];
        bwt_copy_16u(&mut u[1..], a, index - 1);
        bwt_copy_16u(
            &mut u[index as usize..],
            &a[index as usize..],
            n as SaSint - index,
        );
    }
    index
}

/// Constructs the Burrows-Wheeler transformed 16-bit string (BWT) of a given 16-bit string.
///
/// - `t` (`[0..n-1]`): the input 16-bit string.
/// - `u` (`[0..n-1]`): the output 16-bit string (can alias `t`).
/// - `a` (`[0..n-1+fs]`): the temporary array.
/// - `fs`: extra space available at the end of `a` (0 should be enough for most cases).
/// - `freq` (`[0..65535]`): optional output symbol frequency table.
///
/// Returns the primary index on success, -1 or -2 on error.
pub fn libsais16_bwt(
    t: &[u16],
    u: &mut [u16],
    a: &mut [SaSint],
    fs: SaSint,
    freq: Option<&mut [SaSint]>,
) -> SaSint {
    build_bwt(t, u, a, fs, freq, 1)
}

fn build_bwt_aux(
    t: &[u16],
    u: &mut [u16],
    a: &mut [SaSint],
    fs: SaSint,
    freq: Option<&mut [SaSint]>,
    r: SaSint,
    i: &mut [SaSint],
    threads: SaSint,
) -> SaSint {
    if threads < 0 || r < 2 || (r & (r - 1)) != 0 {
        return -1;
    }
    let samples = if t.is_empty() {
        1
    } else {
        (t.len() - 1) / r as usize + 1
    };
    if i.len() < samples {
        return -1;
    }
    let n = t.len();
    if n <= 1 {
        fill_freq(t, freq);
        if n == 1 {
            u[0] = t[0];
        }
        i[0] = n as SaSint;
        return 0;
    }

    let index = main_16u_alloc(t, a, LIBSAIS_FLAGS_BWT, r, Some(i), fs, freq, threads);
    if index == 0 {
        u[0] = t[n - 1];
        bwt_copy_16u(&mut u[1..], a, i[0] - 1);
        bwt_copy_16u(
            &mut u[i[0] as usize..],
            &a[i[0] as usize..],
            n as SaSint - i[0],
        );
    }
    index
}

/// Constructs the Burrows-Wheeler transformed 16-bit string (BWT) of a given 16-bit string with auxiliary indexes.
///
/// - `t` (`[0..n-1]`): the input 16-bit string.
/// - `u` (`[0..n-1]`): the output 16-bit string (can alias `t`).
/// - `a` (`[0..n-1+fs]`): the temporary array.
/// - `fs`: extra space available at the end of `a` (0 should be enough for most cases).
/// - `freq` (`[0..65535]`): optional output symbol frequency table.
/// - `r`: sampling rate for the auxiliary indexes (must be a power of two).
/// - `i` (`[0..(n-1)/r]`): output auxiliary indexes.
///
/// Returns 0 on success, -1 or -2 on error.
pub fn libsais16_bwt_aux(
    t: &[u16],
    u: &mut [u16],
    a: &mut [SaSint],
    fs: SaSint,
    freq: Option<&mut [SaSint]>,
    r: SaSint,
    i: &mut [SaSint],
) -> SaSint {
    build_bwt_aux(t, u, a, fs, freq, r, i, 1)
}

/// Constructs the Burrows-Wheeler transformed 16-bit string (BWT) of a given 16-bit string using a libsais16 context.
///
/// - `ctx`: the libsais16 context.
/// - `t` (`[0..n-1]`): the input 16-bit string.
/// - `u` (`[0..n-1]`): the output 16-bit string (can alias `t`).
/// - `a` (`[0..n-1+fs]`): the temporary array.
/// - `fs`: extra space available at the end of `a` (0 should be enough for most cases).
/// - `freq` (`[0..65535]`): optional output symbol frequency table.
///
/// Returns the primary index on success, -1 or -2 on error.
pub fn libsais16_bwt_ctx(
    ctx: &mut Context,
    t: &[u16],
    u: &mut [u16],
    a: &mut [SaSint],
    fs: SaSint,
    freq: Option<&mut [SaSint]>,
) -> SaSint {
    if fs < 0
        || u.len() < t.len()
        || a.len()
            < t.len()
                .saturating_add(usize::try_from(fs).unwrap_or(usize::MAX))
        || freq.as_ref().is_some_and(|freq| freq.len() < ALPHABET_SIZE)
    {
        return -1;
    }
    if t.len() <= 1 {
        fill_freq(t, freq);
        if t.len() == 1 {
            u[0] = t[0];
        }
        return t.len() as SaSint;
    }

    let n = t.len();
    let mut index = main_16u_ctx(ctx, t, a, LIBSAIS_FLAGS_BWT, 0, None, fs, freq);
    if index >= 0 {
        index += 1;
        u[0] = t[n - 1];
        bwt_copy_16u(&mut u[1..], a, index - 1);
        bwt_copy_16u(
            &mut u[index as usize..],
            &a[index as usize..],
            n as SaSint - index,
        );
    }
    index
}

/// Constructs the BWT of a given 16-bit string with auxiliary indexes using a libsais16 context.
///
/// - `ctx`: the libsais16 context.
/// - `t` (`[0..n-1]`): the input 16-bit string.
/// - `u` (`[0..n-1]`): the output 16-bit string (can alias `t`).
/// - `a` (`[0..n-1+fs]`): the temporary array.
/// - `fs`: extra space available at the end of `a` (0 should be enough for most cases).
/// - `freq` (`[0..65535]`): optional output symbol frequency table.
/// - `r`: sampling rate for the auxiliary indexes (must be a power of two).
/// - `i` (`[0..(n-1)/r]`): output auxiliary indexes.
///
/// Returns 0 on success, -1 or -2 on error.
pub fn libsais16_bwt_aux_ctx(
    ctx: &mut Context,
    t: &[u16],
    u: &mut [u16],
    a: &mut [SaSint],
    fs: SaSint,
    freq: Option<&mut [SaSint]>,
    r: SaSint,
    i: &mut [SaSint],
) -> SaSint {
    if fs < 0 || r < 2 || (r & (r - 1)) != 0 {
        return -1;
    }
    let samples = if t.is_empty() {
        1
    } else {
        (t.len() - 1) / r as usize + 1
    };
    if u.len() < t.len()
        || a.len()
            < t.len()
                .saturating_add(usize::try_from(fs).unwrap_or(usize::MAX))
        || i.len() < samples
        || freq.as_ref().is_some_and(|freq| freq.len() < ALPHABET_SIZE)
    {
        return -1;
    }
    if t.len() <= 1 {
        fill_freq(t, freq);
        if t.len() == 1 {
            u[0] = t[0];
        }
        i[0] = t.len() as SaSint;
        return 0;
    }

    let n = t.len();
    let index = main_16u_ctx(ctx, t, a, LIBSAIS_FLAGS_BWT, r, Some(i), fs, freq);
    if index == 0 {
        u[0] = t[n - 1];
        bwt_copy_16u(&mut u[1..], a, i[0] - 1);
        bwt_copy_16u(
            &mut u[i[0] as usize..],
            &a[i[0] as usize..],
            n as SaSint - i[0],
        );
    }
    index
}

/// Constructs the Burrows-Wheeler transformed 16-bit string (BWT) of a given 16-bit string in parallel using OpenMP-style threading.
///
/// - `t` (`[0..n-1]`): the input 16-bit string.
/// - `u` (`[0..n-1]`): the output 16-bit string (can alias `t`).
/// - `a` (`[0..n-1+fs]`): the temporary array.
/// - `fs`: extra space available at the end of `a` (0 should be enough for most cases).
/// - `freq` (`[0..65535]`): optional output symbol frequency table.
/// - `threads`: number of worker threads (can be 0 for the implementation default).
///
/// Returns the primary index on success, -1 or -2 on error.
pub fn libsais16_bwt_omp(
    t: &[u16],
    u: &mut [u16],
    a: &mut [SaSint],
    fs: SaSint,
    freq: Option<&mut [SaSint]>,
    threads: SaSint,
) -> SaSint {
    if threads < 0 {
        -1
    } else {
        build_bwt(t, u, a, fs, freq, threads)
    }
}

/// Constructs the BWT of a given 16-bit string with auxiliary indexes in parallel using OpenMP-style threading.
///
/// - `t` (`[0..n-1]`): the input 16-bit string.
/// - `u` (`[0..n-1]`): the output 16-bit string (can alias `t`).
/// - `a` (`[0..n-1+fs]`): the temporary array.
/// - `fs`: extra space available at the end of `a` (0 should be enough for most cases).
/// - `freq` (`[0..65535]`): optional output symbol frequency table.
/// - `r`: sampling rate for the auxiliary indexes (must be a power of two).
/// - `i` (`[0..(n-1)/r]`): output auxiliary indexes.
/// - `threads`: number of worker threads (can be 0 for the implementation default).
///
/// Returns 0 on success, -1 or -2 on error.
pub fn libsais16_bwt_aux_omp(
    t: &[u16],
    u: &mut [u16],
    a: &mut [SaSint],
    fs: SaSint,
    freq: Option<&mut [SaSint]>,
    r: SaSint,
    i: &mut [SaSint],
    threads: SaSint,
) -> SaSint {
    if threads < 0 {
        -1
    } else {
        build_bwt_aux(t, u, a, fs, freq, r, i, threads)
    }
}

fn validate_unbwt_aux(
    t: &[u16],
    u: &[u16],
    a: &[SaSint],
    freq: Option<&[SaSint]>,
    r: SaSint,
    i: &[SaSint],
) -> SaSint {
    let n = t.len();
    if u.len() < n
        || a.len() < n
        || freq.is_some_and(|freq| freq.len() < ALPHABET_SIZE)
        || ((r != n as SaSint) && (r < 2 || (r & (r - 1)) != 0))
        || i.is_empty()
    {
        return -1;
    }
    if n <= 1 {
        return if i[0] == n as SaSint { 0 } else { -1 };
    }

    let samples = (n - 1) / r as usize + 1;
    if i.len() < samples {
        return -1;
    }

    for &index in &i[..samples] {
        if index <= 0 || index as usize > n {
            return -1;
        }
    }
    0
}

fn unbwt_compute_histogram(t: &[u16], count: &mut [usize]) {
    for &symbol in t {
        count[symbol as usize] += 1;
    }
}

fn unbwt_shift(n: usize) -> usize {
    let mut shift = 0usize;
    while (n >> shift) > (1usize << UNBWT_FASTBITS) {
        shift += 1;
    }
    shift
}

fn unbwt_calculate_fastbits(bucket2: &mut [usize], fastbits: &mut [u16], shift: usize) {
    let mut v = 0usize;
    let mut sum = 1usize;
    for (w, bucket) in bucket2.iter_mut().enumerate().take(ALPHABET_SIZE) {
        let prev = sum;
        sum += *bucket;
        *bucket = prev;
        if prev != sum {
            while v <= ((sum - 1) >> shift) {
                fastbits[v] = w as u16;
                v += 1;
            }
        }
    }
}

fn unbwt_calculate_p(t: &[u16], p: &mut [usize], bucket2: &mut [usize], index: usize) {
    for row in 0..index {
        let symbol = t[row] as usize;
        p[bucket2[symbol]] = row;
        bucket2[symbol] += 1;
    }

    for row in index + 1..=t.len() {
        let symbol = t[row - 1] as usize;
        p[bucket2[symbol]] = row;
        bucket2[symbol] += 1;
    }
}

#[allow(dead_code, non_snake_case)]
fn unbwt_calculate_P(
    t: &[u16],
    p: &mut [usize],
    bucket2: &mut [usize],
    index: usize,
    block_start: usize,
    block_end: usize,
) {
    let first_end = index.min(block_end);
    for row in block_start..first_end {
        let symbol = t[row] as usize;
        p[bucket2[symbol]] = row;
        bucket2[symbol] += 1;
    }

    let second_start = block_start.max(index) + 1;
    for row in second_start..=block_end {
        let symbol = t[row - 1] as usize;
        p[bucket2[symbol]] = row;
        bucket2[symbol] += 1;
    }
}

fn unbwt_init_single(
    t: &[u16],
    p: &mut [usize],
    freq: Option<&[SaSint]>,
    i: &[SaSint],
    bucket2: &mut [usize],
    fastbits: &mut [u16],
) {
    let shift = unbwt_shift(t.len());
    if let Some(freq) = freq {
        for c in 0..ALPHABET_SIZE {
            bucket2[c] = freq[c] as usize;
        }
    } else {
        bucket2.fill(0);
        unbwt_compute_histogram(t, bucket2);
    }

    unbwt_calculate_fastbits(bucket2, fastbits, shift);
    unbwt_calculate_p(t, p, bucket2, i[0] as usize);
}

#[allow(dead_code)]
fn unbwt_init_parallel(
    t: &[u16],
    p: &mut [usize],
    freq: Option<&[SaSint]>,
    i: &[SaSint],
    bucket2: &mut [usize],
    fastbits: &mut [u16],
    buckets: &mut [usize],
    threads: SaSint,
) {
    let n = t.len();
    let available_threads = buckets.len() / ALPHABET_SIZE;
    let num_threads = if threads > 1 && n >= 65_536 && available_threads > 1 {
        usize::try_from(threads)
            .expect("threads must be non-negative")
            .min(available_threads)
            .max(1)
    } else {
        1
    };

    if num_threads == 1 {
        unbwt_init_single(t, p, freq, i, bucket2, fastbits);
        return;
    }

    let index = usize::try_from(i[0]).expect("primary index must be non-negative");
    let shift = unbwt_shift(n);
    let block_stride = (n / num_threads) & !15usize;

    for thread in 0..num_threads {
        let block_start = thread * block_stride;
        let block_size = if thread + 1 < num_threads {
            block_stride
        } else {
            n - block_start
        };
        let local = &mut buckets[thread * ALPHABET_SIZE..(thread + 1) * ALPHABET_SIZE];
        local.fill(0);
        unbwt_compute_histogram(&t[block_start..block_start + block_size], local);
    }

    bucket2.fill(0);
    for thread in 0..num_threads {
        let local = &mut buckets[thread * ALPHABET_SIZE..(thread + 1) * ALPHABET_SIZE];
        for c in 0..ALPHABET_SIZE {
            let a = bucket2[c];
            let b = local[c];
            bucket2[c] = a + b;
            local[c] = a;
        }
    }

    unbwt_calculate_fastbits(bucket2, fastbits, shift);

    for thread in 0..num_threads {
        let block_start = thread * block_stride;
        let block_size = if thread + 1 < num_threads {
            block_stride
        } else {
            n - block_start
        };
        let local = &mut buckets[thread * ALPHABET_SIZE..(thread + 1) * ALPHABET_SIZE];
        for c in 0..ALPHABET_SIZE {
            local[c] += bucket2[c];
        }
        unbwt_calculate_P(t, p, local, index, block_start, block_start + block_size);
    }

    let last_local = &buckets[(num_threads - 1) * ALPHABET_SIZE..num_threads * ALPHABET_SIZE];
    bucket2.copy_from_slice(last_local);
}

fn unbwt_decode_symbol(
    p0: usize,
    p: &[usize],
    bucket2: &[usize],
    fastbits: &[u16],
    shift: usize,
) -> (u16, usize) {
    let mut c0 = fastbits[p0 >> shift] as usize;
    if bucket2[c0] <= p0 {
        while bucket2[c0] <= p0 {
            c0 += 1;
        }
    }
    (c0 as u16, p[p0])
}

#[allow(dead_code)]
fn unbwt_decode_1(
    u: &mut [u16],
    p: &[usize],
    bucket2: &[usize],
    fastbits: &[u16],
    shift: usize,
    i0: &mut usize,
    k: usize,
) {
    let mut cursors = [*i0];
    unbwt_decode_lanes::<1>(u, p, bucket2, fastbits, shift, k, &mut cursors, k);
    *i0 = cursors[0];
}

#[allow(dead_code)]
fn unbwt_decode_2(
    u: &mut [u16],
    p: &[usize],
    bucket2: &[usize],
    fastbits: &[u16],
    shift: usize,
    r: usize,
    i0: &mut usize,
    i1: &mut usize,
    k: usize,
) {
    let mut cursors = [*i0, *i1];
    unbwt_decode_lanes::<2>(u, p, bucket2, fastbits, shift, r, &mut cursors, k);
    *i0 = cursors[0];
    *i1 = cursors[1];
}

#[allow(dead_code)]
fn unbwt_decode_3(
    u: &mut [u16],
    p: &[usize],
    bucket2: &[usize],
    fastbits: &[u16],
    shift: usize,
    r: usize,
    i0: &mut usize,
    i1: &mut usize,
    i2: &mut usize,
    k: usize,
) {
    let mut cursors = [*i0, *i1, *i2];
    unbwt_decode_lanes::<3>(u, p, bucket2, fastbits, shift, r, &mut cursors, k);
    *i0 = cursors[0];
    *i1 = cursors[1];
    *i2 = cursors[2];
}

#[allow(dead_code)]
fn unbwt_decode_4(
    u: &mut [u16],
    p: &[usize],
    bucket2: &[usize],
    fastbits: &[u16],
    shift: usize,
    r: usize,
    i0: &mut usize,
    i1: &mut usize,
    i2: &mut usize,
    i3: &mut usize,
    k: usize,
) {
    let mut cursors = [*i0, *i1, *i2, *i3];
    unbwt_decode_lanes::<4>(u, p, bucket2, fastbits, shift, r, &mut cursors, k);
    *i0 = cursors[0];
    *i1 = cursors[1];
    *i2 = cursors[2];
    *i3 = cursors[3];
}

#[allow(dead_code)]
fn unbwt_decode_5(
    u: &mut [u16],
    p: &[usize],
    bucket2: &[usize],
    fastbits: &[u16],
    shift: usize,
    r: usize,
    cursors: &mut [usize; 5],
    k: usize,
) {
    unbwt_decode_lanes::<5>(u, p, bucket2, fastbits, shift, r, cursors, k);
}

#[allow(dead_code)]
fn unbwt_decode_6(
    u: &mut [u16],
    p: &[usize],
    bucket2: &[usize],
    fastbits: &[u16],
    shift: usize,
    r: usize,
    cursors: &mut [usize; 6],
    k: usize,
) {
    unbwt_decode_lanes::<6>(u, p, bucket2, fastbits, shift, r, cursors, k);
}

#[allow(dead_code)]
fn unbwt_decode_7(
    u: &mut [u16],
    p: &[usize],
    bucket2: &[usize],
    fastbits: &[u16],
    shift: usize,
    r: usize,
    cursors: &mut [usize; 7],
    k: usize,
) {
    unbwt_decode_lanes::<7>(u, p, bucket2, fastbits, shift, r, cursors, k);
}

#[allow(dead_code)]
fn unbwt_decode_8(
    u: &mut [u16],
    p: &[usize],
    bucket2: &[usize],
    fastbits: &[u16],
    shift: usize,
    r: usize,
    cursors: &mut [usize; 8],
    k: usize,
) {
    unbwt_decode_lanes::<8>(u, p, bucket2, fastbits, shift, r, cursors, k);
}

fn unbwt_decode(
    u: &mut [u16],
    p: &[usize],
    n: usize,
    r: usize,
    i: &[SaSint],
    bucket2: &[usize],
    fastbits: &[u16],
) {
    let shift = unbwt_shift(n);
    let blocks = 1 + (n - 1) / r;
    let remainder = n - r * (blocks - 1);
    unbwt_decode_blocks(u, p, r, i, bucket2, fastbits, shift, blocks, remainder);
}

fn unbwt_decode_blocks(
    u: &mut [u16],
    p: &[usize],
    r: usize,
    i: &[SaSint],
    bucket2: &[usize],
    fastbits: &[u16],
    shift: usize,
    blocks: usize,
    remainder: usize,
) {
    let mut blocks_left = blocks;
    let mut i_offset = 0usize;
    let mut u_offset = 0usize;

    while blocks_left > 8 {
        let mut cursors = [
            i[i_offset] as usize,
            i[i_offset + 1] as usize,
            i[i_offset + 2] as usize,
            i[i_offset + 3] as usize,
            i[i_offset + 4] as usize,
            i[i_offset + 5] as usize,
            i[i_offset + 6] as usize,
            i[i_offset + 7] as usize,
        ];
        unbwt_decode_lanes::<8>(
            &mut u[u_offset..],
            p,
            bucket2,
            fastbits,
            shift,
            r,
            &mut cursors,
            r,
        );
        i_offset += 8;
        blocks_left -= 8;
        u_offset += 8 * r;
    }

    match blocks_left {
        1 => {
            let mut cursors = [i[i_offset] as usize];
            unbwt_decode_lanes::<1>(
                &mut u[u_offset..],
                p,
                bucket2,
                fastbits,
                shift,
                r,
                &mut cursors,
                remainder,
            );
        }
        2 => {
            let mut cursors = [i[i_offset] as usize, i[i_offset + 1] as usize];
            unbwt_decode_lanes::<2>(
                &mut u[u_offset..],
                p,
                bucket2,
                fastbits,
                shift,
                r,
                &mut cursors,
                remainder,
            );
            let mut first = [cursors[0]];
            unbwt_decode_lanes::<1>(
                &mut u[u_offset + remainder..],
                p,
                bucket2,
                fastbits,
                shift,
                r,
                &mut first,
                r - remainder,
            );
        }
        3 => {
            let mut cursors = [
                i[i_offset] as usize,
                i[i_offset + 1] as usize,
                i[i_offset + 2] as usize,
            ];
            unbwt_decode_lanes::<3>(
                &mut u[u_offset..],
                p,
                bucket2,
                fastbits,
                shift,
                r,
                &mut cursors,
                remainder,
            );
            let mut first = [cursors[0], cursors[1]];
            unbwt_decode_lanes::<2>(
                &mut u[u_offset + remainder..],
                p,
                bucket2,
                fastbits,
                shift,
                r,
                &mut first,
                r - remainder,
            );
        }
        4 => {
            let mut cursors = [
                i[i_offset] as usize,
                i[i_offset + 1] as usize,
                i[i_offset + 2] as usize,
                i[i_offset + 3] as usize,
            ];
            unbwt_decode_lanes::<4>(
                &mut u[u_offset..],
                p,
                bucket2,
                fastbits,
                shift,
                r,
                &mut cursors,
                remainder,
            );
            let mut first = [cursors[0], cursors[1], cursors[2]];
            unbwt_decode_lanes::<3>(
                &mut u[u_offset + remainder..],
                p,
                bucket2,
                fastbits,
                shift,
                r,
                &mut first,
                r - remainder,
            );
        }
        5 => {
            let mut cursors = [
                i[i_offset] as usize,
                i[i_offset + 1] as usize,
                i[i_offset + 2] as usize,
                i[i_offset + 3] as usize,
                i[i_offset + 4] as usize,
            ];
            unbwt_decode_lanes::<5>(
                &mut u[u_offset..],
                p,
                bucket2,
                fastbits,
                shift,
                r,
                &mut cursors,
                remainder,
            );
            let mut first = [cursors[0], cursors[1], cursors[2], cursors[3]];
            unbwt_decode_lanes::<4>(
                &mut u[u_offset + remainder..],
                p,
                bucket2,
                fastbits,
                shift,
                r,
                &mut first,
                r - remainder,
            );
        }
        6 => {
            let mut cursors = [
                i[i_offset] as usize,
                i[i_offset + 1] as usize,
                i[i_offset + 2] as usize,
                i[i_offset + 3] as usize,
                i[i_offset + 4] as usize,
                i[i_offset + 5] as usize,
            ];
            unbwt_decode_lanes::<6>(
                &mut u[u_offset..],
                p,
                bucket2,
                fastbits,
                shift,
                r,
                &mut cursors,
                remainder,
            );
            let mut first = [cursors[0], cursors[1], cursors[2], cursors[3], cursors[4]];
            unbwt_decode_lanes::<5>(
                &mut u[u_offset + remainder..],
                p,
                bucket2,
                fastbits,
                shift,
                r,
                &mut first,
                r - remainder,
            );
        }
        7 => {
            let mut cursors = [
                i[i_offset] as usize,
                i[i_offset + 1] as usize,
                i[i_offset + 2] as usize,
                i[i_offset + 3] as usize,
                i[i_offset + 4] as usize,
                i[i_offset + 5] as usize,
                i[i_offset + 6] as usize,
            ];
            unbwt_decode_lanes::<7>(
                &mut u[u_offset..],
                p,
                bucket2,
                fastbits,
                shift,
                r,
                &mut cursors,
                remainder,
            );
            let mut first = [
                cursors[0], cursors[1], cursors[2], cursors[3], cursors[4], cursors[5],
            ];
            unbwt_decode_lanes::<6>(
                &mut u[u_offset + remainder..],
                p,
                bucket2,
                fastbits,
                shift,
                r,
                &mut first,
                r - remainder,
            );
        }
        _ => {
            let mut cursors = [
                i[i_offset] as usize,
                i[i_offset + 1] as usize,
                i[i_offset + 2] as usize,
                i[i_offset + 3] as usize,
                i[i_offset + 4] as usize,
                i[i_offset + 5] as usize,
                i[i_offset + 6] as usize,
                i[i_offset + 7] as usize,
            ];
            unbwt_decode_lanes::<8>(
                &mut u[u_offset..],
                p,
                bucket2,
                fastbits,
                shift,
                r,
                &mut cursors,
                remainder,
            );
            let mut first = [
                cursors[0], cursors[1], cursors[2], cursors[3], cursors[4], cursors[5], cursors[6],
            ];
            unbwt_decode_lanes::<7>(
                &mut u[u_offset + remainder..],
                p,
                bucket2,
                fastbits,
                shift,
                r,
                &mut first,
                r - remainder,
            );
        }
    }
}

fn unbwt_decode_omp(
    u: &mut [u16],
    p: &[usize],
    n: usize,
    r: usize,
    i: &[SaSint],
    bucket2: &[usize],
    fastbits: &[u16],
    threads: SaSint,
) {
    let blocks = 1 + (n - 1) / r;
    let remainder = n - r * (blocks - 1);
    let num_threads = if threads > 1 && n >= 65_536 {
        usize::try_from(threads)
            .expect("threads must be non-negative")
            .min(blocks)
            .max(1)
    } else {
        1
    };

    if num_threads == 1 {
        unbwt_decode(u, p, n, r, i, bucket2, fastbits);
        return;
    }

    let shift = unbwt_shift(n);
    let block_stride = blocks / num_threads;
    let block_remainder = blocks % num_threads;
    let u_ptr = SyncMutPtr::new(u);
    run_rayon_with_threads(num_threads, || {
        (0..num_threads).into_par_iter().for_each(|thread| {
            let block_count = block_stride + usize::from(thread < block_remainder);
            let block_start = block_stride * thread + thread.min(block_remainder);
            let tail = if thread + 1 < num_threads {
                r
            } else {
                remainder
            };
            let u = unsafe { u_ptr.as_slice() };
            unbwt_decode_blocks(
                &mut u[r * block_start..],
                p,
                r,
                &i[block_start..],
                bucket2,
                fastbits,
                shift,
                block_count,
                tail,
            );
        });
    });
}

fn unbwt_decode_lanes<const LANES: usize>(
    u: &mut [u16],
    p: &[usize],
    bucket2: &[usize],
    fastbits: &[u16],
    shift: usize,
    r: usize,
    cursors: &mut [usize; LANES],
    k: usize,
) {
    for pos in 0..k {
        for lane in 0..LANES {
            let (symbol, next) = unbwt_decode_symbol(cursors[lane], p, bucket2, fastbits, shift);
            cursors[lane] = next;
            u[lane * r + pos] = symbol;
        }
    }
}

fn unbwt_core(
    t: &[u16],
    u: &mut [u16],
    a: &mut [SaSint],
    freq: Option<&[SaSint]>,
    r: SaSint,
    i: &[SaSint],
) -> SaSint {
    let n = t.len();
    let shift = unbwt_shift(n);
    let mut bucket2 = vec![0usize; ALPHABET_SIZE];
    let mut fastbits = vec![0u16; 1 + (n >> shift)];

    unbwt_core_with_buffers(t, u, a, freq, r, i, &mut bucket2, &mut fastbits, 1)
}

fn unbwt_core_with_buffers(
    t: &[u16],
    u: &mut [u16],
    a: &mut [SaSint],
    freq: Option<&[SaSint]>,
    r: SaSint,
    i: &[SaSint],
    bucket2: &mut [usize],
    fastbits: &mut [u16],
    threads: SaSint,
) -> SaSint {
    let n = t.len();
    let shift = unbwt_shift(n);
    if bucket2.len() < ALPHABET_SIZE || fastbits.len() < 1 + (n >> shift) {
        return -2;
    }

    let mut p = vec![0usize; n + 1];
    unbwt_init_single(
        t,
        &mut p,
        freq,
        i,
        &mut bucket2[..ALPHABET_SIZE],
        &mut fastbits[..1 + (n >> shift)],
    );
    unbwt_decode_omp(
        u,
        &p,
        n,
        r as usize,
        i,
        &bucket2[..ALPHABET_SIZE],
        &fastbits[..1 + (n >> shift)],
        threads,
    );

    for (dst, &src) in a.iter_mut().zip(p.iter().skip(1)) {
        *dst = src as SaSint;
    }
    0
}

fn inverse_bwt(
    t: &[u16],
    u: &mut [u16],
    a: &mut [SaSint],
    freq: Option<&[SaSint]>,
    primary: SaSint,
) -> SaSint {
    let n = t.len();
    let i = [primary];
    let rc = validate_unbwt_aux(t, u, a, freq, n as SaSint, &i);
    if rc != 0 {
        return rc;
    }
    if n <= 1 {
        if n == 1 {
            u[0] = t[0];
        }
        return 0;
    }
    unbwt_core(t, u, a, freq, n as SaSint, &i)
}

/// Reconstructs the original 16-bit string from a given BWT and primary index.
///
/// - `t` (`[0..n-1]`): the input 16-bit string.
/// - `u` (`[0..n-1]`): the output 16-bit string (can alias `t`).
/// - `a` (`[0..n]`): the temporary array (must have length `n + 1`).
/// - `freq` (`[0..65535]`): optional input symbol frequency table.
/// - `i`: the primary index.
///
/// Returns 0 on success, -1 or -2 on error.
pub fn libsais16_unbwt(
    t: &[u16],
    u: &mut [u16],
    a: &mut [SaSint],
    freq: Option<&[SaSint]>,
    i: SaSint,
) -> SaSint {
    inverse_bwt(t, u, a, freq, i)
}

/// Reconstructs the original 16-bit string from a given BWT and primary index using a libsais16 reverse-BWT context.
///
/// - `ctx`: the libsais16 reverse-BWT context.
/// - `t` (`[0..n-1]`): the input 16-bit string.
/// - `u` (`[0..n-1]`): the output 16-bit string (can alias `t`).
/// - `a` (`[0..n]`): the temporary array (must have length `n + 1`).
/// - `freq` (`[0..65535]`): optional input symbol frequency table.
/// - `i`: the primary index.
///
/// Returns 0 on success, -1 or -2 on error.
pub fn libsais16_unbwt_ctx(
    ctx: &mut UnbwtContext,
    t: &[u16],
    u: &mut [u16],
    a: &mut [SaSint],
    freq: Option<&[SaSint]>,
    i: SaSint,
) -> SaSint {
    libsais16_unbwt_aux_ctx(ctx, t, u, a, freq, t.len() as SaSint, &[i])
}

/// Reconstructs the original 16-bit string from a given BWT with auxiliary indexes.
///
/// - `t` (`[0..n-1]`): the input 16-bit string.
/// - `u` (`[0..n-1]`): the output 16-bit string (can alias `t`).
/// - `a` (`[0..n]`): the temporary array (must have length `n + 1`).
/// - `freq` (`[0..65535]`): optional input symbol frequency table.
/// - `r`: sampling rate for the auxiliary indexes (must be a power of two).
/// - `i` (`[0..(n-1)/r]`): input auxiliary indexes.
///
/// Returns 0 on success, -1 or -2 on error.
pub fn libsais16_unbwt_aux(
    t: &[u16],
    u: &mut [u16],
    a: &mut [SaSint],
    freq: Option<&[SaSint]>,
    r: SaSint,
    i: &[SaSint],
) -> SaSint {
    let rc = validate_unbwt_aux(t, u, a, freq, r, i);
    if rc != 0 {
        return rc;
    }
    if t.len() <= 1 {
        if t.len() == 1 {
            u[0] = t[0];
        }
        return 0;
    }
    unbwt_core(t, u, a, freq, r, i)
}

/// Reconstructs the original 16-bit string from a given BWT with auxiliary indexes using a libsais16 reverse-BWT context.
///
/// - `ctx`: the libsais16 reverse-BWT context.
/// - `t` (`[0..n-1]`): the input 16-bit string.
/// - `u` (`[0..n-1]`): the output 16-bit string (can alias `t`).
/// - `a` (`[0..n]`): the temporary array (must have length `n + 1`).
/// - `freq` (`[0..65535]`): optional input symbol frequency table.
/// - `r`: sampling rate for the auxiliary indexes (must be a power of two).
/// - `i` (`[0..(n-1)/r]`): input auxiliary indexes.
///
/// Returns 0 on success, -1 or -2 on error.
pub fn libsais16_unbwt_aux_ctx(
    ctx: &mut UnbwtContext,
    t: &[u16],
    u: &mut [u16],
    a: &mut [SaSint],
    freq: Option<&[SaSint]>,
    r: SaSint,
    i: &[SaSint],
) -> SaSint {
    let rc = validate_unbwt_aux(t, u, a, freq, r, i);
    if rc != 0 {
        return rc;
    }
    if t.len() <= 1 {
        if t.len() == 1 {
            u[0] = t[0];
        }
        return 0;
    }
    unbwt_core_with_buffers(
        t,
        u,
        a,
        freq,
        r,
        i,
        &mut ctx.bucket2,
        &mut ctx.fastbits,
        ctx.threads,
    )
}

/// Reconstructs the original 16-bit string from a given BWT and primary index in parallel using OpenMP-style threading.
///
/// - `t` (`[0..n-1]`): the input 16-bit string.
/// - `u` (`[0..n-1]`): the output 16-bit string (can alias `t`).
/// - `a` (`[0..n]`): the temporary array (must have length `n + 1`).
/// - `freq` (`[0..65535]`): optional input symbol frequency table.
/// - `i`: the primary index.
/// - `threads`: number of worker threads (can be 0 for the implementation default).
///
/// Returns 0 on success, -1 or -2 on error.
pub fn libsais16_unbwt_omp(
    t: &[u16],
    u: &mut [u16],
    a: &mut [SaSint],
    freq: Option<&[SaSint]>,
    i: SaSint,
    threads: SaSint,
) -> SaSint {
    if threads < 0 {
        -1
    } else {
        let primary = [i];
        libsais16_unbwt_aux_omp(t, u, a, freq, t.len() as SaSint, &primary, threads)
    }
}

/// Reconstructs the original 16-bit string from a given BWT with auxiliary indexes in parallel using OpenMP-style threading.
///
/// - `t` (`[0..n-1]`): the input 16-bit string.
/// - `u` (`[0..n-1]`): the output 16-bit string (can alias `t`).
/// - `a` (`[0..n]`): the temporary array (must have length `n + 1`).
/// - `freq` (`[0..65535]`): optional input symbol frequency table.
/// - `r`: sampling rate for the auxiliary indexes (must be a power of two).
/// - `i` (`[0..(n-1)/r]`): input auxiliary indexes.
/// - `threads`: number of worker threads (can be 0 for the implementation default).
///
/// Returns 0 on success, -1 or -2 on error.
pub fn libsais16_unbwt_aux_omp(
    t: &[u16],
    u: &mut [u16],
    a: &mut [SaSint],
    freq: Option<&[SaSint]>,
    r: SaSint,
    i: &[SaSint],
    threads: SaSint,
) -> SaSint {
    if threads < 0 {
        -1
    } else {
        let rc = validate_unbwt_aux(t, u, a, freq, r, i);
        if rc != 0 {
            return rc;
        }
        if t.len() <= 1 {
            if t.len() == 1 {
                u[0] = t[0];
            }
            return 0;
        }
        let n = t.len();
        let shift = unbwt_shift(n);
        let mut bucket2 = vec![0usize; ALPHABET_SIZE];
        let mut fastbits = vec![0u16; 1 + (n >> shift)];
        unbwt_core_with_buffers(
            t,
            u,
            a,
            freq,
            r,
            i,
            &mut bucket2,
            &mut fastbits,
            normalize_threads(threads),
        )
    }
}

/// Constructs the permuted longest common prefix array (PLCP) of a given 16-bit string and suffix array.
///
/// - `t` (`[0..n-1]`): the input 16-bit string.
/// - `sa` (`[0..n-1]`): the input suffix array.
/// - `plcp` (`[0..n-1]`): the output permuted longest common prefix array.
///
/// Returns 0 on success, -1 on error.
pub fn libsais16_plcp(t: &[u16], sa: &[SaSint], plcp: &mut [SaSint]) -> SaSint {
    compute_plcp(t, sa, plcp, false)
}

/// Constructs the PLCP of a given 16-bit string set and generalized suffix array (GSA).
///
/// - `t` (`[0..n-1]`): the input 16-bit string set using 0 as separators (`t[n-1]` must be 0).
/// - `sa` (`[0..n-1]`): the input generalized suffix array.
/// - `plcp` (`[0..n-1]`): the output permuted longest common prefix array.
///
/// Returns 0 on success, -1 on error.
pub fn libsais16_plcp_gsa(t: &[u16], sa: &[SaSint], plcp: &mut [SaSint]) -> SaSint {
    if t.last().copied().unwrap_or(0) != 0 {
        -1
    } else {
        compute_plcp(t, sa, plcp, true)
    }
}

fn compute_plcp(t: &[u16], sa: &[SaSint], plcp: &mut [SaSint], gsa: bool) -> SaSint {
    if sa.len() != t.len() || plcp.len() != t.len() {
        return -1;
    }
    if t.len() <= 1 {
        if t.len() == 1 {
            plcp[0] = 0;
        }
        return 0;
    }

    if compute_phi(sa, plcp) != 0 {
        return -1;
    }

    compute_plcp_from_phi(t, plcp, gsa)
}

fn compute_phi(sa: &[SaSint], plcp: &mut [SaSint]) -> SaSint {
    let n = sa.len();
    let mut previous = n as SaSint;
    for &suffix_value in sa {
        let Some(suffix) = suffix_index(suffix_value, n) else {
            return -1;
        };
        plcp[suffix] = previous;
        previous = suffix_value;
    }
    0
}

fn compute_plcp_from_phi(t: &[u16], plcp: &mut [SaSint], gsa: bool) -> SaSint {
    let n = t.len();
    let mut l = 0usize;
    for i in 0..t.len() {
        let previous = plcp[i];
        if previous == n as SaSint {
            plcp[i] = 0;
            l = 0;
            continue;
        }

        let Some(prev) = suffix_index(previous, n) else {
            return -1;
        };

        while i + l < t.len()
            && prev + l < t.len()
            && t[i + l] == t[prev + l]
            && (!gsa || t[i + l] != 0)
        {
            l += 1;
        }
        plcp[i] = l as SaSint;
        l = l.saturating_sub(1);
    }
    0
}

fn compute_phi_omp(sa: &[SaSint], plcp: &mut [SaSint], n: SaSint, threads: SaSint) -> SaSint {
    let n_usize = n as usize;
    if threads == 1 || n < 65_536 {
        return compute_phi(&sa[..n_usize], &mut plcp[..n_usize]);
    }

    let block_stride = (n / threads) & !15;
    for thread in 0..threads {
        let block_start = thread * block_stride;
        let block_size = if thread < threads - 1 {
            block_stride
        } else {
            n - block_start
        };
        let start = block_start as usize;
        let end = (block_start + block_size) as usize;
        let mut previous = if start > 0 { sa[start - 1] } else { n };
        for &suffix_value in &sa[start..end] {
            let Some(suffix) = suffix_index(suffix_value, n_usize) else {
                return -1;
            };
            plcp[suffix] = previous;
            previous = suffix_value;
        }
    }
    0
}

fn compute_plcp_omp(t: &[u16], plcp: &mut [SaSint], n: SaSint, threads: SaSint) -> SaSint {
    if threads == 1 || n < 65_536 {
        let n = n as usize;
        return compute_plcp_from_phi(&t[..n], &mut plcp[..n], false);
    }

    let block_stride = (n / threads) & !15;
    for thread in 0..threads {
        let block_start = thread * block_stride;
        let block_size = if thread < threads - 1 {
            block_stride
        } else {
            n - block_start
        };
        let rc = compute_plcp_range(
            t,
            plcp,
            n as usize,
            block_start as isize,
            block_size as isize,
            false,
        );
        if rc != 0 {
            return rc;
        }
    }
    0
}

fn compute_plcp_range(
    t: &[u16],
    plcp: &mut [SaSint],
    n: usize,
    omp_block_start: isize,
    omp_block_size: isize,
    gsa: bool,
) -> SaSint {
    let mut l = 0usize;
    let end = (omp_block_start + omp_block_size) as usize;
    for i in omp_block_start as usize..end {
        let previous = plcp[i];
        if previous == n as SaSint {
            plcp[i] = 0;
            l = 0;
            continue;
        }

        let Some(prev) = suffix_index(previous, n) else {
            return -1;
        };

        while i + l < t.len()
            && prev + l < t.len()
            && t[i + l] == t[prev + l]
            && (!gsa || t[i + l] != 0)
        {
            l += 1;
        }
        plcp[i] = l as SaSint;
        l = l.saturating_sub(1);
    }
    0
}

fn compute_plcp_gsa(
    t: &[u16],
    plcp: &mut [SaSint],
    omp_block_start: isize,
    omp_block_size: isize,
) -> SaSint {
    let n = t.len();
    let mut l = 0usize;
    let end = (omp_block_start + omp_block_size) as usize;
    for i in omp_block_start as usize..end {
        let previous = plcp[i];
        if previous == n as SaSint {
            plcp[i] = 0;
            l = 0;
            continue;
        }

        let Some(prev) = suffix_index(previous, n) else {
            return -1;
        };

        while i + l < t.len() && prev + l < t.len() && t[i + l] == t[prev + l] && t[i + l] != 0 {
            l += 1;
        }
        plcp[i] = l as SaSint;
        l = l.saturating_sub(1);
    }
    0
}

fn compute_plcp_gsa_omp(t: &[u16], plcp: &mut [SaSint], n: SaSint, threads: SaSint) -> SaSint {
    if threads == 1 || n < 65_536 {
        return compute_plcp_gsa(t, plcp, 0, n as isize);
    }

    let block_stride = (n / threads) & !15;
    for thread in 0..threads {
        let block_start = thread * block_stride;
        let block_size = if thread < threads - 1 {
            block_stride
        } else {
            n - block_start
        };
        let rc = compute_plcp_gsa(t, plcp, block_start as isize, block_size as isize);
        if rc != 0 {
            return rc;
        }
    }
    0
}

fn compute_lcp(
    plcp: &[SaSint],
    sa: &[SaSint],
    lcp: &mut [SaSint],
    omp_block_start: isize,
    omp_block_size: isize,
) -> SaSint {
    let end = (omp_block_start + omp_block_size) as usize;
    for row in omp_block_start as usize..end {
        let Some(suffix) = suffix_index(sa[row], plcp.len()) else {
            return -1;
        };
        lcp[row] = plcp[suffix];
    }
    0
}

fn compute_lcp_omp(
    plcp: &[SaSint],
    sa: &[SaSint],
    lcp: &mut [SaSint],
    n: SaSint,
    threads: SaSint,
) -> SaSint {
    if threads == 1 || n < 65_536 {
        return compute_lcp(plcp, sa, lcp, 0, n as isize);
    }

    let block_stride = (n / threads) & !15;
    for thread in 0..threads {
        let block_start = thread * block_stride;
        let block_size = if thread < threads - 1 {
            block_stride
        } else {
            n - block_start
        };
        let rc = compute_lcp(plcp, sa, lcp, block_start as isize, block_size as isize);
        if rc != 0 {
            return rc;
        }
    }
    0
}

/// Constructs the longest common prefix array (LCP) from a PLCP and suffix array.
///
/// - `plcp` (`[0..n-1]`): the input permuted longest common prefix array.
/// - `sa` (`[0..n-1]`): the input suffix array or generalized suffix array (GSA).
/// - `lcp` (`[0..n-1]`): the output longest common prefix array (can alias `sa`).
///
/// Returns 0 on success, -1 on error.
pub fn libsais16_lcp(plcp: &[SaSint], sa: &[SaSint], lcp: &mut [SaSint]) -> SaSint {
    if plcp.len() != sa.len() || lcp.len() != sa.len() {
        return -1;
    }
    for (row, &suffix) in sa.iter().enumerate() {
        let Some(suffix) = suffix_index(suffix, plcp.len()) else {
            return -1;
        };
        lcp[row] = plcp[suffix];
    }
    0
}

fn suffix_index(value: SaSint, len: usize) -> Option<usize> {
    usize::try_from(value).ok().filter(|&index| index < len)
}

/// Constructs the PLCP of a given 16-bit string and suffix array in parallel using OpenMP-style threading.
///
/// - `t` (`[0..n-1]`): the input 16-bit string.
/// - `sa` (`[0..n-1]`): the input suffix array.
/// - `plcp` (`[0..n-1]`): the output permuted longest common prefix array.
/// - `threads`: number of worker threads (can be 0 for the implementation default).
///
/// Returns 0 on success, -1 on error.
pub fn libsais16_plcp_omp(
    t: &[u16],
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
    if t.len() <= 1 {
        if t.len() == 1 {
            plcp[0] = 0;
        }
        return 0;
    }

    let n = t.len() as SaSint;
    let threads = normalize_threads(threads);
    if compute_phi_omp(sa, plcp, n, threads) != 0 {
        return -1;
    }
    compute_plcp_omp(t, plcp, n, threads)
}

/// Constructs the PLCP of a given 16-bit string set and GSA in parallel using OpenMP-style threading.
///
/// - `t` (`[0..n-1]`): the input 16-bit string set using 0 as separators (`t[n-1]` must be 0).
/// - `sa` (`[0..n-1]`): the input generalized suffix array.
/// - `plcp` (`[0..n-1]`): the output permuted longest common prefix array.
/// - `threads`: number of worker threads (can be 0 for the implementation default).
///
/// Returns 0 on success, -1 on error.
pub fn libsais16_plcp_gsa_omp(
    t: &[u16],
    sa: &[SaSint],
    plcp: &mut [SaSint],
    threads: SaSint,
) -> SaSint {
    if threads < 0 {
        return -1;
    }
    if t.last().copied().unwrap_or(0) != 0 {
        return -1;
    }
    if sa.len() != t.len() || plcp.len() != t.len() {
        return -1;
    }
    if t.len() <= 1 {
        if t.len() == 1 {
            plcp[0] = 0;
        }
        return 0;
    }

    let n = t.len() as SaSint;
    let threads = normalize_threads(threads);
    if compute_phi_omp(sa, plcp, n, threads) != 0 {
        return -1;
    }
    compute_plcp_gsa_omp(t, plcp, n, threads)
}

/// Constructs the LCP from a PLCP and suffix array in parallel using OpenMP-style threading.
///
/// - `plcp` (`[0..n-1]`): the input permuted longest common prefix array.
/// - `sa` (`[0..n-1]`): the input suffix array or generalized suffix array (GSA).
/// - `lcp` (`[0..n-1]`): the output longest common prefix array (can alias `sa`).
/// - `threads`: number of worker threads (can be 0 for the implementation default).
///
/// Returns 0 on success, -1 on error.
pub fn libsais16_lcp_omp(
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

    compute_lcp_omp(
        plcp,
        sa,
        lcp,
        sa.len() as SaSint,
        normalize_threads(threads),
    )
}

#[cfg(all(test, feature = "upstream-c"))]
mod tests {
    use super::*;

    unsafe extern "C" {
        fn probe_public_libsais16(t: *const u16, sa: *mut SaSint, n: SaSint, fs: SaSint) -> SaSint;
        fn probe_public_libsais16_freq(
            t: *const u16,
            sa: *mut SaSint,
            n: SaSint,
            fs: SaSint,
            freq: *mut SaSint,
        ) -> SaSint;
        fn probe_public_libsais16_gsa(
            t: *const u16,
            sa: *mut SaSint,
            n: SaSint,
            fs: SaSint,
        ) -> SaSint;
        fn probe_public_libsais16_gsa_freq(
            t: *const u16,
            sa: *mut SaSint,
            n: SaSint,
            fs: SaSint,
            freq: *mut SaSint,
        ) -> SaSint;
        fn probe_public_libsais16_int(
            t: *mut SaSint,
            sa: *mut SaSint,
            n: SaSint,
            k: SaSint,
            fs: SaSint,
        ) -> SaSint;
        fn probe_libsais16_main_32s_entry(
            t: *mut SaSint,
            sa: *mut SaSint,
            n: SaSint,
            k: SaSint,
            fs: SaSint,
            threads: SaSint,
        ) -> SaSint;
        fn probe_libsais16_final_sorting_scan_left_to_right_32s(
            t: *const SaSint,
            sa: *mut SaSint,
            induction_bucket: *mut SaSint,
            omp_block_start: SaSint,
            omp_block_size: SaSint,
        );
        fn probe_libsais16_final_sorting_scan_right_to_left_32s(
            t: *const SaSint,
            sa: *mut SaSint,
            induction_bucket: *mut SaSint,
            omp_block_start: SaSint,
            omp_block_size: SaSint,
        );
        fn probe_libsais16_clear_lms_suffixes_omp(
            sa: *mut SaSint,
            n: SaSint,
            k: SaSint,
            bucket_start: *mut SaSint,
            bucket_end: *mut SaSint,
            threads: SaSint,
        );
        fn probe_libsais16_flip_suffix_markers_omp(sa: *mut SaSint, l: SaSint, threads: SaSint);
        fn probe_libsais16_induce_final_order_32s_6k(
            t: *const SaSint,
            sa: *mut SaSint,
            n: SaSint,
            k: SaSint,
            buckets: *mut SaSint,
            threads: SaSint,
        );
        fn probe_libsais16_induce_final_order_32s_4k(
            t: *const SaSint,
            sa: *mut SaSint,
            n: SaSint,
            k: SaSint,
            buckets: *mut SaSint,
            threads: SaSint,
        );
        fn probe_libsais16_induce_final_order_32s_2k(
            t: *const SaSint,
            sa: *mut SaSint,
            n: SaSint,
            k: SaSint,
            buckets: *mut SaSint,
            threads: SaSint,
        );
        fn probe_libsais16_induce_final_order_32s_1k(
            t: *const SaSint,
            sa: *mut SaSint,
            n: SaSint,
            k: SaSint,
            buckets: *mut SaSint,
            threads: SaSint,
        );
        fn probe_libsais16_induce_partial_order_32s_6k_omp(
            t: *const SaSint,
            sa: *mut SaSint,
            n: SaSint,
            k: SaSint,
            buckets: *mut SaSint,
            first_lms_suffix: SaSint,
            left_suffixes_count: SaSint,
            threads: SaSint,
        );
        fn probe_libsais16_induce_partial_order_32s_4k_omp(
            t: *const SaSint,
            sa: *mut SaSint,
            n: SaSint,
            k: SaSint,
            buckets: *mut SaSint,
            threads: SaSint,
        );
        fn probe_libsais16_induce_partial_order_32s_2k_omp(
            t: *const SaSint,
            sa: *mut SaSint,
            n: SaSint,
            k: SaSint,
            buckets: *mut SaSint,
            threads: SaSint,
        );
        fn probe_libsais16_induce_partial_order_32s_1k_omp(
            t: *const SaSint,
            sa: *mut SaSint,
            n: SaSint,
            k: SaSint,
            buckets: *mut SaSint,
            threads: SaSint,
        );
        fn probe_libsais16_induce_partial_order_16u_omp(
            t: *const u16,
            sa: *mut SaSint,
            n: SaSint,
            k: SaSint,
            flags: SaSint,
            buckets: *mut SaSint,
            first_lms_suffix: SaSint,
            left_suffixes_count: SaSint,
            threads: SaSint,
        );
        fn probe_libsais16_induce_final_order_16u_omp(
            t: *const u16,
            sa: *mut SaSint,
            n: SaSint,
            k: SaSint,
            flags: SaSint,
            r: SaSint,
            i: *mut SaSint,
            buckets: *mut SaSint,
            threads: SaSint,
        ) -> SaSint;
        fn probe_public_libsais16_bwt(
            t: *const u16,
            u: *mut u16,
            a: *mut SaSint,
            n: SaSint,
            fs: SaSint,
        ) -> SaSint;
        fn probe_public_libsais16_bwt_freq(
            t: *const u16,
            u: *mut u16,
            a: *mut SaSint,
            n: SaSint,
            fs: SaSint,
            freq: *mut SaSint,
        ) -> SaSint;
        fn probe_public_libsais16_bwt_aux(
            t: *const u16,
            u: *mut u16,
            a: *mut SaSint,
            n: SaSint,
            fs: SaSint,
            r: SaSint,
            i: *mut SaSint,
        ) -> SaSint;
        fn probe_public_libsais16_bwt_aux_freq(
            t: *const u16,
            u: *mut u16,
            a: *mut SaSint,
            n: SaSint,
            fs: SaSint,
            freq: *mut SaSint,
            r: SaSint,
            i: *mut SaSint,
        ) -> SaSint;
        fn probe_public_libsais16_unbwt(
            t: *const u16,
            u: *mut u16,
            a: *mut SaSint,
            n: SaSint,
            i: SaSint,
        ) -> SaSint;
        fn probe_public_libsais16_unbwt_freq(
            t: *const u16,
            u: *mut u16,
            a: *mut SaSint,
            n: SaSint,
            freq: *const SaSint,
            i: SaSint,
        ) -> SaSint;
        fn probe_public_libsais16_unbwt_aux(
            t: *const u16,
            u: *mut u16,
            a: *mut SaSint,
            n: SaSint,
            r: SaSint,
            i: *const SaSint,
        ) -> SaSint;
        fn probe_public_libsais16_unbwt_aux_freq(
            t: *const u16,
            u: *mut u16,
            a: *mut SaSint,
            n: SaSint,
            freq: *const SaSint,
            r: SaSint,
            i: *const SaSint,
        ) -> SaSint;
        fn probe_public_libsais16_plcp(
            t: *const u16,
            sa: *const SaSint,
            plcp: *mut SaSint,
            n: SaSint,
        ) -> SaSint;
        fn probe_public_libsais16_plcp_gsa(
            t: *const u16,
            sa: *const SaSint,
            plcp: *mut SaSint,
            n: SaSint,
        ) -> SaSint;
        fn probe_public_libsais16_lcp(
            plcp: *const SaSint,
            sa: *const SaSint,
            lcp: *mut SaSint,
            n: SaSint,
        ) -> SaSint;
        fn probe_libsais16_gather_lms_suffixes_16u(
            t: *const u16,
            sa: *mut SaSint,
            n: SaSint,
            m: SaSint,
            omp_block_start: SaSint,
            omp_block_size: SaSint,
        );
        fn probe_libsais16_count_and_gather_lms_suffixes_16u(
            t: *const u16,
            sa: *mut SaSint,
            n: SaSint,
            buckets: *mut SaSint,
            omp_block_start: SaSint,
            omp_block_size: SaSint,
        ) -> SaSint;
        fn probe_libsais16_initialize_buckets_start_and_end_16u(
            buckets: *mut SaSint,
            freq: *mut SaSint,
        ) -> SaSint;
        fn probe_libsais16_initialize_buckets_for_lms_suffixes_radix_sort_16u(
            t: *const u16,
            buckets: *mut SaSint,
            first_lms_suffix: SaSint,
        ) -> SaSint;
        fn probe_libsais16_radix_sort_lms_suffixes_16u(
            t: *const u16,
            sa: *mut SaSint,
            induction_bucket: *mut SaSint,
            omp_block_start: SaSint,
            omp_block_size: SaSint,
        );
        fn probe_libsais16_initialize_buckets_for_partial_sorting_16u(
            t: *const u16,
            buckets: *mut SaSint,
            first_lms_suffix: SaSint,
            left_suffixes_count: SaSint,
        );
        fn probe_libsais16_partial_sorting_scan_left_to_right_16u(
            t: *const u16,
            sa: *mut SaSint,
            buckets: *mut SaSint,
            d: SaSint,
            omp_block_start: SaSint,
            omp_block_size: SaSint,
        ) -> SaSint;
        fn probe_libsais16_partial_sorting_scan_right_to_left_16u(
            t: *const u16,
            sa: *mut SaSint,
            buckets: *mut SaSint,
            d: SaSint,
            omp_block_start: SaSint,
            omp_block_size: SaSint,
        ) -> SaSint;
        fn probe_libsais16_partial_gsa_scan_right_to_left_16u(
            t: *const u16,
            sa: *mut SaSint,
            buckets: *mut SaSint,
            d: SaSint,
            omp_block_start: SaSint,
            omp_block_size: SaSint,
        ) -> SaSint;
        fn probe_libsais16_partial_sorting_shift_markers_16u_omp(
            sa: *mut SaSint,
            n: SaSint,
            buckets: *const SaSint,
            threads: SaSint,
        );
        fn probe_libsais16_final_sorting_scan_left_to_right_16u(
            t: *const u16,
            sa: *mut SaSint,
            induction_bucket: *mut SaSint,
            omp_block_start: SaSint,
            omp_block_size: SaSint,
        );
        fn probe_libsais16_final_sorting_scan_right_to_left_16u(
            t: *const u16,
            sa: *mut SaSint,
            induction_bucket: *mut SaSint,
            omp_block_start: SaSint,
            omp_block_size: SaSint,
        );
        fn probe_libsais16_final_gsa_scan_right_to_left_16u(
            t: *const u16,
            sa: *mut SaSint,
            induction_bucket: *mut SaSint,
            omp_block_start: SaSint,
            omp_block_size: SaSint,
        );
        fn probe_libsais16_final_bwt_scan_left_to_right_16u(
            t: *const u16,
            sa: *mut SaSint,
            induction_bucket: *mut SaSint,
            omp_block_start: SaSint,
            omp_block_size: SaSint,
        );
        fn probe_libsais16_final_bwt_scan_right_to_left_16u(
            t: *const u16,
            sa: *mut SaSint,
            induction_bucket: *mut SaSint,
            omp_block_start: SaSint,
            omp_block_size: SaSint,
        ) -> SaSint;
        fn probe_libsais16_final_bwt_aux_scan_left_to_right_16u(
            t: *const u16,
            sa: *mut SaSint,
            rm: SaSint,
            i_sample: *mut SaSint,
            induction_bucket: *mut SaSint,
            omp_block_start: SaSint,
            omp_block_size: SaSint,
        );
        fn probe_libsais16_final_bwt_aux_scan_right_to_left_16u(
            t: *const u16,
            sa: *mut SaSint,
            rm: SaSint,
            i_sample: *mut SaSint,
            induction_bucket: *mut SaSint,
            omp_block_start: SaSint,
            omp_block_size: SaSint,
        );
        fn probe_libsais16_renumber_lms_suffixes_16u(
            sa: *mut SaSint,
            m: SaSint,
            name: SaSint,
            omp_block_start: SaSint,
            omp_block_size: SaSint,
        ) -> SaSint;
        fn probe_libsais16_place_lms_suffixes_interval_16u(
            sa: *mut SaSint,
            n: SaSint,
            m: SaSint,
            flags: SaSint,
            buckets: *mut SaSint,
        );
        fn probe_libsais16_bwt_copy_16u(u: *mut u16, a: *mut SaSint, n: SaSint);
        fn probe_libsais16_gather_lms_suffixes_16u_omp(
            t: *const u16,
            sa: *mut SaSint,
            n: SaSint,
            threads: SaSint,
        );
        fn probe_libsais16_count_and_gather_lms_suffixes_16u_omp(
            t: *const u16,
            sa: *mut SaSint,
            n: SaSint,
            buckets: *mut SaSint,
            threads: SaSint,
        ) -> SaSint;
        fn probe_libsais16_radix_sort_lms_suffixes_16u_omp(
            t: *const u16,
            sa: *mut SaSint,
            n: SaSint,
            m: SaSint,
            flags: SaSint,
            buckets: *mut SaSint,
            threads: SaSint,
        );
        fn probe_libsais16_partial_sorting_scan_left_to_right_16u_omp(
            t: *const u16,
            sa: *mut SaSint,
            n: SaSint,
            k: SaSint,
            buckets: *mut SaSint,
            left_suffixes_count: SaSint,
            d: SaSint,
            threads: SaSint,
        ) -> SaSint;
        fn probe_libsais16_partial_sorting_scan_right_to_left_16u_omp(
            t: *const u16,
            sa: *mut SaSint,
            n: SaSint,
            k: SaSint,
            buckets: *mut SaSint,
            first_lms_suffix: SaSint,
            left_suffixes_count: SaSint,
            d: SaSint,
            threads: SaSint,
        );
        fn probe_libsais16_partial_gsa_scan_right_to_left_16u_omp(
            t: *const u16,
            sa: *mut SaSint,
            n: SaSint,
            k: SaSint,
            buckets: *mut SaSint,
            first_lms_suffix: SaSint,
            left_suffixes_count: SaSint,
            d: SaSint,
            threads: SaSint,
        );
        fn probe_libsais16_renumber_lms_suffixes_16u_omp(
            sa: *mut SaSint,
            m: SaSint,
            threads: SaSint,
        ) -> SaSint;
        fn probe_libsais16_final_bwt_scan_left_to_right_16u_omp(
            t: *const u16,
            sa: *mut SaSint,
            n: SaSint,
            k: SaSint,
            induction_bucket: *mut SaSint,
            threads: SaSint,
        );
        fn probe_libsais16_final_bwt_aux_scan_left_to_right_16u_omp(
            t: *const u16,
            sa: *mut SaSint,
            n: SaSint,
            k: SaSint,
            rm: SaSint,
            i_sample: *mut SaSint,
            induction_bucket: *mut SaSint,
            threads: SaSint,
        );
        fn probe_libsais16_final_sorting_scan_left_to_right_16u_omp(
            t: *const u16,
            sa: *mut SaSint,
            n: SaSint,
            k: SaSint,
            induction_bucket: *mut SaSint,
            threads: SaSint,
        );
        fn probe_libsais16_final_bwt_scan_right_to_left_16u_omp(
            t: *const u16,
            sa: *mut SaSint,
            n: SaSint,
            k: SaSint,
            induction_bucket: *mut SaSint,
            threads: SaSint,
        ) -> SaSint;
        fn probe_libsais16_final_bwt_aux_scan_right_to_left_16u_omp(
            t: *const u16,
            sa: *mut SaSint,
            n: SaSint,
            k: SaSint,
            rm: SaSint,
            i_sample: *mut SaSint,
            induction_bucket: *mut SaSint,
            threads: SaSint,
        );
        fn probe_libsais16_final_sorting_scan_right_to_left_16u_omp(
            t: *const u16,
            sa: *mut SaSint,
            omp_block_start: SaSint,
            omp_block_size: SaSint,
            k: SaSint,
            induction_bucket: *mut SaSint,
            threads: SaSint,
        );
        fn probe_libsais16_final_gsa_scan_right_to_left_16u_omp(
            t: *const u16,
            sa: *mut SaSint,
            omp_block_start: SaSint,
            omp_block_size: SaSint,
            k: SaSint,
            induction_bucket: *mut SaSint,
            threads: SaSint,
        );
        fn probe_libsais16_bwt_copy_16u_omp(
            u: *mut u16,
            a: *mut SaSint,
            n: SaSint,
            threads: SaSint,
        );
        fn probe_libsais16_gather_marked_lms_suffixes(
            sa: *mut SaSint,
            m: SaSint,
            l: SaSint,
            omp_block_start: SaSint,
            omp_block_size: SaSint,
        ) -> SaSint;
        fn probe_libsais16_gather_marked_lms_suffixes_omp(
            sa: *mut SaSint,
            n: SaSint,
            m: SaSint,
            fs: SaSint,
            threads: SaSint,
        );
        fn probe_libsais16_renumber_and_gather_lms_suffixes_omp(
            sa: *mut SaSint,
            n: SaSint,
            m: SaSint,
            fs: SaSint,
            threads: SaSint,
        ) -> SaSint;
        fn probe_libsais16_reconstruct_lms_suffixes(
            sa: *mut SaSint,
            n: SaSint,
            m: SaSint,
            omp_block_start: SaSint,
            omp_block_size: SaSint,
        );
        fn probe_libsais16_reconstruct_lms_suffixes_omp(
            sa: *mut SaSint,
            n: SaSint,
            m: SaSint,
            threads: SaSint,
        );
        fn probe_libsais16_renumber_distinct_lms_suffixes_32s_4k(
            sa: *mut SaSint,
            m: SaSint,
            name: SaSint,
            omp_block_start: SaSint,
            omp_block_size: SaSint,
        ) -> SaSint;
        fn probe_libsais16_mark_distinct_lms_suffixes_32s(
            sa: *mut SaSint,
            m: SaSint,
            omp_block_start: SaSint,
            omp_block_size: SaSint,
        );
        fn probe_libsais16_clamp_lms_suffixes_length_32s(
            sa: *mut SaSint,
            m: SaSint,
            omp_block_start: SaSint,
            omp_block_size: SaSint,
        );
        fn probe_libsais16_renumber_distinct_lms_suffixes_32s_4k_omp(
            sa: *mut SaSint,
            m: SaSint,
            threads: SaSint,
        ) -> SaSint;
        fn probe_libsais16_mark_distinct_lms_suffixes_32s_omp(
            sa: *mut SaSint,
            n: SaSint,
            m: SaSint,
            threads: SaSint,
        );
        fn probe_libsais16_clamp_lms_suffixes_length_32s_omp(
            sa: *mut SaSint,
            n: SaSint,
            m: SaSint,
            threads: SaSint,
        );
        fn probe_libsais16_renumber_and_mark_distinct_lms_suffixes_32s_4k_omp(
            sa: *mut SaSint,
            n: SaSint,
            m: SaSint,
            threads: SaSint,
        ) -> SaSint;
        fn probe_libsais16_renumber_unique_and_nonunique_lms_suffixes_32s(
            t: *mut SaSint,
            sa: *mut SaSint,
            m: SaSint,
            f: SaSint,
            omp_block_start: SaSint,
            omp_block_size: SaSint,
        ) -> SaSint;
        fn probe_libsais16_compact_unique_and_nonunique_lms_suffixes_32s(
            sa: *mut SaSint,
            m: SaSint,
            pl: *mut SaSint,
            pr: *mut SaSint,
            omp_block_start: SaSint,
            omp_block_size: SaSint,
        );
        fn probe_libsais16_renumber_unique_and_nonunique_lms_suffixes_32s_omp(
            t: *mut SaSint,
            sa: *mut SaSint,
            m: SaSint,
            threads: SaSint,
        ) -> SaSint;
        fn probe_libsais16_compact_unique_and_nonunique_lms_suffixes_32s_omp(
            sa: *mut SaSint,
            n: SaSint,
            m: SaSint,
            fs: SaSint,
            f: SaSint,
            threads: SaSint,
        );
        fn probe_libsais16_compact_lms_suffixes_32s_omp(
            t: *mut SaSint,
            sa: *mut SaSint,
            n: SaSint,
            m: SaSint,
            fs: SaSint,
            threads: SaSint,
        ) -> SaSint;
        fn probe_libsais16_merge_unique_lms_suffixes_32s(
            t: *mut SaSint,
            sa: *mut SaSint,
            n: SaSint,
            m: SaSint,
            l: SaSint,
            omp_block_start: SaSint,
            omp_block_size: SaSint,
        );
        fn probe_libsais16_merge_nonunique_lms_suffixes_32s(
            sa: *mut SaSint,
            n: SaSint,
            m: SaSint,
            l: SaSint,
            omp_block_start: SaSint,
            omp_block_size: SaSint,
        );
        fn probe_libsais16_merge_unique_lms_suffixes_32s_omp(
            t: *mut SaSint,
            sa: *mut SaSint,
            n: SaSint,
            m: SaSint,
            threads: SaSint,
        );
        fn probe_libsais16_merge_nonunique_lms_suffixes_32s_omp(
            sa: *mut SaSint,
            n: SaSint,
            m: SaSint,
            f: SaSint,
            threads: SaSint,
        );
        fn probe_libsais16_merge_compacted_lms_suffixes_32s_omp(
            t: *mut SaSint,
            sa: *mut SaSint,
            n: SaSint,
            m: SaSint,
            f: SaSint,
            threads: SaSint,
        );
        fn probe_libsais16_radix_sort_lms_suffixes_32s_6k(
            t: *const SaSint,
            sa: *mut SaSint,
            induction_bucket: *mut SaSint,
            omp_block_start: SaSint,
            omp_block_size: SaSint,
        );
        fn probe_libsais16_radix_sort_lms_suffixes_32s_2k(
            t: *const SaSint,
            sa: *mut SaSint,
            induction_bucket: *mut SaSint,
            omp_block_start: SaSint,
            omp_block_size: SaSint,
        );
        fn probe_libsais16_radix_sort_lms_suffixes_32s_6k_omp(
            t: *const SaSint,
            sa: *mut SaSint,
            n: SaSint,
            m: SaSint,
            induction_bucket: *mut SaSint,
            threads: SaSint,
        );
        fn probe_libsais16_radix_sort_lms_suffixes_32s_2k_omp(
            t: *const SaSint,
            sa: *mut SaSint,
            n: SaSint,
            m: SaSint,
            induction_bucket: *mut SaSint,
            threads: SaSint,
        );
        fn probe_libsais16_radix_sort_lms_suffixes_32s_1k(
            t: *const SaSint,
            sa: *mut SaSint,
            n: SaSint,
            buckets: *mut SaSint,
        ) -> SaSint;
        fn probe_libsais16_radix_sort_set_markers_32s_6k(
            sa: *mut SaSint,
            induction_bucket: *mut SaSint,
            omp_block_start: SaSint,
            omp_block_size: SaSint,
        );
        fn probe_libsais16_radix_sort_set_markers_32s_4k(
            sa: *mut SaSint,
            induction_bucket: *mut SaSint,
            omp_block_start: SaSint,
            omp_block_size: SaSint,
        );
        fn probe_libsais16_radix_sort_set_markers_32s_6k_omp(
            sa: *mut SaSint,
            k: SaSint,
            induction_bucket: *mut SaSint,
            threads: SaSint,
        );
        fn probe_libsais16_radix_sort_set_markers_32s_4k_omp(
            sa: *mut SaSint,
            k: SaSint,
            induction_bucket: *mut SaSint,
            threads: SaSint,
        );
        fn probe_libsais16_place_lms_suffixes_histogram_32s_6k(
            sa: *mut SaSint,
            n: SaSint,
            k: SaSint,
            m: SaSint,
            buckets: *const SaSint,
        );
        fn probe_libsais16_place_lms_suffixes_histogram_32s_4k(
            sa: *mut SaSint,
            n: SaSint,
            k: SaSint,
            m: SaSint,
            buckets: *const SaSint,
        );
        fn probe_libsais16_place_lms_suffixes_histogram_32s_2k(
            sa: *mut SaSint,
            n: SaSint,
            k: SaSint,
            m: SaSint,
            buckets: *const SaSint,
        );
        fn probe_libsais16_gather_lms_suffixes_32s(
            t: *const SaSint,
            sa: *mut SaSint,
            n: SaSint,
        ) -> SaSint;
        fn probe_libsais16_gather_compacted_lms_suffixes_32s(
            t: *const SaSint,
            sa: *mut SaSint,
            n: SaSint,
        ) -> SaSint;
        fn probe_libsais16_count_lms_suffixes_32s_2k(
            t: *const SaSint,
            n: SaSint,
            k: SaSint,
            buckets: *mut SaSint,
        );
        fn probe_libsais16_count_and_gather_lms_suffixes_32s_4k(
            t: *const SaSint,
            sa: *mut SaSint,
            n: SaSint,
            k: SaSint,
            buckets: *mut SaSint,
            omp_block_start: SaSint,
            omp_block_size: SaSint,
        ) -> SaSint;
        fn probe_libsais16_count_and_gather_lms_suffixes_32s_4k_omp(
            t: *const SaSint,
            sa: *mut SaSint,
            n: SaSint,
            k: SaSint,
            buckets: *mut SaSint,
            local_buckets: SaSint,
            threads: SaSint,
        ) -> SaSint;
        fn probe_libsais16_count_suffixes_32s(
            t: *const SaSint,
            n: SaSint,
            k: SaSint,
            buckets: *mut SaSint,
        );
        fn probe_libsais16_initialize_buckets_start_and_end_32s_6k(k: SaSint, buckets: *mut SaSint);
        fn probe_libsais16_initialize_buckets_start_and_end_32s_4k(k: SaSint, buckets: *mut SaSint);
        fn probe_libsais16_initialize_buckets_end_32s_2k(k: SaSint, buckets: *mut SaSint);
        fn probe_libsais16_initialize_buckets_start_and_end_32s_2k(k: SaSint, buckets: *mut SaSint);
        fn probe_libsais16_initialize_buckets_start_32s_1k(k: SaSint, buckets: *mut SaSint);
        fn probe_libsais16_initialize_buckets_end_32s_1k(k: SaSint, buckets: *mut SaSint);
        fn probe_libsais16_initialize_buckets_for_lms_suffixes_radix_sort_32s_2k(
            t: *const SaSint,
            k: SaSint,
            buckets: *mut SaSint,
            first_lms_suffix: SaSint,
        );
        fn probe_libsais16_initialize_buckets_for_lms_suffixes_radix_sort_32s_6k(
            t: *const SaSint,
            k: SaSint,
            buckets: *mut SaSint,
            first_lms_suffix: SaSint,
        ) -> SaSint;
        fn probe_libsais16_initialize_buckets_for_radix_and_partial_sorting_32s_4k(
            t: *const SaSint,
            k: SaSint,
            buckets: *mut SaSint,
            first_lms_suffix: SaSint,
        );
        fn probe_libsais16_place_lms_suffixes_interval_32s_4k(
            sa: *mut SaSint,
            n: SaSint,
            k: SaSint,
            m: SaSint,
            buckets: *const SaSint,
        );
        fn probe_libsais16_place_lms_suffixes_interval_32s_2k(
            sa: *mut SaSint,
            n: SaSint,
            k: SaSint,
            m: SaSint,
            buckets: *const SaSint,
        );
        fn probe_libsais16_place_lms_suffixes_interval_32s_1k(
            t: *const SaSint,
            sa: *mut SaSint,
            k: SaSint,
            m: SaSint,
            buckets: *mut SaSint,
        );
        fn probe_libsais16_renumber_and_mark_distinct_lms_suffixes_32s_1k_omp(
            t: *mut SaSint,
            sa: *mut SaSint,
            n: SaSint,
            m: SaSint,
            threads: SaSint,
        ) -> SaSint;
        fn probe_libsais16_partial_sorting_shift_markers_32s_6k_omp(
            sa: *mut SaSint,
            k: SaSint,
            buckets: *const SaSint,
            threads: SaSint,
        );
        fn probe_libsais16_partial_sorting_shift_markers_32s_4k(sa: *mut SaSint, n: SaSint);
        fn probe_libsais16_partial_sorting_shift_buckets_32s_6k(k: SaSint, buckets: *mut SaSint);
        fn probe_libsais16_partial_sorting_scan_left_to_right_32s_6k(
            t: *const SaSint,
            sa: *mut SaSint,
            buckets: *mut SaSint,
            d: SaSint,
            omp_block_start: SaSint,
            omp_block_size: SaSint,
        ) -> SaSint;
        fn probe_libsais16_partial_sorting_scan_left_to_right_32s_4k(
            t: *const SaSint,
            sa: *mut SaSint,
            k: SaSint,
            buckets: *mut SaSint,
            d: SaSint,
            omp_block_start: SaSint,
            omp_block_size: SaSint,
        ) -> SaSint;
        fn probe_libsais16_partial_sorting_scan_left_to_right_32s_1k(
            t: *const SaSint,
            sa: *mut SaSint,
            buckets: *mut SaSint,
            omp_block_start: SaSint,
            omp_block_size: SaSint,
        );
        fn probe_libsais16_partial_sorting_scan_left_to_right_32s_6k_omp(
            t: *const SaSint,
            sa: *mut SaSint,
            n: SaSint,
            buckets: *mut SaSint,
            left_suffixes_count: SaSint,
            d: SaSint,
            threads: SaSint,
        ) -> SaSint;
        fn probe_libsais16_partial_sorting_scan_left_to_right_32s_4k_omp(
            t: *const SaSint,
            sa: *mut SaSint,
            n: SaSint,
            k: SaSint,
            buckets: *mut SaSint,
            d: SaSint,
            threads: SaSint,
        ) -> SaSint;
        fn probe_libsais16_partial_sorting_scan_left_to_right_32s_1k_omp(
            t: *const SaSint,
            sa: *mut SaSint,
            n: SaSint,
            buckets: *mut SaSint,
            threads: SaSint,
        );
        fn probe_libsais16_partial_sorting_scan_right_to_left_32s_6k(
            t: *const SaSint,
            sa: *mut SaSint,
            buckets: *mut SaSint,
            d: SaSint,
            omp_block_start: SaSint,
            omp_block_size: SaSint,
        ) -> SaSint;
        fn probe_libsais16_partial_sorting_scan_right_to_left_32s_4k(
            t: *const SaSint,
            sa: *mut SaSint,
            k: SaSint,
            buckets: *mut SaSint,
            d: SaSint,
            omp_block_start: SaSint,
            omp_block_size: SaSint,
        ) -> SaSint;
        fn probe_libsais16_partial_sorting_scan_right_to_left_32s_1k(
            t: *const SaSint,
            sa: *mut SaSint,
            buckets: *mut SaSint,
            omp_block_start: SaSint,
            omp_block_size: SaSint,
        );
        fn probe_libsais16_partial_sorting_scan_right_to_left_32s_6k_omp(
            t: *const SaSint,
            sa: *mut SaSint,
            n: SaSint,
            buckets: *mut SaSint,
            first_lms_suffix: SaSint,
            left_suffixes_count: SaSint,
            d: SaSint,
            threads: SaSint,
        ) -> SaSint;
        fn probe_libsais16_partial_sorting_scan_right_to_left_32s_4k_omp(
            t: *const SaSint,
            sa: *mut SaSint,
            n: SaSint,
            k: SaSint,
            buckets: *mut SaSint,
            d: SaSint,
            threads: SaSint,
        ) -> SaSint;
        fn probe_libsais16_partial_sorting_scan_right_to_left_32s_1k_omp(
            t: *const SaSint,
            sa: *mut SaSint,
            n: SaSint,
            buckets: *mut SaSint,
            threads: SaSint,
        );
        fn probe_libsais16_partial_sorting_gather_lms_suffixes_32s_4k(
            sa: *mut SaSint,
            omp_block_start: SaSint,
            omp_block_size: SaSint,
        ) -> SaSint;
        fn probe_libsais16_partial_sorting_gather_lms_suffixes_32s_1k(
            sa: *mut SaSint,
            omp_block_start: SaSint,
            omp_block_size: SaSint,
        ) -> SaSint;
        fn probe_libsais16_partial_sorting_gather_lms_suffixes_32s_4k_omp(
            sa: *mut SaSint,
            n: SaSint,
            threads: SaSint,
        );
        fn probe_libsais16_partial_sorting_gather_lms_suffixes_32s_1k_omp(
            sa: *mut SaSint,
            n: SaSint,
            threads: SaSint,
        );
        fn probe_libsais16_count_and_gather_lms_suffixes_32s_2k(
            t: *const SaSint,
            sa: *mut SaSint,
            n: SaSint,
            k: SaSint,
            buckets: *mut SaSint,
            omp_block_start: SaSint,
            omp_block_size: SaSint,
        ) -> SaSint;
        fn probe_libsais16_count_and_gather_compacted_lms_suffixes_32s_2k(
            t: *const SaSint,
            sa: *mut SaSint,
            n: SaSint,
            k: SaSint,
            buckets: *mut SaSint,
            omp_block_start: SaSint,
            omp_block_size: SaSint,
        ) -> SaSint;
        fn probe_libsais16_count_and_gather_lms_suffixes_32s_2k_omp(
            t: *const SaSint,
            sa: *mut SaSint,
            n: SaSint,
            k: SaSint,
            buckets: *mut SaSint,
            local_buckets: SaSint,
            threads: SaSint,
        ) -> SaSint;
        fn probe_libsais16_count_and_gather_compacted_lms_suffixes_32s_2k_omp(
            t: *const SaSint,
            sa: *mut SaSint,
            n: SaSint,
            k: SaSint,
            buckets: *mut SaSint,
            local_buckets: SaSint,
            threads: SaSint,
        );
        fn probe_libsais16_reconstruct_compacted_lms_suffixes_32s_2k_omp(
            t: *mut SaSint,
            sa: *mut SaSint,
            n: SaSint,
            k: SaSint,
            m: SaSint,
            fs: SaSint,
            f: SaSint,
            buckets: *mut SaSint,
            local_buckets: SaSint,
            threads: SaSint,
        );
        fn probe_libsais16_reconstruct_compacted_lms_suffixes_32s_1k_omp(
            t: *mut SaSint,
            sa: *mut SaSint,
            n: SaSint,
            m: SaSint,
            fs: SaSint,
            f: SaSint,
            threads: SaSint,
        );
    }

    fn brute_sa(t: &[u16]) -> Vec<SaSint> {
        let mut sa: Vec<_> = (0..t.len() as SaSint).collect();
        sa.sort_by(|&a, &b| t[a as usize..].cmp(&t[b as usize..]));
        sa
    }

    #[test]
    fn libsais16_gather_lms_suffixes_16u_matches_c() {
        let cases: &[&[u16]] = &[
            &[2, 1, 3, 1, 2, 0],
            &[7, 7, 7, 7, 0],
            &[3, 1, 2, 1, 0, 4, 1, 0],
            &[9, 1, 9, 1, 9, 0, 2, 2, 0],
        ];

        for &text in cases {
            let n = text.len() as SaSint;
            let mut rust_sa = vec![-99; text.len()];
            let mut c_sa = rust_sa.clone();

            gather_lms_suffixes_16u(text, &mut rust_sa, n, n - 1, 0, n);
            unsafe {
                probe_libsais16_gather_lms_suffixes_16u(
                    text.as_ptr(),
                    c_sa.as_mut_ptr(),
                    n,
                    n - 1,
                    0,
                    n,
                );
            }

            assert_eq!(rust_sa, c_sa);
        }
    }

    #[test]
    fn libsais16_count_and_gather_lms_suffixes_16u_matches_c() {
        let cases: &[&[u16]] = &[
            &[2, 1, 3, 1, 2, 0],
            &[7, 7, 7, 7, 0],
            &[3, 1, 2, 1, 0, 4, 1, 0],
            &[9, 1, 9, 1, 9, 0, 2, 2, 0],
        ];

        for &text in cases {
            let n = text.len() as SaSint;
            let mut rust_sa = vec![-99; text.len()];
            let mut c_sa = rust_sa.clone();
            let mut rust_buckets = vec![-1; 4 * ALPHABET_SIZE];
            let mut c_buckets = rust_buckets.clone();

            let rust_m =
                count_and_gather_lms_suffixes_16u(text, &mut rust_sa, n, &mut rust_buckets, 0, n);
            let c_m = unsafe {
                probe_libsais16_count_and_gather_lms_suffixes_16u(
                    text.as_ptr(),
                    c_sa.as_mut_ptr(),
                    n,
                    c_buckets.as_mut_ptr(),
                    0,
                    n,
                )
            };

            assert_eq!(rust_m, c_m);
            assert_eq!(rust_sa, c_sa);
            assert_eq!(rust_buckets, c_buckets);
        }
    }

    #[test]
    fn libsais16_initialize_buckets_start_and_end_16u_matches_c() {
        let mut rust_buckets = vec![0; 8 * ALPHABET_SIZE];
        for (symbol, counts) in [
            (0usize, [1, 0, 0, 2]),
            (1, [0, 3, 1, 0]),
            (7, [2, 1, 0, 1]),
            (1024, [0, 0, 5, 0]),
        ] {
            for state in 0..4 {
                rust_buckets[buckets_index4(symbol, state)] = counts[state];
            }
        }
        let mut c_buckets = rust_buckets.clone();
        let mut rust_freq = vec![-1; ALPHABET_SIZE];
        let mut c_freq = rust_freq.clone();

        let rust_k = initialize_buckets_start_and_end_16u(&mut rust_buckets, Some(&mut rust_freq));
        let c_k = unsafe {
            probe_libsais16_initialize_buckets_start_and_end_16u(
                c_buckets.as_mut_ptr(),
                c_freq.as_mut_ptr(),
            )
        };

        assert_eq!(rust_k, c_k);
        assert_eq!(rust_buckets, c_buckets);
        assert_eq!(rust_freq, c_freq);

        let mut rust_buckets_no_freq = vec![0; 8 * ALPHABET_SIZE];
        rust_buckets_no_freq[..4 * ALPHABET_SIZE]
            .copy_from_slice(&rust_buckets[..4 * ALPHABET_SIZE]);
        let mut c_buckets_no_freq = rust_buckets_no_freq.clone();

        let rust_k = initialize_buckets_start_and_end_16u(&mut rust_buckets_no_freq, None);
        let c_k = unsafe {
            probe_libsais16_initialize_buckets_start_and_end_16u(
                c_buckets_no_freq.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };

        assert_eq!(rust_k, c_k);
        assert_eq!(rust_buckets_no_freq, c_buckets_no_freq);
    }

    #[test]
    fn libsais16_lms_radix_bucket_initialization_matches_c() {
        let text = [3, 1, 2, 1, 0, 4, 1, 0];
        let n = text.len() as SaSint;
        let mut rust_sa = vec![-99; text.len()];
        let mut rust_buckets = vec![0; 8 * ALPHABET_SIZE];
        let m = count_and_gather_lms_suffixes_16u(
            &text,
            &mut rust_sa,
            n,
            &mut rust_buckets[..4 * ALPHABET_SIZE],
            0,
            n,
        );
        initialize_buckets_start_and_end_16u(&mut rust_buckets, None);
        let first_lms_suffix = rust_sa[(n - m) as usize];

        let mut c_buckets = rust_buckets.clone();
        let rust_count = initialize_buckets_for_lms_suffixes_radix_sort_16u(
            &text,
            &mut rust_buckets,
            first_lms_suffix,
        );
        let c_count = unsafe {
            probe_libsais16_initialize_buckets_for_lms_suffixes_radix_sort_16u(
                text.as_ptr(),
                c_buckets.as_mut_ptr(),
                first_lms_suffix,
            )
        };

        assert_eq!(rust_count, c_count);
        assert_eq!(rust_buckets, c_buckets);
    }

    #[test]
    fn libsais16_radix_sort_lms_suffixes_16u_matches_c() {
        let text = [3, 1, 2, 1, 0, 4, 1, 0];
        let n = text.len() as SaSint;
        let mut rust_sa = vec![-99; text.len()];
        let mut rust_buckets = vec![0; 8 * ALPHABET_SIZE];
        let m = count_and_gather_lms_suffixes_16u(
            &text,
            &mut rust_sa,
            n,
            &mut rust_buckets[..4 * ALPHABET_SIZE],
            0,
            n,
        );
        initialize_buckets_start_and_end_16u(&mut rust_buckets, None);
        let first_lms_suffix = rust_sa[(n - m) as usize];
        initialize_buckets_for_lms_suffixes_radix_sort_16u(
            &text,
            &mut rust_buckets,
            first_lms_suffix,
        );

        let mut c_sa = rust_sa.clone();
        let mut c_buckets = rust_buckets.clone();
        {
            let induction_bucket = &mut rust_buckets[4 * ALPHABET_SIZE..];
            radix_sort_lms_suffixes_16u(&text, &mut rust_sa, induction_bucket, n - m + 1, m - 1);
        }
        unsafe {
            probe_libsais16_radix_sort_lms_suffixes_16u(
                text.as_ptr(),
                c_sa.as_mut_ptr(),
                c_buckets[4 * ALPHABET_SIZE..].as_mut_ptr(),
                n - m + 1,
                m - 1,
            );
        }

        assert_eq!(rust_sa, c_sa);
        assert_eq!(rust_buckets, c_buckets);
    }

    #[test]
    fn libsais16_initialize_buckets_for_partial_sorting_16u_matches_c() {
        let text = [3, 1, 2, 1, 0, 4, 1, 0];
        let n = text.len() as SaSint;
        let mut rust_sa = vec![-99; text.len()];
        let mut rust_buckets = vec![0; 8 * ALPHABET_SIZE];
        let m = count_and_gather_lms_suffixes_16u(
            &text,
            &mut rust_sa,
            n,
            &mut rust_buckets[..4 * ALPHABET_SIZE],
            0,
            n,
        );
        initialize_buckets_start_and_end_16u(&mut rust_buckets, None);
        let first_lms_suffix = rust_sa[(n - m) as usize];
        let left_suffixes_count = initialize_buckets_for_lms_suffixes_radix_sort_16u(
            &text,
            &mut rust_buckets,
            first_lms_suffix,
        );
        let mut c_buckets = rust_buckets.clone();

        initialize_buckets_for_partial_sorting_16u(
            &text,
            &mut rust_buckets,
            first_lms_suffix,
            left_suffixes_count,
        );
        unsafe {
            probe_libsais16_initialize_buckets_for_partial_sorting_16u(
                text.as_ptr(),
                c_buckets.as_mut_ptr(),
                first_lms_suffix,
                left_suffixes_count,
            );
        }

        assert_eq!(rust_buckets, c_buckets);
    }

    fn partial_scan_fixture() -> ([u16; 10], Vec<SaSint>, Vec<SaSint>) {
        let text = [1, 0, 2, 1, 3, 0, 2, 4, 1, 0];
        let mut sa = vec![0; 128];
        sa[..5].copy_from_slice(&[3, 5 | SAINT_MIN, 7, 2, 9 | SAINT_MIN]);

        let mut buckets = vec![0; 6 * ALPHABET_SIZE];
        for v in 0..32 {
            buckets[v] = 80 + (v as SaSint) * 4;
            buckets[2 * ALPHABET_SIZE + v] = if v % 3 == 0 { 2 } else { 0 };
            buckets[4 * ALPHABET_SIZE + v] = 20 + (v as SaSint) * 4;
        }

        (text, sa, buckets)
    }

    #[test]
    fn libsais16_partial_sorting_scan_left_to_right_16u_matches_c() {
        let (text, mut rust_sa, mut rust_buckets) = partial_scan_fixture();
        let mut c_sa = rust_sa.clone();
        let mut c_buckets = rust_buckets.clone();

        let rust_d =
            partial_sorting_scan_left_to_right_16u(&text, &mut rust_sa, &mut rust_buckets, 3, 0, 5);
        let c_d = unsafe {
            probe_libsais16_partial_sorting_scan_left_to_right_16u(
                text.as_ptr(),
                c_sa.as_mut_ptr(),
                c_buckets.as_mut_ptr(),
                3,
                0,
                5,
            )
        };

        assert_eq!(rust_d, c_d);
        assert_eq!(rust_sa, c_sa);
        assert_eq!(rust_buckets, c_buckets);
    }

    #[test]
    fn libsais16_partial_sorting_scan_right_to_left_16u_matches_c() {
        let (text, mut rust_sa, mut rust_buckets) = partial_scan_fixture();
        let mut c_sa = rust_sa.clone();
        let mut c_buckets = rust_buckets.clone();

        let rust_d =
            partial_sorting_scan_right_to_left_16u(&text, &mut rust_sa, &mut rust_buckets, 3, 0, 5);
        let c_d = unsafe {
            probe_libsais16_partial_sorting_scan_right_to_left_16u(
                text.as_ptr(),
                c_sa.as_mut_ptr(),
                c_buckets.as_mut_ptr(),
                3,
                0,
                5,
            )
        };

        assert_eq!(rust_d, c_d);
        assert_eq!(rust_sa, c_sa);
        assert_eq!(rust_buckets, c_buckets);
    }

    #[test]
    fn libsais16_partial_gsa_scan_right_to_left_16u_matches_c() {
        let (text, mut rust_sa, mut rust_buckets) = partial_scan_fixture();
        let mut c_sa = rust_sa.clone();
        let mut c_buckets = rust_buckets.clone();

        let rust_d =
            partial_gsa_scan_right_to_left_16u(&text, &mut rust_sa, &mut rust_buckets, 3, 0, 5);
        let c_d = unsafe {
            probe_libsais16_partial_gsa_scan_right_to_left_16u(
                text.as_ptr(),
                c_sa.as_mut_ptr(),
                c_buckets.as_mut_ptr(),
                3,
                0,
                5,
            )
        };

        assert_eq!(rust_d, c_d);
        assert_eq!(rust_sa, c_sa);
        assert_eq!(rust_buckets, c_buckets);
    }

    #[test]
    fn libsais16_partial_sorting_shift_markers_16u_matches_c() {
        let mut rust_sa = vec![0; 16];
        rust_sa[2..6].copy_from_slice(&[1, 2 | SAINT_MIN, 3 | SAINT_MIN, 4]);
        rust_sa[8..12].copy_from_slice(&[5 | SAINT_MIN, 6, 7 | SAINT_MIN, 8]);
        let mut c_sa = rust_sa.clone();

        let mut buckets = vec![0; 6 * ALPHABET_SIZE];
        buckets[0] = 2;
        buckets[2] = 8;
        buckets[4 * ALPHABET_SIZE + 2] = 6;
        buckets[4 * ALPHABET_SIZE + 4] = 12;

        let n = rust_sa.len() as SaSint;
        partial_sorting_shift_markers_16u_omp(&mut rust_sa, n, &buckets, 1);
        unsafe {
            probe_libsais16_partial_sorting_shift_markers_16u_omp(
                c_sa.as_mut_ptr(),
                c_sa.len() as SaSint,
                buckets.as_ptr(),
                1,
            );
        }

        assert_eq!(rust_sa, c_sa);
    }

    #[test]
    fn libsais16_partial_left_to_right_16u_block_omp_uses_cache_pipeline() {
        let block_size = 65_536usize;
        let k = 512usize;
        let text: Vec<u16> = (0..block_size + 2)
            .map(|i| 1 + ((i * 17 + i / 7) % (k - 1)) as u16)
            .collect();
        let sa_len = block_size + 2 * k * 100;
        let mut base_sa = vec![0; sa_len];
        for (i, slot) in base_sa.iter_mut().take(block_size).enumerate() {
            *slot = (i + 2) as SaSint;
        }
        let mut base_buckets = vec![0; 8 * ALPHABET_SIZE];
        for v in 0..2 * k {
            base_buckets[4 * ALPHABET_SIZE + v] = (block_size + v * 100) as SaSint;
        }

        let mut scalar_sa = base_sa.clone();
        let mut threaded_sa = base_sa;
        let mut scalar_buckets = base_buckets.clone();
        let mut threaded_buckets = base_buckets;
        let mut thread_state = alloc_thread_state(4).unwrap();
        let scalar_d = partial_sorting_scan_left_to_right_16u(
            &text,
            &mut scalar_sa,
            &mut scalar_buckets,
            0,
            0,
            block_size as SaSint,
        );
        let threaded_d = partial_sorting_scan_left_to_right_16u_block_omp(
            &text,
            &mut threaded_sa,
            k as SaSint,
            &mut threaded_buckets,
            0,
            0,
            block_size as SaSint,
            4,
            &mut thread_state,
        );

        assert_eq!(threaded_d, scalar_d);
        assert_eq!(threaded_sa, scalar_sa);
        assert_eq!(threaded_buckets, scalar_buckets);
    }

    #[test]
    fn libsais16_partial_left_to_right_16u_omp_uses_block_pipeline() {
        let block_size = 65_536usize;
        let k = 512usize;
        let text: Vec<u16> = (0..block_size + 2)
            .map(|i| 1 + ((i * 17 + i / 7) % (k - 1)) as u16)
            .collect();
        let sa_len = block_size + 2 * k * 100;
        let mut base_sa = vec![0; sa_len];
        for (i, slot) in base_sa.iter_mut().take(block_size).enumerate() {
            let value = (i + 2) as SaSint;
            *slot = if i % 17 == 0 {
                value | SAINT_MIN
            } else {
                value
            };
        }
        let mut base_buckets = vec![0; 8 * ALPHABET_SIZE];
        for v in 0..2 * k {
            base_buckets[4 * ALPHABET_SIZE + v] = (block_size + v * 100) as SaSint;
            base_buckets[2 * ALPHABET_SIZE + v] = if v % 5 == 0 { 3 } else { 0 };
        }

        let mut scalar_sa = base_sa.clone();
        let mut threaded_sa = base_sa;
        let mut scalar_buckets = base_buckets.clone();
        let mut threaded_buckets = base_buckets;
        let scalar_d = partial_sorting_scan_left_to_right_16u_omp(
            &text,
            &mut scalar_sa,
            text.len() as SaSint,
            k as SaSint,
            &mut scalar_buckets,
            block_size as SaSint,
            7,
            1,
        );
        let threaded_d = partial_sorting_scan_left_to_right_16u_omp(
            &text,
            &mut threaded_sa,
            text.len() as SaSint,
            k as SaSint,
            &mut threaded_buckets,
            block_size as SaSint,
            7,
            4,
        );

        assert_eq!(threaded_d, scalar_d);
        assert_eq!(threaded_sa, scalar_sa);
        assert_eq!(threaded_buckets, scalar_buckets);
    }

    #[test]
    fn libsais16_partial_right_to_left_16u_block_omp_uses_cache_pipeline() {
        let block_size = 65_536usize;
        let k = 512usize;
        let width = 2 * k;
        let block_start = width * 200 + 1024;
        let text: Vec<u16> = (0..block_size + 2)
            .map(|i| 1 + ((i * 17 + i / 7) % (k - 1)) as u16)
            .collect();
        let sa_len = block_start + block_size + 1;
        let mut base_sa = vec![0; sa_len];
        for i in 0..block_size {
            let value = (i + 2) as SaSint;
            base_sa[block_start + i] = if i % 17 == 0 {
                value | SAINT_MIN
            } else {
                value
            };
        }
        let mut base_buckets = vec![0; 8 * ALPHABET_SIZE];
        for v in 0..width {
            base_buckets[v] = ((v + 1) * 200) as SaSint;
            base_buckets[2 * ALPHABET_SIZE + v] = if v % 5 == 0 { 3 } else { 0 };
        }

        let mut scalar_sa = base_sa.clone();
        let mut threaded_sa = base_sa.clone();
        let mut scalar_buckets = base_buckets.clone();
        let mut threaded_buckets = base_buckets.clone();
        let mut thread_state = alloc_thread_state(4).unwrap();
        let scalar_d = partial_sorting_scan_right_to_left_16u(
            &text,
            &mut scalar_sa,
            &mut scalar_buckets,
            7,
            block_start as SaSint,
            block_size as SaSint,
        );
        let threaded_d = partial_sorting_scan_right_to_left_16u_block_omp(
            &text,
            &mut threaded_sa,
            k as SaSint,
            &mut threaded_buckets,
            7,
            block_start as SaSint,
            block_size as SaSint,
            4,
            &mut thread_state,
        );
        assert_eq!(threaded_d, scalar_d);
        assert_eq!(threaded_sa, scalar_sa);
        assert_eq!(threaded_buckets, scalar_buckets);

        let mut scalar_sa = base_sa;
        let mut threaded_sa = scalar_sa.clone();
        let mut scalar_buckets = base_buckets.clone();
        let mut threaded_buckets = base_buckets;
        let scalar_d = partial_gsa_scan_right_to_left_16u(
            &text,
            &mut scalar_sa,
            &mut scalar_buckets,
            7,
            block_start as SaSint,
            block_size as SaSint,
        );
        let threaded_d = partial_gsa_scan_right_to_left_16u_block_omp(
            &text,
            &mut threaded_sa,
            k as SaSint,
            &mut threaded_buckets,
            7,
            block_start as SaSint,
            block_size as SaSint,
            4,
            &mut thread_state,
        );
        assert_eq!(threaded_d, scalar_d);
        assert_eq!(threaded_sa, scalar_sa);
        assert_eq!(threaded_buckets, scalar_buckets);
    }

    #[test]
    fn libsais16_partial_right_to_left_16u_omp_uses_block_pipeline() {
        let block_size = 65_536usize;
        let k = 512usize;
        let width = 2 * k;
        let block_start = width * 200 + 1024;
        let text: Vec<u16> = (0..block_size + 2)
            .map(|i| 1 + ((i * 17 + i / 7) % (k - 1)) as u16)
            .collect();
        let sa_len = block_start + block_size + 1;
        let n = sa_len as SaSint;
        let first_lms_suffix = n - (block_start + block_size) as SaSint;
        let left_suffixes_count = block_start as SaSint - 1;
        let mut base_sa = vec![0; sa_len];
        for i in 0..block_size {
            let value = (i + 2) as SaSint;
            base_sa[block_start + i] = if i % 17 == 0 {
                value | SAINT_MIN
            } else {
                value
            };
        }
        let mut base_buckets = vec![0; 8 * ALPHABET_SIZE];
        for v in 0..width {
            base_buckets[v] = ((v + 1) * 200) as SaSint;
            base_buckets[2 * ALPHABET_SIZE + v] = if v % 5 == 0 { 3 } else { 0 };
        }

        let mut scalar_sa = base_sa.clone();
        let mut threaded_sa = base_sa.clone();
        let mut scalar_buckets = base_buckets.clone();
        let mut threaded_buckets = base_buckets.clone();
        partial_sorting_scan_right_to_left_16u_omp(
            &text,
            &mut scalar_sa,
            n,
            k as SaSint,
            &mut scalar_buckets,
            first_lms_suffix,
            left_suffixes_count,
            7,
            1,
        );
        partial_sorting_scan_right_to_left_16u_omp(
            &text,
            &mut threaded_sa,
            n,
            k as SaSint,
            &mut threaded_buckets,
            first_lms_suffix,
            left_suffixes_count,
            7,
            4,
        );
        assert_eq!(threaded_sa, scalar_sa);
        assert_eq!(threaded_buckets, scalar_buckets);

        let mut scalar_sa = base_sa;
        let mut threaded_sa = scalar_sa.clone();
        let mut scalar_buckets = base_buckets.clone();
        let mut threaded_buckets = base_buckets;
        partial_gsa_scan_right_to_left_16u_omp(
            &text,
            &mut scalar_sa,
            n,
            k as SaSint,
            &mut scalar_buckets,
            first_lms_suffix,
            left_suffixes_count,
            7,
            1,
        );
        partial_gsa_scan_right_to_left_16u_omp(
            &text,
            &mut threaded_sa,
            n,
            k as SaSint,
            &mut threaded_buckets,
            first_lms_suffix,
            left_suffixes_count,
            7,
            4,
        );
        assert_eq!(threaded_sa, scalar_sa);
        assert_eq!(threaded_buckets, scalar_buckets);
    }

    fn final_scan_fixture() -> ([u16; 10], Vec<SaSint>, Vec<SaSint>) {
        let text = [1, 0, 2, 1, 3, 0, 2, 4, 1, 0];
        let mut sa = vec![0; 96];
        sa[..6].copy_from_slice(&[3, 0, 5 | SAINT_MIN, 7, 2, 9 | SAINT_MIN]);

        let mut induction_bucket = vec![0; ALPHABET_SIZE];
        for c in 0..8 {
            induction_bucket[c] = 24 + (c as SaSint) * 6;
        }

        (text, sa, induction_bucket)
    }

    fn final_order_buckets(induction_bucket: &[SaSint]) -> Vec<SaSint> {
        let mut buckets = vec![0; 8 * ALPHABET_SIZE];
        buckets[6 * ALPHABET_SIZE..7 * ALPHABET_SIZE].copy_from_slice(induction_bucket);
        buckets[7 * ALPHABET_SIZE..8 * ALPHABET_SIZE].copy_from_slice(induction_bucket);
        buckets
    }

    #[test]
    fn libsais16_final_sorting_scan_left_to_right_16u_matches_c() {
        let (text, mut rust_sa, mut rust_bucket) = final_scan_fixture();
        let mut c_sa = rust_sa.clone();
        let mut c_bucket = rust_bucket.clone();

        final_sorting_scan_left_to_right_16u(&text, &mut rust_sa, &mut rust_bucket, 0, 6);
        unsafe {
            probe_libsais16_final_sorting_scan_left_to_right_16u(
                text.as_ptr(),
                c_sa.as_mut_ptr(),
                c_bucket.as_mut_ptr(),
                0,
                6,
            );
        }

        assert_eq!(rust_sa, c_sa);
        assert_eq!(rust_bucket, c_bucket);
    }

    #[test]
    fn libsais16_final_sorting_scan_right_to_left_16u_matches_c() {
        let (text, mut rust_sa, mut rust_bucket) = final_scan_fixture();
        let mut c_sa = rust_sa.clone();
        let mut c_bucket = rust_bucket.clone();

        final_sorting_scan_right_to_left_16u(&text, &mut rust_sa, &mut rust_bucket, 0, 6);
        unsafe {
            probe_libsais16_final_sorting_scan_right_to_left_16u(
                text.as_ptr(),
                c_sa.as_mut_ptr(),
                c_bucket.as_mut_ptr(),
                0,
                6,
            );
        }

        assert_eq!(rust_sa, c_sa);
        assert_eq!(rust_bucket, c_bucket);
    }

    #[test]
    fn libsais16_final_gsa_scan_right_to_left_16u_matches_c() {
        let (text, mut rust_sa, mut rust_bucket) = final_scan_fixture();
        let mut c_sa = rust_sa.clone();
        let mut c_bucket = rust_bucket.clone();

        final_gsa_scan_right_to_left_16u(&text, &mut rust_sa, &mut rust_bucket, 0, 6);
        unsafe {
            probe_libsais16_final_gsa_scan_right_to_left_16u(
                text.as_ptr(),
                c_sa.as_mut_ptr(),
                c_bucket.as_mut_ptr(),
                0,
                6,
            );
        }

        assert_eq!(rust_sa, c_sa);
        assert_eq!(rust_bucket, c_bucket);
    }

    #[test]
    fn libsais16_final_sorting_32s_helpers_behave_like_upstream_shapes() {
        let t = vec![0, 1, 2, 1, 0, 1, 2, 1, 0];

        let mut rust_sa = vec![1, 0, 0];
        let mut rust_bucket = vec![0, 1, 3];
        let mut c_sa = rust_sa.clone();
        let mut c_bucket = rust_bucket.clone();
        final_sorting_scan_left_to_right_32s(&t, &mut rust_sa, &mut rust_bucket, 0, 1);
        unsafe {
            probe_libsais16_final_sorting_scan_left_to_right_32s(
                t.as_ptr(),
                c_sa.as_mut_ptr(),
                c_bucket.as_mut_ptr(),
                0,
                1,
            );
        }
        assert_eq!(rust_sa, c_sa);
        assert_eq!(rust_bucket, c_bucket);

        let mut rust_sa = vec![0, 2, 0];
        let mut rust_bucket = vec![1, 2, 3];
        let mut c_sa = rust_sa.clone();
        let mut c_bucket = rust_bucket.clone();
        final_sorting_scan_right_to_left_32s(&t, &mut rust_sa, &mut rust_bucket, 0, 2);
        unsafe {
            probe_libsais16_final_sorting_scan_right_to_left_32s(
                t.as_ptr(),
                c_sa.as_mut_ptr(),
                c_bucket.as_mut_ptr(),
                0,
                2,
            );
        }
        assert_eq!(rust_sa, c_sa);
        assert_eq!(rust_bucket, c_bucket);

        let mut sa = vec![1, 2, 0, 0];
        let mut induction_bucket = vec![0, 1, 3];
        let mut cache = vec![ThreadCache::default(); PER_THREAD_CACHE_SIZE];
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

        let mut sa = vec![0, 2, 0, 0];
        let mut induction_bucket = vec![1, 2, 3];
        let mut cache = vec![ThreadCache::default(); PER_THREAD_CACHE_SIZE];
        final_sorting_scan_right_to_left_32s_block_omp(
            &t,
            &mut sa,
            &mut induction_bucket,
            &mut cache,
            0,
            2,
            2,
        );
        assert_eq!(sa[1] & SAINT_MAX, 1);
        assert_eq!(induction_bucket[1], 1);
    }

    #[test]
    fn libsais16_final_left_to_right_16u_block_omp_uses_cache_pipeline() {
        let block_size = 65_536usize;
        let k = 512usize;
        let text: Vec<u16> = (0..=block_size).map(|i| 1 + (i % (k - 1)) as u16).collect();
        let sa_len = block_size + k * 200;
        let mut base_sa = vec![0; sa_len];
        for (i, slot) in base_sa.iter_mut().take(block_size).enumerate() {
            *slot = (i + 1) as SaSint;
        }
        let mut base_bucket = vec![0; k];
        for c in 0..k {
            base_bucket[c] = (block_size + c * 200) as SaSint;
        }

        let mut scalar_sa = base_sa.clone();
        let mut threaded_sa = base_sa.clone();
        let mut scalar_bucket = base_bucket.clone();
        let mut threaded_bucket = base_bucket.clone();
        let mut thread_state = alloc_thread_state(4).unwrap();
        final_bwt_scan_left_to_right_16u(
            &text,
            &mut scalar_sa,
            &mut scalar_bucket,
            0,
            block_size as SaSint,
        );
        final_bwt_scan_left_to_right_16u_block_omp(
            &text,
            &mut threaded_sa,
            k as SaSint,
            &mut threaded_bucket,
            0,
            block_size as SaSint,
            4,
            &mut thread_state,
        );
        assert_eq!(threaded_sa, scalar_sa);
        assert_eq!(threaded_bucket, scalar_bucket);

        let rm = 3;
        let mut scalar_sa = base_sa.clone();
        let mut threaded_sa = base_sa.clone();
        let mut scalar_bucket = base_bucket.clone();
        let mut threaded_bucket = base_bucket.clone();
        let mut scalar_i = vec![-1; (block_size / (rm as usize + 1)) + 2];
        let mut threaded_i = scalar_i.clone();
        final_bwt_aux_scan_left_to_right_16u(
            &text,
            &mut scalar_sa,
            rm,
            &mut scalar_i,
            &mut scalar_bucket,
            0,
            block_size as SaSint,
        );
        final_bwt_aux_scan_left_to_right_16u_block_omp(
            &text,
            &mut threaded_sa,
            k as SaSint,
            rm,
            &mut threaded_i,
            &mut threaded_bucket,
            0,
            block_size as SaSint,
            4,
            &mut thread_state,
        );
        assert_eq!(threaded_sa, scalar_sa);
        assert_eq!(threaded_i, scalar_i);
        assert_eq!(threaded_bucket, scalar_bucket);

        let mut scalar_sa = base_sa;
        let mut threaded_sa = scalar_sa.clone();
        let mut scalar_bucket = base_bucket.clone();
        let mut threaded_bucket = base_bucket;
        final_sorting_scan_left_to_right_16u(
            &text,
            &mut scalar_sa,
            &mut scalar_bucket,
            0,
            block_size as SaSint,
        );
        final_sorting_scan_left_to_right_16u_block_omp(
            &text,
            &mut threaded_sa,
            k as SaSint,
            &mut threaded_bucket,
            0,
            block_size as SaSint,
            4,
            &mut thread_state,
        );
        assert_eq!(threaded_sa, scalar_sa);
        assert_eq!(threaded_bucket, scalar_bucket);
    }

    #[test]
    fn libsais16_final_right_to_left_16u_block_omp_uses_cache_pipeline() {
        let block_size = 65_536usize;
        let k = 512usize;
        let block_start = k * 200 + 1024;
        let text: Vec<u16> = (0..=block_size + 1)
            .map(|i| 1 + (i % (k - 1)) as u16)
            .collect();
        let sa_len = block_start + block_size + 1;
        let mut base_sa = vec![0; sa_len];
        for i in 0..block_size {
            base_sa[block_start + i] = (i + 1) as SaSint;
        }
        let mut base_bucket = vec![0; k];
        for c in 0..k {
            base_bucket[c] = ((c + 1) * 200) as SaSint;
        }

        let mut scalar_sa = base_sa.clone();
        let mut threaded_sa = base_sa.clone();
        let mut scalar_bucket = base_bucket.clone();
        let mut threaded_bucket = base_bucket.clone();
        let mut thread_state = alloc_thread_state(4).unwrap();
        final_bwt_scan_right_to_left_16u(
            &text,
            &mut scalar_sa,
            &mut scalar_bucket,
            block_start as SaSint,
            block_size as SaSint,
        );
        final_bwt_scan_right_to_left_16u_block_omp(
            &text,
            &mut threaded_sa,
            k as SaSint,
            &mut threaded_bucket,
            block_start as SaSint,
            block_size as SaSint,
            4,
            &mut thread_state,
        );
        assert_eq!(threaded_sa, scalar_sa);
        assert_eq!(threaded_bucket, scalar_bucket);

        let rm = 3;
        let mut scalar_sa = base_sa.clone();
        let mut threaded_sa = base_sa.clone();
        let mut scalar_bucket = base_bucket.clone();
        let mut threaded_bucket = base_bucket.clone();
        let mut scalar_i = vec![-1; (block_size / (rm as usize + 1)) + 2];
        let mut threaded_i = scalar_i.clone();
        final_bwt_aux_scan_right_to_left_16u(
            &text,
            &mut scalar_sa,
            rm,
            &mut scalar_i,
            &mut scalar_bucket,
            block_start as SaSint,
            block_size as SaSint,
        );
        final_bwt_aux_scan_right_to_left_16u_block_omp(
            &text,
            &mut threaded_sa,
            k as SaSint,
            rm,
            &mut threaded_i,
            &mut threaded_bucket,
            block_start as SaSint,
            block_size as SaSint,
            4,
            &mut thread_state,
        );
        assert_eq!(threaded_sa, scalar_sa);
        assert_eq!(threaded_i, scalar_i);
        assert_eq!(threaded_bucket, scalar_bucket);

        let mut scalar_sa = base_sa.clone();
        let mut threaded_sa = base_sa.clone();
        let mut scalar_bucket = base_bucket.clone();
        let mut threaded_bucket = base_bucket.clone();
        final_sorting_scan_right_to_left_16u(
            &text,
            &mut scalar_sa,
            &mut scalar_bucket,
            block_start as SaSint,
            block_size as SaSint,
        );
        final_sorting_scan_right_to_left_16u_block_omp(
            &text,
            &mut threaded_sa,
            k as SaSint,
            &mut threaded_bucket,
            block_start as SaSint,
            block_size as SaSint,
            4,
            &mut thread_state,
        );
        assert_eq!(threaded_sa, scalar_sa);
        assert_eq!(threaded_bucket, scalar_bucket);

        let mut scalar_sa = base_sa;
        let mut threaded_sa = scalar_sa.clone();
        let mut scalar_bucket = base_bucket.clone();
        let mut threaded_bucket = base_bucket;
        final_gsa_scan_right_to_left_16u(
            &text,
            &mut scalar_sa,
            &mut scalar_bucket,
            block_start as SaSint,
            block_size as SaSint,
        );
        final_gsa_scan_right_to_left_16u_block_omp(
            &text,
            &mut threaded_sa,
            k as SaSint,
            &mut threaded_bucket,
            block_start as SaSint,
            block_size as SaSint,
            4,
            &mut thread_state,
        );
        assert_eq!(threaded_sa, scalar_sa);
        assert_eq!(threaded_bucket, scalar_bucket);
    }

    #[test]
    fn libsais16_clear_lms_suffixes_omp_zeroes_requested_bucket_ranges() {
        let mut rust_sa = vec![5, 4, 3, 2, 1, 9];
        let mut c_sa = rust_sa.clone();
        let n = rust_sa.len() as SaSint;
        let mut bucket_start = vec![1, 4, 5];
        let mut bucket_end = vec![3, 5, 5];

        clear_lms_suffixes_omp(&mut rust_sa, n, 3, &bucket_start, &bucket_end, 2);
        unsafe {
            probe_libsais16_clear_lms_suffixes_omp(
                c_sa.as_mut_ptr(),
                n,
                3,
                bucket_start.as_mut_ptr(),
                bucket_end.as_mut_ptr(),
                2,
            );
        }

        assert_eq!(rust_sa, c_sa);
    }

    #[test]
    fn libsais16_partial_order_wrapper_helpers_match_manual_sequence() {
        let mut rust_sa = vec![1, 2, 3, 4];
        let mut c_sa = rust_sa.clone();
        flip_suffix_markers_omp(&mut rust_sa, 3, 2);
        unsafe {
            probe_libsais16_flip_suffix_markers_omp(c_sa.as_mut_ptr(), 3, 2);
        }
        assert_eq!(rust_sa, c_sa);

        let t = vec![0, 1, 2, 1, 0, 1, 2, 1, 0];
        let n = t.len() as SaSint;
        let k = 3;
        let mut wrapped_sa = vec![0; t.len()];
        let mut wrapped_buckets = vec![0; k as usize];
        let mut wrapped_state = alloc_thread_state(1).unwrap();
        induce_partial_order_32s_1k_omp(
            &t,
            &mut wrapped_sa,
            n,
            k,
            &mut wrapped_buckets,
            1,
            &mut wrapped_state,
        );

        let mut manual_sa = vec![0; t.len()];
        let mut manual_buckets = vec![0; k as usize];
        let mut manual_state = alloc_thread_state(1).unwrap();
        count_suffixes_32s(&t, n, k, &mut manual_buckets);
        initialize_buckets_start_32s_1k(k, &mut manual_buckets);
        partial_sorting_scan_left_to_right_32s_1k_omp(
            &t,
            &mut manual_sa,
            n,
            &mut manual_buckets,
            1,
            &mut manual_state,
        );
        count_suffixes_32s(&t, n, k, &mut manual_buckets);
        initialize_buckets_end_32s_1k(k, &mut manual_buckets);
        partial_sorting_scan_right_to_left_32s_1k_omp(
            &t,
            &mut manual_sa,
            n,
            &mut manual_buckets,
            1,
            &mut manual_state,
        );
        partial_sorting_gather_lms_suffixes_32s_1k_omp(&mut manual_sa, n, 1, &mut manual_state);

        assert_eq!(wrapped_sa, manual_sa);
        assert_eq!(wrapped_buckets, manual_buckets);
    }

    #[test]
    fn libsais16_induce_partial_order_32s_wrappers_match_c() {
        let t = make_main_32s_stress_text(128, 24);
        let n = t.len() as SaSint;
        let k = 24;
        let threads = 1;

        let mut rust_sa = vec![0; t.len()];
        let mut rust_buckets = vec![0; 6 * k as usize];
        let mut rust_state = alloc_thread_state(threads).unwrap();
        let m = count_and_gather_lms_suffixes_32s_4k_omp(
            &t,
            &mut rust_sa,
            n,
            k,
            &mut rust_buckets,
            1,
            threads,
            &mut rust_state,
        );
        assert!(m > 1);
        rust_sa[..(n - m) as usize].fill(0);
        let first_lms_suffix = rust_sa[(n - m) as usize];
        let left_suffixes_count = initialize_buckets_for_lms_suffixes_radix_sort_32s_6k(
            &t,
            k,
            &mut rust_buckets,
            first_lms_suffix,
        );
        let (_, induction_bucket) = rust_buckets.split_at_mut(4 * k as usize);
        radix_sort_lms_suffixes_32s_6k_omp(&t, &mut rust_sa, n, m, induction_bucket, threads);
        radix_sort_set_markers_32s_6k_omp(&mut rust_sa, k, induction_bucket, threads);
        initialize_buckets_for_partial_sorting_32s_6k(
            &t,
            k,
            &mut rust_buckets,
            first_lms_suffix,
            left_suffixes_count,
        );
        let mut c_sa = rust_sa.clone();
        let mut c_buckets = rust_buckets.clone();
        induce_partial_order_32s_6k_omp(
            &t,
            &mut rust_sa,
            n,
            k,
            &mut rust_buckets,
            first_lms_suffix,
            left_suffixes_count,
            threads,
            &mut rust_state,
        );
        unsafe {
            probe_libsais16_induce_partial_order_32s_6k_omp(
                t.as_ptr(),
                c_sa.as_mut_ptr(),
                n,
                k,
                c_buckets.as_mut_ptr(),
                first_lms_suffix,
                left_suffixes_count,
                threads,
            );
        }
        assert_eq!(rust_sa, c_sa);
        assert_eq!(rust_buckets, c_buckets);

        let mut rust_sa = vec![0; t.len()];
        let mut rust_buckets = vec![0; 4 * k as usize];
        let mut rust_state = alloc_thread_state(threads).unwrap();
        let m = count_and_gather_lms_suffixes_32s_2k_omp(
            &t,
            &mut rust_sa,
            n,
            k,
            &mut rust_buckets,
            1,
            threads,
            &mut rust_state,
        );
        assert!(m > 1);
        let first_lms_suffix = rust_sa[(n - m) as usize];
        initialize_buckets_for_radix_and_partial_sorting_32s_4k(
            &t,
            k,
            &mut rust_buckets,
            first_lms_suffix,
        );
        let (_, induction_bucket) = rust_buckets.split_at_mut(1);
        radix_sort_lms_suffixes_32s_2k_omp(&t, &mut rust_sa, n, m, induction_bucket, threads);
        radix_sort_set_markers_32s_4k_omp(&mut rust_sa, k, induction_bucket, threads);
        place_lms_suffixes_interval_32s_4k(&mut rust_sa, n, k, m - 1, &rust_buckets);
        let mut c_sa = rust_sa.clone();
        let mut c_buckets = rust_buckets.clone();
        induce_partial_order_32s_4k_omp(
            &t,
            &mut rust_sa,
            n,
            k,
            &mut rust_buckets,
            threads,
            &mut rust_state,
        );
        unsafe {
            probe_libsais16_induce_partial_order_32s_4k_omp(
                t.as_ptr(),
                c_sa.as_mut_ptr(),
                n,
                k,
                c_buckets.as_mut_ptr(),
                threads,
            );
        }
        assert_eq!(rust_sa, c_sa);
        assert_eq!(rust_buckets, c_buckets);

        let mut rust_sa = vec![0; t.len()];
        let mut rust_buckets = vec![0; 2 * k as usize];
        let mut rust_state = alloc_thread_state(threads).unwrap();
        let m = count_and_gather_lms_suffixes_32s_2k_omp(
            &t,
            &mut rust_sa,
            n,
            k,
            &mut rust_buckets,
            1,
            threads,
            &mut rust_state,
        );
        assert!(m > 1);
        let first_lms_suffix = rust_sa[(n - m) as usize];
        initialize_buckets_for_lms_suffixes_radix_sort_32s_2k(
            &t,
            k,
            &mut rust_buckets,
            first_lms_suffix,
        );
        let (_, induction_bucket) = rust_buckets.split_at_mut(1);
        radix_sort_lms_suffixes_32s_2k_omp(&t, &mut rust_sa, n, m, induction_bucket, threads);
        place_lms_suffixes_interval_32s_2k(&mut rust_sa, n, k, m - 1, &rust_buckets);
        initialize_buckets_start_and_end_32s_2k(k, &mut rust_buckets);
        let mut c_sa = rust_sa.clone();
        let mut c_buckets = rust_buckets.clone();
        induce_partial_order_32s_2k_omp(
            &t,
            &mut rust_sa,
            n,
            k,
            &mut rust_buckets,
            threads,
            &mut rust_state,
        );
        unsafe {
            probe_libsais16_induce_partial_order_32s_2k_omp(
                t.as_ptr(),
                c_sa.as_mut_ptr(),
                n,
                k,
                c_buckets.as_mut_ptr(),
                threads,
            );
        }
        assert_eq!(rust_sa, c_sa);
        assert_eq!(rust_buckets, c_buckets);

        let mut rust_sa = vec![0; t.len()];
        let mut rust_buckets = vec![0; k as usize];
        let mut rust_state = alloc_thread_state(threads).unwrap();
        count_suffixes_32s(&t, n, k, &mut rust_buckets);
        initialize_buckets_end_32s_1k(k, &mut rust_buckets);
        let m = radix_sort_lms_suffixes_32s_1k(&t, &mut rust_sa, n, &mut rust_buckets);
        assert!(m > 1);
        let mut c_sa = rust_sa.clone();
        let mut c_buckets = rust_buckets.clone();
        induce_partial_order_32s_1k_omp(
            &t,
            &mut rust_sa,
            n,
            k,
            &mut rust_buckets,
            threads,
            &mut rust_state,
        );
        unsafe {
            probe_libsais16_induce_partial_order_32s_1k_omp(
                t.as_ptr(),
                c_sa.as_mut_ptr(),
                n,
                k,
                c_buckets.as_mut_ptr(),
                threads,
            );
        }
        assert_eq!(rust_sa, c_sa);
        assert_eq!(rust_buckets, c_buckets);
    }

    #[test]
    fn libsais16_induce_partial_order_16u_omp_matches_c() {
        let text = [3, 1, 2, 1, 0, 4, 1, 0];
        let n = text.len() as SaSint;
        let flags = 0;
        let threads = 1;
        let mut rust_sa = vec![0; text.len()];
        let mut rust_buckets = vec![0; 8 * ALPHABET_SIZE];

        let m = count_and_gather_lms_suffixes_16u_omp(
            &text,
            &mut rust_sa,
            n,
            &mut rust_buckets[..4 * ALPHABET_SIZE],
            threads,
            &mut [],
        );
        let k = initialize_buckets_start_and_end_16u(&mut rust_buckets, None);
        assert!(m > 0);
        let first_lms_suffix = rust_sa[(n - m) as usize];
        let left_suffixes_count = initialize_buckets_for_lms_suffixes_radix_sort_16u(
            &text,
            &mut rust_buckets,
            first_lms_suffix,
        );
        radix_sort_lms_suffixes_16u_omp(
            &text,
            &mut rust_sa,
            n,
            m,
            flags,
            &mut rust_buckets,
            threads,
            &mut [],
        );
        initialize_buckets_for_partial_sorting_16u(
            &text,
            &mut rust_buckets,
            first_lms_suffix,
            left_suffixes_count,
        );

        let mut c_sa = rust_sa.clone();
        let mut c_buckets = rust_buckets.clone();
        induce_partial_order_16u_omp(
            &text,
            &mut rust_sa,
            n,
            k,
            flags,
            &mut rust_buckets,
            first_lms_suffix,
            left_suffixes_count,
            threads,
        );
        unsafe {
            probe_libsais16_induce_partial_order_16u_omp(
                text.as_ptr(),
                c_sa.as_mut_ptr(),
                n,
                k,
                flags,
                c_buckets.as_mut_ptr(),
                first_lms_suffix,
                left_suffixes_count,
                threads,
            );
        }

        assert_eq!(rust_sa, c_sa);
        assert_eq!(rust_buckets, c_buckets);
    }

    fn final_order_32s_fixture() -> (Vec<SaSint>, Vec<SaSint>) {
        (
            vec![0, 1, 2, 1, 0, 1, 2, 1, 0],
            vec![1, 0, 2, 0, 0, 0, 0, 0, 0],
        )
    }

    fn seed_final_order_bucket_sections(buckets: &mut [SaSint], k: usize, branch_k: usize) {
        let left = [0, 1, 3];
        let right = [1, 2, 3];
        let left_section = match branch_k {
            6 => 4 * k,
            4 => 2 * k,
            2 => k,
            _ => 0,
        };
        let right_section = match branch_k {
            6 => 5 * k,
            4 => 3 * k,
            2 => 0,
            _ => 0,
        };
        buckets[left_section..left_section + k].copy_from_slice(&left);
        buckets[right_section..right_section + k].copy_from_slice(&right);
    }

    #[test]
    fn libsais16_induce_final_order_32s_wrappers_match_c() {
        let (t, sa) = final_order_32s_fixture();
        let n = t.len() as SaSint;
        let k = 3;
        let threads = 1;

        let mut rust_sa = sa.clone();
        let mut rust_buckets = vec![0; 6 * k as usize];
        seed_final_order_bucket_sections(&mut rust_buckets, k as usize, 6);
        let mut c_sa = rust_sa.clone();
        let mut c_buckets = rust_buckets.clone();
        let mut rust_state = alloc_thread_state(threads).unwrap();
        induce_final_order_32s_6k(
            &t,
            &mut rust_sa,
            n,
            k,
            &mut rust_buckets,
            threads,
            &mut rust_state,
        );
        unsafe {
            probe_libsais16_induce_final_order_32s_6k(
                t.as_ptr(),
                c_sa.as_mut_ptr(),
                n,
                k,
                c_buckets.as_mut_ptr(),
                threads,
            );
        }
        assert_eq!(rust_sa, c_sa);
        assert_eq!(rust_buckets, c_buckets);

        let mut rust_sa = sa.clone();
        let mut rust_buckets = vec![0; 4 * k as usize];
        seed_final_order_bucket_sections(&mut rust_buckets, k as usize, 4);
        let mut c_sa = rust_sa.clone();
        let mut c_buckets = rust_buckets.clone();
        let mut rust_state = alloc_thread_state(threads).unwrap();
        induce_final_order_32s_4k(
            &t,
            &mut rust_sa,
            n,
            k,
            &mut rust_buckets,
            threads,
            &mut rust_state,
        );
        unsafe {
            probe_libsais16_induce_final_order_32s_4k(
                t.as_ptr(),
                c_sa.as_mut_ptr(),
                n,
                k,
                c_buckets.as_mut_ptr(),
                threads,
            );
        }
        assert_eq!(rust_sa, c_sa);
        assert_eq!(rust_buckets, c_buckets);

        let mut rust_sa = sa.clone();
        let mut rust_buckets = vec![0; 2 * k as usize];
        seed_final_order_bucket_sections(&mut rust_buckets, k as usize, 2);
        let mut c_sa = rust_sa.clone();
        let mut c_buckets = rust_buckets.clone();
        let mut rust_state = alloc_thread_state(threads).unwrap();
        induce_final_order_32s_2k(
            &t,
            &mut rust_sa,
            n,
            k,
            &mut rust_buckets,
            threads,
            &mut rust_state,
        );
        unsafe {
            probe_libsais16_induce_final_order_32s_2k(
                t.as_ptr(),
                c_sa.as_mut_ptr(),
                n,
                k,
                c_buckets.as_mut_ptr(),
                threads,
            );
        }
        assert_eq!(rust_sa, c_sa);
        assert_eq!(rust_buckets, c_buckets);

        let mut rust_sa = sa;
        let mut rust_buckets = vec![0; k as usize];
        let mut c_sa = rust_sa.clone();
        let mut c_buckets = rust_buckets.clone();
        let mut rust_state = alloc_thread_state(threads).unwrap();
        induce_final_order_32s_1k(
            &t,
            &mut rust_sa,
            n,
            k,
            &mut rust_buckets,
            threads,
            &mut rust_state,
        );
        unsafe {
            probe_libsais16_induce_final_order_32s_1k(
                t.as_ptr(),
                c_sa.as_mut_ptr(),
                n,
                k,
                c_buckets.as_mut_ptr(),
                threads,
            );
        }
        assert_eq!(rust_sa, c_sa);
        assert_eq!(rust_buckets, c_buckets);
    }

    #[test]
    fn libsais16_induce_final_order_16u_omp_matches_manual_sequence() {
        let (text, mut wrapped_sa, induction_bucket) = final_scan_fixture();
        let mut wrapped_buckets = final_order_buckets(&induction_bucket);
        let mut c_sa = wrapped_sa.clone();
        let mut c_buckets = wrapped_buckets.clone();
        let mut wrapped_state = alloc_thread_state(1).unwrap();
        let wrapped_index = induce_final_order_16u_omp(
            &text,
            &mut wrapped_sa,
            text.len() as SaSint,
            8,
            0,
            0,
            None,
            &mut wrapped_buckets,
            1,
            &mut wrapped_state,
        );
        let c_index = unsafe {
            probe_libsais16_induce_final_order_16u_omp(
                text.as_ptr(),
                c_sa.as_mut_ptr(),
                text.len() as SaSint,
                8,
                0,
                0,
                std::ptr::null_mut(),
                c_buckets.as_mut_ptr(),
                1,
            )
        };

        let (text, mut manual_sa, induction_bucket) = final_scan_fixture();
        let mut manual_buckets = final_order_buckets(&induction_bucket);
        {
            let (left_buckets, right_tail) = manual_buckets.split_at_mut(7 * ALPHABET_SIZE);
            final_sorting_scan_left_to_right_16u_omp(
                &text,
                &mut manual_sa,
                text.len() as SaSint,
                8,
                &mut left_buckets[6 * ALPHABET_SIZE..7 * ALPHABET_SIZE],
                1,
            );
            final_sorting_scan_right_to_left_16u_omp(
                &text,
                &mut manual_sa,
                0,
                text.len() as SaSint,
                8,
                &mut right_tail[..ALPHABET_SIZE],
                1,
            );
        }

        assert_eq!(wrapped_index, 0);
        assert_eq!(wrapped_index, c_index);
        assert_eq!(wrapped_sa, manual_sa);
        assert_eq!(wrapped_sa, c_sa);
        assert_eq!(wrapped_buckets, manual_buckets);
        assert_eq!(wrapped_buckets, c_buckets);

        let (text, mut wrapped_sa, induction_bucket) = final_scan_fixture();
        let mut wrapped_buckets = final_order_buckets(&induction_bucket);
        let mut c_sa = wrapped_sa.clone();
        let mut c_buckets = wrapped_buckets.clone();
        let mut wrapped_state = alloc_thread_state(1).unwrap();
        let wrapped_index = induce_final_order_16u_omp(
            &text,
            &mut wrapped_sa,
            text.len() as SaSint,
            8,
            LIBSAIS_FLAGS_BWT,
            0,
            None,
            &mut wrapped_buckets,
            1,
            &mut wrapped_state,
        );
        let c_index = unsafe {
            probe_libsais16_induce_final_order_16u_omp(
                text.as_ptr(),
                c_sa.as_mut_ptr(),
                text.len() as SaSint,
                8,
                LIBSAIS_FLAGS_BWT,
                0,
                std::ptr::null_mut(),
                c_buckets.as_mut_ptr(),
                1,
            )
        };

        let (text, mut manual_sa, induction_bucket) = final_scan_fixture();
        let mut manual_buckets = final_order_buckets(&induction_bucket);
        let manual_index = {
            let (left_buckets, right_tail) = manual_buckets.split_at_mut(7 * ALPHABET_SIZE);
            final_bwt_scan_left_to_right_16u_omp(
                &text,
                &mut manual_sa,
                text.len() as SaSint,
                8,
                &mut left_buckets[6 * ALPHABET_SIZE..7 * ALPHABET_SIZE],
                1,
            );
            final_bwt_scan_right_to_left_16u_omp(
                &text,
                &mut manual_sa,
                text.len() as SaSint,
                8,
                &mut right_tail[..ALPHABET_SIZE],
                1,
            )
        };

        assert_eq!(wrapped_index, manual_index);
        assert_eq!(wrapped_index, c_index);
        assert_eq!(wrapped_sa, manual_sa);
        assert_eq!(wrapped_sa, c_sa);
        assert_eq!(wrapped_buckets, manual_buckets);
        assert_eq!(wrapped_buckets, c_buckets);

        let (text, mut wrapped_sa, induction_bucket) = final_scan_fixture();
        let mut wrapped_buckets = final_order_buckets(&induction_bucket);
        let mut c_sa = wrapped_sa.clone();
        let mut c_buckets = wrapped_buckets.clone();
        let mut wrapped_state = alloc_thread_state(1).unwrap();
        let mut wrapped_i = vec![-1; 8];
        let mut c_i = wrapped_i.clone();
        let wrapped_index = induce_final_order_16u_omp(
            &text,
            &mut wrapped_sa,
            text.len() as SaSint,
            8,
            LIBSAIS_FLAGS_BWT,
            2,
            Some(&mut wrapped_i),
            &mut wrapped_buckets,
            1,
            &mut wrapped_state,
        );
        let c_index = unsafe {
            probe_libsais16_induce_final_order_16u_omp(
                text.as_ptr(),
                c_sa.as_mut_ptr(),
                text.len() as SaSint,
                8,
                LIBSAIS_FLAGS_BWT,
                2,
                c_i.as_mut_ptr(),
                c_buckets.as_mut_ptr(),
                1,
            )
        };

        let (text, mut manual_sa, induction_bucket) = final_scan_fixture();
        let mut manual_buckets = final_order_buckets(&induction_bucket);
        let mut manual_i = vec![-1; 8];
        {
            let (left_buckets, right_tail) = manual_buckets.split_at_mut(7 * ALPHABET_SIZE);
            final_bwt_aux_scan_left_to_right_16u_omp(
                &text,
                &mut manual_sa,
                text.len() as SaSint,
                8,
                1,
                &mut manual_i,
                &mut left_buckets[6 * ALPHABET_SIZE..7 * ALPHABET_SIZE],
                1,
            );
            final_bwt_aux_scan_right_to_left_16u_omp(
                &text,
                &mut manual_sa,
                text.len() as SaSint,
                8,
                1,
                &mut manual_i,
                &mut right_tail[..ALPHABET_SIZE],
                1,
            );
        }

        assert_eq!(wrapped_index, 0);
        assert_eq!(wrapped_index, c_index);
        assert_eq!(wrapped_sa, manual_sa);
        assert_eq!(wrapped_sa, c_sa);
        assert_eq!(wrapped_buckets, manual_buckets);
        assert_eq!(wrapped_buckets, c_buckets);
        assert_eq!(wrapped_i, manual_i);
        assert_eq!(wrapped_i, c_i);
    }

    #[test]
    fn libsais16_main_16u_matches_public_c_suffix_array_paths() {
        let text = [3, 1, 4, 1, 5, 9, 0, 2];
        let n = text.len() as SaSint;
        let fs = 32;
        let mut rust_sa = vec![0; text.len() + fs as usize];
        let mut rust_buckets = vec![0; 8 * ALPHABET_SIZE];
        let mut rust_freq = vec![0; ALPHABET_SIZE];
        let mut rust_state = alloc_thread_state(1).unwrap();
        let rust_index = main_16u(
            &text,
            &mut rust_sa,
            n,
            &mut rust_buckets,
            0,
            0,
            None,
            fs,
            Some(&mut rust_freq),
            1,
            &mut rust_state,
        );

        let mut c_sa = vec![0; text.len() + fs as usize];
        let mut c_freq = vec![0; ALPHABET_SIZE];
        let c_index = unsafe {
            probe_public_libsais16_freq(
                text.as_ptr(),
                c_sa.as_mut_ptr(),
                n,
                fs,
                c_freq.as_mut_ptr(),
            )
        };

        assert_eq!(rust_index, c_index);
        assert_eq!(&rust_sa[..text.len()], &c_sa[..text.len()]);
        assert_eq!(rust_freq, c_freq);

        let text = [2, 1, 0, 2, 0];
        let n = text.len() as SaSint;
        let fs = 24;
        let mut rust_sa = vec![0; text.len() + fs as usize];
        let mut rust_buckets = vec![0; 8 * ALPHABET_SIZE];
        let mut rust_freq = vec![0; ALPHABET_SIZE];
        let mut rust_state = alloc_thread_state(1).unwrap();
        let rust_index = main_16u(
            &text,
            &mut rust_sa,
            n,
            &mut rust_buckets,
            LIBSAIS_FLAGS_GSA,
            0,
            None,
            fs,
            Some(&mut rust_freq),
            1,
            &mut rust_state,
        );

        let mut c_sa = vec![0; text.len() + fs as usize];
        let mut c_freq = vec![0; ALPHABET_SIZE];
        let c_index = unsafe {
            probe_public_libsais16_gsa_freq(
                text.as_ptr(),
                c_sa.as_mut_ptr(),
                n,
                fs,
                c_freq.as_mut_ptr(),
            )
        };

        assert_eq!(rust_index, c_index);
        assert_eq!(&rust_sa[..text.len()], &c_sa[..text.len()]);
        assert_eq!(rust_freq, c_freq);
    }

    fn make_main_32s_stress_text(len: usize, alphabet: SaSint) -> Vec<SaSint> {
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

    fn make_recursive_main_32s_text(repeats: usize) -> Vec<SaSint> {
        let motif = [9, 4, 9, 2, 9, 4, 9, 1];
        let mut t = Vec::with_capacity(repeats * motif.len() + 1);
        for _ in 0..repeats {
            t.extend_from_slice(&motif);
        }
        t.push(0);
        t
    }

    fn assert_main_32s_entry_matches_c(mut t: Vec<SaSint>, k: SaSint, fs: SaSint) {
        let n = t.len() as SaSint;
        let threads = 1;
        let mut sa = vec![0; t.len() + fs as usize];
        let initial_t = t.clone();
        let initial_sa = sa.clone();

        let c_result = unsafe {
            probe_libsais16_main_32s_entry(t.as_mut_ptr(), sa.as_mut_ptr(), n, k, fs, threads)
        };
        let c_t = t.clone();
        let c_sa = sa.clone();

        t.copy_from_slice(&initial_t);
        sa.copy_from_slice(&initial_sa);

        let mut thread_state = alloc_thread_state(threads).unwrap();
        let rust_result = main_32s_entry(
            t.as_mut_ptr(),
            &mut sa,
            n,
            k,
            fs,
            threads,
            &mut thread_state,
        );

        assert_eq!(rust_result, c_result);
        assert_eq!(t, c_t);
        assert_eq!(sa, c_sa);
    }

    #[test]
    fn libsais16_main_32s_entry_matches_c_for_local_32s_paths() {
        assert_main_32s_entry_matches_c(make_main_32s_stress_text(1024, 300), 300, 2048);
        assert_main_32s_entry_matches_c(make_main_32s_stress_text(1024, 400), 400, 2048);
        assert_main_32s_entry_matches_c(make_main_32s_stress_text(1024, 700), 700, 2048);
        assert_main_32s_entry_matches_c(make_main_32s_stress_text(1024, 1501), 1501, 2048);
        assert_main_32s_entry_matches_c(make_recursive_main_32s_text(24), 300, 0);
        assert_main_32s_entry_matches_c(make_recursive_main_32s_text(24), 1501, 0);
    }

    #[test]
    fn libsais16_final_bwt_scan_left_to_right_16u_matches_c() {
        let (text, mut rust_sa, mut rust_bucket) = final_scan_fixture();
        let mut c_sa = rust_sa.clone();
        let mut c_bucket = rust_bucket.clone();

        final_bwt_scan_left_to_right_16u(&text, &mut rust_sa, &mut rust_bucket, 0, 6);
        unsafe {
            probe_libsais16_final_bwt_scan_left_to_right_16u(
                text.as_ptr(),
                c_sa.as_mut_ptr(),
                c_bucket.as_mut_ptr(),
                0,
                6,
            );
        }

        assert_eq!(rust_sa, c_sa);
        assert_eq!(rust_bucket, c_bucket);
    }

    #[test]
    fn libsais16_final_bwt_scan_right_to_left_16u_matches_c() {
        let (text, mut rust_sa, mut rust_bucket) = final_scan_fixture();
        let mut c_sa = rust_sa.clone();
        let mut c_bucket = rust_bucket.clone();

        let rust_index =
            final_bwt_scan_right_to_left_16u(&text, &mut rust_sa, &mut rust_bucket, 0, 6);
        let c_index = unsafe {
            probe_libsais16_final_bwt_scan_right_to_left_16u(
                text.as_ptr(),
                c_sa.as_mut_ptr(),
                c_bucket.as_mut_ptr(),
                0,
                6,
            )
        };

        assert_eq!(rust_index, c_index);
        assert_eq!(rust_sa, c_sa);
        assert_eq!(rust_bucket, c_bucket);
    }

    #[test]
    fn libsais16_final_bwt_aux_scan_left_to_right_16u_matches_c() {
        let (text, mut rust_sa, mut rust_bucket) = final_scan_fixture();
        let mut c_sa = rust_sa.clone();
        let mut c_bucket = rust_bucket.clone();
        let mut rust_i = vec![-1; 8];
        let mut c_i = rust_i.clone();

        final_bwt_aux_scan_left_to_right_16u(
            &text,
            &mut rust_sa,
            1,
            &mut rust_i,
            &mut rust_bucket,
            0,
            6,
        );
        unsafe {
            probe_libsais16_final_bwt_aux_scan_left_to_right_16u(
                text.as_ptr(),
                c_sa.as_mut_ptr(),
                1,
                c_i.as_mut_ptr(),
                c_bucket.as_mut_ptr(),
                0,
                6,
            );
        }

        assert_eq!(rust_sa, c_sa);
        assert_eq!(rust_bucket, c_bucket);
        assert_eq!(rust_i, c_i);
    }

    #[test]
    fn libsais16_final_bwt_aux_scan_right_to_left_16u_matches_c() {
        let (text, mut rust_sa, mut rust_bucket) = final_scan_fixture();
        let mut c_sa = rust_sa.clone();
        let mut c_bucket = rust_bucket.clone();
        let mut rust_i = vec![-1; 8];
        let mut c_i = rust_i.clone();

        final_bwt_aux_scan_right_to_left_16u(
            &text,
            &mut rust_sa,
            1,
            &mut rust_i,
            &mut rust_bucket,
            0,
            6,
        );
        unsafe {
            probe_libsais16_final_bwt_aux_scan_right_to_left_16u(
                text.as_ptr(),
                c_sa.as_mut_ptr(),
                1,
                c_i.as_mut_ptr(),
                c_bucket.as_mut_ptr(),
                0,
                6,
            );
        }

        assert_eq!(rust_sa, c_sa);
        assert_eq!(rust_bucket, c_bucket);
        assert_eq!(rust_i, c_i);
    }

    #[test]
    fn libsais16_renumber_lms_suffixes_16u_matches_c() {
        let m = 6;
        let mut rust_sa = vec![0; 20];
        rust_sa[..m].copy_from_slice(&[2, 4 | SAINT_MIN, 6, 8 | SAINT_MIN, 10, 12 | SAINT_MIN]);
        let mut c_sa = rust_sa.clone();

        let rust_name = renumber_lms_suffixes_16u(&mut rust_sa, m as SaSint, 5, 0, m as SaSint);
        let c_name = unsafe {
            probe_libsais16_renumber_lms_suffixes_16u(
                c_sa.as_mut_ptr(),
                m as SaSint,
                5,
                0,
                m as SaSint,
            )
        };

        assert_eq!(rust_name, c_name);
        assert_eq!(rust_sa, c_sa);
    }

    fn lms_interval_fixture() -> (Vec<SaSint>, Vec<SaSint>) {
        let mut sa = vec![-7; 16];
        sa[4..8].copy_from_slice(&[41, 42, 61, 62]);

        let mut buckets = vec![0; 8 * ALPHABET_SIZE];
        buckets[buckets_index2(2, 1)] = 0;
        buckets[buckets_index2(3, 1)] = 2;
        buckets[buckets_index2(4, 1)] = 2;
        buckets[buckets_index2(5, 1)] = 2;
        buckets[buckets_index2(6, 1)] = 4;
        buckets[buckets_index2(7, 1)] = 4;
        buckets[7 * ALPHABET_SIZE + 2] = 6;
        buckets[7 * ALPHABET_SIZE + 5] = 12;

        (sa, buckets)
    }

    #[test]
    fn libsais16_place_lms_suffixes_interval_16u_matches_c() {
        for flags in [0, LIBSAIS_FLAGS_GSA] {
            let (mut rust_sa, mut rust_buckets) = lms_interval_fixture();
            let mut c_sa = rust_sa.clone();
            let mut c_buckets = rust_buckets.clone();

            place_lms_suffixes_interval_16u(&mut rust_sa, 16, 8, flags, &mut rust_buckets);
            unsafe {
                probe_libsais16_place_lms_suffixes_interval_16u(
                    c_sa.as_mut_ptr(),
                    16,
                    8,
                    flags,
                    c_buckets.as_mut_ptr(),
                );
            }

            assert_eq!(rust_sa, c_sa);
            assert_eq!(rust_buckets, c_buckets);
        }
    }

    #[test]
    fn libsais16_bwt_copy_16u_matches_c() {
        let mut a = vec![0, 1, 65535, 65536, -1, -2, 70000, 17, 131071, -65536];
        let mut rust_u = vec![999; a.len()];
        let mut c_u = rust_u.clone();

        bwt_copy_16u(&mut rust_u, &a, a.len() as SaSint);
        unsafe {
            probe_libsais16_bwt_copy_16u(c_u.as_mut_ptr(), a.as_mut_ptr(), a.len() as SaSint);
        }

        assert_eq!(rust_u, c_u);
    }

    #[test]
    fn libsais16_early_omp_wrappers_match_c() {
        let text = [3, 1, 2, 1, 0, 4, 1, 0];
        let n = text.len() as SaSint;

        let mut rust_sa = vec![-99; text.len()];
        let mut c_sa = rust_sa.clone();
        gather_lms_suffixes_16u_omp(&text, &mut rust_sa, n, 1, &mut []);
        unsafe {
            probe_libsais16_gather_lms_suffixes_16u_omp(text.as_ptr(), c_sa.as_mut_ptr(), n, 1);
        }
        assert_eq!(rust_sa, c_sa);

        let mut rust_sa = vec![-99; text.len()];
        let mut c_sa = rust_sa.clone();
        let mut rust_buckets = vec![-1; 4 * ALPHABET_SIZE];
        let mut c_buckets = rust_buckets.clone();
        let rust_m = count_and_gather_lms_suffixes_16u_omp(
            &text,
            &mut rust_sa,
            n,
            &mut rust_buckets,
            1,
            &mut [],
        );
        let c_m = unsafe {
            probe_libsais16_count_and_gather_lms_suffixes_16u_omp(
                text.as_ptr(),
                c_sa.as_mut_ptr(),
                n,
                c_buckets.as_mut_ptr(),
                1,
            )
        };
        assert_eq!(rust_m, c_m);
        assert_eq!(rust_sa, c_sa);
        assert_eq!(rust_buckets, c_buckets);

        let mut rust_buckets = vec![0; 8 * ALPHABET_SIZE];
        let m = count_and_gather_lms_suffixes_16u(
            &text,
            &mut rust_sa,
            n,
            &mut rust_buckets[..4 * ALPHABET_SIZE],
            0,
            n,
        );
        initialize_buckets_start_and_end_16u(&mut rust_buckets, None);
        let first_lms_suffix = rust_sa[(n - m) as usize];
        initialize_buckets_for_lms_suffixes_radix_sort_16u(
            &text,
            &mut rust_buckets,
            first_lms_suffix,
        );
        let mut c_sa = rust_sa.clone();
        let mut c_buckets = rust_buckets.clone();
        radix_sort_lms_suffixes_16u_omp(
            &text,
            &mut rust_sa,
            n,
            m,
            0,
            &mut rust_buckets,
            1,
            &mut [],
        );
        unsafe {
            probe_libsais16_radix_sort_lms_suffixes_16u_omp(
                text.as_ptr(),
                c_sa.as_mut_ptr(),
                n,
                m,
                0,
                c_buckets.as_mut_ptr(),
                1,
            );
        }
        assert_eq!(rust_sa, c_sa);
        assert_eq!(rust_buckets, c_buckets);
    }

    #[test]
    fn libsais16_early_omp_wrappers_use_block_partition_for_large_inputs() {
        let n = 65_600usize;
        let text: Vec<u16> = (0..n)
            .map(|i| 1 + ((i * 37 + i / 17) % 509) as u16)
            .collect();

        let mut gathered_threaded = vec![-99; n];
        let mut gathered_scalar = vec![-99; n];
        let mut thread_state = alloc_thread_state(4).unwrap();
        let mut count_sa = vec![-99; n];
        let mut count_buckets = vec![0; 4 * ALPHABET_SIZE];
        count_and_gather_lms_suffixes_16u_omp(
            &text,
            &mut count_sa,
            n as SaSint,
            &mut count_buckets,
            4,
            &mut thread_state,
        );
        gather_lms_suffixes_16u_omp(
            &text,
            &mut gathered_threaded,
            n as SaSint,
            4,
            &mut thread_state,
        );
        gather_lms_suffixes_16u(
            &text,
            &mut gathered_scalar,
            n as SaSint,
            n as SaSint - 1,
            0,
            n as SaSint,
        );
        assert_eq!(gathered_threaded, gathered_scalar);

        let mut sa_threaded = vec![-99; n];
        let mut sa_scalar = vec![-99; n];
        let mut buckets_threaded = vec![0; 4 * ALPHABET_SIZE];
        let mut buckets_scalar = vec![0; 4 * ALPHABET_SIZE];
        let m_threaded = count_and_gather_lms_suffixes_16u_omp(
            &text,
            &mut sa_threaded,
            n as SaSint,
            &mut buckets_threaded,
            4,
            &mut thread_state,
        );
        let m_scalar = count_and_gather_lms_suffixes_16u(
            &text,
            &mut sa_scalar,
            n as SaSint,
            &mut buckets_scalar,
            0,
            n as SaSint,
        );
        assert_eq!(m_threaded, m_scalar);
        assert_eq!(
            &sa_threaded[n - m_threaded as usize..],
            &sa_scalar[n - m_scalar as usize..]
        );
        assert_eq!(buckets_threaded, buckets_scalar);
    }

    #[test]
    fn libsais16_late_omp_wrappers_match_c() {
        let m = 6;
        let mut rust_sa = vec![0; 20];
        rust_sa[..m].copy_from_slice(&[2, 4 | SAINT_MIN, 6, 8 | SAINT_MIN, 10, 12 | SAINT_MIN]);
        let mut c_sa = rust_sa.clone();
        let mut rust_thread_state = alloc_thread_state(1).unwrap();
        let rust_name =
            renumber_lms_suffixes_16u_omp(&mut rust_sa, m as SaSint, 1, &mut rust_thread_state);
        let c_name = unsafe {
            probe_libsais16_renumber_lms_suffixes_16u_omp(c_sa.as_mut_ptr(), m as SaSint, 1)
        };
        assert_eq!(rust_name, c_name);
        assert_eq!(rust_sa, c_sa);

        let mut a = vec![0, 1, 65535, 65536, -1, -2, 70000, 17, 131071, -65536];
        let mut rust_u = vec![999; a.len()];
        let mut c_u = rust_u.clone();
        bwt_copy_16u_omp(&mut rust_u, &a, a.len() as SaSint, 1);
        unsafe {
            probe_libsais16_bwt_copy_16u_omp(
                c_u.as_mut_ptr(),
                a.as_mut_ptr(),
                a.len() as SaSint,
                1,
            );
        }
        assert_eq!(rust_u, c_u);
    }

    #[test]
    fn libsais16_gather_marked_lms_suffixes_matches_c() {
        let mut rust_sa = vec![0, 0, 3 | SAINT_MIN, 4, 5 | SAINT_MIN, 6, -7, 8];
        let mut c_sa = rust_sa.clone();

        let rust_l = gather_marked_lms_suffixes(&mut rust_sa, 2, 8, 0, 4) as SaSint;
        let c_l =
            unsafe { probe_libsais16_gather_marked_lms_suffixes(c_sa.as_mut_ptr(), 2, 8, 0, 4) };

        assert_eq!(rust_l, c_l);
        assert_eq!(rust_sa, c_sa);
    }

    #[test]
    fn libsais16_gather_marked_lms_suffixes_omp_matches_c() {
        let mut rust_sa = vec![0; 10];
        rust_sa[4..8].copy_from_slice(&[2 | SAINT_MIN, 4, 6 | SAINT_MIN, 8]);
        let mut c_sa = rust_sa.clone();

        let mut rust_thread_state = alloc_thread_state(1).unwrap();
        gather_marked_lms_suffixes_omp(&mut rust_sa, 8, 4, 2, 1, &mut rust_thread_state);
        unsafe {
            probe_libsais16_gather_marked_lms_suffixes_omp(c_sa.as_mut_ptr(), 8, 4, 2, 1);
        }

        assert_eq!(rust_sa, c_sa);
    }

    #[test]
    fn libsais16_renumber_and_gather_lms_suffixes_omp_matches_c() {
        let mut rust_sa = vec![0; 10];
        rust_sa[..4].copy_from_slice(&[2, 4 | SAINT_MIN, 6, 8 | SAINT_MIN]);
        let mut c_sa = rust_sa.clone();

        let mut rust_thread_state = alloc_thread_state(1).unwrap();
        let rust_name =
            renumber_and_gather_lms_suffixes_omp(&mut rust_sa, 8, 4, 2, 1, &mut rust_thread_state);
        let c_name = unsafe {
            probe_libsais16_renumber_and_gather_lms_suffixes_omp(c_sa.as_mut_ptr(), 8, 4, 2, 1)
        };

        assert_eq!(rust_name, c_name);
        assert_eq!(rust_sa, c_sa);
    }

    #[test]
    fn libsais16_reconstruct_lms_suffixes_matches_c() {
        let mut rust_sa = vec![2, 0, 1, 77, 88, 10, 11, 12];
        let mut c_sa = rust_sa.clone();

        reconstruct_lms_suffixes(&mut rust_sa, 8, 3, 0, 3);
        unsafe {
            probe_libsais16_reconstruct_lms_suffixes(c_sa.as_mut_ptr(), 8, 3, 0, 3);
        }

        assert_eq!(rust_sa, c_sa);

        let mut rust_sa = vec![2, 0, 1, 77, 88, 10, 11, 12];
        let mut c_sa = rust_sa.clone();
        reconstruct_lms_suffixes_omp(&mut rust_sa, 8, 3, 1);
        unsafe {
            probe_libsais16_reconstruct_lms_suffixes_omp(c_sa.as_mut_ptr(), 8, 3, 1);
        }

        assert_eq!(rust_sa, c_sa);
    }

    #[test]
    fn libsais16_lms_late_omp_wrappers_use_block_partition() {
        let m = 65_536usize;
        let mut scalar = vec![0; 2 * m + 8];
        for i in 0..m {
            let value = (2 * i) as SaSint;
            scalar[i] = if i % 7 == 0 { value | SAINT_MIN } else { value };
        }
        let mut threaded = scalar.clone();

        let mut scalar_state = alloc_thread_state(1).unwrap();
        let mut threaded_state = alloc_thread_state(4).unwrap();
        let scalar_name =
            renumber_lms_suffixes_16u_omp(&mut scalar, m as SaSint, 1, &mut scalar_state);
        let threaded_name =
            renumber_lms_suffixes_16u_omp(&mut threaded, m as SaSint, 4, &mut threaded_state);
        assert_eq!(threaded_name, scalar_name);
        assert_eq!(threaded, scalar);

        let n = 131_072usize;
        let m = 65_536usize;
        let fs = 128usize;
        let mut scalar = vec![0; n + fs];
        for i in 0..(n >> 1) {
            let value = (i as SaSint + 1) & SAINT_MAX;
            scalar[m + i] = if i % 7 == 0 { value | SAINT_MIN } else { value };
        }
        let marked_count = (0..(n >> 1)).filter(|i| i % 7 == 0).count();
        let mut threaded = scalar.clone();

        let mut scalar_state = alloc_thread_state(1).unwrap();
        let mut threaded_state = alloc_thread_state(4).unwrap();
        gather_marked_lms_suffixes_omp(
            &mut scalar,
            n as SaSint,
            m as SaSint,
            fs as SaSint,
            1,
            &mut scalar_state,
        );
        gather_marked_lms_suffixes_omp(
            &mut threaded,
            n as SaSint,
            m as SaSint,
            fs as SaSint,
            4,
            &mut threaded_state,
        );
        assert_eq!(
            &threaded[n + fs - marked_count..n + fs],
            &scalar[n + fs - marked_count..n + fs]
        );

        let m = 65_536usize;
        let n = 2 * m;
        let mut scalar = vec![0; n];
        for i in 0..m {
            scalar[i] = i as SaSint;
            scalar[n - m + i] = 1_000_000 + i as SaSint;
        }
        let mut threaded = scalar.clone();

        reconstruct_lms_suffixes_omp(&mut scalar, n as SaSint, m as SaSint, 1);
        reconstruct_lms_suffixes_omp(&mut threaded, n as SaSint, m as SaSint, 4);
        assert_eq!(threaded, scalar);
    }

    #[test]
    fn libsais16_distinct_lms_helpers_match_c() {
        let m = 6;
        let mut rust_sa = vec![0; 18];
        rust_sa[..m].copy_from_slice(&[
            2 | SAINT_MIN,
            4 | SAINT_MIN,
            6,
            8 | SAINT_MIN,
            10,
            12 | SAINT_MIN,
        ]);
        let mut c_sa = rust_sa.clone();
        let rust_name =
            renumber_distinct_lms_suffixes_32s_4k(&mut rust_sa, m as SaSint, 1, 0, m as isize);
        let c_name = unsafe {
            probe_libsais16_renumber_distinct_lms_suffixes_32s_4k(
                c_sa.as_mut_ptr(),
                m as SaSint,
                1,
                0,
                m as SaSint,
            )
        };
        assert_eq!(rust_name, c_name);
        assert_eq!(rust_sa, c_sa);

        let mut rust_sa = vec![0; 12];
        rust_sa[m..m + 6].copy_from_slice(&[SAINT_MIN | 1, 0, SAINT_MIN | 2, 0, 3, 0]);
        let mut c_sa = rust_sa.clone();
        mark_distinct_lms_suffixes_32s(&mut rust_sa, m as SaSint, 0, 6);
        unsafe {
            probe_libsais16_mark_distinct_lms_suffixes_32s(c_sa.as_mut_ptr(), m as SaSint, 0, 6);
        }
        assert_eq!(rust_sa, c_sa);

        let mut rust_sa = vec![0; 12];
        rust_sa[m..m + 6].copy_from_slice(&[SAINT_MIN | 1, 7, SAINT_MIN | 2, 0, -5, 9]);
        let mut c_sa = rust_sa.clone();
        clamp_lms_suffixes_length_32s(&mut rust_sa, m as SaSint, 0, 6);
        unsafe {
            probe_libsais16_clamp_lms_suffixes_length_32s(c_sa.as_mut_ptr(), m as SaSint, 0, 6);
        }
        assert_eq!(rust_sa, c_sa);
    }

    #[test]
    fn libsais16_distinct_lms_omp_wrappers_match_c() {
        let n = 12;
        let m = 6;
        let mut rust_sa = vec![0; 18];
        rust_sa[..m].copy_from_slice(&[
            2 | SAINT_MIN,
            4 | SAINT_MIN,
            6,
            8 | SAINT_MIN,
            10,
            12 | SAINT_MIN,
        ]);
        let mut c_sa = rust_sa.clone();
        let mut rust_thread_state = alloc_thread_state(1).unwrap();
        let rust_name = renumber_distinct_lms_suffixes_32s_4k_omp(
            &mut rust_sa,
            m as SaSint,
            1,
            &mut rust_thread_state,
        );
        let c_name = unsafe {
            probe_libsais16_renumber_distinct_lms_suffixes_32s_4k_omp(
                c_sa.as_mut_ptr(),
                m as SaSint,
                1,
            )
        };
        assert_eq!(rust_name, c_name);
        assert_eq!(rust_sa, c_sa);

        let mut rust_sa = vec![0; 18];
        rust_sa[m..m + 6].copy_from_slice(&[SAINT_MIN | 1, 0, SAINT_MIN | 2, 0, 3, 0]);
        let mut c_sa = rust_sa.clone();
        mark_distinct_lms_suffixes_32s_omp(&mut rust_sa, n, m as SaSint, 1);
        unsafe {
            probe_libsais16_mark_distinct_lms_suffixes_32s_omp(
                c_sa.as_mut_ptr(),
                n,
                m as SaSint,
                1,
            );
        }
        assert_eq!(rust_sa, c_sa);

        let mut rust_sa = vec![0; 18];
        rust_sa[m..m + 6].copy_from_slice(&[SAINT_MIN | 1, 7, SAINT_MIN | 2, 0, -5, 9]);
        let mut c_sa = rust_sa.clone();
        clamp_lms_suffixes_length_32s_omp(&mut rust_sa, n, m as SaSint, 1);
        unsafe {
            probe_libsais16_clamp_lms_suffixes_length_32s_omp(c_sa.as_mut_ptr(), n, m as SaSint, 1);
        }
        assert_eq!(rust_sa, c_sa);

        let mut rust_sa = vec![0; 18];
        rust_sa[..m].copy_from_slice(&[
            2 | SAINT_MIN,
            4 | SAINT_MIN,
            6,
            8 | SAINT_MIN,
            10,
            12 | SAINT_MIN,
        ]);
        let mut c_sa = rust_sa.clone();
        let mut rust_thread_state = alloc_thread_state(1).unwrap();
        let rust_name = renumber_and_mark_distinct_lms_suffixes_32s_4k_omp(
            &mut rust_sa,
            n,
            m as SaSint,
            1,
            &mut rust_thread_state,
        );
        let c_name = unsafe {
            probe_libsais16_renumber_and_mark_distinct_lms_suffixes_32s_4k_omp(
                c_sa.as_mut_ptr(),
                n,
                m as SaSint,
                1,
            )
        };
        assert_eq!(rust_name, c_name);
        assert_eq!(rust_sa, c_sa);
    }

    #[test]
    fn libsais16_distinct_lms_omp_wrappers_use_block_partition() {
        let m = 65_536usize;
        let mut scalar = vec![0; 2 * m];
        for i in 0..m {
            let value = (2 * i) as SaSint;
            scalar[i] = if i % 7 == 0 { value | SAINT_MIN } else { value };
        }
        let mut threaded = scalar.clone();

        let mut scalar_state = alloc_thread_state(1).unwrap();
        let mut threaded_state = alloc_thread_state(4).unwrap();
        let scalar_name = renumber_distinct_lms_suffixes_32s_4k_omp(
            &mut scalar,
            m as SaSint,
            1,
            &mut scalar_state,
        );
        let threaded_name = renumber_distinct_lms_suffixes_32s_4k_omp(
            &mut threaded,
            m as SaSint,
            4,
            &mut threaded_state,
        );
        assert_eq!(threaded_name, scalar_name);
        assert_eq!(threaded, scalar);

        let n = 131_072usize;
        let m = 65_536usize;
        let mut scalar = vec![0; n];
        for i in 0..(n >> 1) {
            scalar[m + i] = if i % 5 == 0 {
                SAINT_MIN | (i as SaSint + 1)
            } else if i % 11 == 0 {
                0
            } else {
                i as SaSint + 1
            };
        }
        let mut threaded = scalar.clone();
        mark_distinct_lms_suffixes_32s_omp(&mut scalar, n as SaSint, m as SaSint, 1);
        mark_distinct_lms_suffixes_32s_omp(&mut threaded, n as SaSint, m as SaSint, 4);
        assert_eq!(&threaded[m..n], &scalar[m..n]);

        let mut scalar = vec![0; n];
        for i in 0..(n >> 1) {
            scalar[m + i] = if i % 5 == 0 {
                SAINT_MIN | (i as SaSint + 1)
            } else {
                i as SaSint + 1
            };
        }
        let mut threaded = scalar.clone();
        clamp_lms_suffixes_length_32s_omp(&mut scalar, n as SaSint, m as SaSint, 1);
        clamp_lms_suffixes_length_32s_omp(&mut threaded, n as SaSint, m as SaSint, 4);
        assert_eq!(&threaded[m..n], &scalar[m..n]);
    }

    #[test]
    fn libsais16_unique_nonunique_lms_helpers_match_c() {
        let m = 4;
        let mut rust_t = vec![0; 12];
        let mut rust_sa = vec![0; 12];
        rust_sa[..m].copy_from_slice(&[2, 4, 6, 8]);
        rust_sa[m + 1] = SAINT_MIN | 11;
        rust_sa[m + 2] = 22;
        rust_sa[m + 3] = SAINT_MIN | 33;
        rust_sa[m + 4] = 44;
        let mut c_t = rust_t.clone();
        let mut c_sa = rust_sa.clone();

        let rust_f = renumber_unique_and_nonunique_lms_suffixes_32s(
            &mut rust_t,
            &mut rust_sa,
            m as SaSint,
            0,
            0,
            m as isize,
        );
        let c_f = unsafe {
            probe_libsais16_renumber_unique_and_nonunique_lms_suffixes_32s(
                c_t.as_mut_ptr(),
                c_sa.as_mut_ptr(),
                m as SaSint,
                0,
                0,
                m as SaSint,
            )
        };
        assert_eq!(rust_f, c_f);
        assert_eq!(rust_t, c_t);
        assert_eq!(rust_sa, c_sa);

        let mut rust_sa = vec![0; 10];
        rust_sa[m..m + 4].copy_from_slice(&[SAINT_MIN | 3, 4, SAINT_MIN | 5, 6]);
        let mut c_sa = rust_sa.clone();
        let mut rust_l = m as isize;
        let mut rust_r = 10isize;
        let mut c_l = rust_l as SaSint;
        let mut c_r = rust_r as SaSint;
        compact_unique_and_nonunique_lms_suffixes_32s(
            &mut rust_sa,
            m as SaSint,
            &mut rust_l,
            &mut rust_r,
            0,
            4,
        );
        unsafe {
            probe_libsais16_compact_unique_and_nonunique_lms_suffixes_32s(
                c_sa.as_mut_ptr(),
                m as SaSint,
                &mut c_l,
                &mut c_r,
                0,
                4,
            );
        }
        assert_eq!(rust_l as SaSint, c_l);
        assert_eq!(rust_r as SaSint, c_r);
        assert_eq!(rust_sa, c_sa);
    }

    #[test]
    fn libsais16_unique_nonunique_lms_omp_wrappers_match_c() {
        let n = 8;
        let m = 4;
        let fs = 4;
        let mut rust_t = vec![0; 12];
        let mut rust_sa = vec![0; 12];
        rust_sa[..m].copy_from_slice(&[2, 4, 6, 8]);
        rust_sa[m + 1] = SAINT_MIN | 11;
        rust_sa[m + 2] = 22;
        rust_sa[m + 3] = SAINT_MIN | 33;
        rust_sa[m + 4] = 44;
        let mut c_t = rust_t.clone();
        let mut c_sa = rust_sa.clone();

        let rust_f = renumber_unique_and_nonunique_lms_suffixes_32s_omp(
            &mut rust_t,
            &mut rust_sa,
            m as SaSint,
            1,
        );
        let c_f = unsafe {
            probe_libsais16_renumber_unique_and_nonunique_lms_suffixes_32s_omp(
                c_t.as_mut_ptr(),
                c_sa.as_mut_ptr(),
                m as SaSint,
                1,
            )
        };
        assert_eq!(rust_f, c_f);
        assert_eq!(rust_t, c_t);
        assert_eq!(rust_sa, c_sa);

        let mut rust_sa = vec![0; 12];
        rust_sa[m..m + 4].copy_from_slice(&[SAINT_MIN | 3, 4, SAINT_MIN | 5, 6]);
        rust_sa[m - 2..m].copy_from_slice(&[101, 102]);
        let mut c_sa = rust_sa.clone();
        compact_unique_and_nonunique_lms_suffixes_32s_omp(&mut rust_sa, n, m as SaSint, fs, 2, 1);
        unsafe {
            probe_libsais16_compact_unique_and_nonunique_lms_suffixes_32s_omp(
                c_sa.as_mut_ptr(),
                n,
                m as SaSint,
                fs,
                2,
                1,
            );
        }
        assert_eq!(rust_sa, c_sa);

        let mut rust_t = vec![0; 12];
        let mut rust_sa = vec![0; 12];
        rust_sa[..m].copy_from_slice(&[2, 4, 6, 8]);
        rust_sa[m + 1] = SAINT_MIN | 11;
        rust_sa[m + 2] = 22;
        rust_sa[m + 3] = SAINT_MIN | 33;
        rust_sa[m + 4] = 44;
        let mut c_t = rust_t.clone();
        let mut c_sa = rust_sa.clone();
        let rust_f = compact_lms_suffixes_32s_omp(&mut rust_t, &mut rust_sa, n, m as SaSint, fs, 1);
        let c_f = unsafe {
            probe_libsais16_compact_lms_suffixes_32s_omp(
                c_t.as_mut_ptr(),
                c_sa.as_mut_ptr(),
                n,
                m as SaSint,
                fs,
                1,
            )
        };
        assert_eq!(rust_f, c_f);
        assert_eq!(rust_t, c_t);
        assert_eq!(rust_sa, c_sa);
    }

    #[test]
    fn libsais16_unique_nonunique_lms_omp_wrappers_use_block_partition() {
        let m = 65_536usize;
        let mut scalar_t = vec![0; 2 * m];
        let mut scalar_sa = vec![0; 2 * m];
        for i in 0..m {
            scalar_sa[i] = (2 * i) as SaSint;
            scalar_sa[m + i] = if i % 5 == 0 {
                SAINT_MIN | (i as SaSint + 3)
            } else {
                i as SaSint + 3
            };
        }
        let mut threaded_t = scalar_t.clone();
        let mut threaded_sa = scalar_sa.clone();

        let scalar_f = renumber_unique_and_nonunique_lms_suffixes_32s_omp(
            &mut scalar_t,
            &mut scalar_sa,
            m as SaSint,
            1,
        );
        let threaded_f = renumber_unique_and_nonunique_lms_suffixes_32s_omp(
            &mut threaded_t,
            &mut threaded_sa,
            m as SaSint,
            4,
        );
        assert_eq!(threaded_f, scalar_f);
        assert_eq!(threaded_t, scalar_t);
        assert_eq!(threaded_sa, scalar_sa);

        let n = 131_072usize;
        let m = 4_096usize;
        let fs = 8_192usize;
        let mut scalar_sa = vec![0; n + fs];
        for i in 0..(n >> 1) {
            scalar_sa[m + i] = if i % 32 == 0 {
                SAINT_MIN | (i as SaSint + 1)
            } else {
                i as SaSint + 1
            };
        }
        let f = 1_024usize;
        for i in 0..f {
            scalar_sa[m - f + i] = 1_000_000 + i as SaSint;
        }
        let mut threaded_sa = scalar_sa.clone();

        compact_unique_and_nonunique_lms_suffixes_32s_omp(
            &mut scalar_sa,
            n as SaSint,
            m as SaSint,
            fs as SaSint,
            f as SaSint,
            1,
        );
        compact_unique_and_nonunique_lms_suffixes_32s_omp(
            &mut threaded_sa,
            n as SaSint,
            m as SaSint,
            fs as SaSint,
            f as SaSint,
            4,
        );
        assert_eq!(&threaded_sa[..m], &scalar_sa[..m]);
        assert_eq!(
            &threaded_sa[n + fs - m..n + fs],
            &scalar_sa[n + fs - m..n + fs]
        );
    }

    #[test]
    fn libsais16_merge_lms_helpers_match_c() {
        let n = 10;
        let m = 3;
        let mut rust_t = vec![0; n as usize];
        rust_t[1] = SAINT_MIN | 11;
        rust_t[3] = SAINT_MIN | 22;
        rust_t[7] = SAINT_MIN | 33;
        let mut rust_sa = vec![0; n as usize];
        rust_sa[6..10].copy_from_slice(&[2, 5, 8, 9]);
        let mut c_t = rust_t.clone();
        let mut c_sa = rust_sa.clone();
        merge_unique_lms_suffixes_32s(&mut rust_t, &mut rust_sa, n, m, 0, 0, n as isize);
        unsafe {
            probe_libsais16_merge_unique_lms_suffixes_32s(
                c_t.as_mut_ptr(),
                c_sa.as_mut_ptr(),
                n,
                m,
                0,
                0,
                n,
            );
        }
        assert_eq!(rust_t, c_t);
        assert_eq!(rust_sa, c_sa);

        let n = 10;
        let m = 5;
        let mut rust_sa = vec![9, 0, 8, 0, 0, 7, 31, 32, 33, 34];
        let mut c_sa = rust_sa.clone();
        merge_nonunique_lms_suffixes_32s(&mut rust_sa, n, m, 2, 0, m as isize);
        unsafe {
            probe_libsais16_merge_nonunique_lms_suffixes_32s(c_sa.as_mut_ptr(), n, m, 2, 0, m);
        }
        assert_eq!(rust_sa, c_sa);
    }

    #[test]
    fn libsais16_merge_lms_omp_wrappers_match_c() {
        let n = 12;
        let m = 4;
        let f = 2;
        let mut rust_t = vec![0; n as usize];
        rust_t[1] = SAINT_MIN | 11;
        rust_t[5] = SAINT_MIN | 22;
        let mut rust_sa = vec![0; n as usize];
        rust_sa[1] = 41;
        rust_sa[7..12].copy_from_slice(&[2, 6, 21, 22, 23]);
        let mut c_t = rust_t.clone();
        let mut c_sa = rust_sa.clone();
        merge_unique_lms_suffixes_32s_omp(&mut rust_t, &mut rust_sa, n, m, 1);
        unsafe {
            probe_libsais16_merge_unique_lms_suffixes_32s_omp(
                c_t.as_mut_ptr(),
                c_sa.as_mut_ptr(),
                n,
                m,
                1,
            );
        }
        assert_eq!(rust_t, c_t);
        assert_eq!(rust_sa, c_sa);

        let mut rust_sa = vec![0, 41, 1, 0, 55, 66, 77, 2, 6, 21, 22, 23];
        let mut c_sa = rust_sa.clone();
        merge_nonunique_lms_suffixes_32s_omp(&mut rust_sa, n, m, f, 1);
        unsafe {
            probe_libsais16_merge_nonunique_lms_suffixes_32s_omp(c_sa.as_mut_ptr(), n, m, f, 1);
        }
        assert_eq!(rust_sa, c_sa);

        let mut rust_t = vec![0; n as usize];
        rust_t[1] = SAINT_MIN | 11;
        rust_t[5] = SAINT_MIN | 22;
        let mut rust_sa = vec![0; n as usize];
        rust_sa[1] = 41;
        rust_sa[7..12].copy_from_slice(&[2, 6, 21, 22, 23]);
        let mut c_t = rust_t.clone();
        let mut c_sa = rust_sa.clone();
        merge_compacted_lms_suffixes_32s_omp(&mut rust_t, &mut rust_sa, n, m, f, 1);
        unsafe {
            probe_libsais16_merge_compacted_lms_suffixes_32s_omp(
                c_t.as_mut_ptr(),
                c_sa.as_mut_ptr(),
                n,
                m,
                f,
                1,
            );
        }
        assert_eq!(rust_t, c_t);
        assert_eq!(rust_sa, c_sa);
    }

    #[test]
    fn libsais16_merge_lms_omp_wrappers_use_block_partition() {
        let n = 65_536usize;
        let m = 10_000usize;
        let mut scalar_t = vec![0; n];
        for i in (0..n).step_by(17) {
            scalar_t[i] = SAINT_MIN | (i as SaSint + 1);
        }
        let unique_count = scalar_t.iter().filter(|&&value| value < 0).count();
        let mut scalar_sa = vec![0; n];
        let source = n - m - 1;
        for i in 0..=unique_count {
            scalar_sa[source + i] = ((i * 13 + 7) % n) as SaSint;
        }
        let mut threaded_t = scalar_t.clone();
        let mut threaded_sa = scalar_sa.clone();

        merge_unique_lms_suffixes_32s_omp(
            &mut scalar_t,
            &mut scalar_sa,
            n as SaSint,
            m as SaSint,
            1,
        );
        merge_unique_lms_suffixes_32s_omp(
            &mut threaded_t,
            &mut threaded_sa,
            n as SaSint,
            m as SaSint,
            4,
        );
        assert_eq!(threaded_t, scalar_t);
        assert_eq!(threaded_sa, scalar_sa);

        let n = 131_072usize;
        let m = 65_536usize;
        let f = 100usize;
        let mut scalar_sa = vec![1; n];
        for i in (0..m).step_by(9) {
            scalar_sa[i] = 0;
        }
        let zero_count = scalar_sa[..m].iter().filter(|&&value| value == 0).count();
        let source = n - m - 1 + f;
        for i in 0..=zero_count {
            scalar_sa[source + i] = 2_000_000 + i as SaSint;
        }
        let mut threaded_sa = scalar_sa.clone();

        merge_nonunique_lms_suffixes_32s_omp(
            &mut scalar_sa,
            n as SaSint,
            m as SaSint,
            f as SaSint,
            1,
        );
        merge_nonunique_lms_suffixes_32s_omp(
            &mut threaded_sa,
            n as SaSint,
            m as SaSint,
            f as SaSint,
            4,
        );
        assert_eq!(threaded_sa, scalar_sa);
    }

    #[test]
    fn libsais16_radix_sort_lms_suffixes_32s_match_c() {
        let t = vec![0, 1, 2, 3, 1, 2, 3, 0];
        let mut rust_sa = vec![0, 0, 0, 0, 0, 1, 2, 3];
        let mut c_sa = rust_sa.clone();
        let mut rust_bucket = vec![0, 6, 7, 8];
        let mut c_bucket = rust_bucket.clone();
        radix_sort_lms_suffixes_32s_6k(&t, &mut rust_sa, &mut rust_bucket, 5, 3);
        unsafe {
            probe_libsais16_radix_sort_lms_suffixes_32s_6k(
                t.as_ptr(),
                c_sa.as_mut_ptr(),
                c_bucket.as_mut_ptr(),
                5,
                3,
            );
        }
        assert_eq!(rust_sa, c_sa);
        assert_eq!(rust_bucket, c_bucket);

        let mut rust_sa = vec![0, 0, 0, 0, 0, 1, 2, 3];
        let mut c_sa = rust_sa.clone();
        let mut rust_bucket = vec![0, 0, 6, 0, 7, 0, 8, 0];
        let mut c_bucket = rust_bucket.clone();
        radix_sort_lms_suffixes_32s_2k(&t, &mut rust_sa, &mut rust_bucket, 5, 3);
        unsafe {
            probe_libsais16_radix_sort_lms_suffixes_32s_2k(
                t.as_ptr(),
                c_sa.as_mut_ptr(),
                c_bucket.as_mut_ptr(),
                5,
                3,
            );
        }
        assert_eq!(rust_sa, c_sa);
        assert_eq!(rust_bucket, c_bucket);

        let mut cache = vec![ThreadCache::default(); 8];
        let sa = vec![0, 0, 0, 0, 0, 1, 2, 3];
        radix_sort_lms_suffixes_32s_block_gather(&t, &sa, &mut cache, 5, 3);
        assert_eq!(cache[5].index, 1);
        assert_eq!(cache[5].symbol, 1);
        assert_eq!(cache[6].index, 2);
        assert_eq!(cache[6].symbol, 2);
        assert_eq!(cache[7].index, 3);
        assert_eq!(cache[7].symbol, 3);

        let mut bucket = vec![0, 6, 7, 8];
        radix_sort_lms_suffixes_32s_6k_block_sort(&mut bucket, &mut cache, 5, 3);
        assert_eq!(bucket, vec![0, 5, 6, 7]);
        assert_eq!(cache[5].symbol, 5);
        assert_eq!(cache[6].symbol, 6);
        assert_eq!(cache[7].symbol, 7);

        let mut cache = vec![ThreadCache::default(); 8];
        radix_sort_lms_suffixes_32s_block_gather(&t, &sa, &mut cache, 5, 3);
        let mut bucket = vec![0, 0, 6, 0, 7, 0, 8, 0];
        radix_sort_lms_suffixes_32s_2k_block_sort(&mut bucket, &mut cache, 5, 3);
        assert_eq!(bucket, vec![0, 0, 5, 0, 6, 0, 7, 0]);
        assert_eq!(cache[5].symbol, 5);
        assert_eq!(cache[6].symbol, 6);
        assert_eq!(cache[7].symbol, 7);

        let mut rust_sa = vec![0, 0, 0, 0, 0, 1, 2, 3];
        let mut c_sa = rust_sa.clone();
        let mut rust_bucket = vec![0, 6, 7, 8];
        let mut c_bucket = rust_bucket.clone();
        radix_sort_lms_suffixes_32s_6k_omp(&t, &mut rust_sa, 8, 4, &mut rust_bucket, 1);
        unsafe {
            probe_libsais16_radix_sort_lms_suffixes_32s_6k_omp(
                t.as_ptr(),
                c_sa.as_mut_ptr(),
                8,
                4,
                c_bucket.as_mut_ptr(),
                1,
            );
        }
        assert_eq!(rust_sa, c_sa);
        assert_eq!(rust_bucket, c_bucket);

        let mut rust_sa = vec![0, 0, 0, 0, 0, 1, 2, 3];
        let mut c_sa = rust_sa.clone();
        let mut rust_bucket = vec![0, 0, 6, 0, 7, 0, 8, 0];
        let mut c_bucket = rust_bucket.clone();
        radix_sort_lms_suffixes_32s_2k_omp(&t, &mut rust_sa, 8, 4, &mut rust_bucket, 1);
        unsafe {
            probe_libsais16_radix_sort_lms_suffixes_32s_2k_omp(
                t.as_ptr(),
                c_sa.as_mut_ptr(),
                8,
                4,
                c_bucket.as_mut_ptr(),
                1,
            );
        }
        assert_eq!(rust_sa, c_sa);
        assert_eq!(rust_bucket, c_bucket);

        let t = vec![2, 1, 3, 1, 0];
        let mut rust_sa = vec![0; t.len()];
        let mut c_sa = rust_sa.clone();
        let mut rust_bucket = vec![0, 2, 4, 5];
        let mut c_bucket = rust_bucket.clone();
        let rust_m =
            radix_sort_lms_suffixes_32s_1k(&t, &mut rust_sa, t.len() as SaSint, &mut rust_bucket);
        let c_m = unsafe {
            probe_libsais16_radix_sort_lms_suffixes_32s_1k(
                t.as_ptr(),
                c_sa.as_mut_ptr(),
                t.len() as SaSint,
                c_bucket.as_mut_ptr(),
            )
        };
        assert_eq!(rust_m, c_m);
        assert_eq!(rust_sa, c_sa);
        assert_eq!(rust_bucket, c_bucket);
    }

    #[test]
    fn libsais16_radix_sort_set_markers_32s_match_c() {
        let mut rust_sa = vec![0; 8];
        let mut c_sa = rust_sa.clone();
        let mut induction_bucket = vec![1, 3, 5, 7];
        radix_sort_set_markers_32s_6k(&mut rust_sa, &induction_bucket, 0, 4);
        unsafe {
            probe_libsais16_radix_sort_set_markers_32s_6k(
                c_sa.as_mut_ptr(),
                induction_bucket.as_mut_ptr(),
                0,
                4,
            );
        }
        assert_eq!(rust_sa, c_sa);

        let mut rust_sa = vec![0; 8];
        let mut c_sa = rust_sa.clone();
        radix_sort_set_markers_32s_6k_omp(&mut rust_sa, 5, &induction_bucket, 1);
        unsafe {
            probe_libsais16_radix_sort_set_markers_32s_6k_omp(
                c_sa.as_mut_ptr(),
                5,
                induction_bucket.as_mut_ptr(),
                1,
            );
        }
        assert_eq!(rust_sa, c_sa);

        let mut rust_sa = vec![0; 8];
        let mut c_sa = rust_sa.clone();
        let mut induction_bucket = vec![1, 0, 3, 0, 5, 0, 7, 0];
        radix_sort_set_markers_32s_4k(&mut rust_sa, &induction_bucket, 0, 4);
        unsafe {
            probe_libsais16_radix_sort_set_markers_32s_4k(
                c_sa.as_mut_ptr(),
                induction_bucket.as_mut_ptr(),
                0,
                4,
            );
        }
        assert_eq!(rust_sa, c_sa);

        let mut rust_sa = vec![0; 8];
        let mut c_sa = rust_sa.clone();
        radix_sort_set_markers_32s_4k_omp(&mut rust_sa, 5, &induction_bucket, 1);
        unsafe {
            probe_libsais16_radix_sort_set_markers_32s_4k_omp(
                c_sa.as_mut_ptr(),
                5,
                induction_bucket.as_mut_ptr(),
                1,
            );
        }
        assert_eq!(rust_sa, c_sa);
    }

    #[test]
    fn libsais16_radix_sort_set_markers_32s_omp_partitions_large_inputs() {
        let k = 65_600usize;
        let induction_bucket_6k: Vec<SaSint> = (0..k).map(|i| i as SaSint).collect();
        let mut single = vec![0; k];
        let mut threaded = vec![0; k];
        radix_sort_set_markers_32s_6k_omp(&mut single, k as SaSint, &induction_bucket_6k, 1);
        radix_sort_set_markers_32s_6k_omp(&mut threaded, k as SaSint, &induction_bucket_6k, 4);
        assert_eq!(threaded, single);

        let mut induction_bucket_4k = vec![0; 2 * k];
        for i in 0..k {
            induction_bucket_4k[buckets_index2(i, 0)] = i as SaSint;
        }
        let mut single = vec![0; k];
        let mut threaded = vec![0; k];
        radix_sort_set_markers_32s_4k_omp(&mut single, k as SaSint, &induction_bucket_4k, 1);
        radix_sort_set_markers_32s_4k_omp(&mut threaded, k as SaSint, &induction_bucket_4k, 4);
        assert_eq!(threaded, single);
    }

    #[test]
    fn libsais16_partial_sorting_32s_helpers_match_c() {
        let k = 3;
        let mut rust_sa = vec![0, SAINT_MIN, 2, SAINT_MIN, 4, SAINT_MIN];
        let mut c_sa = rust_sa.clone();
        let mut buckets = vec![0; 6 * k as usize];
        buckets[buckets_index4(1, 0)] = 3;
        buckets[buckets_index4(2, 0)] = 6;
        buckets[4 * k as usize + buckets_index2(0, 0)] = 0;
        buckets[4 * k as usize + buckets_index2(1, 0)] = 1;
        partial_sorting_shift_markers_32s_6k_omp(&mut rust_sa, k, &buckets, 1);
        unsafe {
            probe_libsais16_partial_sorting_shift_markers_32s_6k_omp(
                c_sa.as_mut_ptr(),
                k,
                buckets.as_ptr(),
                1,
            );
        }
        assert_eq!(rust_sa, c_sa);

        let mut rust_sa = vec![
            1 | SUFFIX_GROUP_MARKER,
            2,
            3 | SUFFIX_GROUP_MARKER,
            4 | SUFFIX_GROUP_MARKER,
            5,
            6,
        ];
        let mut c_sa = rust_sa.clone();
        partial_sorting_shift_markers_32s_4k(&mut rust_sa, 6);
        unsafe { probe_libsais16_partial_sorting_shift_markers_32s_4k(c_sa.as_mut_ptr(), 6) };
        assert_eq!(rust_sa, c_sa);

        let mut rust_buckets = vec![0; 6 * k as usize];
        for (i, value) in rust_buckets[4 * k as usize..].iter_mut().enumerate() {
            *value = 100 + i as SaSint;
        }
        let mut c_buckets = rust_buckets.clone();
        partial_sorting_shift_buckets_32s_6k(k, &mut rust_buckets);
        unsafe { probe_libsais16_partial_sorting_shift_buckets_32s_6k(k, c_buckets.as_mut_ptr()) };
        assert_eq!(rust_buckets, c_buckets);

        let mut rust_sa = vec![1 | SUFFIX_GROUP_MARKER, -3, 5 | SUFFIX_GROUP_MARKER, -7];
        let mut c_sa = rust_sa.clone();
        let rust_l = partial_sorting_gather_lms_suffixes_32s_4k(&mut rust_sa, 0, 4);
        let c_l = unsafe {
            probe_libsais16_partial_sorting_gather_lms_suffixes_32s_4k(c_sa.as_mut_ptr(), 0, 4)
        };
        assert_eq!(rust_l, c_l);
        assert_eq!(rust_sa, c_sa);

        let mut rust_sa = vec![1, -3, 5, -7];
        let mut c_sa = rust_sa.clone();
        let rust_l = partial_sorting_gather_lms_suffixes_32s_1k(&mut rust_sa, 0, 4);
        let c_l = unsafe {
            probe_libsais16_partial_sorting_gather_lms_suffixes_32s_1k(c_sa.as_mut_ptr(), 0, 4)
        };
        assert_eq!(rust_l, c_l);
        assert_eq!(rust_sa, c_sa);

        let mut rust_state = alloc_thread_state(1).unwrap();
        let mut rust_sa = vec![1 | SUFFIX_GROUP_MARKER, -3, 5 | SUFFIX_GROUP_MARKER, -7];
        let mut c_sa = rust_sa.clone();
        partial_sorting_gather_lms_suffixes_32s_4k_omp(&mut rust_sa, 4, 1, &mut rust_state);
        unsafe {
            probe_libsais16_partial_sorting_gather_lms_suffixes_32s_4k_omp(c_sa.as_mut_ptr(), 4, 1);
        }
        assert_eq!(rust_sa, c_sa);

        let mut rust_state = alloc_thread_state(1).unwrap();
        let mut rust_sa = vec![1, -3, 5, -7];
        let mut c_sa = rust_sa.clone();
        partial_sorting_gather_lms_suffixes_32s_1k_omp(&mut rust_sa, 4, 1, &mut rust_state);
        unsafe {
            probe_libsais16_partial_sorting_gather_lms_suffixes_32s_1k_omp(c_sa.as_mut_ptr(), 4, 1);
        }
        assert_eq!(rust_sa, c_sa);
    }

    #[test]
    fn libsais16_partial_sorting_gather_lms_suffixes_32s_omp_uses_block_partition() {
        let n = 65_536usize;
        let mut base_4k = vec![0; n];
        let mut base_1k = vec![0; n];
        for i in 0..n {
            let value = (i as SaSint + 1) & SAINT_MAX;
            base_4k[i] = if i % 7 == 0 {
                value | SAINT_MIN | SUFFIX_GROUP_MARKER
            } else if i % 11 == 0 {
                value | SUFFIX_GROUP_MARKER
            } else {
                value
            };
            base_1k[i] = if i % 7 == 0 { value | SAINT_MIN } else { value };
        }
        let lms_count = base_1k.iter().filter(|&&v| v < 0).count();

        let mut scalar = base_4k.clone();
        let mut threaded = base_4k;
        let mut scalar_state = alloc_thread_state(1).unwrap();
        let mut threaded_state = alloc_thread_state(4).unwrap();
        partial_sorting_gather_lms_suffixes_32s_4k_omp(
            &mut scalar,
            n as SaSint,
            1,
            &mut scalar_state,
        );
        partial_sorting_gather_lms_suffixes_32s_4k_omp(
            &mut threaded,
            n as SaSint,
            4,
            &mut threaded_state,
        );
        assert_eq!(&threaded[..lms_count], &scalar[..lms_count]);

        let mut scalar = base_1k.clone();
        let mut threaded = base_1k;
        partial_sorting_gather_lms_suffixes_32s_1k_omp(
            &mut scalar,
            n as SaSint,
            1,
            &mut scalar_state,
        );
        partial_sorting_gather_lms_suffixes_32s_1k_omp(
            &mut threaded,
            n as SaSint,
            4,
            &mut threaded_state,
        );
        assert_eq!(&threaded[..lms_count], &scalar[..lms_count]);
    }

    #[test]
    fn libsais16_partial_sorting_32s_block_helpers_behave_like_upstream_shapes() {
        let t = vec![0, 1, 2, 1, 0];
        let k = 3;

        let mut sa = vec![0, 4 | SAINT_MIN, 0];
        let mut cache = vec![ThreadCache::default(); sa.len()];
        partial_sorting_scan_right_to_left_32s_6k_block_gather(&t, &mut sa, &mut cache, 1, 1);
        assert_eq!(cache[1].index, 4 | SAINT_MIN);
        assert_eq!(cache[1].symbol, buckets_index4(1, 1) as SaSint);

        let mut sa = vec![0, 4 | SUFFIX_GROUP_MARKER, 0];
        let mut cache = vec![ThreadCache::default(); sa.len()];
        partial_sorting_scan_right_to_left_32s_4k_block_gather(&t, &mut sa, &mut cache, 1, 1);
        assert_eq!(sa[1], 0);
        assert_eq!(cache[1].index, 4 | SUFFIX_GROUP_MARKER);
        assert_eq!(cache[1].symbol, buckets_index2(1, 1) as SaSint);

        let mut sa = vec![0, 4, 0];
        let mut cache = vec![ThreadCache::default(); sa.len()];
        partial_sorting_scan_right_to_left_32s_1k_block_gather(&t, &mut sa, &mut cache, 1, 1);
        assert_eq!(sa[1], 0);
        assert_eq!(cache[1].index, 3 | SAINT_MIN);
        assert_eq!(cache[1].symbol, 1);

        let mut sa = vec![4 | SAINT_MIN, 0, 0];
        let mut cache = vec![ThreadCache::default(); sa.len()];
        partial_sorting_scan_left_to_right_32s_6k_block_gather(&t, &mut sa, &mut cache, 0, 1);
        assert_eq!(cache[0].index, 4 | SAINT_MIN);
        assert_eq!(cache[0].symbol, buckets_index4(1, 1) as SaSint);

        let mut sa = vec![4 | SUFFIX_GROUP_MARKER, 0, 0];
        let mut cache = vec![ThreadCache::default(); sa.len()];
        partial_sorting_scan_left_to_right_32s_4k_block_gather(&t, &mut sa, &mut cache, 0, 1);
        assert_eq!(sa[0], 0);
        assert_eq!(cache[0].index, 4 | SUFFIX_GROUP_MARKER);
        assert_eq!(cache[0].symbol, buckets_index2(1, 0) as SaSint);

        let mut sa = vec![4, 0, 0];
        let mut cache = vec![ThreadCache::default(); sa.len()];
        partial_sorting_scan_left_to_right_32s_1k_block_gather(&t, &mut sa, &mut cache, 0, 1);
        assert_eq!(sa[0], 0);
        assert_eq!(cache[0].index, 3);
        assert_eq!(cache[0].symbol, 1);

        let mut cache = vec![ThreadCache::default(); 3];
        cache[1].index = 4 | SAINT_MIN;
        cache[1].symbol = buckets_index4(1, 1) as SaSint;
        let mut buckets = vec![0; 4 * k];
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
        assert_eq!(cache[1].index, 3 | SAINT_MIN);
        assert_eq!(buckets[buckets_index4(1, 1)], 1);
        assert_eq!(buckets[buckets_index4(1, 1) + 2], 1);

        let mut cache = vec![ThreadCache::default(); 3];
        cache[0].index = 4 | SAINT_MIN;
        cache[0].symbol = buckets_index4(1, 1) as SaSint;
        let mut buckets = vec![0; 4 * k];
        buckets[buckets_index4(1, 1)] = 1;
        let d = partial_sorting_scan_left_to_right_32s_6k_block_sort(
            &t,
            &mut buckets,
            0,
            &mut cache,
            0,
            1,
        );
        assert_eq!(d, 1);
        assert_eq!(cache[0].index, 3 | SAINT_MIN);
        assert_eq!(buckets[buckets_index4(1, 1)], 2);
        assert_eq!(buckets[buckets_index4(1, 1) + 2], 1);

        let mut cache = vec![ThreadCache::default(); 3];
        cache[1].index = 4 | SUFFIX_GROUP_MARKER;
        cache[1].symbol = buckets_index2(1, 1) as SaSint;
        let mut buckets = vec![0; 4 * k];
        buckets[3 * k + 1] = 2;
        let d = partial_sorting_scan_right_to_left_32s_4k_block_sort(
            &t,
            k as SaSint,
            &mut buckets,
            0,
            &mut cache,
            1,
            1,
        );
        assert_eq!(d, 1);
        assert_eq!(cache[1].symbol, 1);
        assert_eq!(buckets[3 * k + 1], 1);

        let mut cache = vec![ThreadCache::default(); 3];
        cache[0].index = 4 | SUFFIX_GROUP_MARKER;
        cache[0].symbol = buckets_index2(1, 0) as SaSint;
        let mut buckets = vec![0; 4 * k];
        buckets[2 * k + 1] = 1;
        let d = partial_sorting_scan_left_to_right_32s_4k_block_sort(
            &t,
            k as SaSint,
            &mut buckets,
            0,
            &mut cache,
            0,
            1,
        );
        assert_eq!(d, 1);
        assert_eq!(cache[0].symbol, 1);
        assert_eq!(buckets[2 * k + 1], 2);

        let mut cache = vec![ThreadCache::default(); 3];
        cache[1].index = 4;
        cache[1].symbol = 1;
        let mut buckets = vec![0; k];
        buckets[1] = 2;
        partial_sorting_scan_right_to_left_32s_1k_block_sort(&t, &mut buckets, &mut cache, 1, 1);
        assert_eq!(cache[1].symbol, 1);
        assert_eq!(buckets[1], 1);

        let mut cache = vec![ThreadCache::default(); 3];
        cache[0].index = 4;
        cache[0].symbol = 1;
        let mut buckets = vec![0; k];
        buckets[1] = 1;
        partial_sorting_scan_left_to_right_32s_1k_block_sort(&t, &mut buckets, &mut cache, 0, 1);
        assert_eq!(cache[0].symbol, 1);
        assert_eq!(buckets[1], 2);
    }

    #[test]
    fn libsais16_partial_sorting_scan_32s_match_c() {
        let t = vec![0, 1, 2, 1, 3, 0];
        let k = 4;

        let mut rust_sa = vec![3, 4, 0, 0, 9, 9, 9, 9];
        let mut c_sa = rust_sa.clone();
        let mut rust_buckets = vec![0; 6 * k as usize];
        rust_buckets[buckets_index4(2, 0)] = 4;
        rust_buckets[buckets_index4(1, 1)] = 5;
        let mut c_buckets = rust_buckets.clone();
        let rust_d =
            partial_sorting_scan_left_to_right_32s_6k(&t, &mut rust_sa, &mut rust_buckets, 0, 0, 2);
        let c_d = unsafe {
            probe_libsais16_partial_sorting_scan_left_to_right_32s_6k(
                t.as_ptr(),
                c_sa.as_mut_ptr(),
                c_buckets.as_mut_ptr(),
                0,
                0,
                2,
            )
        };
        assert_eq!(rust_d, c_d);
        assert_eq!(rust_sa, c_sa);
        assert_eq!(rust_buckets, c_buckets);

        let mut rust_sa = vec![3, 4, 0, 0, 9, 9, 9, 9];
        let mut c_sa = rust_sa.clone();
        let mut rust_buckets = vec![0; 4 * k as usize];
        rust_buckets[2 * k as usize + 2] = 4;
        rust_buckets[2 * k as usize + 1] = 5;
        let mut c_buckets = rust_buckets.clone();
        let rust_d = partial_sorting_scan_left_to_right_32s_4k(
            &t,
            &mut rust_sa,
            k,
            &mut rust_buckets,
            0,
            0,
            2,
        );
        let c_d = unsafe {
            probe_libsais16_partial_sorting_scan_left_to_right_32s_4k(
                t.as_ptr(),
                c_sa.as_mut_ptr(),
                k,
                c_buckets.as_mut_ptr(),
                0,
                0,
                2,
            )
        };
        assert_eq!(rust_d, c_d);
        assert_eq!(rust_sa, c_sa);
        assert_eq!(rust_buckets, c_buckets);

        let mut rust_sa = vec![3, 4, 0, 0, 9, 9, 9, 9];
        let mut c_sa = rust_sa.clone();
        let mut rust_buckets = vec![0, 5, 4, 0];
        let mut c_buckets = rust_buckets.clone();
        partial_sorting_scan_left_to_right_32s_1k(&t, &mut rust_sa, &mut rust_buckets, 0, 2);
        unsafe {
            probe_libsais16_partial_sorting_scan_left_to_right_32s_1k(
                t.as_ptr(),
                c_sa.as_mut_ptr(),
                c_buckets.as_mut_ptr(),
                0,
                2,
            );
        }
        assert_eq!(rust_sa, c_sa);
        assert_eq!(rust_buckets, c_buckets);

        let mut rust_sa = vec![3, 4, 0, 0, 9, 9, 9, 9];
        let mut c_sa = rust_sa.clone();
        let mut rust_buckets = vec![0; 6 * k as usize];
        rust_buckets[buckets_index4(2, 0)] = 7;
        rust_buckets[buckets_index4(1, 1)] = 6;
        let mut c_buckets = rust_buckets.clone();
        let rust_d =
            partial_sorting_scan_right_to_left_32s_6k(&t, &mut rust_sa, &mut rust_buckets, 0, 0, 2);
        let c_d = unsafe {
            probe_libsais16_partial_sorting_scan_right_to_left_32s_6k(
                t.as_ptr(),
                c_sa.as_mut_ptr(),
                c_buckets.as_mut_ptr(),
                0,
                0,
                2,
            )
        };
        assert_eq!(rust_d, c_d);
        assert_eq!(rust_sa, c_sa);
        assert_eq!(rust_buckets, c_buckets);

        let mut rust_sa = vec![3, 4, 0, 0, 9, 9, 9, 9];
        let mut c_sa = rust_sa.clone();
        let mut rust_buckets = vec![0; 4 * k as usize];
        rust_buckets[3 * k as usize + 2] = 7;
        rust_buckets[3 * k as usize + 1] = 6;
        let mut c_buckets = rust_buckets.clone();
        let rust_d = partial_sorting_scan_right_to_left_32s_4k(
            &t,
            &mut rust_sa,
            k,
            &mut rust_buckets,
            0,
            0,
            2,
        );
        let c_d = unsafe {
            probe_libsais16_partial_sorting_scan_right_to_left_32s_4k(
                t.as_ptr(),
                c_sa.as_mut_ptr(),
                k,
                c_buckets.as_mut_ptr(),
                0,
                0,
                2,
            )
        };
        assert_eq!(rust_d, c_d);
        assert_eq!(rust_sa, c_sa);
        assert_eq!(rust_buckets, c_buckets);

        let mut rust_sa = vec![3, 4, 0, 0, 9, 9, 9, 9];
        let mut c_sa = rust_sa.clone();
        let mut rust_buckets = vec![0, 6, 7, 0];
        let mut c_buckets = rust_buckets.clone();
        partial_sorting_scan_right_to_left_32s_1k(&t, &mut rust_sa, &mut rust_buckets, 0, 2);
        unsafe {
            probe_libsais16_partial_sorting_scan_right_to_left_32s_1k(
                t.as_ptr(),
                c_sa.as_mut_ptr(),
                c_buckets.as_mut_ptr(),
                0,
                2,
            );
        }
        assert_eq!(rust_sa, c_sa);
        assert_eq!(rust_buckets, c_buckets);

        let mut state = alloc_thread_state(1).unwrap();
        let mut rust_sa = vec![3, 4, 0, 0, 9, 9, 7, 9];
        let mut c_sa = rust_sa.clone();
        let mut rust_buckets = vec![0; 6 * k as usize];
        rust_buckets[buckets_index4(2, 0)] = 4;
        rust_buckets[buckets_index4(1, 1)] = 5;
        rust_buckets[buckets_index4(3, 0)] = 6;
        let mut c_buckets = rust_buckets.clone();
        let rust_d = partial_sorting_scan_left_to_right_32s_6k_omp(
            &t,
            &mut rust_sa,
            5,
            &mut rust_buckets,
            2,
            0,
            1,
            &mut state,
        );
        let c_d = unsafe {
            probe_libsais16_partial_sorting_scan_left_to_right_32s_6k_omp(
                t.as_ptr(),
                c_sa.as_mut_ptr(),
                5,
                c_buckets.as_mut_ptr(),
                2,
                0,
                1,
            )
        };
        assert_eq!(rust_d, c_d);
        assert_eq!(rust_sa, c_sa);
        assert_eq!(rust_buckets, c_buckets);

        let mut state = alloc_thread_state(1).unwrap();
        let mut rust_sa = vec![3, 4, 0, 0, 9, 9, 7, 9];
        let mut c_sa = rust_sa.clone();
        let mut rust_buckets = vec![0; 4 * k as usize];
        rust_buckets[2 * k as usize + 2] = 4;
        rust_buckets[2 * k as usize + 1] = 5;
        rust_buckets[2 * k as usize + 3] = 6;
        let mut c_buckets = rust_buckets.clone();
        let rust_d = partial_sorting_scan_left_to_right_32s_4k_omp(
            &t,
            &mut rust_sa,
            5,
            k,
            &mut rust_buckets,
            0,
            1,
            &mut state,
        );
        let c_d = unsafe {
            probe_libsais16_partial_sorting_scan_left_to_right_32s_4k_omp(
                t.as_ptr(),
                c_sa.as_mut_ptr(),
                5,
                k,
                c_buckets.as_mut_ptr(),
                0,
                1,
            )
        };
        assert_eq!(rust_d, c_d);
        assert_eq!(rust_sa, c_sa);
        assert_eq!(rust_buckets, c_buckets);

        let mut state = alloc_thread_state(1).unwrap();
        let mut rust_sa = vec![3, 4, 0, 0, 9, 9, 7, 9];
        let mut c_sa = rust_sa.clone();
        let mut rust_buckets = vec![0, 5, 4, 6];
        let mut c_buckets = rust_buckets.clone();
        partial_sorting_scan_left_to_right_32s_1k_omp(
            &t,
            &mut rust_sa,
            5,
            &mut rust_buckets,
            1,
            &mut state,
        );
        unsafe {
            probe_libsais16_partial_sorting_scan_left_to_right_32s_1k_omp(
                t.as_ptr(),
                c_sa.as_mut_ptr(),
                5,
                c_buckets.as_mut_ptr(),
                1,
            );
        }
        assert_eq!(rust_sa, c_sa);
        assert_eq!(rust_buckets, c_buckets);

        let mut state = alloc_thread_state(1).unwrap();
        let mut rust_sa = vec![0, 0, 3, 4, 9, 9, 9, 9];
        let mut c_sa = rust_sa.clone();
        let mut rust_buckets = vec![0; 6 * k as usize];
        rust_buckets[buckets_index4(2, 0)] = 7;
        rust_buckets[buckets_index4(1, 1)] = 6;
        let mut c_buckets = rust_buckets.clone();
        let rust_d = partial_sorting_scan_right_to_left_32s_6k_omp(
            &t,
            &mut rust_sa,
            5,
            &mut rust_buckets,
            1,
            1,
            0,
            1,
            &mut state,
        );
        let c_d = unsafe {
            probe_libsais16_partial_sorting_scan_right_to_left_32s_6k_omp(
                t.as_ptr(),
                c_sa.as_mut_ptr(),
                5,
                c_buckets.as_mut_ptr(),
                1,
                1,
                0,
                1,
            )
        };
        assert_eq!(rust_d, c_d);
        assert_eq!(rust_sa, c_sa);
        assert_eq!(rust_buckets, c_buckets);

        let mut state = alloc_thread_state(1).unwrap();
        let mut rust_sa = vec![3, 4, 0, 0, 9, 9, 9, 9];
        let mut c_sa = rust_sa.clone();
        let mut rust_buckets = vec![0; 4 * k as usize];
        rust_buckets[3 * k as usize + 2] = 7;
        rust_buckets[3 * k as usize + 1] = 6;
        let mut c_buckets = rust_buckets.clone();
        let rust_d = partial_sorting_scan_right_to_left_32s_4k_omp(
            &t,
            &mut rust_sa,
            2,
            k,
            &mut rust_buckets,
            0,
            1,
            &mut state,
        );
        let c_d = unsafe {
            probe_libsais16_partial_sorting_scan_right_to_left_32s_4k_omp(
                t.as_ptr(),
                c_sa.as_mut_ptr(),
                2,
                k,
                c_buckets.as_mut_ptr(),
                0,
                1,
            )
        };
        assert_eq!(rust_d, c_d);
        assert_eq!(rust_sa, c_sa);
        assert_eq!(rust_buckets, c_buckets);

        let mut state = alloc_thread_state(1).unwrap();
        let mut rust_sa = vec![3, 4, 0, 0, 9, 9, 9, 9];
        let mut c_sa = rust_sa.clone();
        let mut rust_buckets = vec![0, 6, 7, 0];
        let mut c_buckets = rust_buckets.clone();
        partial_sorting_scan_right_to_left_32s_1k_omp(
            &t,
            &mut rust_sa,
            2,
            &mut rust_buckets,
            1,
            &mut state,
        );
        unsafe {
            probe_libsais16_partial_sorting_scan_right_to_left_32s_1k_omp(
                t.as_ptr(),
                c_sa.as_mut_ptr(),
                2,
                c_buckets.as_mut_ptr(),
                1,
            );
        }
        assert_eq!(rust_sa, c_sa);
        assert_eq!(rust_buckets, c_buckets);
    }

    #[test]
    fn libsais16_place_lms_suffixes_histogram_32s_match_c() {
        let n = 12;
        let k = 4;
        let m = 4;
        let mut rust_sa = vec![101, 102, 103, 104, 9, 9, 9, 9, 9, 9, 9, 9];
        let mut c_sa = rust_sa.clone();
        let mut buckets = vec![0; 2 * k as usize];
        buckets[buckets_index2(1, 0)] = 7;
        buckets[buckets_index2(1, 1)] = 2;
        buckets[buckets_index2(2, 0)] = 10;
        buckets[buckets_index2(2, 1)] = 1;
        place_lms_suffixes_histogram_32s_2k(&mut rust_sa, n, k, m, &buckets);
        unsafe {
            probe_libsais16_place_lms_suffixes_histogram_32s_2k(
                c_sa.as_mut_ptr(),
                n,
                k,
                m,
                buckets.as_ptr(),
            );
        }
        assert_eq!(rust_sa, c_sa);

        let mut rust_sa = vec![101, 102, 103, 104, 9, 9, 9, 9, 9, 9, 9, 9];
        let mut c_sa = rust_sa.clone();
        let mut buckets = vec![0; 4 * k as usize];
        buckets[buckets_index2(1, 1)] = 2;
        buckets[buckets_index2(2, 1)] = 1;
        buckets[3 * k as usize + 1] = 7;
        buckets[3 * k as usize + 2] = 10;
        place_lms_suffixes_histogram_32s_4k(&mut rust_sa, n, k, m, &buckets);
        unsafe {
            probe_libsais16_place_lms_suffixes_histogram_32s_4k(
                c_sa.as_mut_ptr(),
                n,
                k,
                m,
                buckets.as_ptr(),
            );
        }
        assert_eq!(rust_sa, c_sa);

        let mut rust_sa = vec![101, 102, 103, 104, 9, 9, 9, 9, 9, 9, 9, 9];
        let mut c_sa = rust_sa.clone();
        let mut buckets = vec![0; 6 * k as usize];
        buckets[buckets_index4(1, 1)] = 2;
        buckets[buckets_index4(2, 1)] = 1;
        buckets[5 * k as usize + 1] = 7;
        buckets[5 * k as usize + 2] = 10;
        place_lms_suffixes_histogram_32s_6k(&mut rust_sa, n, k, m, &buckets);
        unsafe {
            probe_libsais16_place_lms_suffixes_histogram_32s_6k(
                c_sa.as_mut_ptr(),
                n,
                k,
                m,
                buckets.as_ptr(),
            );
        }
        assert_eq!(rust_sa, c_sa);
    }

    #[test]
    fn libsais16_count_gather_lms_suffixes_32s_match_c() {
        let t = vec![2, 1, 3, 1, 2, 0, 1, 0];
        let n = t.len() as SaSint;
        let k = 4;

        let mut rust_sa = vec![0; t.len()];
        let mut c_sa = rust_sa.clone();
        let rust_m = gather_lms_suffixes_32s(&t, &mut rust_sa, n);
        let c_m =
            unsafe { probe_libsais16_gather_lms_suffixes_32s(t.as_ptr(), c_sa.as_mut_ptr(), n) };
        assert_eq!(rust_m, c_m);
        assert_eq!(rust_sa, c_sa);

        let compact_t = vec![2, SAINT_MIN | 1, 3, 1, SAINT_MIN | 2, 0, 1, 0];
        let mut rust_sa = vec![0; compact_t.len()];
        let mut c_sa = rust_sa.clone();
        let rust_m = gather_compacted_lms_suffixes_32s(&compact_t, &mut rust_sa, n);
        let c_m = unsafe {
            probe_libsais16_gather_compacted_lms_suffixes_32s(
                compact_t.as_ptr(),
                c_sa.as_mut_ptr(),
                n,
            )
        };
        assert_eq!(rust_m, c_m);
        assert_eq!(rust_sa, c_sa);

        let mut rust_buckets = vec![99; 2 * k as usize];
        let mut c_buckets = rust_buckets.clone();
        count_lms_suffixes_32s_2k(&t, n, k, &mut rust_buckets);
        unsafe {
            probe_libsais16_count_lms_suffixes_32s_2k(t.as_ptr(), n, k, c_buckets.as_mut_ptr());
        }
        assert_eq!(rust_buckets, c_buckets);

        let mut rust_sa = vec![0; t.len()];
        let mut c_sa = rust_sa.clone();
        let mut rust_buckets = vec![0; 2 * k as usize];
        let mut c_buckets = rust_buckets.clone();
        let rust_m = count_and_gather_lms_suffixes_32s_2k(
            &t,
            &mut rust_sa,
            n,
            k,
            &mut rust_buckets,
            0,
            n as isize,
        );
        let c_m = unsafe {
            probe_libsais16_count_and_gather_lms_suffixes_32s_2k(
                t.as_ptr(),
                c_sa.as_mut_ptr(),
                n,
                k,
                c_buckets.as_mut_ptr(),
                0,
                n,
            )
        };
        assert_eq!(rust_m, c_m);
        assert_eq!(rust_sa, c_sa);
        assert_eq!(rust_buckets, c_buckets);

        let mut rust_sa = vec![0; compact_t.len()];
        let mut c_sa = rust_sa.clone();
        let mut rust_buckets = vec![0; 2 * k as usize];
        let mut c_buckets = rust_buckets.clone();
        let rust_m = count_and_gather_compacted_lms_suffixes_32s_2k(
            &compact_t,
            &mut rust_sa,
            n,
            k,
            &mut rust_buckets,
            0,
            n as isize,
        );
        let c_m = unsafe {
            probe_libsais16_count_and_gather_compacted_lms_suffixes_32s_2k(
                compact_t.as_ptr(),
                c_sa.as_mut_ptr(),
                n,
                k,
                c_buckets.as_mut_ptr(),
                0,
                n,
            )
        };
        assert_eq!(rust_m, c_m);
        assert_eq!(rust_sa, c_sa);
        assert_eq!(rust_buckets, c_buckets);
    }

    #[test]
    fn libsais16_small_openmp_leaf_helpers_match_upstream_shapes() {
        let sa = [-1, 0, 3, SAINT_MIN, 0, 7, -5];
        assert_eq!(count_negative_marked_suffixes(&sa, 1, 5), 1);
        assert_eq!(count_zero_marked_suffixes(&sa, 1, 5), 2);

        let mut buckets = vec![1, 2, 3, 0, 4, 5, 6, 0, 7, 8, 9, 0, 10, 11, 12, 0];
        accumulate_counts_s32_4(&mut buckets, 12, 3, 4);
        assert_eq!(&buckets[12..15], &[22, 26, 30]);

        let mut many = Vec::new();
        for bucket in 0..10 {
            many.extend([bucket, bucket + 1, bucket + 2, 0]);
        }
        accumulate_counts_s32(&mut many, 36, 3, 4, 10);
        assert_eq!(&many[36..39], &[45, 55, 65]);

        let t = [1, SAINT_MIN | 2, 0];
        let mut compacted_buckets = vec![0; 6];
        count_compacted_lms_suffixes_32s_2k(&t, t.len() as SaSint, 3, &mut compacted_buckets);
        assert_eq!(compacted_buckets, vec![1, 0, 1, 0, 0, 1]);

        let unique_sa = [0, 2, 4, 6, 0, -10, 20, -30];
        assert_eq!(count_unique_suffixes(&unique_sa, 4, 0, 4), 2);

        assert_eq!(get_bucket_stride(20_000, 1000, 4), 1024);
        assert_eq!(get_bucket_stride(3024, 1001, 4), 1008);
        assert_eq!(get_bucket_stride(3000, 1001, 4), 1001);
    }

    #[test]
    fn libsais16_count_gather_lms_suffixes_32s_omp_wrappers_match_c() {
        let t = vec![2, 1, 3, 1, 2, 0, 1, 0];
        let n = t.len() as SaSint;
        let k = 4;
        let mut rust_sa = vec![0; t.len()];
        let mut c_sa = rust_sa.clone();
        let mut rust_buckets = vec![0; 2 * k as usize];
        let mut c_buckets = rust_buckets.clone();
        let mut rust_state = alloc_thread_state(1).unwrap();
        let rust_m = count_and_gather_lms_suffixes_32s_2k_omp(
            &t,
            &mut rust_sa,
            n,
            k,
            &mut rust_buckets,
            0,
            1,
            &mut rust_state,
        );
        let c_m = unsafe {
            probe_libsais16_count_and_gather_lms_suffixes_32s_2k_omp(
                t.as_ptr(),
                c_sa.as_mut_ptr(),
                n,
                k,
                c_buckets.as_mut_ptr(),
                0,
                1,
            )
        };
        assert_eq!(rust_m, c_m);
        assert_eq!(rust_sa, c_sa);
        assert_eq!(rust_buckets, c_buckets);

        let compact_t = vec![2, SAINT_MIN | 1, 3, 1, SAINT_MIN | 2, 0, 1, 0];
        let mut rust_sa = vec![0; compact_t.len()];
        let mut c_sa = rust_sa.clone();
        let mut rust_buckets = vec![0; 2 * k as usize];
        let mut c_buckets = rust_buckets.clone();
        let mut rust_state = alloc_thread_state(1).unwrap();
        count_and_gather_compacted_lms_suffixes_32s_2k_omp(
            &compact_t,
            &mut rust_sa,
            n,
            k,
            &mut rust_buckets,
            0,
            1,
            &mut rust_state,
        );
        unsafe {
            probe_libsais16_count_and_gather_compacted_lms_suffixes_32s_2k_omp(
                compact_t.as_ptr(),
                c_sa.as_mut_ptr(),
                n,
                k,
                c_buckets.as_mut_ptr(),
                0,
                1,
            );
        }
        assert_eq!(rust_sa, c_sa);
        assert_eq!(rust_buckets, c_buckets);
    }

    #[test]
    fn libsais16_count_gather_lms_suffixes_32s_4k_match_c() {
        let t = vec![2, 1, 3, 1, 2, 0, 1, 0];
        let n = t.len() as SaSint;
        let k = 4;

        let mut rust_buckets = vec![77; 4 * k as usize];
        let mut c_buckets = vec![0; 4 * k as usize];
        let mut c_sa_for_count = vec![0; t.len()];
        count_lms_suffixes_32s_4k(&t, n, k, &mut rust_buckets);
        unsafe {
            probe_libsais16_count_and_gather_lms_suffixes_32s_4k(
                t.as_ptr(),
                c_sa_for_count.as_mut_ptr(),
                n,
                k,
                c_buckets.as_mut_ptr(),
                0,
                n,
            );
        }
        assert_eq!(rust_buckets, c_buckets);

        let mut rust_sa = vec![0; t.len()];
        let mut c_sa = rust_sa.clone();
        let mut rust_buckets = vec![0; 4 * k as usize];
        let mut c_buckets = rust_buckets.clone();
        let rust_m = count_and_gather_lms_suffixes_32s_4k(
            &t,
            &mut rust_sa,
            n,
            k,
            &mut rust_buckets,
            0,
            n as isize,
        );
        let c_m = unsafe {
            probe_libsais16_count_and_gather_lms_suffixes_32s_4k(
                t.as_ptr(),
                c_sa.as_mut_ptr(),
                n,
                k,
                c_buckets.as_mut_ptr(),
                0,
                n,
            )
        };
        assert_eq!(rust_m, c_m);
        assert_eq!(rust_sa, c_sa);
        assert_eq!(rust_buckets, c_buckets);

        let mut rust_sa = vec![0; t.len()];
        let mut c_sa = rust_sa.clone();
        let mut rust_buckets = vec![0; 4 * k as usize];
        let mut c_buckets = rust_buckets.clone();
        let mut rust_state = alloc_thread_state(1).unwrap();
        let rust_m = count_and_gather_lms_suffixes_32s_4k_omp(
            &t,
            &mut rust_sa,
            n,
            k,
            &mut rust_buckets,
            0,
            1,
            &mut rust_state,
        );
        let c_m = unsafe {
            probe_libsais16_count_and_gather_lms_suffixes_32s_4k_omp(
                t.as_ptr(),
                c_sa.as_mut_ptr(),
                n,
                k,
                c_buckets.as_mut_ptr(),
                0,
                1,
            )
        };
        assert_eq!(rust_m, c_m);
        assert_eq!(rust_sa, c_sa);
        assert_eq!(rust_buckets, c_buckets);

        let mut rust_buckets = vec![91; k as usize];
        let mut c_buckets = rust_buckets.clone();
        count_suffixes_32s(&t, n, k, &mut rust_buckets);
        unsafe {
            probe_libsais16_count_suffixes_32s(t.as_ptr(), n, k, c_buckets.as_mut_ptr());
        }
        assert_eq!(rust_buckets, c_buckets);
    }

    #[test]
    fn libsais16_initialize_buckets_32s_match_c() {
        let k = 4;

        let base_6k = vec![
            1, 2, 0, 1, 0, 1, 2, 0, 3, 0, 1, 1, 2, 1, 0, 0, 9, 9, 9, 9, 8, 8, 8, 8,
        ];
        let mut rust = base_6k.clone();
        let mut c = base_6k.clone();
        initialize_buckets_start_and_end_32s_6k(k, &mut rust);
        unsafe { probe_libsais16_initialize_buckets_start_and_end_32s_6k(k, c.as_mut_ptr()) };
        assert_eq!(rust, c);

        let base_4k = vec![1, 2, 0, 1, 3, 0, 2, 1, 9, 9, 9, 9, 8, 8, 8, 8];
        let mut rust = base_4k.clone();
        let mut c = base_4k.clone();
        initialize_buckets_start_and_end_32s_4k(k, &mut rust);
        unsafe { probe_libsais16_initialize_buckets_start_and_end_32s_4k(k, c.as_mut_ptr()) };
        assert_eq!(rust, c);

        let base_2k = vec![1, 2, 0, 1, 3, 0, 2, 1];
        let mut rust = base_2k.clone();
        let mut c = base_2k.clone();
        initialize_buckets_end_32s_2k(k, &mut rust);
        unsafe { probe_libsais16_initialize_buckets_end_32s_2k(k, c.as_mut_ptr()) };
        assert_eq!(rust, c);

        let mut rust = base_2k.clone();
        let mut c = base_2k.clone();
        initialize_buckets_start_and_end_32s_2k(k, &mut rust);
        unsafe { probe_libsais16_initialize_buckets_start_and_end_32s_2k(k, c.as_mut_ptr()) };
        assert_eq!(rust, c);

        let base_1k = vec![2, 1, 3, 2];
        let mut rust = base_1k.clone();
        let mut c = base_1k.clone();
        initialize_buckets_start_32s_1k(k, &mut rust);
        unsafe { probe_libsais16_initialize_buckets_start_32s_1k(k, c.as_mut_ptr()) };
        assert_eq!(rust, c);

        let mut rust = base_1k.clone();
        let mut c = base_1k.clone();
        initialize_buckets_end_32s_1k(k, &mut rust);
        unsafe { probe_libsais16_initialize_buckets_end_32s_1k(k, c.as_mut_ptr()) };
        assert_eq!(rust, c);

        let t = vec![2, 1, 3, 1, 2, 0, 1, 0];
        let mut rust = vec![1, 2, 0, 1, 3, 0, 2, 1];
        let mut c = rust.clone();
        initialize_buckets_for_lms_suffixes_radix_sort_32s_2k(&t, k, &mut rust, 4);
        unsafe {
            probe_libsais16_initialize_buckets_for_lms_suffixes_radix_sort_32s_2k(
                t.as_ptr(),
                k,
                c.as_mut_ptr(),
                4,
            );
        }
        assert_eq!(rust, c);

        let mut rust = vec![
            1, 2, 0, 1, 3, 0, 2, 1, 1, 0, 2, 0, 0, 1, 1, 0, 9, 9, 9, 9, 8, 8, 8, 8,
        ];
        let mut c = rust.clone();
        let rust_sum = initialize_buckets_for_lms_suffixes_radix_sort_32s_6k(&t, k, &mut rust, 4);
        let c_sum = unsafe {
            probe_libsais16_initialize_buckets_for_lms_suffixes_radix_sort_32s_6k(
                t.as_ptr(),
                k,
                c.as_mut_ptr(),
                4,
            )
        };
        assert_eq!(rust_sum, c_sum);
        assert_eq!(rust, c);

        let mut rust = base_4k.clone();
        let mut c = base_4k;
        initialize_buckets_for_radix_and_partial_sorting_32s_4k(&t, k, &mut rust, 4);
        unsafe {
            probe_libsais16_initialize_buckets_for_radix_and_partial_sorting_32s_4k(
                t.as_ptr(),
                k,
                c.as_mut_ptr(),
                4,
            );
        }
        assert_eq!(rust, c);
    }

    #[test]
    fn libsais16_place_lms_suffixes_interval_32s_match_c() {
        let n = 12;
        let k = 4;
        let m = 4;

        let mut rust_sa = vec![101, 102, 103, 104, 9, 9, 9, 9, 9, 9, 9, 9];
        let mut c_sa = rust_sa.clone();
        let mut buckets = vec![0; 4 * k as usize];
        buckets[buckets_index2(0, 1)] = 2;
        buckets[buckets_index2(1, 1)] = 2;
        buckets[buckets_index2(2, 1)] = 3;
        buckets[buckets_index2(2, 1) + buckets_index2(1, 0)] = 4;
        buckets[3 * k as usize + 1] = 7;
        buckets[3 * k as usize + 2] = 10;
        place_lms_suffixes_interval_32s_4k(&mut rust_sa, n, k, m, &buckets);
        unsafe {
            probe_libsais16_place_lms_suffixes_interval_32s_4k(
                c_sa.as_mut_ptr(),
                n,
                k,
                m,
                buckets.as_ptr(),
            );
        }
        assert_eq!(rust_sa, c_sa);

        let mut rust_sa = vec![101, 102, 103, 104, 9, 9, 9, 9, 9, 9, 9, 9];
        let mut c_sa = rust_sa.clone();
        let mut buckets = vec![0; 2 * k as usize];
        buckets[buckets_index2(1, 0)] = 7;
        buckets[buckets_index2(0, 1)] = 1;
        buckets[buckets_index2(1, 1)] = 1;
        buckets[buckets_index2(2, 0)] = 10;
        buckets[buckets_index2(2, 1)] = 2;
        buckets[buckets_index2(3, 1)] = 3;
        place_lms_suffixes_interval_32s_2k(&mut rust_sa, n, k, m, &buckets);
        unsafe {
            probe_libsais16_place_lms_suffixes_interval_32s_2k(
                c_sa.as_mut_ptr(),
                n,
                k,
                m,
                buckets.as_ptr(),
            );
        }
        assert_eq!(rust_sa, c_sa);

        let t = vec![0, 1, 2, 1, 2, 3, 1, 3, 0, 0, 0, 0];
        let mut rust_sa = vec![1, 3, 4, 7, 9, 9, 9, 9, 9, 9, 9, 9];
        let mut c_sa = rust_sa.clone();
        let rust_buckets = vec![0, 3, 6, 10];
        let mut c_buckets = rust_buckets.clone();
        place_lms_suffixes_interval_32s_1k(&t, &mut rust_sa, k, m, &rust_buckets);
        unsafe {
            probe_libsais16_place_lms_suffixes_interval_32s_1k(
                t.as_ptr(),
                c_sa.as_mut_ptr(),
                k,
                m,
                c_buckets.as_mut_ptr(),
            );
        }
        assert_eq!(rust_sa, c_sa);
        assert_eq!(rust_buckets, c_buckets);
    }

    #[test]
    fn libsais16_renumber_and_mark_distinct_lms_suffixes_32s_1k_matches_c() {
        let rust_t = vec![2, 1, 3, 1, 2, 0, 1, 0];
        let n = rust_t.len() as SaSint;
        let mut probe_sa = vec![0; rust_t.len()];
        let m = gather_lms_suffixes_32s(&rust_t, &mut probe_sa, n);
        let mut rust_sa = vec![0; rust_t.len()];
        let mut c_t = rust_t.clone();
        let mut c_sa = rust_sa.clone();

        let rust_name =
            renumber_and_mark_distinct_lms_suffixes_32s_1k_omp(&rust_t, &mut rust_sa, n, m, 1);
        let c_name = unsafe {
            probe_libsais16_renumber_and_mark_distinct_lms_suffixes_32s_1k_omp(
                c_t.as_mut_ptr(),
                c_sa.as_mut_ptr(),
                n,
                m,
                1,
            )
        };
        assert_eq!(rust_name, c_name);
        assert_eq!(rust_t, c_t);
        assert_eq!(rust_sa, c_sa);
    }

    #[test]
    fn libsais16_reconstruct_compacted_lms_suffixes_32s_match_c() {
        let n = 8;
        let k = 4;
        let fs = 0;
        let f = 0;
        let mut m_probe_sa = vec![0; n as usize];
        let m = gather_lms_suffixes_32s(&[2, 1, 3, 1, 2, 0, 1, 0], &mut m_probe_sa, n);

        let mut rust_t = vec![2, 1, 3, 1, 2, 0, 1, 0];
        let mut c_t = rust_t.clone();
        let mut rust_sa = vec![0; n as usize];
        let mut c_sa = rust_sa.clone();
        let mut rust_buckets = vec![0; 2 * k as usize];
        let mut c_buckets = rust_buckets.clone();
        let mut rust_thread_state = alloc_thread_state(1).unwrap();
        reconstruct_compacted_lms_suffixes_32s_2k_omp(
            &mut rust_t,
            &mut rust_sa,
            n,
            k,
            m,
            fs,
            f,
            &mut rust_buckets,
            0,
            1,
            &mut rust_thread_state,
        );
        unsafe {
            probe_libsais16_reconstruct_compacted_lms_suffixes_32s_2k_omp(
                c_t.as_mut_ptr(),
                c_sa.as_mut_ptr(),
                n,
                k,
                m,
                fs,
                f,
                c_buckets.as_mut_ptr(),
                0,
                1,
            );
        }
        assert_eq!(rust_t, c_t);
        assert_eq!(rust_sa, c_sa);
        assert_eq!(rust_buckets, c_buckets);

        let mut rust_t = vec![2, 1, 3, 1, 2, 0, 1, 0];
        let mut c_t = rust_t.clone();
        let mut rust_sa = vec![0; n as usize];
        let mut c_sa = rust_sa.clone();
        reconstruct_compacted_lms_suffixes_32s_1k_omp(&mut rust_t, &mut rust_sa, n, m, fs, f, 1);
        unsafe {
            probe_libsais16_reconstruct_compacted_lms_suffixes_32s_1k_omp(
                c_t.as_mut_ptr(),
                c_sa.as_mut_ptr(),
                n,
                m,
                fs,
                f,
                1,
            );
        }
        assert_eq!(rust_t, c_t);
        assert_eq!(rust_sa, c_sa);
    }

    #[test]
    fn libsais16_partial_omp_wrappers_match_c() {
        let (text, mut rust_sa, mut rust_buckets) = partial_scan_fixture();
        let mut c_sa = rust_sa.clone();
        let mut c_buckets = rust_buckets.clone();

        let rust_d = partial_sorting_scan_left_to_right_16u_omp(
            &text,
            &mut rust_sa,
            text.len() as SaSint,
            8,
            &mut rust_buckets,
            5,
            3,
            1,
        );
        let c_d = unsafe {
            probe_libsais16_partial_sorting_scan_left_to_right_16u_omp(
                text.as_ptr(),
                c_sa.as_mut_ptr(),
                text.len() as SaSint,
                8,
                c_buckets.as_mut_ptr(),
                5,
                3,
                1,
            )
        };
        assert_eq!(rust_d, c_d);
        assert_eq!(rust_sa, c_sa);
        assert_eq!(rust_buckets, c_buckets);

        let (text, mut rust_sa, mut rust_buckets) = partial_scan_fixture();
        rust_sa[6..10].copy_from_slice(&[3, 5 | SAINT_MIN, 7, 9 | SAINT_MIN]);
        let mut c_sa = rust_sa.clone();
        let mut c_buckets = rust_buckets.clone();
        partial_sorting_scan_right_to_left_16u_omp(
            &text,
            &mut rust_sa,
            text.len() as SaSint,
            8,
            &mut rust_buckets,
            0,
            5,
            3,
            1,
        );
        unsafe {
            probe_libsais16_partial_sorting_scan_right_to_left_16u_omp(
                text.as_ptr(),
                c_sa.as_mut_ptr(),
                text.len() as SaSint,
                8,
                c_buckets.as_mut_ptr(),
                0,
                5,
                3,
                1,
            );
        }
        assert_eq!(rust_sa, c_sa);
        assert_eq!(rust_buckets, c_buckets);

        let (text, mut rust_sa, mut rust_buckets) = partial_scan_fixture();
        rust_sa[6..10].copy_from_slice(&[3, 5 | SAINT_MIN, 7, 9 | SAINT_MIN]);
        let mut c_sa = rust_sa.clone();
        let mut c_buckets = rust_buckets.clone();
        partial_gsa_scan_right_to_left_16u_omp(
            &text,
            &mut rust_sa,
            text.len() as SaSint,
            8,
            &mut rust_buckets,
            0,
            5,
            3,
            1,
        );
        unsafe {
            probe_libsais16_partial_gsa_scan_right_to_left_16u_omp(
                text.as_ptr(),
                c_sa.as_mut_ptr(),
                text.len() as SaSint,
                8,
                c_buckets.as_mut_ptr(),
                0,
                5,
                3,
                1,
            );
        }
        assert_eq!(rust_sa, c_sa);
        assert_eq!(rust_buckets, c_buckets);
    }

    #[test]
    fn libsais16_final_omp_wrappers_match_c() {
        let (text, mut rust_sa, mut rust_bucket) = final_scan_fixture();
        let mut c_sa = rust_sa.clone();
        let mut c_bucket = rust_bucket.clone();
        final_bwt_scan_left_to_right_16u_omp(
            &text,
            &mut rust_sa,
            text.len() as SaSint,
            8,
            &mut rust_bucket,
            1,
        );
        unsafe {
            probe_libsais16_final_bwt_scan_left_to_right_16u_omp(
                text.as_ptr(),
                c_sa.as_mut_ptr(),
                text.len() as SaSint,
                8,
                c_bucket.as_mut_ptr(),
                1,
            );
        }
        assert_eq!(rust_sa, c_sa);
        assert_eq!(rust_bucket, c_bucket);

        let (text, mut rust_sa, mut rust_bucket) = final_scan_fixture();
        let mut c_sa = rust_sa.clone();
        let mut c_bucket = rust_bucket.clone();
        let mut rust_i = vec![-1; 8];
        let mut c_i = rust_i.clone();
        final_bwt_aux_scan_left_to_right_16u_omp(
            &text,
            &mut rust_sa,
            text.len() as SaSint,
            8,
            1,
            &mut rust_i,
            &mut rust_bucket,
            1,
        );
        unsafe {
            probe_libsais16_final_bwt_aux_scan_left_to_right_16u_omp(
                text.as_ptr(),
                c_sa.as_mut_ptr(),
                text.len() as SaSint,
                8,
                1,
                c_i.as_mut_ptr(),
                c_bucket.as_mut_ptr(),
                1,
            );
        }
        assert_eq!(rust_sa, c_sa);
        assert_eq!(rust_bucket, c_bucket);
        assert_eq!(rust_i, c_i);

        let (text, mut rust_sa, mut rust_bucket) = final_scan_fixture();
        let mut c_sa = rust_sa.clone();
        let mut c_bucket = rust_bucket.clone();
        final_sorting_scan_left_to_right_16u_omp(
            &text,
            &mut rust_sa,
            text.len() as SaSint,
            8,
            &mut rust_bucket,
            1,
        );
        unsafe {
            probe_libsais16_final_sorting_scan_left_to_right_16u_omp(
                text.as_ptr(),
                c_sa.as_mut_ptr(),
                text.len() as SaSint,
                8,
                c_bucket.as_mut_ptr(),
                1,
            );
        }
        assert_eq!(rust_sa, c_sa);
        assert_eq!(rust_bucket, c_bucket);

        let (text, mut rust_sa, mut rust_bucket) = final_scan_fixture();
        let mut c_sa = rust_sa.clone();
        let mut c_bucket = rust_bucket.clone();
        let rust_index = final_bwt_scan_right_to_left_16u_omp(
            &text,
            &mut rust_sa,
            text.len() as SaSint,
            8,
            &mut rust_bucket,
            1,
        );
        let c_index = unsafe {
            probe_libsais16_final_bwt_scan_right_to_left_16u_omp(
                text.as_ptr(),
                c_sa.as_mut_ptr(),
                text.len() as SaSint,
                8,
                c_bucket.as_mut_ptr(),
                1,
            )
        };
        assert_eq!(rust_index, c_index);
        assert_eq!(rust_sa, c_sa);
        assert_eq!(rust_bucket, c_bucket);

        let (text, mut rust_sa, mut rust_bucket) = final_scan_fixture();
        let mut c_sa = rust_sa.clone();
        let mut c_bucket = rust_bucket.clone();
        let mut rust_i = vec![-1; 8];
        let mut c_i = rust_i.clone();
        final_bwt_aux_scan_right_to_left_16u_omp(
            &text,
            &mut rust_sa,
            text.len() as SaSint,
            8,
            1,
            &mut rust_i,
            &mut rust_bucket,
            1,
        );
        unsafe {
            probe_libsais16_final_bwt_aux_scan_right_to_left_16u_omp(
                text.as_ptr(),
                c_sa.as_mut_ptr(),
                text.len() as SaSint,
                8,
                1,
                c_i.as_mut_ptr(),
                c_bucket.as_mut_ptr(),
                1,
            );
        }
        assert_eq!(rust_sa, c_sa);
        assert_eq!(rust_bucket, c_bucket);
        assert_eq!(rust_i, c_i);

        let (text, mut rust_sa, mut rust_bucket) = final_scan_fixture();
        let mut c_sa = rust_sa.clone();
        let mut c_bucket = rust_bucket.clone();
        final_sorting_scan_right_to_left_16u_omp(&text, &mut rust_sa, 0, 6, 8, &mut rust_bucket, 1);
        unsafe {
            probe_libsais16_final_sorting_scan_right_to_left_16u_omp(
                text.as_ptr(),
                c_sa.as_mut_ptr(),
                0,
                6,
                8,
                c_bucket.as_mut_ptr(),
                1,
            );
        }
        assert_eq!(rust_sa, c_sa);
        assert_eq!(rust_bucket, c_bucket);

        let (text, mut rust_sa, mut rust_bucket) = final_scan_fixture();
        let mut c_sa = rust_sa.clone();
        let mut c_bucket = rust_bucket.clone();
        final_gsa_scan_right_to_left_16u_omp(&text, &mut rust_sa, 0, 6, 8, &mut rust_bucket, 1);
        unsafe {
            probe_libsais16_final_gsa_scan_right_to_left_16u_omp(
                text.as_ptr(),
                c_sa.as_mut_ptr(),
                0,
                6,
                8,
                c_bucket.as_mut_ptr(),
                1,
            );
        }
        assert_eq!(rust_sa, c_sa);
        assert_eq!(rust_bucket, c_bucket);
    }

    #[test]
    fn libsais16_matches_bruteforce() {
        let t = [3, 1, 4, 1, 5, 9, 0, 2];
        let mut sa = vec![0; t.len()];
        let mut freq = vec![0; ALPHABET_SIZE];
        assert_eq!(libsais16(&t, &mut sa, 0, Some(&mut freq)), 0);
        assert_eq!(sa, brute_sa(&t));
        assert_eq!(freq[1], 2);
        assert_eq!(freq[9], 1);
    }

    #[test]
    fn libsais16_bwt_round_trips() {
        let t = [2, 1, 3, 1, 2, 4, 1, 0];
        let mut bwt = vec![0; t.len()];
        let mut work = vec![0; t.len()];
        let primary = libsais16_bwt(&t, &mut bwt, &mut work, 0, None);
        assert!(primary > 0);

        let mut restored = vec![0; t.len()];
        assert_eq!(
            libsais16_unbwt(&bwt, &mut restored, &mut work, None, primary),
            0
        );
        assert_eq!(restored, t);
    }

    #[test]
    fn libsais16_plcp_lcp_are_consistent() {
        let t = [2, 1, 2, 1, 0];
        let sa = brute_sa(&t);
        let mut plcp = vec![0; t.len()];
        let mut lcp = vec![0; t.len()];
        assert_eq!(libsais16_plcp(&t, &sa, &mut plcp), 0);
        assert_eq!(libsais16_lcp(&plcp, &sa, &mut lcp), 0);
        assert_eq!(lcp[0], 0);

        let mut named_plcp = vec![0; t.len()];
        assert_eq!(
            compute_phi_omp(&sa, &mut named_plcp, t.len() as SaSint, 1),
            0
        );
        assert_eq!(
            compute_plcp_omp(&t, &mut named_plcp, t.len() as SaSint, 1),
            0
        );
        assert_eq!(named_plcp, plcp);

        let mut named_lcp = vec![0; t.len()];
        assert_eq!(
            compute_lcp_omp(&named_plcp, &sa, &mut named_lcp, t.len() as SaSint, 1),
            0
        );
        assert_eq!(named_lcp, lcp);

        let mut gsa_plcp = vec![0; t.len()];
        let mut named_gsa_plcp = vec![0; t.len()];
        assert_eq!(libsais16_plcp_gsa(&t, &sa, &mut gsa_plcp), 0);
        assert_eq!(
            compute_phi_omp(&sa, &mut named_gsa_plcp, t.len() as SaSint, 1),
            0
        );
        assert_eq!(
            compute_plcp_gsa_omp(&t, &mut named_gsa_plcp, t.len() as SaSint, 1),
            0
        );
        assert_eq!(named_gsa_plcp, gsa_plcp);
    }

    #[test]
    fn libsais16_bwt_copy_16u_omp_uses_block_partition_for_large_inputs() {
        let n = 65_600usize;
        let a: Vec<SaSint> = (0..n).map(|i| (i * 17) as SaSint).collect();
        let mut threaded = vec![0; n];
        let mut sequential = vec![0; n];

        bwt_copy_16u_omp(&mut threaded, &a, n as SaSint, 4);
        bwt_copy_16u(&mut sequential, &a, n as SaSint);

        assert_eq!(threaded, sequential);
    }

    #[test]
    fn libsais16_plcp_lcp_omp_wrappers_match_single_thread_on_large_inputs() {
        let n = 65_600usize;
        let text: Vec<u16> = (0..n).map(|i| 1 + (i % 251) as u16).collect();
        let sa: Vec<SaSint> = (0..n as SaSint).collect();

        let mut plcp_single = vec![0; n];
        let mut plcp_threaded = vec![0; n];
        assert_eq!(compute_phi_omp(&sa, &mut plcp_single, n as SaSint, 1), 0);
        assert_eq!(compute_phi_omp(&sa, &mut plcp_threaded, n as SaSint, 4), 0);
        assert_eq!(plcp_threaded, plcp_single);

        assert_eq!(compute_plcp_omp(&text, &mut plcp_single, n as SaSint, 1), 0);
        assert_eq!(
            compute_plcp_omp(&text, &mut plcp_threaded, n as SaSint, 4),
            0
        );
        assert_eq!(plcp_threaded, plcp_single);

        let mut lcp_single = vec![0; n];
        let mut lcp_threaded = vec![0; n];
        assert_eq!(
            compute_lcp_omp(&plcp_single, &sa, &mut lcp_single, n as SaSint, 1),
            0
        );
        assert_eq!(
            compute_lcp_omp(&plcp_threaded, &sa, &mut lcp_threaded, n as SaSint, 4),
            0
        );
        assert_eq!(lcp_threaded, lcp_single);
    }

    #[test]
    fn libsais16_context_allocates_upstream_shaped_buffers() {
        let ctx = create_ctx().unwrap();
        assert_eq!(ctx.threads, 1);
        assert_eq!(ctx.buckets.len(), 8 * ALPHABET_SIZE);
        assert!(ctx.thread_state.is_none());

        let ctx = create_ctx_omp(2).unwrap();
        assert_eq!(ctx.threads, 2);
        assert_eq!(ctx.buckets.len(), 8 * ALPHABET_SIZE);
        let thread_state = ctx.thread_state.as_ref().unwrap();
        assert_eq!(thread_state.len(), 2);
        assert_eq!(thread_state[0].buckets.len(), 4 * ALPHABET_SIZE);
        assert_eq!(thread_state[0].cache_entries, PER_THREAD_CACHE_SIZE);

        let ctx = create_ctx_omp(0).unwrap();
        assert_eq!(ctx.threads, 1);
        assert!(ctx.thread_state.is_none());
    }

    #[test]
    fn libsais16_unbwt_context_allocates_upstream_shaped_buffers() {
        let ctx = unbwt_create_ctx().unwrap();
        assert_eq!(ctx.threads, 1);
        assert_eq!(ctx.bucket2.len(), ALPHABET_SIZE);
        assert_eq!(ctx.fastbits.len(), 1 + (1 << UNBWT_FASTBITS));
        assert!(ctx.buckets.is_none());

        let ctx = unbwt_create_ctx_omp(3).unwrap();
        assert_eq!(ctx.threads, 3);
        assert_eq!(ctx.bucket2.len(), ALPHABET_SIZE);
        assert_eq!(ctx.fastbits.len(), 1 + (1 << UNBWT_FASTBITS));
        assert_eq!(ctx.buckets.as_ref().unwrap().len(), 3 * ALPHABET_SIZE);
    }

    #[test]
    fn libsais16_named_unbwt_helpers_follow_decode_shapes() {
        let t = [0, 1, 2];
        let mut p = vec![usize::MAX; 4];
        let mut bucket2 = vec![0; ALPHABET_SIZE];
        bucket2[0] = 1;
        bucket2[1] = 2;
        bucket2[2] = 3;
        unbwt_calculate_P(&t, &mut p, &mut bucket2, 2, 1, 3);
        assert_eq!(p[2], 1);
        assert_eq!(p[3], 3);

        let p = [1usize, 2, 0];
        let mut bucket2 = vec![3; ALPHABET_SIZE];
        bucket2[0] = 1;
        bucket2[1] = 2;
        bucket2[2] = 3;
        let fastbits = vec![0; 3];

        let mut u = vec![99; 3];
        let mut i0 = 0;
        unbwt_decode_1(&mut u, &p, &bucket2, &fastbits, 0, &mut i0, 3);
        assert_eq!(u, vec![0, 1, 2]);
        assert_eq!(i0, 0);

        let mut u = vec![99; 6];
        let (mut i0, mut i1) = (0, 1);
        unbwt_decode_2(&mut u, &p, &bucket2, &fastbits, 0, 3, &mut i0, &mut i1, 2);
        assert_eq!(&u[..2], &[0, 1]);
        assert_eq!(&u[3..5], &[1, 2]);
        assert_eq!((i0, i1), (2, 0));

        let mut u = vec![99; 8];
        let mut cursors = [0; 8];
        unbwt_decode_8(&mut u, &p, &bucket2, &fastbits, 0, 1, &mut cursors, 1);
        assert_eq!(u, vec![0; 8]);
        assert_eq!(cursors, [1; 8]);
    }

    #[test]
    fn libsais16_unbwt_init_parallel_uses_block_partition() {
        let n = 70_003usize;
        let t: Vec<u16> = (0..n)
            .map(|i| ((i.wrapping_mul(37).wrapping_add(i >> 3)) % 251) as u16)
            .collect();
        let i = [12_345];

        let mut single_p = vec![0; n + 1];
        let mut threaded_p = vec![0; n + 1];
        let mut single_bucket2 = vec![0; ALPHABET_SIZE];
        let mut threaded_bucket2 = vec![0; ALPHABET_SIZE];
        let mut single_fastbits = vec![0; 1 + (1 << UNBWT_FASTBITS)];
        let mut threaded_fastbits = vec![0; 1 + (1 << UNBWT_FASTBITS)];
        let mut buckets = vec![0; 4 * ALPHABET_SIZE];

        unbwt_init_single(
            &t,
            &mut single_p,
            None,
            &i,
            &mut single_bucket2,
            &mut single_fastbits,
        );
        unbwt_init_parallel(
            &t,
            &mut threaded_p,
            None,
            &i,
            &mut threaded_bucket2,
            &mut threaded_fastbits,
            &mut buckets,
            4,
        );

        assert_eq!(threaded_p, single_p);
        assert_eq!(threaded_bucket2, single_bucket2);
        assert_eq!(threaded_fastbits, single_fastbits);
    }

    fn assert_libsais16_matches_c(text: &[u16]) {
        let mut rust_sa = vec![0; text.len()];
        let mut c_sa = vec![0; text.len()];

        let rust_rc = libsais16(text, &mut rust_sa, 0, None);
        let c_rc = unsafe {
            probe_public_libsais16(text.as_ptr(), c_sa.as_mut_ptr(), text.len() as SaSint, 0)
        };

        assert_eq!(rust_rc, c_rc);
        assert_eq!(rust_sa, c_sa);
    }

    fn assert_libsais16_gsa_matches_c(text: &[u16]) {
        let mut rust_sa = vec![0; text.len()];
        let mut c_sa = vec![0; text.len()];

        let rust_rc = libsais16_gsa(text, &mut rust_sa, 0, None);
        let c_rc = unsafe {
            probe_public_libsais16_gsa(text.as_ptr(), c_sa.as_mut_ptr(), text.len() as SaSint, 0)
        };

        assert_eq!(rust_rc, c_rc);
        assert_eq!(rust_sa, c_sa);
    }

    fn assert_libsais16_int_matches_c(text: &[SaSint], k: SaSint) {
        let mut rust_t = text.to_vec();
        let mut c_t = text.to_vec();
        let mut rust_sa = vec![0; text.len()];
        let mut c_sa = vec![0; text.len()];

        let rust_rc = libsais16_int(&mut rust_t, &mut rust_sa, k, 0);
        let c_rc = unsafe {
            probe_public_libsais16_int(
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

    fn assert_libsais16_bwt_matches_c(text: &[u16]) {
        let mut rust_u = vec![0; text.len()];
        let mut rust_a = vec![0; text.len()];
        let mut c_u = vec![0; text.len()];
        let mut c_a = vec![0; text.len()];

        let rust_rc = libsais16_bwt(text, &mut rust_u, &mut rust_a, 0, None);
        let c_rc = unsafe {
            probe_public_libsais16_bwt(
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

    fn assert_libsais16_bwt_aux_matches_c(text: &[u16], r: SaSint) {
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

        let rust_rc = libsais16_bwt_aux(text, &mut rust_u, &mut rust_a, 0, None, r, &mut rust_i);
        let c_rc = unsafe {
            probe_public_libsais16_bwt_aux(
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

    fn assert_libsais16_freq_outputs_match_c(text: &[u16], gsa_text: &[u16]) {
        let mut rust_sa = vec![0; text.len()];
        let mut c_sa = vec![0; text.len()];
        let mut rust_freq = vec![-1; ALPHABET_SIZE];
        let mut c_freq = vec![-1; ALPHABET_SIZE];

        let rust_rc = libsais16(text, &mut rust_sa, 0, Some(&mut rust_freq));
        let c_rc = unsafe {
            probe_public_libsais16_freq(
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
        let rust_rc = libsais16_gsa(gsa_text, &mut rust_gsa, 0, Some(&mut rust_freq));
        let c_rc = unsafe {
            probe_public_libsais16_gsa_freq(
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
        let rust_rc = libsais16_bwt(text, &mut rust_u, &mut rust_a, 0, Some(&mut rust_freq));
        let c_rc = unsafe {
            probe_public_libsais16_bwt_freq(
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
        let rust_rc = libsais16_bwt_aux(
            text,
            &mut rust_u,
            &mut rust_a,
            0,
            Some(&mut rust_freq),
            r,
            &mut rust_i,
        );
        let c_rc = unsafe {
            probe_public_libsais16_bwt_aux_freq(
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

    fn assert_libsais16_unbwt_matches_c(text: &[u16]) {
        let mut bwt = vec![0; text.len()];
        let mut work = vec![0; text.len()];
        let primary = libsais16_bwt(text, &mut bwt, &mut work, 0, None);
        assert!(primary >= 0);

        let mut rust_u = vec![0; text.len()];
        let mut rust_a = vec![0; text.len() + 1];
        let mut c_u = vec![0; text.len()];
        let mut c_a = vec![0; text.len() + 1];

        let rust_rc = libsais16_unbwt(&bwt, &mut rust_u, &mut rust_a, None, primary);
        let c_rc = unsafe {
            probe_public_libsais16_unbwt(
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

    fn assert_libsais16_unbwt_aux_matches_c(text: &[u16], r: SaSint) {
        let mut bwt = vec![0; text.len()];
        let mut work = vec![0; text.len()];
        let mut aux = vec![0; (text.len() - 1) / r as usize + 1];
        let bwt_rc = libsais16_bwt_aux(text, &mut bwt, &mut work, 0, None, r, &mut aux);
        assert_eq!(bwt_rc, 0);

        let mut rust_u = vec![0; text.len()];
        let mut rust_a = vec![0; text.len() + 1];
        let mut c_u = vec![0; text.len()];
        let mut c_a = vec![0; text.len() + 1];

        let rust_rc = libsais16_unbwt_aux(&bwt, &mut rust_u, &mut rust_a, None, r, &aux);
        let c_rc = unsafe {
            probe_public_libsais16_unbwt_aux(
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

    fn assert_libsais16_unbwt_freq_matches_c(text: &[u16]) {
        let mut freq = vec![0; ALPHABET_SIZE];
        let mut bwt = vec![0; text.len()];
        let mut work = vec![0; text.len()];
        let primary = libsais16_bwt(text, &mut bwt, &mut work, 0, Some(&mut freq));
        assert!(primary >= 0);

        let mut rust_u = vec![0; text.len()];
        let mut rust_a = vec![0; text.len() + 1];
        let mut c_u = vec![0; text.len()];
        let mut c_a = vec![0; text.len() + 1];

        let rust_rc = libsais16_unbwt(&bwt, &mut rust_u, &mut rust_a, Some(&freq), primary);
        let c_rc = unsafe {
            probe_public_libsais16_unbwt_freq(
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
        let bwt_rc = libsais16_bwt_aux(text, &mut bwt, &mut work, 0, Some(&mut freq), r, &mut aux);
        assert_eq!(bwt_rc, 0);

        rust_u.fill(0);
        rust_a.fill(0);
        c_u.fill(0);
        c_a.fill(0);
        let rust_rc = libsais16_unbwt_aux(&bwt, &mut rust_u, &mut rust_a, Some(&freq), r, &aux);
        let c_rc = unsafe {
            probe_public_libsais16_unbwt_aux_freq(
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

    fn assert_libsais16_plcp_lcp_matches_c(text: &[u16]) {
        let mut sa = vec![0; text.len()];
        assert_eq!(libsais16(text, &mut sa, 0, None), 0);

        let mut rust_plcp = vec![0; text.len()];
        let mut c_plcp = vec![0; text.len()];
        let rust_rc = libsais16_plcp(text, &sa, &mut rust_plcp);
        let c_rc = unsafe {
            probe_public_libsais16_plcp(
                text.as_ptr(),
                sa.as_ptr(),
                c_plcp.as_mut_ptr(),
                text.len() as SaSint,
            )
        };
        assert_eq!(rust_rc, c_rc);
        assert_eq!(rust_plcp, c_plcp);

        let mut rust_lcp = vec![0; text.len()];
        let mut c_lcp = vec![0; text.len()];
        let rust_rc = libsais16_lcp(&rust_plcp, &sa, &mut rust_lcp);
        let c_rc = unsafe {
            probe_public_libsais16_lcp(
                c_plcp.as_ptr(),
                sa.as_ptr(),
                c_lcp.as_mut_ptr(),
                text.len() as SaSint,
            )
        };
        assert_eq!(rust_rc, c_rc);
        assert_eq!(rust_lcp, c_lcp);
    }

    fn assert_libsais16_plcp_gsa_matches_c(text: &[u16]) {
        let mut sa = vec![0; text.len()];
        assert_eq!(libsais16_gsa(text, &mut sa, 0, None), 0);

        let mut rust_plcp = vec![0; text.len()];
        let mut c_plcp = vec![0; text.len()];
        let rust_rc = libsais16_plcp_gsa(text, &sa, &mut rust_plcp);
        let c_rc = unsafe {
            probe_public_libsais16_plcp_gsa(
                text.as_ptr(),
                sa.as_ptr(),
                c_plcp.as_mut_ptr(),
                text.len() as SaSint,
            )
        };
        assert_eq!(rust_rc, c_rc);
        assert_eq!(rust_plcp, c_plcp);
    }

    #[test]
    fn public_libsais16_matches_upstream_c() {
        for text in [
            [].as_slice(),
            &[1][..],
            &[2, 1, 3, 1, 2, 0],
            &[2, 1, 3, 1, 2, 4, 1, 0],
            &[65_535, 1, 65_534, 1, 0],
            &[7, 7, 7, 7, 7, 0],
        ] {
            assert_libsais16_matches_c(text);
        }
    }

    #[test]
    fn public_libsais16_bwt_matches_upstream_c() {
        for text in [
            [].as_slice(),
            &[1][..],
            &[2, 1, 3, 1, 2, 0],
            &[2, 1, 3, 1, 2, 4, 1, 0],
            &[65_535, 1, 65_534, 1, 0],
            &[7, 7, 7, 7, 7, 0],
        ] {
            assert_libsais16_bwt_matches_c(text);
        }
    }

    #[test]
    fn public_libsais16_gsa_matches_upstream_c() {
        for text in [&[0][..], &[2, 1, 0], &[2, 1, 0, 3, 1, 0], &[7, 7, 0, 7, 0]] {
            assert_libsais16_gsa_matches_c(text);
        }
    }

    #[test]
    fn public_libsais16_int_matches_upstream_c() {
        for (text, k) in [
            (&[][..], 0),
            (&[0][..], 1),
            (&[1, 2, 1, 0][..], 3),
            (&[2, 1, 2, 1, 0][..], 3),
            (&[3, 3, 3, 2, 1, 0][..], 4),
        ] {
            assert_libsais16_int_matches_c(text, k);
        }
    }

    #[test]
    fn public_libsais16_plcp_lcp_matches_upstream_c() {
        for text in [
            &[2, 1, 3, 1, 2, 0][..],
            &[2, 1, 3, 1, 2, 4, 1, 0],
            &[65_535, 1, 65_534, 1, 0],
            &[7, 7, 7, 7, 7, 0],
        ] {
            assert_libsais16_plcp_lcp_matches_c(text);
        }
    }

    #[test]
    fn public_libsais16_plcp_gsa_matches_upstream_c() {
        for text in [&[0][..], &[2, 1, 0], &[2, 1, 0, 3, 1, 0], &[7, 7, 0, 7, 0]] {
            assert_libsais16_plcp_gsa_matches_c(text);
        }
    }

    #[test]
    fn public_libsais16_bwt_aux_matches_upstream_c() {
        for text in [
            &[2, 1, 3, 1, 2, 0][..],
            &[2, 1, 3, 1, 2, 4, 1, 0],
            &[65_535, 1, 65_534, 1, 0],
            &[7, 7, 7, 7, 7, 0],
        ] {
            assert_libsais16_bwt_aux_matches_c(text, 4);
        }
    }

    #[test]
    fn public_libsais16_frequency_outputs_match_upstream_c() {
        assert_libsais16_freq_outputs_match_c(&[65_535, 1, 2, 1, 0], &[65_535, 1, 0, 2, 1, 0]);
    }

    #[test]
    fn public_libsais16_unbwt_with_frequency_matches_upstream_c() {
        assert_libsais16_unbwt_freq_matches_c(&[65_535, 1, 2, 1, 0]);
    }

    #[test]
    fn public_libsais16_unbwt_matches_upstream_c() {
        for text in [
            &[1][..],
            &[2, 1, 3, 1, 2, 0],
            &[2, 1, 3, 1, 2, 4, 1, 0],
            &[65_535, 1, 65_534, 1, 0],
            &[7, 7, 7, 7, 7, 0],
        ] {
            assert_libsais16_unbwt_matches_c(text);
        }
    }

    #[test]
    fn public_libsais16_unbwt_aux_matches_upstream_c() {
        for text in [
            &[2, 1, 3, 1, 2, 0][..],
            &[2, 1, 3, 1, 2, 4, 1, 0],
            &[65_535, 1, 65_534, 1, 0],
            &[7, 7, 7, 7, 7, 0],
        ] {
            assert_libsais16_unbwt_aux_matches_c(text, 4);
        }
    }

    #[test]
    fn public_libsais16_unbwt_aux_exercises_decode_dispatch_cases() {
        for len in [2usize, 5, 9, 13, 17, 21, 25, 29, 33, 37] {
            let text = (0..len)
                .map(|i| ((i * 37 + 11) % 65_535 + 1) as u16)
                .collect::<Vec<_>>();
            assert_libsais16_unbwt_aux_matches_c(&text, 4);
        }
    }

    #[test]
    fn libsais16_lcp_helpers_reject_invalid_suffix_entries() {
        let text = [2, 1, 2, 1, 0];
        let mut plcp = vec![0; text.len()];
        let mut lcp = vec![0; text.len()];

        assert_eq!(libsais16_plcp(&text, &[0, 1, -1, 3, 4], &mut plcp), -1);
        assert_eq!(libsais16_plcp(&text, &[0, 1, 2, 3, 5], &mut plcp), -1);
        assert_eq!(libsais16_lcp(&plcp, &[0, 1, -1, 3, 4], &mut lcp), -1);
        assert_eq!(libsais16_lcp(&plcp, &[0, 1, 2, 3, 5], &mut lcp), -1);
    }

    #[test]
    fn libsais16_rejects_invalid_public_arguments() {
        let text = [2, 1, 3, 1, 2, 0];
        let int_text = [1, 2, 1, 0];
        let mut int_text_for_short_sa = int_text.to_vec();
        let mut int_text_for_negative_fs = int_text.to_vec();
        let mut sa = vec![0; text.len() - 1];
        let mut int_sa = vec![0; int_text.len() - 1];
        let mut full_int_sa = vec![0; int_text.len()];
        let mut freq = vec![0; ALPHABET_SIZE - 1];
        let mut u = vec![0; text.len() - 1];
        let mut a = vec![0; text.len() - 1];
        let mut full_u = vec![0; text.len()];
        let mut full_a = vec![0; text.len()];
        let mut aux = vec![0; 1];

        assert_eq!(libsais16(&text, &mut sa, 0, None), -1);
        assert_eq!(libsais16(&text, &mut full_a, 0, Some(&mut freq)), -1);
        assert_eq!(libsais16_gsa(&[1, 2, 3], &mut full_a[..3], 0, None), -1);
        assert_eq!(
            libsais16_int(&mut int_text_for_short_sa, &mut int_sa, 3, 0),
            -1
        );
        assert_eq!(
            libsais16_int(&mut int_text_for_negative_fs, &mut full_int_sa, 3, -1),
            -1
        );
        assert_eq!(libsais16_bwt(&text, &mut u, &mut full_a, 0, None), -1);
        assert_eq!(libsais16_bwt(&text, &mut full_u, &mut a, 0, None), -1);
        assert_eq!(
            libsais16_bwt_aux(&text, &mut full_u, &mut full_a, 0, None, 0, &mut aux),
            -1
        );
        assert_eq!(
            libsais16_bwt_aux(&text, &mut full_u, &mut full_a, 0, None, 3, &mut aux),
            -1
        );
        assert_eq!(
            libsais16_bwt_aux(&text, &mut full_u, &mut full_a, 0, None, 4, &mut aux),
            -1
        );
        assert_eq!(create_ctx_omp(-1), None);
        assert_eq!(unbwt_create_ctx_omp(-1), None);
    }

    #[test]
    fn libsais16_unbwt_rejects_invalid_public_arguments() {
        let text = [2, 1, 3, 1, 2, 0];
        let mut bwt = vec![0; text.len()];
        let mut work = vec![0; text.len()];
        let primary = libsais16_bwt(&text, &mut bwt, &mut work, 0, None);

        let mut short_u = vec![0; text.len() - 1];
        let mut short_a = vec![0; text.len() - 1];
        let mut full_u = vec![0; text.len()];
        let mut full_a = vec![0; text.len()];
        let short_freq = vec![0; ALPHABET_SIZE - 1];
        let short_aux = vec![primary];
        let bad_aux = vec![0, 0];
        let good_aux = vec![primary, 4];

        assert_eq!(
            libsais16_unbwt(&bwt, &mut short_u, &mut full_a, None, primary),
            -1
        );
        assert_eq!(
            libsais16_unbwt(&bwt, &mut full_u, &mut short_a, None, primary),
            -1
        );
        assert_eq!(
            libsais16_unbwt(&bwt, &mut full_u, &mut full_a, Some(&short_freq), primary),
            -1
        );
        assert_eq!(libsais16_unbwt(&bwt, &mut full_u, &mut full_a, None, 0), -1);
        assert_eq!(
            libsais16_unbwt(
                &bwt,
                &mut full_u,
                &mut full_a,
                None,
                text.len() as SaSint + 1
            ),
            -1
        );
        assert_eq!(
            libsais16_unbwt_aux(&bwt, &mut full_u, &mut full_a, None, 0, &good_aux),
            -1
        );
        assert_eq!(
            libsais16_unbwt_aux(&bwt, &mut full_u, &mut full_a, None, 3, &good_aux),
            -1
        );
        assert_eq!(
            libsais16_unbwt_aux(&bwt, &mut full_u, &mut full_a, None, 4, &short_aux),
            -1
        );
        assert_eq!(
            libsais16_unbwt_aux(&bwt, &mut full_u, &mut full_a, None, 4, &bad_aux),
            -1
        );
    }

    #[test]
    fn libsais16_ctx_rejects_invalid_public_arguments() {
        let text = [2, 1, 3, 1, 2, 0];
        let mut ctx = create_ctx().unwrap();
        let mut sa = vec![0; text.len() - 1];
        let mut freq = vec![0; ALPHABET_SIZE - 1];
        let mut u = vec![0; text.len() - 1];
        let mut a = vec![0; text.len() - 1];
        let mut full_u = vec![0; text.len()];
        let mut full_a = vec![0; text.len()];
        let mut aux = vec![0; 1];

        assert_eq!(libsais16_ctx(&mut ctx, &text, &mut sa, 0, None), -1);
        assert_eq!(
            libsais16_ctx(&mut ctx, &text, &mut full_a, 0, Some(&mut freq)),
            -1
        );
        assert_eq!(
            libsais16_gsa_ctx(&mut ctx, &[1, 2, 3], &mut full_a[..3], 0, None),
            -1
        );
        assert_eq!(
            libsais16_bwt_ctx(&mut ctx, &text, &mut u, &mut full_a, 0, None),
            -1
        );
        assert_eq!(
            libsais16_bwt_ctx(&mut ctx, &text, &mut full_u, &mut a, 0, None),
            -1
        );
        assert_eq!(
            libsais16_bwt_aux_ctx(
                &mut ctx,
                &text,
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
            libsais16_bwt_aux_ctx(
                &mut ctx,
                &text,
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
            libsais16_bwt_aux_ctx(
                &mut ctx,
                &text,
                &mut full_u,
                &mut full_a,
                0,
                None,
                4,
                &mut aux
            ),
            -1
        );

        let mut default_ctx = Context::default();
        assert_eq!(
            libsais16_ctx(&mut default_ctx, &text, &mut full_a, 0, None),
            -2
        );

        let mut bad_bucket_ctx = create_ctx().unwrap();
        bad_bucket_ctx.buckets.clear();
        assert_eq!(
            libsais16_ctx(&mut bad_bucket_ctx, &text, &mut full_a, 0, None),
            -2
        );

        let mut short_thread_state_ctx = create_ctx_omp(2).unwrap();
        short_thread_state_ctx
            .thread_state
            .as_mut()
            .unwrap()
            .truncate(1);
        assert_eq!(
            libsais16_ctx(&mut short_thread_state_ctx, &text, &mut full_a, 0, None),
            -2
        );
    }

    #[test]
    fn libsais16_unbwt_ctx_rejects_invalid_public_arguments() {
        let text = [2, 1, 3, 1, 2, 0];
        let mut bwt = vec![0; text.len()];
        let mut work = vec![0; text.len()];
        let primary = libsais16_bwt(&text, &mut bwt, &mut work, 0, None);
        let mut ctx = unbwt_create_ctx().unwrap();

        let mut short_u = vec![0; text.len() - 1];
        let mut short_a = vec![0; text.len() - 1];
        let mut full_u = vec![0; text.len()];
        let mut full_a = vec![0; text.len()];
        let short_freq = vec![0; ALPHABET_SIZE - 1];
        let short_aux = vec![primary];
        let bad_aux = vec![0, 0];
        let good_aux = vec![primary, 4];

        assert_eq!(
            libsais16_unbwt_ctx(&mut ctx, &bwt, &mut short_u, &mut full_a, None, primary),
            -1
        );
        assert_eq!(
            libsais16_unbwt_ctx(&mut ctx, &bwt, &mut full_u, &mut short_a, None, primary),
            -1
        );
        assert_eq!(
            libsais16_unbwt_ctx(
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
            libsais16_unbwt_ctx(&mut ctx, &bwt, &mut full_u, &mut full_a, None, 0),
            -1
        );
        assert_eq!(
            libsais16_unbwt_aux_ctx(&mut ctx, &bwt, &mut full_u, &mut full_a, None, 0, &good_aux),
            -1
        );
        assert_eq!(
            libsais16_unbwt_aux_ctx(&mut ctx, &bwt, &mut full_u, &mut full_a, None, 3, &good_aux),
            -1
        );
        assert_eq!(
            libsais16_unbwt_aux_ctx(
                &mut ctx,
                &bwt,
                &mut full_u,
                &mut full_a,
                None,
                4,
                &short_aux
            ),
            -1
        );
        assert_eq!(
            libsais16_unbwt_aux_ctx(&mut ctx, &bwt, &mut full_u, &mut full_a, None, 4, &bad_aux),
            -1
        );
    }

    #[test]
    fn libsais16_context_wrappers_match_direct_calls() {
        let text = [2, 1, 3, 1, 2, 0];
        let mut ctx = create_ctx().unwrap();

        let mut direct_sa = vec![0; text.len()];
        let mut ctx_sa = vec![0; text.len()];
        assert_eq!(libsais16(&text, &mut direct_sa, 0, None), 0);
        assert_eq!(libsais16_ctx(&mut ctx, &text, &mut ctx_sa, 0, None), 0);
        assert_eq!(ctx_sa, direct_sa);

        let mut direct_bwt = vec![0; text.len()];
        let mut direct_work = vec![0; text.len()];
        let mut ctx_bwt = vec![0; text.len()];
        let mut ctx_work = vec![0; text.len()];
        assert_eq!(
            libsais16_bwt(&text, &mut direct_bwt, &mut direct_work, 0, None),
            libsais16_bwt_ctx(&mut ctx, &text, &mut ctx_bwt, &mut ctx_work, 0, None)
        );
        assert_eq!(ctx_bwt, direct_bwt);

        let mut direct_aux = vec![0; 2];
        let mut ctx_aux = vec![0; 2];
        assert_eq!(
            libsais16_bwt_aux(
                &text,
                &mut direct_bwt,
                &mut direct_work,
                0,
                None,
                4,
                &mut direct_aux
            ),
            libsais16_bwt_aux_ctx(
                &mut ctx,
                &text,
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
    fn libsais16_unbwt_context_wrappers_match_direct_calls() {
        let text = [2, 1, 3, 1, 2, 0];
        let mut bwt = vec![0; text.len()];
        let mut work = vec![0; text.len()];
        let primary = libsais16_bwt(&text, &mut bwt, &mut work, 0, None);

        let mut ctx = unbwt_create_ctx().unwrap();
        let mut direct = vec![0; text.len()];
        let mut direct_work = vec![0; text.len()];
        let mut via_ctx = vec![0; text.len()];
        let mut ctx_work = vec![0; text.len()];

        assert_eq!(
            libsais16_unbwt(&bwt, &mut direct, &mut direct_work, None, primary),
            0
        );
        assert_eq!(
            libsais16_unbwt_ctx(&mut ctx, &bwt, &mut via_ctx, &mut ctx_work, None, primary),
            0
        );
        assert_eq!(via_ctx, direct);

        let mut aux = vec![0; 2];
        assert_eq!(
            libsais16_bwt_aux(&text, &mut bwt, &mut work, 0, None, 4, &mut aux),
            0
        );
        assert_eq!(
            libsais16_unbwt_aux(&bwt, &mut direct, &mut direct_work, None, 4, &aux),
            0
        );
        assert_eq!(
            libsais16_unbwt_aux_ctx(&mut ctx, &bwt, &mut via_ctx, &mut ctx_work, None, 4, &aux),
            0
        );
        assert_eq!(via_ctx, direct);
    }

    #[test]
    fn libsais16_ctx_frequency_wrappers_match_direct_calls() {
        let text = [2, 1, 3, 1, 2, 0];
        let gsa_text = [2, 1, 0, 3, 1, 0];
        let mut ctx = create_ctx().unwrap();

        let mut direct_sa = vec![0; text.len()];
        let mut ctx_sa = vec![0; text.len()];
        let mut direct_freq = vec![-1; ALPHABET_SIZE];
        let mut ctx_freq = vec![-1; ALPHABET_SIZE];
        assert_eq!(
            libsais16(&text, &mut direct_sa, 0, Some(&mut direct_freq)),
            0
        );
        assert_eq!(
            libsais16_ctx(&mut ctx, &text, &mut ctx_sa, 0, Some(&mut ctx_freq)),
            0
        );
        assert_eq!(ctx_sa, direct_sa);
        assert_eq!(ctx_freq, direct_freq);

        let mut direct_gsa = vec![0; gsa_text.len()];
        let mut ctx_gsa = vec![0; gsa_text.len()];
        direct_freq.fill(-1);
        ctx_freq.fill(-1);
        assert_eq!(
            libsais16_gsa(&gsa_text, &mut direct_gsa, 0, Some(&mut direct_freq)),
            0
        );
        assert_eq!(
            libsais16_gsa_ctx(&mut ctx, &gsa_text, &mut ctx_gsa, 0, Some(&mut ctx_freq)),
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
            libsais16_bwt(
                &text,
                &mut direct_bwt,
                &mut direct_work,
                0,
                Some(&mut direct_freq)
            ),
            libsais16_bwt_ctx(
                &mut ctx,
                &text,
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
            libsais16_bwt_aux(
                &text,
                &mut direct_bwt,
                &mut direct_work,
                0,
                Some(&mut direct_freq),
                4,
                &mut direct_aux
            ),
            libsais16_bwt_aux_ctx(
                &mut ctx,
                &text,
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
    fn libsais16_unbwt_ctx_frequency_wrappers_match_direct_calls() {
        let text = [2, 1, 3, 1, 2, 0];
        let mut freq = vec![0; ALPHABET_SIZE];
        let mut bwt = vec![0; text.len()];
        let mut work = vec![0; text.len()];
        let primary = libsais16_bwt(&text, &mut bwt, &mut work, 0, Some(&mut freq));
        assert!(primary >= 0);

        let mut ctx = unbwt_create_ctx().unwrap();
        let mut direct = vec![0; text.len()];
        let mut direct_work = vec![0; text.len() + 1];
        let mut via_ctx = vec![0; text.len()];
        let mut ctx_work = vec![0; text.len() + 1];
        assert_eq!(
            libsais16_unbwt(&bwt, &mut direct, &mut direct_work, Some(&freq), primary),
            libsais16_unbwt_ctx(
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
            libsais16_bwt_aux(&text, &mut bwt, &mut work, 0, Some(&mut freq), 4, &mut aux),
            0
        );
        direct.fill(0);
        direct_work.fill(0);
        via_ctx.fill(0);
        ctx_work.fill(0);
        assert_eq!(
            libsais16_unbwt_aux(&bwt, &mut direct, &mut direct_work, Some(&freq), 4, &aux),
            libsais16_unbwt_aux_ctx(
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
    fn libsais16_omp_wrappers_match_direct_calls_and_reject_negative_threads() {
        let text = [2, 1, 3, 1, 2, 0];
        let gsa_text = [2, 1, 0, 3, 1, 0];
        let mut direct_sa = vec![0; text.len()];
        let mut omp_sa = vec![0; text.len()];
        assert_eq!(libsais16(&text, &mut direct_sa, 0, None), 0);
        assert_eq!(libsais16_omp(&text, &mut omp_sa, 0, None, 2), 0);
        assert_eq!(omp_sa, direct_sa);
        assert_eq!(libsais16_omp(&text, &mut omp_sa, 0, None, -1), -1);

        let mut direct_gsa = vec![0; gsa_text.len()];
        let mut omp_gsa = vec![0; gsa_text.len()];
        assert_eq!(libsais16_gsa(&gsa_text, &mut direct_gsa, 0, None), 0);
        assert_eq!(libsais16_gsa_omp(&gsa_text, &mut omp_gsa, 0, None, 2), 0);
        assert_eq!(omp_gsa, direct_gsa);
        assert_eq!(libsais16_gsa_omp(&gsa_text, &mut omp_gsa, 0, None, -1), -1);

        let int_text = [1, 2, 1, 0];
        let mut direct_int_text = int_text.to_vec();
        let mut omp_int_text = int_text.to_vec();
        let mut direct_int_sa = vec![0; int_text.len()];
        let mut omp_int_sa = vec![0; int_text.len()];
        assert_eq!(
            libsais16_int(&mut direct_int_text, &mut direct_int_sa, 3, 0),
            0
        );
        assert_eq!(
            libsais16_int_omp(&mut omp_int_text, &mut omp_int_sa, 3, 0, 2),
            0
        );
        assert_eq!(omp_int_text, direct_int_text);
        assert_eq!(omp_int_sa, direct_int_sa);
        assert_eq!(
            libsais16_int_omp(&mut omp_int_text, &mut omp_int_sa, 3, 0, -1),
            -1
        );

        let mut direct_bwt = vec![0; text.len()];
        let mut direct_work = vec![0; text.len()];
        let mut omp_bwt = vec![0; text.len()];
        let mut omp_work = vec![0; text.len()];
        assert_eq!(
            libsais16_bwt(&text, &mut direct_bwt, &mut direct_work, 0, None),
            libsais16_bwt_omp(&text, &mut omp_bwt, &mut omp_work, 0, None, 2)
        );
        assert_eq!(omp_bwt, direct_bwt);
        assert_eq!(
            libsais16_bwt_omp(&text, &mut omp_bwt, &mut omp_work, 0, None, -1),
            -1
        );

        let mut direct_aux = vec![0; 2];
        let mut omp_aux = vec![0; 2];
        assert_eq!(
            libsais16_bwt_aux(
                &text,
                &mut direct_bwt,
                &mut direct_work,
                0,
                None,
                4,
                &mut direct_aux
            ),
            libsais16_bwt_aux_omp(
                &text,
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
            libsais16_bwt_aux_omp(
                &text,
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
    fn libsais16_omp_frequency_wrappers_match_direct_calls() {
        let text = [2, 1, 3, 1, 2, 0];
        let gsa_text = [2, 1, 0, 3, 1, 0];
        let mut direct_sa = vec![0; text.len()];
        let mut omp_sa = vec![0; text.len()];
        let mut direct_freq = vec![-1; ALPHABET_SIZE];
        let mut omp_freq = vec![-1; ALPHABET_SIZE];
        assert_eq!(
            libsais16(&text, &mut direct_sa, 0, Some(&mut direct_freq)),
            0
        );
        assert_eq!(
            libsais16_omp(&text, &mut omp_sa, 0, Some(&mut omp_freq), 2),
            0
        );
        assert_eq!(omp_sa, direct_sa);
        assert_eq!(omp_freq, direct_freq);

        let mut direct_gsa = vec![0; gsa_text.len()];
        let mut omp_gsa = vec![0; gsa_text.len()];
        direct_freq.fill(-1);
        omp_freq.fill(-1);
        assert_eq!(
            libsais16_gsa(&gsa_text, &mut direct_gsa, 0, Some(&mut direct_freq)),
            0
        );
        assert_eq!(
            libsais16_gsa_omp(&gsa_text, &mut omp_gsa, 0, Some(&mut omp_freq), 2),
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
            libsais16_bwt(
                &text,
                &mut direct_bwt,
                &mut direct_work,
                0,
                Some(&mut direct_freq)
            ),
            libsais16_bwt_omp(
                &text,
                &mut omp_bwt,
                &mut omp_work,
                0,
                Some(&mut omp_freq),
                2
            )
        );
        assert_eq!(omp_bwt, direct_bwt);
        assert_eq!(omp_freq, direct_freq);

        let mut direct_aux = vec![0; 2];
        let mut omp_aux = vec![0; 2];
        direct_freq.fill(-1);
        omp_freq.fill(-1);
        assert_eq!(
            libsais16_bwt_aux(
                &text,
                &mut direct_bwt,
                &mut direct_work,
                0,
                Some(&mut direct_freq),
                4,
                &mut direct_aux
            ),
            libsais16_bwt_aux_omp(
                &text,
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
    fn libsais16_unbwt_omp_frequency_wrappers_match_direct_calls() {
        let text = [2, 1, 3, 1, 2, 0];
        let mut freq = vec![0; ALPHABET_SIZE];
        let mut bwt = vec![0; text.len()];
        let mut work = vec![0; text.len()];
        let primary = libsais16_bwt(&text, &mut bwt, &mut work, 0, Some(&mut freq));
        assert!(primary >= 0);

        let mut direct = vec![0; text.len()];
        let mut direct_work = vec![0; text.len() + 1];
        let mut omp = vec![0; text.len()];
        let mut omp_work = vec![0; text.len() + 1];
        assert_eq!(
            libsais16_unbwt(&bwt, &mut direct, &mut direct_work, Some(&freq), primary),
            libsais16_unbwt_omp(&bwt, &mut omp, &mut omp_work, Some(&freq), primary, 2)
        );
        assert_eq!(omp, direct);
        assert_eq!(omp, text);

        let mut aux = vec![0; (text.len() - 1) / 4 + 1];
        assert_eq!(
            libsais16_bwt_aux(&text, &mut bwt, &mut work, 0, Some(&mut freq), 4, &mut aux),
            0
        );
        direct.fill(0);
        direct_work.fill(0);
        omp.fill(0);
        omp_work.fill(0);
        assert_eq!(
            libsais16_unbwt_aux(&bwt, &mut direct, &mut direct_work, Some(&freq), 4, &aux),
            libsais16_unbwt_aux_omp(&bwt, &mut omp, &mut omp_work, Some(&freq), 4, &aux, 2)
        );
        assert_eq!(omp, direct);
        assert_eq!(omp, text);
    }

    #[test]
    fn libsais16_lcp_and_unbwt_omp_wrappers_match_direct_calls() {
        let text = [2, 1, 3, 1, 2, 0];
        let mut sa = vec![0; text.len()];
        assert_eq!(libsais16(&text, &mut sa, 0, None), 0);

        let mut direct_plcp = vec![0; text.len()];
        let mut omp_plcp = vec![0; text.len()];
        assert_eq!(libsais16_plcp(&text, &sa, &mut direct_plcp), 0);
        assert_eq!(libsais16_plcp_omp(&text, &sa, &mut omp_plcp, 2), 0);
        assert_eq!(omp_plcp, direct_plcp);
        assert_eq!(libsais16_plcp_omp(&text, &sa, &mut omp_plcp, -1), -1);

        let gsa_text = [2, 1, 0, 1, 2, 0];
        let mut gsa = vec![0; gsa_text.len()];
        assert_eq!(libsais16_gsa(&gsa_text, &mut gsa, 0, None), 0);
        let mut direct_gsa_plcp = vec![0; gsa_text.len()];
        let mut omp_gsa_plcp = vec![0; gsa_text.len()];
        assert_eq!(libsais16_plcp_gsa(&gsa_text, &gsa, &mut direct_gsa_plcp), 0);
        assert_eq!(
            libsais16_plcp_gsa_omp(&gsa_text, &gsa, &mut omp_gsa_plcp, 2),
            0
        );
        assert_eq!(omp_gsa_plcp, direct_gsa_plcp);
        assert_eq!(
            libsais16_plcp_gsa_omp(&gsa_text, &gsa, &mut omp_gsa_plcp, -1),
            -1
        );

        let mut direct_lcp = vec![0; text.len()];
        let mut omp_lcp = vec![0; text.len()];
        assert_eq!(libsais16_lcp(&direct_plcp, &sa, &mut direct_lcp), 0);
        assert_eq!(libsais16_lcp_omp(&direct_plcp, &sa, &mut omp_lcp, 2), 0);
        assert_eq!(omp_lcp, direct_lcp);
        assert_eq!(libsais16_lcp_omp(&direct_plcp, &sa, &mut omp_lcp, -1), -1);

        let mut bwt = vec![0; text.len()];
        let mut work = vec![0; text.len()];
        let primary = libsais16_bwt(&text, &mut bwt, &mut work, 0, None);
        let mut direct = vec![0; text.len()];
        let mut omp = vec![0; text.len()];
        let mut direct_work = vec![0; text.len()];
        let mut omp_work = vec![0; text.len()];
        assert_eq!(
            libsais16_unbwt(&bwt, &mut direct, &mut direct_work, None, primary),
            0
        );
        assert_eq!(
            libsais16_unbwt_omp(&bwt, &mut omp, &mut omp_work, None, primary, 2),
            0
        );
        assert_eq!(omp, direct);
        assert_eq!(
            libsais16_unbwt_omp(&bwt, &mut omp, &mut omp_work, None, primary, -1),
            -1
        );
    }
}
