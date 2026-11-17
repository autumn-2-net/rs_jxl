//! Independent source-shaped v7 bands on an owned Rayon pool.
//! PJA pjamodel.pas CodeBand/BandCount and pjaarith.pas arithmetic recurrence,
//! MIT, revision 1058f63fbfb9c60cc6640b7bb130d6f7c7ea7dbd. LICENSE-PJA/NOTICE-PJA.
use super::*;
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};

fn worker_count(threads: usize) -> Result<usize> {
    if !(1..=256).contains(&threads) {
        Err(Error::Invalid("coefficient thread count"))
    } else {
        Ok(threads)
    }
}
fn pool(threads: usize) -> Result<rayon::ThreadPool> {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .map_err(|_| Error::Limit("coefficient worker pool"))
}

pub(crate) fn encode_parallel(
    components: &[Component],
    max_output: usize,
    threads: usize,
) -> Result<Vec<Vec<u8>>> {
    worker_count(threads)?;
    let (rows, bands) = validate(components)?;
    if threads == 1 || bands == 1 {
        return encode(components, max_output);
    }
    encode_with_bands(components, rows, bands, max_output, threads, kernels)
}
pub(crate) fn decode_parallel(
    components: &mut [Component],
    streams: &[&[u8]],
    threads: usize,
) -> Result<()> {
    worker_count(threads)?;
    let (rows, bands) = validate(components)?;
    if streams.len() != bands {
        return Err(Error::Invalid("coefficient band count"));
    }
    if threads == 1 || bands == 1 {
        return decode_strict(components, streams);
    }
    decode_with_bands(components, rows, streams, threads, kernels)
}

/// Validated custom-band entry; shared output budget across all workers.
pub(super) fn encode_custom_bands(
    components: &[Component],
    bands: usize,
    max_output: usize,
    threads: usize,
) -> Result<Vec<Vec<u8>>> {
    worker_count(threads)?;
    let (rows, _) = validate(components)?;
    if bands == 0 || bands > 16 || bands > rows {
        return Err(Error::Invalid("parallel coefficient band count"));
    }
    encode_with_bands(components, rows, bands, max_output, threads, kernels)
}

/// No coefficients are published until every band passes strict termination.
pub(super) fn decode_custom_bands(
    components: &mut [Component],
    streams: &[&[u8]],
    threads: usize,
) -> Result<()> {
    worker_count(threads)?;
    let (rows, _) = validate(components)?;
    if streams.is_empty() || streams.len() > 16 || streams.len() > rows {
        return Err(Error::Invalid("parallel coefficient band count"));
    }
    decode_with_bands(components, rows, streams, threads, kernels)
}

struct Band {
    descriptors: Vec<Component>,
    coef: Vec<Vec<i16>>,
    origin: usize,
    rows: usize,
}
fn band(components: &[Component], m0: usize, m1: usize, encoding: bool) -> Result<Band> {
    let t = tables();
    let mut descriptors = Vec::with_capacity(components.len());
    let mut coef = Vec::with_capacity(components.len());
    for c in components {
        let first = m0 * c.v_samp * c.width * 64;
        let end = m1 * c.v_samp * c.width * 64;
        let mut v = Vec::new();
        v.try_reserve_exact(end - first)
            .map_err(|_| Error::Limit("coefficient band allocation"))?;
        v.resize(end - first, 0);
        if encoding {
            for (src, dst) in c.coefficients[first..end]
                .chunks_exact(64)
                .zip(v.chunks_exact_mut(64))
            {
                for k in 0..64 {
                    dst[k] = src[t.zz_table[k] as usize];
                }
            }
        }
        descriptors.push(Component {
            width: c.width,
            height: (m1 - m0) * c.v_samp,
            h_samp: c.h_samp,
            v_samp: c.v_samp,
            quant: c.quant,
            coefficients: Vec::new(),
        });
        coef.push(v);
    }
    Ok(Band {
        descriptors,
        coef,
        origin: m0,
        rows: m1 - m0,
    })
}
impl Band {
    fn code<C: BitCoder>(&mut self, cd: &mut C, kernel: Kernel) -> Result<()> {
        code_band_with(
            cd,
            &self.descriptors,
            &mut self.coef,
            0,
            self.rows,
            BandConfig {
                mcu_origin: self.origin,
                kernel,
            },
        )
    }
}

fn encode_with_bands(
    components: &[Component],
    rows: usize,
    bands: usize,
    max_output: usize,
    threads: usize,
    kernel: Kernel,
) -> Result<Vec<Vec<u8>>> {
    let remaining = AtomicUsize::new(max_output);
    let workers = pool(threads.min(bands))?;
    workers.install(|| {
        (0..bands)
            .into_par_iter()
            .map(|i| {
                let mut band = band(components, rows * i / bands, rows * (i + 1) / bands, true)?;
                let mut cd = SharedEncoder::new(&remaining);
                band.code(&mut cd, kernel)?;
                cd.finish()
            })
            .collect()
    })
}
fn decode_with_bands(
    components: &mut [Component],
    rows: usize,
    streams: &[&[u8]],
    threads: usize,
    kernel: Kernel,
) -> Result<()> {
    let n = streams.len();
    let workers = pool(threads.min(n))?;
    let decoded: Vec<Band> = workers.install(|| {
        streams
            .par_iter()
            .enumerate()
            .map(|(i, stream)| {
                let mut band = band(components, rows * i / n, rows * (i + 1) / n, false)?;
                let mut cd = Decoder::new(stream);
                band.code(&mut cd, kernel)?;
                cd.finish_check()?;
                Ok(band)
            })
            .collect::<Result<Vec<_>>>()
    })?;
    // Keep the same transactional coefficient publication as serial decode.
    // Across all results there is exactly one image's coefficient storage.
    let t = tables();
    for band in decoded {
        for (c, coef) in components.iter_mut().zip(band.coef) {
            let first = band.origin * c.v_samp * c.width * 64;
            let end = first + coef.len();
            for (src, dst) in coef
                .chunks_exact(64)
                .zip(c.coefficients[first..end].chunks_exact_mut(64))
            {
                for k in 0..64 {
                    dst[t.zz_table[k] as usize] = src[k];
                }
            }
        }
    }
    Ok(())
}

/// Same arithmetic recurrence as arith::Encoder, with one shared emitted-byte
/// budget for all concurrent streams. A worker never allocates max_output on
/// its own. Small fallible reserve chunks add at most 4095 unused requested
/// capacity bytes per active/finished band, rather than multiplying the cap.
struct SharedEncoder<'a> {
    low: u32,
    high: u32,
    output: Vec<u8>,
    remaining: &'a AtomicUsize,
    exceeded: bool,
}
impl<'a> SharedEncoder<'a> {
    fn new(remaining: &'a AtomicUsize) -> Self {
        Self {
            low: 0,
            high: u32::MAX,
            output: Vec::new(),
            remaining,
            exceeded: false,
        }
    }
    fn put(&mut self, v: u8) {
        if self.exceeded {
            return;
        }
        if self
            .remaining
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_sub(1))
            .is_err()
        {
            self.exceeded = true;
            return;
        }
        if self.output.len() == self.output.capacity() {
            let extra = self
                .remaining
                .load(Ordering::Relaxed)
                .saturating_add(1)
                .min(4096);
            if self.output.try_reserve_exact(extra).is_err() {
                self.exceeded = true;
                return;
            }
        }
        self.output.push(v);
    }
    fn finish(mut self) -> Result<Vec<u8>> {
        for _ in 0..4 {
            self.put((self.low >> 24) as u8);
            self.low <<= 8;
        }
        if self.exceeded {
            Err(Error::Limit("parallel coefficient output"))
        } else {
            Ok(self.output)
        }
    }
}
impl BitCoder for SharedEncoder<'_> {
    fn bit(&mut self, prob: u32, value: i32) -> i32 {
        let mid =
            self.low + (((self.high - self.low) as u64 * prob.clamp(1, 65535) as u64) >> 16) as u32;
        let bit = i32::from(value != 0);
        if bit != 0 {
            self.high = mid;
        } else {
            self.low = mid + 1;
        }
        while (self.low ^ self.high) & 0xff00_0000 == 0 {
            self.put((self.high >> 24) as u8);
            self.low <<= 8;
            self.high = (self.high << 8) | 255;
        }
        bit
    }
    fn encoding(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn source_streams(
        original: &[Component],
        rows: usize,
        n: usize,
        kernel: Kernel,
    ) -> Vec<Vec<u8>> {
        let mut coef = zigzag(original, true);
        (0..n)
            .map(|i| {
                let mut cd = Encoder::new(1 << 24);
                code_band_with(
                    &mut cd,
                    original,
                    &mut coef,
                    rows * i / n,
                    rows * (i + 1) / n,
                    BandConfig {
                        mcu_origin: 0,
                        kernel,
                    },
                )
                .unwrap();
                cd.finish().unwrap()
            })
            .collect()
    }
    fn check(original: Vec<Component>, n: usize) {
        let rows = validate(&original).unwrap().0;
        let scalar = source_streams(&original, rows, n, kernels_scalar);
        let runtime = source_streams(&original, rows, n, kernels);
        assert_eq!(
            runtime, scalar,
            "runtime AVX2/scalar kernel stream equality"
        );
        let size = scalar.iter().map(Vec::len).sum();
        for threads in [1, 2, 4] {
            let encoded = encode_with_bands(&original, rows, n, size, threads, kernels).unwrap();
            assert_eq!(encoded, scalar, "threads={threads}");
            let mut back = original.clone();
            for c in &mut back {
                c.coefficients.fill(0);
            }
            let refs = encoded.iter().map(Vec::as_slice).collect::<Vec<_>>();
            decode_with_bands(&mut back, rows, &refs, threads, kernels).unwrap();
            assert_eq!(back, original, "threads={threads}");
            let mut scalar_back = original.clone();
            for c in &mut scalar_back {
                c.coefficients.fill(0);
            }
            decode_with_bands(&mut scalar_back, rows, &refs, threads, kernels_scalar).unwrap();
            assert_eq!(scalar_back, original);
        }
        assert!(matches!(
            encode_with_bands(&original, rows, n, size - 1, 4, kernels),
            Err(Error::Limit(_))
        ));
    }
    #[test]
    fn disjoint_bands_match_source_global_coordinates() {
        check(
            vec![
                super::super::tests::fixture(4, 10, 2, 2, 1),
                super::super::tests::fixture(2, 5, 1, 1, 2),
                super::super::tests::fixture(2, 5, 1, 1, 3),
            ],
            3,
        );
    }
    #[test]
    fn fractional_luma_mapping_matches_full_global_zero_contexts() {
        check(
            vec![
                super::super::tests::fixture(6, 15, 3, 3, 1),
                super::super::tests::fixture(4, 10, 2, 2, 2),
                super::super::tests::fixture(4, 10, 2, 2, 3),
            ],
            3,
        );
    }
    #[test]
    fn public_threads_and_budget_contract() {
        let original = vec![super::super::tests::fixture(2, 3, 1, 1, 0)];
        let encoded = encode(&original, 1 << 20).unwrap();
        assert_eq!(encode_parallel(&original, 1 << 20, 4).unwrap(), encoded);
        assert!(encode_parallel(&original, 1 << 20, 0).is_err());
        let mut back = original.clone();
        back[0].coefficients.fill(0);
        decode_parallel(
            &mut back,
            &encoded.iter().map(Vec::as_slice).collect::<Vec<_>>(),
            4,
        )
        .unwrap();
        assert_eq!(back, original);
        assert!(decode_parallel(&mut back, &[], 4).is_err());
    }
}
