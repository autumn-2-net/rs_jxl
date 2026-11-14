use super::*;
use crate::execution;
use crate::modular::tree::Tree;

#[derive(Debug, PartialEq)]
struct Outcome {
    signature: Vec<(usize, u32)>,
    bits: u64,
    coarsened_views: usize,
    histograms: Box<Histograms>,
}

fn hash(i: usize) -> i32 {
    let mut x = (i as u32).wrapping_add(0x9e37_79b9);
    x = (x ^ (x >> 16)).wrapping_mul(0x85eb_ca6b);
    x = (x ^ (x >> 13)).wrapping_mul(0xc2b2_ae35);
    ((x ^ (x >> 16)) & 65535) as i32
}

fn evaluate(limit: usize) -> Outcome {
    // Keep odd dimensions and non-dense IDs, with just enough total work to
    // activate parallel domain dispatch and deep serial spatial descendants.
    let dimensions = [(257, 129), (257, 129), (17, 9)];
    let pixels: Vec<Vec<i32>> = dimensions
        .iter()
        .enumerate()
        .map(|(kind, &(width, height))| {
            (0..width * height)
                .map(|i| hash(match kind {
                    0 => i / width,
                    1 => i % width,
                    _ => i,
                }))
                .collect()
        })
        .collect();
    let views: Vec<_> = dimensions
        .iter()
        .enumerate()
        .map(|(c, &(width, height))| ChannelView {
            channel: if c == 1 { 2 } else { 0 },
            group: if c == 2 { 7 } else { 1 },
            width,
            height,
            stride: width,
            pixels: &pixels[c],
            is_meta: false,
        })
        .collect();
    let ordered: Vec<_> = views.iter().collect();
    let candidate = domain(&ordered, limit, 16).unwrap();
    let mut nodes = Vec::new();
    candidate.plan.append(&mut nodes);
    let tree = Tree::new(nodes).unwrap();
    Outcome {
        signature: tree
            .tokens()
            .iter()
            .map(|t| (t.context, t.value))
            .collect(),
        bits: candidate.bits.to_bits(),
        coarsened_views: candidate.coarsened_views,
        histograms: candidate.histograms,
    }
}

#[test]
fn candidates_stay_small_and_combine_reuses_lower_histograms() {
    assert!(std::mem::size_of::<Candidate>() <= 128);
    let mut lower = empty_histograms();
    let mut upper = empty_histograms();
    for histogram in lower.iter_mut() {
        histogram.add(0, false);
    }
    for histogram in upper.iter_mut() {
        histogram.add(128, false);
    }
    let allocation = std::ptr::from_ref(&*lower);
    let combined = combine(3, 15, leaf(lower, 0), leaf(upper, 0));
    assert_eq!(std::ptr::from_ref(&*combined.histograms), allocation);
    assert!(combined.histograms.iter().all(|h| h.count == 2));
}

#[test]
fn deep_domain_and_spatial_plans_fit_small_serial_stack() {
    let limits = [1, 4096];
    let expected = execution::serial(|| limits.map(evaluate));
    let actual = std::thread::Builder::new()
        .name("block-plan-small-stack".into())
        .stack_size(256 << 10)
        .spawn(move || execution::serial(|| limits.map(evaluate)))
        .unwrap()
        .join()
        .unwrap();
    assert_eq!(actual, expected);
}

#[test]
fn deep_plans_keep_serial_signatures_on_one_and_four_small_stack_workers() {
    let limits = [3, 4096];
    let expected = execution::serial(|| limits.map(evaluate));
    for (workers, stack_bytes) in [(1, 256 << 10), (4, 512 << 10)] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .stack_size(stack_bytes)
            .build()
            .unwrap();
        let actual = execution::with_pool(&pool, || limits.map(evaluate));
        assert_eq!(actual, expected, "workers={workers}");
    }
}
