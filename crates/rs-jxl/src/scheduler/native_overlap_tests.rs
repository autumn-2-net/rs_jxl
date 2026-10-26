use super::*;
use crate::scheduler::{params::Candidate, probe::ProbeLimits};
use crate::{PixelData, PixelLayout};

fn image(width: u32, height: u32) -> Image {
    Image::new(
        width,
        height,
        PixelLayout::Rgb,
        PixelData::U8(vec![0; width as usize * height as usize * 3]),
    )
    .unwrap()
}

#[test]
fn scan_gate_requires_continuous_tone_independently_of_literal_evidence() {
    use crate::scheduler::scan::LosslessScan;

    let image = image(512, 256);
    let automatic = CandidateParams::default().backend(9).unwrap();
    let options = DriverOptions::scout();
    let execution = WorkerExecution {
        limits: WorkerLimits {
            tasks: 16,
            scratch_bytes: 256 << 20,
        },
        ..Default::default()
    };
    for (colors, flat, sampled, literal, allowed) in [
        (768, 0, 1024, false, true),
        (768, 0, 1024, true, true),
        (256, 0, 1024, false, false),
        (256, 0, 1024, true, false),
        (512, 0, 1024, false, false),
        (513, 127, 1024, false, true),
        (513, 128, 1024, true, true),
        (685, 134, 1024, true, true),
        (190, 0, 1024, true, false),
        (128, 0, 1024, true, false),
        (768, 511, 1024, false, true),
        (768, 512, 1024, false, false),
        (0, 0, 0, false, false),
    ] {
        let scan = LosslessScan {
            width: image.width() as usize,
            height: image.height() as usize,
            pixels: u64::from(image.width()) * u64::from(image.height()),
            sampled_colors: colors,
            sampled_flat: flat,
            sampled_pixels: sampled,
            literal_candidate: literal,
            ..Default::default()
        };
        let request = PilotRequest {
            image: &image,
            automatic: &automatic,
            plan: &[],
            options: &options,
            scan: Some(&scan),
            level: 9,
            probe_limits: ProbeLimits::default(),
            budget: TrialBudget {
                beat_bytes: None,
                max_output_bytes: 1000,
            },
        };
        assert_eq!(
            reservation(&request, execution, 16).unwrap().is_some(),
            allowed,
            "colors={colors} flat={flat} sampled={sampled} literal={literal}"
        );
    }
}

#[test]
fn overlap_reserves_full_and_samples_before_probe_roots() {
    let automatic = CandidateParams::default().backend(9).unwrap();
    let mut options = DriverOptions::scout();
    options.screen.regions.full_width_strips = true;
    options.screen.regions.strip_rows = Some(32);
    let execution = WorkerExecution {
        limits: WorkerLimits {
            tasks: 64,
            scratch_bytes: 512 << 20,
        },
        probe_workers: Some(64),
        cooperative: true,
        ..Default::default()
    };
    for (width, height, pool, eligible) in [
        (1024, 512, 64, true),
        (1024, 512, 1, false),
        (1024, 513, 16, false),
        (512, 256, 2, true),
    ] {
        let image = image(width, height);
        let request = PilotRequest {
            image: &image,
            automatic: &automatic,
            plan: &[],
            options: &options,
            scan: None,
            level: 9,
            probe_limits: ProbeLimits::default(),
            budget: TrialBudget {
                beat_bytes: None,
                max_output_bytes: 1000,
            },
        };
        let reserved = reservation(&request, execution, pool).unwrap();
        assert_eq!(reserved.is_some(), eligible);
        if let Some(reserved) = reserved {
            assert_eq!(reserved.outer.tasks, pool.min(16));
            assert_eq!(reserved.outer.scratch_bytes, 256 << 20);
            let full = u64::from(width) * u64::from(height) * 96 + (32 << 20);
            let samples = 3 * u64::from(width) * 32 * 12;
            assert_eq!(
                reserved.probes.limits.scratch_bytes,
                (256 << 20) - full - samples
            );
            assert!(reserved.probes.limits.tasks < reserved.outer.tasks);
            assert!(
                reserved.probes.limits.tasks as u64 * PROBE_SCRATCH_BYTES + full + samples
                    <= 256 << 20
            );
            let mut insufficient = execution;
            insufficient.limits.scratch_bytes = full + samples + PROBE_SCRATCH_BYTES - 1;
            assert!(reservation(&request, insufficient, pool).unwrap().is_none());
        }
    }
}

#[test]
fn owned_native_handoff_drains_failed_full_and_every_probe() {
    fn send_sync<T: Send + Sync>() {}
    send_sync::<RootOutput>();
    send_sync::<PilotHandoff<NativeArtifact>>();
    let image = image(512, 256);
    let automatic = CandidateParams::default().backend(9).unwrap();
    let plan = vec![Candidate::new("automatic", automatic.clone())];
    let mut options = DriverOptions::scout();
    options.preflight.enabled = true;
    options.screen.regions.full_width_strips = true;
    options.screen.regions.strip_rows = Some(8);
    let request = PilotRequest {
        image: &image,
        automatic: &automatic,
        plan: &plan,
        options: &options,
        scan: None,
        level: 9,
        probe_limits: ProbeLimits::default(),
        budget: TrialBudget {
            beat_bytes: None,
            max_output_bytes: 1000,
        },
    };
    let base = EncodeOptions {
        threads: 2,
        max_pixels: 1,
        ..Default::default()
    };
    let settings = NativeBackendOptions {
        acceptance: NativeAcceptance::AllowReportedGaps,
        ..Default::default()
    };
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    let mut records = Vec::new();
    let handoff = run(
        &request,
        &base,
        settings,
        &pool,
        None,
        WorkerExecution {
            limits: WorkerLimits {
                tasks: 2,
                scratch_bytes: 256 << 20,
            },
            cooperative: true,
            ..Default::default()
        },
        &mut records,
    )
    .unwrap()
    .unwrap();
    assert!(handoff.automatic.is_err());
    assert!(handoff.front.ranked.is_err());
    assert_eq!(handoff.front.usage.encodes, 3);
    assert_eq!(handoff.front.usage.pixels, 3 * 512 * 8);
    assert_eq!(handoff.front.usage.batches, 1);
    assert_eq!(
        records
            .iter()
            .filter(|r| r.kind == NativeTrialKind::Full)
            .count(),
        1
    );
    assert_eq!(
        records
            .iter()
            .filter(|r| r.kind == NativeTrialKind::Probe)
            .count(),
        3
    );
}
