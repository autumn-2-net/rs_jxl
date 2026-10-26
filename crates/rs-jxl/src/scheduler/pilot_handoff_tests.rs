use super::*;
use crate::scheduler::{policy::winner_neighborhood, scan::scan_lossless_input};
use crate::{PixelData, PixelLayout};

fn fixture() -> (Image, DriverOptions, CandidateParams, Vec<Candidate>) {
    let image = Image::new(
        1024,
        512,
        PixelLayout::Rgb,
        PixelData::U8(
            (0..1024 * 512)
                .flat_map(|i| [i as u8, (i >> 8) as u8, (i >> 16) as u8])
                .collect(),
        ),
    )
    .unwrap();
    let mut options = DriverOptions::scout();
    options.scout_pilot_first = true;
    options.scout_staged_probes = true;
    options.screen.regions.full_width_strips = true;
    options.screen.regions.strip_rows = Some(32);
    options.screen.limits.max_resident_sample_pixels = 3 * 1024 * 32;
    let automatic = CandidateParams::default().backend(9).unwrap();
    let plan = vec![
        Candidate::new("automatic", automatic.clone()),
        Candidate::new("neighborhood", winner_neighborhood(&automatic, 3)),
    ];
    (image, options, automatic, plan)
}

#[test]
fn front_keeps_cumulative_counts_while_releasing_coarse_samples() {
    let (image, mut options, automatic, plan) = fixture();
    let scan = scan_lossless_input(&image, options.scan);
    assert!(scan.likely_continuous_tone() && !scan.literal_candidate);
    let pixels = 2 * 3 * 1024 * (8 + 32);
    options.screen.limits.max_encodes = 12;
    options.screen.limits.max_encoded_pixels = pixels;
    for short in [false, true] {
        let mut limits = options.screen.limits;
        limits.max_encodes -= usize::from(short);
        let request = PilotRequest {
            image: &image,
            automatic: &automatic,
            plan: &plan,
            options: &options,
            scan: Some(&scan),
            level: 9,
            probe_limits: limits,
            budget: TrialBudget {
                beat_bytes: None,
                max_output_bytes: 1000,
            },
        };
        let front = run_front(
            &request,
            Box::new(|tasks, _| {
                Ok(tasks
                    .iter()
                    .map(|t| if t.image.height() == 8 { 100 } else { 1000 })
                    .collect())
            }),
        );
        if short {
            assert!(matches!(front.ranked, Err(Error::Limit(_))));
            assert_eq!(front.usage.encodes, 6);
            assert_eq!(front.usage.batches, 1);
        } else {
            assert_eq!(
                front.usage,
                PilotUsage {
                    encodes: 12,
                    pixels,
                    batches: 2
                }
            );
            let ranked = front.ranked.unwrap();
            assert_eq!(ranked.len(), 2);
            assert!(ranked.iter().all(|(_, scores)| scores == &[1000; 3]));
        }
    }
}

#[test]
fn failed_batch_is_charged_and_never_exposes_partial_rank() {
    let (image, options, automatic, plan) = fixture();
    let request = PilotRequest {
        image: &image,
        automatic: &automatic,
        plan: &plan,
        options: &options,
        scan: None,
        level: 10,
        probe_limits: options.screen.limits,
        budget: TrialBudget {
            beat_bytes: None,
            max_output_bytes: 1000,
        },
    };
    for malformed in [false, true] {
        let front = run_front(
            &request,
            Box::new(move |tasks, _| {
                if malformed {
                    Ok(vec![0; tasks.len()])
                } else {
                    Err(Error::Encoding("injected probe error".into()))
                }
            }),
        );
        assert!(front.ranked.is_err());
        assert_eq!(front.usage.encodes, 6);
        assert_eq!(front.usage.pixels, 2 * 3 * 1024 * 32);
        assert_eq!(front.usage.batches, 1);
        assert!(front.events.is_empty());
    }
}

#[test]
fn decisive_coarse_and_e10_keep_distinct_complete_scales() {
    let (image, mut options, automatic, plan) = fixture();
    options.scout_pilot_consensus = true;
    options.scout_coarse_consensus = true;
    let scan = scan_lossless_input(&image, options.scan);
    for level in [9, 10] {
        let request = PilotRequest {
            image: &image,
            automatic: &automatic,
            plan: &plan,
            options: &options,
            scan: Some(&scan),
            level,
            probe_limits: options.screen.limits,
            budget: TrialBudget {
                beat_bytes: None,
                max_output_bytes: 1000,
            },
        };
        let front = run_front(
            &request,
            Box::new(|tasks, _| {
                Ok(tasks
                    .iter()
                    .map(|t| {
                        let scale = if t.image.height() == 8 { 100 } else { 1000 };
                        scale + usize::from(t.params.group_size_shift.is_some()) * scale
                    })
                    .collect())
            }),
        );
        let ranked = front.ranked.unwrap();
        assert_eq!(front.usage.encodes, 6);
        assert_eq!(front.usage.batches, 1);
        assert_eq!(ranked.len(), if level == 9 { 1 } else { 2 });
        assert_eq!(ranked[0].1, vec![if level == 9 { 100 } else { 1000 }; 3]);
    }
}
