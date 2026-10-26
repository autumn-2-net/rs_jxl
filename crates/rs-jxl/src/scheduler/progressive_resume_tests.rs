use super::*;
use crate::entropy::stream::Coding;
use crate::modular::encode::{FixedTree, ModularOptions};
use crate::modular::palette::PaletteOptions;
use crate::scheduler::NativeAcceptance;
use crate::{PixelData, PixelLayout};

#[test]
fn resume_checks_final_payload_cutoff_before_framing() {
    const WIDTH: u32 = 1025;
    const HEIGHT: u32 = 3;
    const PIXELS: usize = WIDTH as usize * HEIGHT as usize;

    let image = Image::new(
        WIDTH,
        HEIGHT,
        PixelLayout::Gray,
        PixelData::U8((0..PIXELS).map(|i| (i % 251) as u8).collect()),
    )
    .unwrap();
    let options = EncodeOptions {
        threads: 1,
        max_pixels: PIXELS as u64,
        max_output_bytes: 1 << 20,
        group_size_shift: 0,
        shared_tree: true,
        modular: ModularOptions {
            predictor: Predictor::Gradient,
            fixed_tree: Some(FixedTree::GradientLeaf),
            rct: None,
            entropy: Coding::Prefix,
            lz77: false,
            tree_samples: 1,
            max_tree_leaves: 1,
            palette_policy: Some(PaletteOptions {
                groupwise: true,
                palette_colors: 0,
                channel_colors_pre_transform_percent: 0.0,
                channel_colors_percent: 0.0,
                ..Default::default()
            }),
            ..Default::default()
        },
        ..Default::default()
    };
    let params = CandidateParams::default().backend(9).unwrap();
    let (mut artifact, _) = prepare(
        &params,
        &image,
        &options,
        NativeBackendOptions {
            acceptance: NativeAcceptance::AllowReportedGaps,
            patches_off: true,
            ..Default::default()
        },
        NativeTrialKind::Full,
    )
    .unwrap();
    artifact.options = options;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();

    let mut reference = PixelSession::new(&image, &artifact.options)
        .unwrap()
        .unwrap();
    assert_eq!(reference.groups.group_count(), 9);
    crate::execution::with_pool(&pool, || reference.groups.complete_remaining()).unwrap();
    let payload_bytes = reference.groups.retained_bytes();
    assert!(payload_bytes > 0);
    assert!(reference.finish().unwrap().len() > payload_bytes);
    artifact.options.max_output_bytes = payload_bytes;

    // Exercise both an empty missing-group list and a final batch hitting equality.
    for completed in [9usize, 8] {
        let mut session = PixelSession::new(&image, &artifact.options)
            .unwrap()
            .unwrap();
        let ids: Vec<_> = (0..completed).collect();
        crate::execution::with_pool(&pool, || session.groups.encode_groups(&ids)).unwrap();
        assert_eq!(session.groups.completed_groups(), completed);
        let encoded_pixels = ids
            .iter()
            .map(|&id| {
                let rect = session.groups.group_rect(id).unwrap();
                (rect.width * rect.height) as u64
            })
            .sum();
        let mut peak = session.groups.retained_bytes();
        if completed == 9 {
            assert_eq!(peak, payload_bytes);
        } else {
            assert!(peak < payload_bytes);
        }
        let mut contender = Contender {
            candidate: Candidate::new("cutoff-regression", params.clone()),
            artifact: artifact.clone(),
            session: Some(session),
            report: GroupCandidateReport {
                name: "cutoff-regression",
                params: params.clone(),
                state: CandidateState::Sampled,
                total_groups: 9,
                encoded_groups: completed,
                reused_groups: 0,
                encoded_pixels,
                pilot_scores: Vec::new(),
                sampled_region_bytes: Vec::new(),
                estimated_bytes: 0.0,
                seconds: 0.0,
                final_bytes: None,
            },
            completed_group_bytes: None,
        };
        let outcome = resume(
            &mut contender,
            Some(payload_bytes),
            &pool,
            0,
            GroupRaceOptions::default(),
            &mut peak,
        )
        .unwrap_or_else(|error| panic!("resume failed with {completed} completed groups: {error}"));
        assert!(matches!(
            outcome,
            TrialOutput::Bounded { lower_bound } if lower_bound == payload_bytes
        ));
        assert_eq!(contender.report.state, CandidateState::ByteBounded);
        assert_eq!(contender.report.encoded_groups, 9);
        assert_eq!(contender.report.reused_groups, completed);
        assert_eq!(contender.report.encoded_pixels, PIXELS as u64);
        assert_eq!(contender.report.final_bytes, None);
        assert!(contender.session.is_none());
        assert_eq!(peak, payload_bytes);
    }
}
