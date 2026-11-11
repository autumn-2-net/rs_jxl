use super::*;

#[test]
fn zero_run_certificates_match_scalar_prefixes_at_caps_tails_and_overlaps() {
    let mut values = Vec::new();
    for count in [0, 1, 2, 3, 4, 7, 8, 15, 16, 17, 127, 128, 129, 257] {
        for tail in [7, 19] {
            values.extend(std::iter::repeat_n(0, count));
            values.extend([tail, 11, 0, 37]);
        }
    }
    values.extend([0; 259]);
    for window in [4, 16, 128, 512] {
        let layout = Layout {
            window,
            fine_buckets: 0,
            bytes: 0,
        };
        let index = Index::new(&values, 37, &SourceConfig::default(), layout).unwrap();
        for position in 1..values.len() {
            let zeros = index.zeros[position] as usize;
            if zeros < 3 {
                continue;
            }
            for previous in (0..position).step_by(13).chain([position - 1]) {
                for cap in [0, 2, 3, 4, 16, 127, 128, 129, usize::MAX] {
                    let limit = cap.min(values.len() - position);
                    let expected = values[position..position + limit]
                        .iter()
                        .zip(&values[previous..previous + limit])
                        .take_while(|(a, b)| a == b)
                        .count();
                    assert_eq!(
                        index.zero_length(position, previous, limit, zeros),
                        expected,
                        "window={window} position={position} previous={previous} limit={limit}"
                    );
                }
            }
        }
    }
}
