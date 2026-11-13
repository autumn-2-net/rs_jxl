//! Shared clap value parsers for CLI resource limits.

pub(crate) fn parse_threads(value: &str) -> std::result::Result<usize, String> {
    let count = value
        .parse::<usize>()
        .map_err(|_| "threads must be an integer in 1..=256")?;
    if !(1..=256).contains(&count) {
        return Err("threads must be in 1..=256".into());
    }
    Ok(count)
}

pub(crate) fn parse_prev_props(value: &str) -> std::result::Result<u8, String> {
    match value.parse::<u8>() {
        Ok(subset @ (1 | 2 | 4)) => Ok(subset),
        _ => Err("source previous-channel property subset must be 1, 2 or 4".into()),
    }
}

pub(crate) fn parse_sample_limit(value: &str) -> std::result::Result<usize, String> {
    value
        .parse::<usize>()
        .ok()
        .filter(|&limit| limit > 0)
        .ok_or_else(|| "source sample limit must be a positive integer fitting usize".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thread_limits_keep_boundaries_and_error_text() {
        for (value, expected) in [("1", 1), ("256", 256), ("+2", 2), ("002", 2)] {
            assert_eq!(parse_threads(value), Ok(expected));
        }
        for value in ["0", "257", &usize::MAX.to_string()] {
            assert_eq!(
                parse_threads(value).unwrap_err(),
                "threads must be in 1..=256"
            );
        }
        for value in ["", "-1", "abc", "1.5", " 1", "1 ", "18446744073709551616"] {
            assert_eq!(
                parse_threads(value).unwrap_err(),
                "threads must be an integer in 1..=256"
            );
        }
    }

    #[test]
    fn previous_property_subsets_accept_only_upstream_shapes() {
        for (value, expected) in [("1", 1), ("2", 2), ("4", 4)] {
            assert_eq!(parse_prev_props(value), Ok(expected));
        }
        for value in ["", "0", "3", "5", "abc", "1.0", " 2", "-1"] {
            assert_eq!(
                parse_prev_props(value).unwrap_err(),
                "source previous-channel property subset must be 1, 2 or 4"
            );
        }
    }

    #[test]
    fn sample_limits_keep_boundaries_and_error_text() {
        for (value, expected) in [("1", 1), ("+2", 2), ("002", 2)] {
            assert_eq!(parse_sample_limit(value), Ok(expected));
        }
        assert_eq!(parse_sample_limit(&usize::MAX.to_string()), Ok(usize::MAX));
        for value in [
            "",
            "0",
            "-1",
            "abc",
            "1.5",
            " 1",
            "1 ",
            "18446744073709551616",
        ] {
            assert_eq!(
                parse_sample_limit(value).unwrap_err(),
                "source sample limit must be a positive integer fitting usize"
            );
        }
    }
}
