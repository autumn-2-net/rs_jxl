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
