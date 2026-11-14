//! Channel-level ISA selection, independent of encoding policy and CPU count.
use std::sync::OnceLock;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Isa {
    Portable,
    Avx2,
    Avx512,
}

fn choose(requested: Option<&str>, avx2: bool, avx512: bool) -> Isa {
    match requested {
        Some("portable") => Isa::Portable,
        Some("avx512") if avx512 => Isa::Avx512,
        Some("avx512") => Isa::Portable,
        Some("avx2") if avx2 => Isa::Avx2,
        // Wider code generation has not established a cross-image speedup.
        // Keep explicit overrides for diagnosis without changing CPU policy.
        _ => Isa::Portable,
    }
}

pub(super) fn selected() -> Isa {
    static ISA: OnceLock<Isa> = OnceLock::new();
    *ISA.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        let (avx2, avx512) = (
            std::is_x86_feature_detected!("avx2"),
            std::is_x86_feature_detected!("avx2")
                && std::is_x86_feature_detected!("avx512f")
                && std::is_x86_feature_detected!("avx512bw")
                && std::is_x86_feature_detected!("avx512dq")
                && std::is_x86_feature_detected!("avx512vl"),
        );
        #[cfg(not(target_arch = "x86_64"))]
        let (avx2, avx512) = (false, false);
        let requested = std::env::var("RS_JXL_MODULAR_ISA").ok();
        let isa = choose(requested.as_deref(), avx2, avx512);
        crate::profile::note("modular.kernel_isa", format_args!("{isa:?}"));
        isa
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unsupported_and_unknown_requests_never_select_unsafe_isa() {
        for name in [
            None,
            Some("portable"),
            Some("avx2"),
            Some("avx512"),
            Some("invalid"),
        ] {
            assert_eq!(choose(name, false, false), Isa::Portable);
        }
        assert_eq!(choose(Some("avx512"), true, false), Isa::Portable);
        assert_eq!(choose(Some("invalid"), true, true), Isa::Portable);
        assert_eq!(choose(Some("portable"), true, true), Isa::Portable);
        assert_eq!(choose(None, true, true), Isa::Portable);
        assert_eq!(choose(Some("auto"), true, true), Isa::Portable);
        assert_eq!(choose(Some("avx2"), true, true), Isa::Avx2);
        assert_eq!(choose(Some("avx512"), true, true), Isa::Avx512);
    }
}
