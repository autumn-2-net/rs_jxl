use super::*;
fn empty_codestream(packet: u8) -> Vec<u8> {
    let mut d = vec![255, 79, 255, 81, 0, 41, 0, 0];
    for v in [1u32, 1, 0, 0, 1, 1, 0, 0] {
        d.extend(v.to_be_bytes());
    }
    d.extend([0, 1, 7, 1, 1]);
    d.extend([255, 82, 0, 12, 0, 0, 0, 1, 0, 0, 4, 4, 0, 1]);
    d.extend([255, 92, 0, 4, 64, 64]);
    d.extend([
        255, 144, 0, 10, 0, 0, 0, 0, 0, 15, 0, 1, 255, 147, packet, 255, 217,
    ]);
    d
}
#[test]
fn empty_packets_and_header_exception() {
    for packet in [0, 127] {
        let cs = empty_codestream(packet);
        let (blob, stats) = analyze(&cs, false, &Limits::default()).unwrap();
        assert_eq!(rebuild(&blob, false, &Limits::default()).unwrap(), cs);
        assert_eq!(stats.packets, 1);
        assert_eq!(stats.hdr_exceptions, u64::from(packet != 0));
        assert_eq!(blob_siz(&blob).unwrap(), cs[2..45]);
    }
}
#[test]
fn blob_bounds_and_varints() {
    let cs = empty_codestream(0);
    assert!(matches!(
        analyze(
            &cs,
            false,
            &Limits {
                max_input_bytes: cs.len() - 1,
                ..Limits::default()
            }
        ),
        Err(Error::Limit(_))
    ));
    let (blob, _) = analyze(&cs, false, &Limits::default()).unwrap();
    for n in 0..blob.len() {
        assert!(
            rebuild(&blob[..n], false, &Limits::default()).is_err(),
            "accepted truncated blob at {n}"
        );
    }
    let mut damaged = blob.clone();
    damaged.push(0);
    assert!(rebuild(&damaged, false, &Limits::default()).is_err());
}
#[test]
#[ignore = "independent upstream J2K fixtures supplied by RS_JXL_PLUS_PJA_FIXTURES"]
fn upstream_codestream_roundtrips() {
    fn collect(p: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
        if p.is_dir() {
            for e in std::fs::read_dir(p).unwrap() {
                collect(&e.unwrap().path(), out);
            }
        } else if p.extension().is_some_and(|s| {
            matches!(
                s.to_string_lossy().to_ascii_lowercase().as_str(),
                "j2k" | "j2c" | "jhc" | "jph" | "jp2" | "jpx" | "mj2"
            )
        }) {
            out.push(p.to_owned());
        }
    }
    let root = std::path::PathBuf::from(
        std::env::var_os("RS_JXL_PLUS_PJA_FIXTURES").expect("set fixture directory"),
    );
    let mut paths = Vec::new();
    collect(&root, &mut paths);
    paths.sort();
    assert!(!paths.is_empty());
    let mut failed = Vec::new();
    let mut passed = 0;
    for path in paths {
        let data = std::fs::read(&path).unwrap();
        let streams = super::super::codestreams(&data).unwrap_or_default();
        if streams.is_empty() {
            failed.push(format!("{} no codestream", path.display()));
            continue;
        }
        for (i, range) in streams.iter().enumerate() {
            let cs = &data[range.off..range.off + range.len];
            let ht = super::super::is_ht(cs);
            match analyze(cs, ht, &Limits::default()) {
                Ok((blob, st)) => {
                    let back = rebuild(&blob, ht, &Limits::default()).unwrap();
                    assert_eq!(back, cs, "{} stream {i}", path.display());
                    passed += 1;
                    eprintln!(
                        "PASS {} stream{i} ht={ht} bytes={} blob={} packets={} blocks={} raw={} segments={} rules={:?}",
                        path.display(),
                        cs.len(),
                        blob.len(),
                        st.packets,
                        st.cbs,
                        st.cb_raw,
                        st.segs,
                        st.rule
                    );
                }
                Err(e) => failed.push(format!("{} stream{i} ht={ht}: {e}", path.display())),
            }
        }
    }
    for f in &failed {
        eprintln!("FAIL {f}");
    }
    assert!(
        failed.is_empty(),
        "{passed} codestreams passed; {} failed",
        failed.len()
    );
}
