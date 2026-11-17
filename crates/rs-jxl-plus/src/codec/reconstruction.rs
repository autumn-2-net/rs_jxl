//! Source JPEG/sample layout adapters, derived from PJA pjacodec.pas (MIT),
//! 1058f63fbfb9c60cc6640b7bb130d6f7c7ea7dbd; LICENSE-PJA/NOTICE-PJA.
use super::*;
pub(super) fn pixels(width: usize, height: usize, l: &DecodeLimits) -> Result<()> {
    if (width as u64)
        .checked_mul(height as u64)
        .is_none_or(|n| n > l.max_pixels)
    {
        return Err(Error::Limit("codec pixels"));
    }
    Ok(())
}
pub(super) fn jpeg_limits(l: &DecodeLimits) -> jpeg::Limits {
    jpeg::Limits {
        max_input_bytes: l.max_input_bytes.max(l.max_output_bytes),
        max_output_bytes: l.max_output_bytes,
        max_blocks: l.max_coefficient_bytes / 128,
        max_samples: l.max_coefficient_bytes / 2,
        max_scans: l.max_scans,
    }
}
pub(super) fn jls_limits(l: &DecodeLimits) -> jls::Limits {
    jls::Limits {
        max_input_bytes: l.max_input_bytes.max(l.max_output_bytes),
        max_output_bytes: l.max_output_bytes,
        max_samples: l.max_coefficient_bytes / 2,
        max_scans: l.max_scans,
        max_runs: l.max_hint_entries,
    }
}
pub(super) fn components(j: &jpeg::Jpeg) -> Result<Vec<model::Component>> {
    j.components
        .iter()
        .map(|c| {
            let mut quant = [0; 64];
            for k in 0..64 {
                quant[jpeg::ZIGZAG[k] as usize] = j.qt[c.tq][k].max(1);
            }
            let mut coefficients = Vec::new();
            coefficients
                .try_reserve_exact(c.coef.len())
                .map_err(|_| Error::Limit("coefficient copy"))?;
            coefficients.resize(c.coef.len(), 0);
            for (src, dst) in c
                .coef
                .chunks_exact(64)
                .zip(coefficients.chunks_exact_mut(64))
            {
                for k in 0..64 {
                    dst[jpeg::ZIGZAG[k] as usize] = src[k];
                }
            }
            Ok(model::Component {
                width: c.bw,
                height: c.bh,
                h_samp: c.h,
                v_samp: c.v,
                quant,
                coefficients,
            })
        })
        .collect()
}
pub(super) fn install(j: &mut jpeg::Jpeg, components: Vec<model::Component>) {
    for (c, m) in j.components.iter_mut().zip(components) {
        for (src, dst) in m
            .coefficients
            .chunks_exact(64)
            .zip(c.coef.chunks_exact_mut(64))
        {
            for k in 0..64 {
                dst[k] = src[jpeg::ZIGZAG[k] as usize];
            }
        }
    }
}
pub(super) fn lossless_view(j: &jpeg::Jpeg) -> jls::Image {
    jls::Image {
        width: j.width,
        height: j.height,
        bits: j.precision - j.scans[0].al,
        components: j
            .components
            .iter()
            .map(|c| jls::Component {
                id: c.id,
                h: 1,
                v: 1,
                width: c.bw,
                height: c.bh,
                samples: c.samples.clone(),
            })
            .collect(),
        ..Default::default()
    }
}
pub(super) fn apply_scans(j: &mut jpeg::Jpeg, fields: &[ScanReconstruction]) -> Result<()> {
    if fields.len() != j.scans.len() {
        return Err(Error::Invalid("scan reconstruction count"));
    }
    for (i, r) in fields.iter().enumerate() {
        if !(-1..=1).contains(&r.padbit) || r.max_be > jpeg::MAX_BE_LIMIT {
            return Err(Error::Invalid("scan reconstruction values"));
        }
        if r.wrap_pred && !jpeg::may_wrap_pred(j, &j.scans[i]) {
            return Err(Error::Invalid("wrapped lossless predictor scan"));
        }
        if r.ignore_ri && (j.progressive || j.arithmetic || j.scans[i].ri == 0) {
            return Err(Error::Invalid("ignored restart scan"));
        }
        if r.max_be != jpeg::LIBJPEG_MAX_BE
            && (!j.progressive || j.arithmetic || j.scans[i].ah == 0 || j.scans[i].ss == 0)
        {
            return Err(Error::Invalid("correction limit scan"));
        }
        let s = &mut j.scans[i];
        s.padbit = r.padbit;
        s.max_be = r.max_be;
        s.wrap_pred = r.wrap_pred;
        s.ignore_ri = r.ignore_ri;
        if r.ignore_ri {
            s.ri = 0;
        }
    }
    Ok(())
}
