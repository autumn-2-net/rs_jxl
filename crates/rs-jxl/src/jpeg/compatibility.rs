//! Bounded JPEG marker preflight for the current byte-reversible transcoder.
//!
//! This is NOT an entropy decoder, a JPEG conformance validator, or proof that
//! encoding will succeed. Findings describe the current implementation's input
//! restrictions, not necessarily limitations of the JPEG XL standard. See
//! docs/JPEG_COMPATIBILITY.md for the representation/implementation distinction.
//! No pixels, coefficients, metadata payloads or entropy bytes are retained.

use std::fmt;

/// Optional caller budgets; defaults impose no extra resource ceiling.
#[derive(Clone, Debug)]
pub struct ProbeLimits {
    /// Includes data after EOI, even though that data is not scanned.
    pub max_input_bytes: usize,
    /// Includes SOI, EOI and restart markers, but not byte stuffing or FF fill.
    pub max_markers: usize,
    pub max_scans: usize,
    /// Exceeding a limit is an error, never a partial successful report.
    pub max_findings: usize,
}

impl Default for ProbeLimits {
    fn default() -> Self {
        Self {
            max_input_bytes: usize::MAX,
            max_markers: usize::MAX,
            max_scans: usize::MAX,
            max_findings: usize::MAX,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProbeError {
    Truncated {
        offset: usize,
        context: &'static str,
    },
    Invalid {
        offset: usize,
        reason: &'static str,
    },
    Limit {
        offset: usize,
        resource: &'static str,
    },
}

impl fmt::Display for ProbeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Truncated { offset, context } => {
                write!(f, "truncated JPEG {context} at byte {offset}")
            }
            Self::Invalid { offset, reason } => {
                write!(f, "invalid JPEG header at byte {offset}: {reason}")
            }
            Self::Limit { offset, resource } => {
                write!(f, "JPEG probe limit at byte {offset}: {resource}")
            }
        }
    }
}

impl std::error::Error for ProbeError {}

/// Unsupported by the current coefficient/jbrd implementation. This enum is
/// deliberately NOT named `ForbiddenByStandard`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum UnsupportedFeature {
    ArithmeticCoding,
    ArithmeticConditioning,
    LosslessJpeg,
    HierarchicalJpeg,
    NonEightBitPrecision(u8),
    /// Four components, including CMYK/YCCK. See `four_component_hint`.
    FourComponentJpeg,
    ComponentCount(u8),
    DeferredHeight,
    MultipleFrames,
    SamplingFactor {
        component_id: u8,
        horizontal: u8,
        vertical: u8,
    },
    RgbSampling,
    ChangingRestartInterval {
        previous: u16,
        next: u16,
    },
    RestartMarkerFill,
    MoreThanThreeQuantizationDefinitions,
    ExtensionMarker(u8),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Finding {
    /// Offset of the FF immediately preceding the marker code, excluding fill.
    pub offset: usize,
    pub marker: u8,
    pub feature: UnsupportedFeature,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ComponentHeader {
    pub id: u8,
    pub horizontal_sampling: u8,
    pub vertical_sampling: u8,
    pub quantization_table: u8,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FrameHeader {
    pub offset: usize,
    pub marker: u8,
    pub precision: u8,
    pub width: u16,
    /// Zero is retained and reported as requiring unsupported DNL.
    pub height: u16,
    pub components: Vec<ComponentHeader>,
}

/// Interpretation declared by the first exact Adobe APP14 header, not an ICC
/// validation or a pixel-color inference. Four components alone do not prove CMYK.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FourComponentHint {
    Unspecified,
    AdobeCmyk,
    AdobeYcck,
    OtherAdobeTransform(u8),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProbeReport {
    /// First SOF. Subsequent SOFs are counted and diagnosed, not retained.
    pub frame: FrameHeader,
    pub frame_count: usize,
    pub marker_count: usize,
    pub scan_count: usize,
    pub restart_marker_count: usize,
    pub dri_count: usize,
    pub last_restart_interval: Option<u16>,
    pub quantization_definition_count: usize,
    pub adobe_transform: Option<u8>,
    pub four_component_hint: Option<FourComponentHint>,
    pub arithmetic_conditioning_definition_count: usize,
    /// DC L=0/U=1 and AC K=5 are the JPEG decoder defaults.
    pub nondefault_arithmetic_conditioning: bool,
    /// A scan uses selector 4..15, beyond the current jbrd two-bit table fields.
    pub wide_scan_table_selectors: bool,
    pub trailing_bytes: usize,
    pub findings: Vec<Finding>,
}

impl ProbeReport {
    /// No known header-level obstacle was found. Coefficient decoding, scan
    /// progression, quant-table consistency, ICC, reconstruction and encoding
    /// budgets still need validation by `jpeg::transcode`.
    pub fn is_candidate(&self) -> bool {
        self.findings.is_empty()
    }
}

/// Scan one JPEG through EOI, skipping entropy data without interpreting it.
/// Bytes after the first EOI are counted as archival tail data, not another JPEG.
pub fn probe(input: &[u8]) -> Result<ProbeReport, ProbeError> {
    probe_with_limits(input, &ProbeLimits::default())
}

pub fn probe_with_limits(input: &[u8], limits: &ProbeLimits) -> Result<ProbeReport, ProbeError> {
    if input.len() > limits.max_input_bytes {
        return Err(limit(0, "input bytes"));
    }
    if input.len() < 2 {
        return Err(truncated(0, "SOI marker"));
    }
    if input[..2] != [0xff, 0xd8] {
        return Err(invalid(0, "expected SOI marker"));
    }
    if limits.max_markers == 0 {
        return Err(limit(0, "markers"));
    }
    Scanner {
        input,
        limits,
        pos: 2,
        in_scan: false,
        first_frame: None,
        active_ids: [false; 256],
        active_component_count: 0,
        frame_count: 0,
        marker_count: 1,
        scan_count: 0,
        restart_count: 0,
        dri_count: 0,
        restart_interval: None,
        quant_count: 0,
        adobe: None,
        has_app0: false,
        dac_count: 0,
        nondefault_dac: false,
        wide_scan_tables: false,
        findings: Vec::new(),
    }
    .scan()
}

fn invalid(offset: usize, reason: &'static str) -> ProbeError {
    ProbeError::Invalid { offset, reason }
}

fn truncated(offset: usize, context: &'static str) -> ProbeError {
    ProbeError::Truncated { offset, context }
}

fn limit(offset: usize, resource: &'static str) -> ProbeError {
    ProbeError::Limit { offset, resource }
}

struct Scanner<'a> {
    input: &'a [u8],
    limits: &'a ProbeLimits,
    pos: usize,
    in_scan: bool,
    first_frame: Option<FrameHeader>,
    active_ids: [bool; 256],
    active_component_count: usize,
    frame_count: usize,
    marker_count: usize,
    scan_count: usize,
    restart_count: usize,
    dri_count: usize,
    restart_interval: Option<u16>,
    quant_count: usize,
    adobe: Option<u8>,
    has_app0: bool,
    dac_count: usize,
    nondefault_dac: bool,
    wide_scan_tables: bool,
    findings: Vec<Finding>,
}

impl Scanner<'_> {
    fn finding(
        &mut self,
        offset: usize,
        marker: u8,
        feature: UnsupportedFeature,
    ) -> Result<(), ProbeError> {
        if self.findings.len() >= self.limits.max_findings {
            return Err(limit(offset, "findings"));
        }
        self.findings
            .try_reserve(1)
            .map_err(|_| limit(offset, "finding allocation"))?;
        self.findings.push(Finding {
            offset,
            marker,
            feature,
        });
        Ok(())
    }

    fn marker(&mut self) -> Result<(usize, u8, usize), ProbeError> {
        while self.pos < self.input.len() {
            if self.input[self.pos] != 0xff {
                self.pos += 1;
                continue;
            }
            let start = self.pos;
            while self.input.get(self.pos) == Some(&0xff) {
                self.pos += 1;
            }
            let code = *self
                .input
                .get(self.pos)
                .ok_or_else(|| truncated(start, "marker code"))?;
            let offset = self.pos - 1;
            self.pos += 1;
            if code == 0 {
                if self.in_scan && offset != start {
                    return Err(invalid(start, "FF fill before stuffed entropy byte"));
                }
                continue;
            }
            if self.marker_count >= self.limits.max_markers {
                return Err(limit(offset, "markers"));
            }
            self.marker_count += 1;
            return Ok((offset, code, offset - start));
        }
        Err(truncated(self.pos, "EOI marker"))
    }

    fn segment(&mut self) -> Result<&[u8], ProbeError> {
        let start = self.pos;
        let bytes = self
            .input
            .get(start..start.saturating_add(2))
            .ok_or_else(|| truncated(start, "segment length"))?;
        let length = usize::from(u16::from_be_bytes([bytes[0], bytes[1]]));
        if length < 2 {
            return Err(invalid(start, "segment length below two"));
        }
        let end = start
            .checked_add(length)
            .ok_or_else(|| limit(start, "segment length"))?;
        let payload = self
            .input
            .get(start + 2..end)
            .ok_or_else(|| truncated(start, "segment payload"))?;
        self.pos = end;
        Ok(payload)
    }

    fn scan(mut self) -> Result<ProbeReport, ProbeError> {
        loop {
            let (offset, marker, fill) = self.marker()?;
            match marker {
                0xd8 => return Err(invalid(offset, "duplicate SOI")),
                0xd9 => return self.finish(),
                0xd0..=0xd7 => {
                    if !self.in_scan {
                        return Err(invalid(offset, "restart outside entropy scan"));
                    }
                    self.restart_count += 1;
                    if fill != 0 {
                        self.finding(offset, marker, UnsupportedFeature::RestartMarkerFill)?;
                    }
                    continue;
                }
                0x01 => {
                    self.finding(offset, marker, UnsupportedFeature::ExtensionMarker(marker))?;
                    continue;
                }
                _ => {}
            }
            let was_in_scan = self.in_scan;
            self.in_scan = false;
            // Borrow the original slice instead of copying metadata or entropy.
            let payload_start = self.pos + 2;
            self.segment()?;
            let payload = &self.input[payload_start..self.pos];
            match marker {
                0xc0..=0xc3 | 0xc5..=0xc7 | 0xc9..=0xcb | 0xcd..=0xcf => {
                    self.sof(offset, marker, payload)?;
                }
                0xda => self.sos(offset, payload)?,
                0xdb => self.dqt(offset, payload)?,
                0xcc => self.dac(offset, payload)?,
                0xdd => {
                    if payload.len() != 2 {
                        return Err(invalid(offset, "DRI payload length"));
                    }
                    let next = u16::from_be_bytes([payload[0], payload[1]]);
                    if let Some(previous) = self.restart_interval
                        && next != previous
                    {
                        self.finding(
                            offset,
                            marker,
                            UnsupportedFeature::ChangingRestartInterval { previous, next },
                        )?;
                    }
                    self.restart_interval = Some(next);
                    self.dri_count += 1;
                }
                0xdc => {
                    if payload.len() != 2 {
                        return Err(invalid(offset, "DNL payload length"));
                    }
                    self.finding(offset, marker, UnsupportedFeature::DeferredHeight)?;
                    self.in_scan = was_in_scan;
                }
                0xde | 0xdf => {
                    self.finding(offset, marker, UnsupportedFeature::HierarchicalJpeg)?
                }
                0xe0 => self.has_app0 = true,
                0xee if payload.len() == 12 && payload.starts_with(b"Adobe") => {
                    self.adobe.get_or_insert(payload[11]);
                }
                0xc4 | 0xe1..=0xef | 0xfe => {}
                _ => self.finding(offset, marker, UnsupportedFeature::ExtensionMarker(marker))?,
            }
        }
    }

    fn sof(&mut self, offset: usize, marker: u8, p: &[u8]) -> Result<(), ProbeError> {
        if p.len() < 6 {
            return Err(invalid(offset, "short SOF payload"));
        }
        let count = usize::from(p[5]);
        if count == 0 || p.len() != 6 + 3 * count {
            return Err(invalid(offset, "SOF component count or payload length"));
        }
        let height = u16::from_be_bytes([p[1], p[2]]);
        let width = u16::from_be_bytes([p[3], p[4]]);
        if width == 0 || p[0] == 0 {
            return Err(invalid(offset, "zero width or precision"));
        }
        if matches!(marker, 0xc9..=0xcb | 0xcd..=0xcf) {
            self.finding(offset, marker, UnsupportedFeature::ArithmeticCoding)?;
        }
        if matches!(marker, 0xc3 | 0xc7 | 0xcb | 0xcf) {
            self.finding(offset, marker, UnsupportedFeature::LosslessJpeg)?;
        }
        if matches!(marker, 0xc5..=0xc7 | 0xcd..=0xcf) {
            self.finding(offset, marker, UnsupportedFeature::HierarchicalJpeg)?;
        }
        if p[0] != 8 {
            self.finding(
                offset,
                marker,
                UnsupportedFeature::NonEightBitPrecision(p[0]),
            )?;
        }
        if count == 4 {
            self.finding(offset, marker, UnsupportedFeature::FourComponentJpeg)?;
        } else if !matches!(count, 1 | 3) {
            self.finding(
                offset,
                marker,
                UnsupportedFeature::ComponentCount(count as u8),
            )?;
        }
        if height == 0 {
            self.finding(offset, marker, UnsupportedFeature::DeferredHeight)?;
        }
        if self.frame_count != 0 {
            self.finding(offset, marker, UnsupportedFeature::MultipleFrames)?;
        }
        let mut components = Vec::new();
        components
            .try_reserve_exact(count)
            .map_err(|_| limit(offset, "SOF component allocation"))?;
        self.active_ids.fill(false);
        for c in p[6..].chunks_exact(3) {
            let (h, v) = (c[1] >> 4, c[1] & 15);
            if self.active_ids[usize::from(c[0])]
                || !(1..=4).contains(&h)
                || !(1..=4).contains(&v)
                || c[2] > 3
            {
                return Err(invalid(
                    offset,
                    "SOF duplicate ID, sampling or quantizer selector",
                ));
            }
            self.active_ids[usize::from(c[0])] = true;
            if h > 2 || v > 2 {
                self.finding(
                    offset,
                    marker,
                    UnsupportedFeature::SamplingFactor {
                        component_id: c[0],
                        horizontal: h,
                        vertical: v,
                    },
                )?;
            }
            components.push(ComponentHeader {
                id: c[0],
                horizontal_sampling: h,
                vertical_sampling: v,
                quantization_table: c[2],
            });
        }
        self.active_component_count = count;
        self.frame_count += 1;
        if self.first_frame.is_none() {
            self.first_frame = Some(FrameHeader {
                offset,
                marker,
                precision: p[0],
                width,
                height,
                components,
            });
        }
        Ok(())
    }

    fn sos(&mut self, offset: usize, p: &[u8]) -> Result<(), ProbeError> {
        if self.frame_count == 0 || p.is_empty() {
            return Err(invalid(offset, "SOS without frame or component count"));
        }
        let count = usize::from(p[0]);
        if count == 0
            || count > 4
            || count > self.active_component_count
            || p.len() != 4 + 2 * count
        {
            return Err(invalid(offset, "SOS component count or payload length"));
        }
        let mut seen = [false; 256];
        for c in p[1..1 + 2 * count].chunks_exact(2) {
            let id = usize::from(c[0]);
            if !self.active_ids[id] || seen[id] {
                return Err(invalid(offset, "SOS unknown or duplicate component ID"));
            }
            seen[id] = true;
            self.wide_scan_tables |= c[1] >> 4 > 3 || c[1] & 15 > 3;
        }
        if self.scan_count >= self.limits.max_scans {
            return Err(limit(offset, "scans"));
        }
        self.scan_count += 1;
        self.in_scan = true;
        Ok(())
    }

    fn dqt(&mut self, offset: usize, mut p: &[u8]) -> Result<(), ProbeError> {
        if p.is_empty() {
            return Err(invalid(offset, "empty DQT"));
        }
        while !p.is_empty() {
            let precision = p[0] >> 4;
            if precision > 1 || p[0] & 15 > 3 {
                return Err(invalid(offset, "DQT precision or selector"));
            }
            let stride = usize::from(precision) + 1;
            let len = 1 + 64 * stride;
            if p.len() < len {
                return Err(invalid(offset, "short DQT definition"));
            }
            if p[1..len]
                .chunks_exact(stride)
                .any(|value| value.iter().all(|&b| b == 0))
            {
                return Err(invalid(offset, "zero quantization value"));
            }
            self.quant_count += 1;
            if self.quant_count == 4 {
                self.finding(
                    offset,
                    0xdb,
                    UnsupportedFeature::MoreThanThreeQuantizationDefinitions,
                )?;
            }
            p = &p[len..];
        }
        Ok(())
    }

    fn dac(&mut self, offset: usize, p: &[u8]) -> Result<(), ProbeError> {
        if p.is_empty() || !p.len().is_multiple_of(2) {
            return Err(invalid(offset, "DAC payload length"));
        }
        for pair in p.chunks_exact(2) {
            let class = pair[0] >> 4;
            if class > 1
                || (class == 0 && pair[1] & 15 > pair[1] >> 4)
                || (class == 1 && pair[1] > 63)
            {
                return Err(invalid(offset, "DAC class or conditioning value"));
            }
            self.dac_count += 1;
            self.nondefault_dac |= pair[1] != if class == 0 { 0x10 } else { 5 };
        }
        self.finding(offset, 0xcc, UnsupportedFeature::ArithmeticConditioning)
    }

    fn finish(mut self) -> Result<ProbeReport, ProbeError> {
        let frame = self
            .first_frame
            .take()
            .ok_or_else(|| invalid(self.pos - 2, "EOI before SOF"))?;
        if self.scan_count == 0 {
            return Err(invalid(self.pos - 2, "EOI before SOS"));
        }
        // Match the current layout's APP0 / Adobe / RGB-ID precedence exactly.
        let rgb_ids = frame.components.len() == 3
            && frame
                .components
                .iter()
                .zip(b"RGB")
                .all(|(c, &id)| c.id == id);
        let rgb = frame.components.len() == 3
            && !self.has_app0
            && self.adobe.map_or(rgb_ids, |transform| transform == 0);
        if rgb
            && frame
                .components
                .iter()
                .any(|c| c.horizontal_sampling != 1 || c.vertical_sampling != 1)
        {
            self.finding(frame.offset, frame.marker, UnsupportedFeature::RgbSampling)?;
        }
        let four_component_hint = (frame.components.len() == 4).then_some(match self.adobe {
            Some(0) => FourComponentHint::AdobeCmyk,
            Some(2) => FourComponentHint::AdobeYcck,
            Some(other) => FourComponentHint::OtherAdobeTransform(other),
            None => FourComponentHint::Unspecified,
        });
        Ok(ProbeReport {
            frame,
            frame_count: self.frame_count,
            marker_count: self.marker_count,
            scan_count: self.scan_count,
            restart_marker_count: self.restart_count,
            dri_count: self.dri_count,
            last_restart_interval: self.restart_interval,
            quantization_definition_count: self.quant_count,
            adobe_transform: self.adobe,
            four_component_hint,
            arithmetic_conditioning_definition_count: self.dac_count,
            nondefault_arithmetic_conditioning: self.nondefault_dac,
            wide_scan_table_selectors: self.wide_scan_tables,
            trailing_bytes: self.input.len() - self.pos,
            findings: self.findings,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn segment(out: &mut Vec<u8>, marker: u8, payload: &[u8]) {
        out.extend([0xff, marker]);
        out.extend(u16::try_from(payload.len() + 2).unwrap().to_be_bytes());
        out.extend(payload);
    }

    fn header(marker: u8, precision: u8, count: u8) -> Vec<u8> {
        let mut out = vec![0xff, 0xd8];
        let mut sof = vec![precision, 0, 8, 0, 8, count];
        for id in 1..=count {
            sof.extend([id, 0x11, 0]);
        }
        segment(&mut out, marker, &sof);
        out
    }

    fn scan(out: &mut Vec<u8>, entropy: &[u8]) {
        segment(out, 0xda, &[1, 1, 0, 0, 63, 0]);
        out.extend(entropy);
    }

    fn complete(marker: u8, precision: u8, count: u8) -> Vec<u8> {
        let mut out = header(marker, precision, count);
        scan(&mut out, &[0x3f]);
        out.extend([0xff, 0xd9]);
        out
    }

    fn has(report: &ProbeReport, feature: UnsupportedFeature) -> bool {
        report
            .findings
            .iter()
            .any(|finding| finding.feature == feature)
    }

    #[test]
    fn default_findings_are_not_capped_or_truncated() {
        let mut bytes = header(0xc0, 8, 1);
        for _ in 0..257 {
            segment(&mut bytes, 0xdc, &[0, 8]);
        }
        scan(&mut bytes, &[0x3f]);
        bytes.extend([0xff, 0xd9]);
        let report = probe(&bytes).unwrap();
        assert_eq!(report.findings.len(), 257);
        assert!(matches!(
            probe_with_limits(
                &bytes,
                &ProbeLimits {
                    max_findings: 256,
                    ..Default::default()
                }
            ),
            Err(ProbeError::Limit {
                resource: "findings",
                ..
            })
        ));
        assert_eq!(
            probe_with_limits(
                &bytes,
                &ProbeLimits {
                    max_findings: 257,
                    ..Default::default()
                }
            )
            .unwrap(),
            report
        );
    }

    #[test]
    fn candidate_is_not_a_promise_to_encode_missing_tables_or_bad_entropy() {
        for marker in [0xc0, 0xc1, 0xc2] {
            // Deliberately missing DQT/DHT and unvalidated entropy/progression.
            let report = probe(&complete(marker, 8, 3)).unwrap();
            assert!(report.is_candidate());
            assert_eq!(report.frame.precision, 8);
            assert_eq!((report.frame.width, report.frame.height), (8, 8));
            assert_eq!((report.marker_count, report.scan_count), (4, 1));
        }
    }

    #[test]
    fn special_sof_modes_report_all_relevant_features() {
        for marker in [0xc9, 0xca, 0xcb, 0xcd, 0xce, 0xcf] {
            let report = probe(&complete(marker, 12, 4)).unwrap();
            assert!(has(&report, UnsupportedFeature::ArithmeticCoding));
            assert!(has(&report, UnsupportedFeature::NonEightBitPrecision(12)));
            assert!(has(&report, UnsupportedFeature::FourComponentJpeg));
            assert!(!report.is_candidate());
        }
        for marker in [0xc3, 0xc7, 0xcb, 0xcf] {
            assert!(has(
                &probe(&complete(marker, 8, 1)).unwrap(),
                UnsupportedFeature::LosslessJpeg
            ));
        }
        for marker in [0xc5, 0xc6, 0xc7, 0xcd, 0xce, 0xcf] {
            assert!(has(
                &probe(&complete(marker, 8, 1)).unwrap(),
                UnsupportedFeature::HierarchicalJpeg
            ));
        }
        assert!(has(
            &probe(&complete(0xc0, 8, 2)).unwrap(),
            UnsupportedFeature::ComponentCount(2)
        ));
    }

    #[test]
    fn adobe_four_component_hints_do_not_guess_from_count_alone() {
        assert_eq!(
            probe(&complete(0xc0, 8, 4)).unwrap().four_component_hint,
            Some(FourComponentHint::Unspecified)
        );
        for (transform, hint) in [
            (0, FourComponentHint::AdobeCmyk),
            (2, FourComponentHint::AdobeYcck),
            (1, FourComponentHint::OtherAdobeTransform(1)),
        ] {
            let mut bytes = header(0xc0, 8, 4);
            let mut adobe = b"Adobe\0d\0\0\0\0\0".to_vec();
            adobe[11] = transform;
            scan(&mut bytes, &[0]);
            segment(&mut bytes, 0xee, &adobe);
            bytes.extend([0xff, 0xd9]);
            let report = probe(&bytes).unwrap();
            assert_eq!(report.four_component_hint, Some(hint));
            assert_eq!(report.adobe_transform, Some(transform));
        }
    }

    #[test]
    fn arithmetic_conditioning_and_wide_selectors_are_inspectable() {
        for (conditioning, nondefault) in [([0, 0x10, 0x10, 5], false), ([0, 0x21, 0x10, 6], true)]
        {
            let mut bytes = header(0xc9, 8, 1);
            segment(&mut bytes, 0xcc, &conditioning);
            segment(&mut bytes, 0xda, &[1, 1, 0x4f, 0, 63, 0]);
            bytes.extend([0, 0xff, 0xd9]);
            let report = probe(&bytes).unwrap();
            assert_eq!(report.arithmetic_conditioning_definition_count, 2);
            assert_eq!(report.nondefault_arithmetic_conditioning, nondefault);
            assert!(report.wide_scan_table_selectors);
            assert!(has(&report, UnsupportedFeature::ArithmeticConditioning));
        }
    }

    #[test]
    fn restart_presence_is_not_rejection_but_changed_interval_and_fill_are() {
        for (next, filled) in [(7, false), (8, false), (7, true)] {
            let mut bytes = header(0xc2, 8, 1);
            segment(&mut bytes, 0xdd, &[0, 7]);
            scan(
                &mut bytes,
                if filled {
                    &[0xff, 0xff, 0xd0]
                } else {
                    &[0xff, 0xd0]
                },
            );
            segment(&mut bytes, 0xdd, &[0, next]);
            scan(&mut bytes, &[0xff, 0, 0xd1, 0xff, 0xd0]);
            bytes.extend([0xff, 0xd9]);
            let report = probe(&bytes).unwrap();
            assert_eq!(report.restart_marker_count, 2);
            assert_eq!(report.dri_count, 2);
            assert_eq!(report.last_restart_interval, Some(u16::from(next)));
            assert_eq!(report.is_candidate(), next == 7 && !filled);
            assert_eq!(has(&report, UnsupportedFeature::RestartMarkerFill), filled);
            assert_eq!(
                has(
                    &report,
                    UnsupportedFeature::ChangingRestartInterval {
                        previous: 7,
                        next: 8
                    }
                ),
                next == 8
            );
        }
    }

    #[test]
    fn ignores_markers_inside_metadata_stuffing_and_tail() {
        let mut bytes = header(0xc0, 8, 1);
        segment(&mut bytes, 0xfe, &[0xff, 0xc9, 0xff, 0xdc, 0xff, 0xd9]);
        segment(&mut bytes, 0xe1, &[0xff, 0xcb]);
        bytes.extend([0xff, 0xff]);
        scan(&mut bytes, &[0xff, 0, 0xc9, 0xff, 0, 0xd9]);
        bytes.extend([0xff, 0xd9, 0xff, 0xd8, 0xff, 0xc9]);
        let report = probe(&bytes).unwrap();
        assert!(report.is_candidate());
        assert_eq!((report.scan_count, report.trailing_bytes), (1, 4));
    }

    #[test]
    fn dnl_and_hierarchy_are_explicit_and_markers_after_scans_are_seen() {
        let mut bytes = header(0xc0, 8, 1);
        bytes[8] = 0;
        scan(&mut bytes, &[0]);
        segment(&mut bytes, 0xdc, &[0, 8]);
        bytes.extend([0, 0xff, 0xd0]);
        segment(&mut bytes, 0xde, &[8, 0, 8, 0, 8, 1, 1, 0x11, 0]);
        bytes.extend([0xff, 0xd9]);
        let report = probe(&bytes).unwrap();
        assert_eq!(report.frame.height, 0);
        assert!(has(&report, UnsupportedFeature::DeferredHeight));
        assert!(has(&report, UnsupportedFeature::HierarchicalJpeg));
        assert_eq!(report.restart_marker_count, 1);
    }

    #[test]
    fn quantization_precision_does_not_mean_sample_precision() {
        let mut bytes = header(0xc1, 8, 1);
        let mut table = vec![0x10];
        for _ in 0..64 {
            table.extend([1, 0]);
        }
        segment(&mut bytes, 0xdb, &table);
        scan(&mut bytes, &[0]);
        bytes.extend([0xff, 0xd9]);
        let report = probe(&bytes).unwrap();
        assert!(report.is_candidate());
        assert_eq!(report.quantization_definition_count, 1);
        bytes.truncate(bytes.len() - 2);
        for _ in 0..3 {
            segment(&mut bytes, 0xdb, &table);
        }
        bytes.extend([0xff, 0xd9]);
        assert!(has(
            &probe(&bytes).unwrap(),
            UnsupportedFeature::MoreThanThreeQuantizationDefinitions
        ));
    }

    #[test]
    fn rgb_sampling_uses_source_app0_and_adobe_precedence() {
        for (app0, adobe, rejected) in [
            (false, None, true),
            (true, None, false),
            (false, Some(1), false),
            (false, Some(0), true),
            (true, Some(0), false),
        ] {
            let mut bytes = vec![0xff, 0xd8];
            segment(
                &mut bytes,
                0xc0,
                &[
                    8, 0, 8, 0, 8, 3, b'R', 0x22, 0, b'G', 0x22, 0, b'B', 0x22, 0,
                ],
            );
            if app0 {
                segment(&mut bytes, 0xe0, &[]);
            }
            if let Some(transform) = adobe {
                let mut payload = b"Adobe\0d\0\0\0\0\0".to_vec();
                payload[11] = transform;
                segment(&mut bytes, 0xee, &payload);
            }
            segment(&mut bytes, 0xda, &[1, b'R', 0, 0, 63, 0]);
            bytes.extend([0, 0xff, 0xd9]);
            let report = probe(&bytes).unwrap();
            assert_eq!(has(&report, UnsupportedFeature::RgbSampling), rejected);
            assert_eq!(report.is_candidate(), !rejected);
        }
    }

    #[test]
    fn large_dimensions_and_extra_frames_do_not_allocate_pixel_buffers() {
        let mut bytes = header(0xc0, 8, 1);
        bytes[7..11].fill(0xff);
        bytes[13] = 0x31;
        scan(&mut bytes, &[0]);
        segment(&mut bytes, 0xc5, &[8, 0, 1, 0, 1, 1, 2, 0x11, 0]);
        segment(&mut bytes, 0xda, &[1, 2, 0, 0, 63, 0]);
        bytes.extend([0, 0xff, 0xd9]);
        let report = probe(&bytes).unwrap();
        assert_eq!((report.frame.width, report.frame.height), (65535, 65535));
        assert_eq!((report.frame_count, report.scan_count), (2, 2));
        assert_eq!(report.frame.components[0].id, 1);
        assert!(has(&report, UnsupportedFeature::MultipleFrames));
        assert!(has(
            &report,
            UnsupportedFeature::SamplingFactor {
                component_id: 1,
                horizontal: 3,
                vertical: 1,
            }
        ));
    }

    #[test]
    fn restart_markers_consume_work_budget_and_findings_have_real_offsets() {
        let mut bytes = header(0xc9, 8, 1);
        scan(&mut bytes, &[0xff, 0xd0, 0xff, 0xd1]);
        bytes.extend([0xff, 0xd9]);
        let report = probe(&bytes).unwrap();
        assert_eq!(report.marker_count, 6);
        assert_eq!(report.findings[0].offset, 2);
        assert_eq!(report.findings[0].marker, 0xc9);
        assert!(matches!(
            probe_with_limits(
                &bytes,
                &ProbeLimits {
                    max_markers: 5,
                    ..Default::default()
                }
            ),
            Err(ProbeError::Limit {
                resource: "markers",
                ..
            })
        ));
    }

    #[test]
    fn malformed_headers_fail_with_bounded_segment_reads() {
        for (marker, payload) in [
            (0xc0, vec![8]),
            (0xda, vec![]),
            (0xdd, vec![0]),
            (0xdb, vec![0]),
            (0xcc, vec![0, 0x01]),
            (0xcc, vec![0x10, 64]),
            (0xcc, vec![0x20, 0]),
        ] {
            let mut bytes = header(0xc0, 8, 1);
            segment(&mut bytes, marker, &payload);
            bytes.extend_from_slice(&complete(0xc0, 8, 1)[2..]);
            assert!(
                matches!(probe(&bytes), Err(ProbeError::Invalid { .. })),
                "{marker:02x}"
            );
        }
        for length in [0u16, 1, u16::MAX] {
            let mut bytes = header(0xc0, 8, 1);
            bytes.extend([0xff, 0xfe]);
            bytes.extend(length.to_be_bytes());
            assert!(probe(&bytes).is_err());
        }
        let mut bytes = header(0xc0, 8, 1);
        segment(&mut bytes, 0xda, &[2, 1, 0, 1, 0, 0, 63, 0]);
        bytes.extend([0xff, 0xd9]);
        assert!(probe(&bytes).is_err());
    }

    #[test]
    fn every_truncated_prefix_fails_and_resource_limits_are_exact() {
        let bytes = complete(0xc9, 12, 4);
        for end in 0..bytes.len() {
            assert!(probe(&bytes[..end]).is_err(), "prefix {end}");
        }
        let limits = ProbeLimits {
            max_input_bytes: bytes.len(),
            max_markers: 4,
            max_scans: 1,
            max_findings: 3,
        };
        assert!(probe_with_limits(&bytes, &limits).is_ok());
        for limited in [
            ProbeLimits {
                max_input_bytes: bytes.len() - 1,
                ..limits.clone()
            },
            ProbeLimits {
                max_markers: 3,
                ..limits.clone()
            },
            ProbeLimits {
                max_scans: 0,
                ..limits.clone()
            },
            ProbeLimits {
                max_findings: 2,
                ..limits.clone()
            },
        ] {
            assert!(matches!(
                probe_with_limits(&bytes, &limited),
                Err(ProbeError::Limit { .. })
            ));
        }
        assert!(
            probe_with_limits(
                &complete(0xc0, 8, 1),
                &ProbeLimits {
                    max_findings: 0,
                    ..Default::default()
                }
            )
            .is_ok()
        );
    }

    #[test]
    fn deterministic_malformed_inputs_do_not_panic() {
        let mut state = 0x9e37_79b9u32;
        for length in 0..256 {
            for _ in 0..16 {
                let mut input = vec![0xff, 0xd8];
                for _ in 0..length {
                    state ^= state << 13;
                    state ^= state >> 17;
                    state ^= state << 5;
                    input.push(state as u8);
                }
                let _ = probe(&input);
            }
        }
    }
}
