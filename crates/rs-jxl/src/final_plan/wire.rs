// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// BSD-3-Clause; retained notice in docs/PORTING_KERNELS.md.
//! Stream translation of enc_final_plan.cc Read/Write with opt-in budgets.

use std::io::{BufRead, BufReader, Read, Take, Write};

use super::*;

struct Reader<R: Read> {
    input: BufReader<Take<R>>,
    consumed: u64,
    limit: u64,
}

impl<R: Read> Reader<R> {
    fn byte(&mut self) -> Result<Option<u8>> {
        let byte = self.input.fill_buf()?.first().copied();
        if byte.is_some() {
            if self.consumed == self.limit {
                return Err(Error::Limit("final-plan input bytes"));
            }
            self.input.consume(1);
            self.consumed += 1;
        }
        Ok(byte)
    }

    fn token(&mut self, mut consume: impl FnMut(u8) -> Result<()>) -> Result<()> {
        let first = loop {
            match self.byte()? {
                Some(byte) if whitespace(byte) => continue,
                Some(byte) => break byte,
                None => return Err(invalid("truncated final plan")),
            }
        };
        consume(first)?;
        while let Some(byte) = self.byte()? {
            if whitespace(byte) {
                break;
            }
            consume(byte)?;
        }
        Ok(())
    }

    fn name(&mut self) -> Result<Vec<u8>> {
        let mut bytes = Vec::new();
        self.token(|byte| {
            bytes
                .try_reserve(1)
                .map_err(|_| Error::Limit("final-plan allocation"))?;
            bytes.push(byte);
            Ok(())
        })?;
        Ok(bytes)
    }

    // No numeric-token allocation: signed decimal, optional plus/minus and
    // leading zeros, with checked magnitude even for INT64_MIN.
    fn decimal(&mut self, unsigned: bool) -> Result<(bool, u64)> {
        let mut first = true;
        let mut negative = false;
        let mut digits = false;
        let mut magnitude = 0u64;
        self.token(|byte| {
            if first && (byte == b'+' || byte == b'-') {
                first = false;
                negative = byte == b'-';
                if unsigned && negative {
                    return Err(invalid("negative final-plan size"));
                }
                return Ok(());
            }
            first = false;
            if !byte.is_ascii_digit() {
                return Err(invalid("invalid final-plan decimal integer"));
            }
            digits = true;
            magnitude = magnitude
                .checked_mul(10)
                .and_then(|v| v.checked_add(u64::from(byte - b'0')))
                .ok_or_else(|| invalid("final-plan integer overflow"))?;
            if !unsigned && magnitude > (i64::MAX as u64) + u64::from(negative) {
                return Err(invalid("final-plan integer overflow"));
            }
            Ok(())
        })?;
        if !digits {
            return Err(invalid("missing final-plan integer digits"));
        }
        Ok((negative, magnitude))
    }

    fn integer(&mut self) -> Result<i64> {
        let (negative, magnitude) = self.decimal(false)?;
        Ok(if negative {
            (magnitude as i64).wrapping_neg()
        } else {
            magnitude as i64
        })
    }

    fn size(&mut self) -> Result<usize> {
        usize::try_from(self.decimal(true)?.1).map_err(|_| Error::Limit("final-plan size"))
    }

    fn end(&mut self) -> Result<()> {
        while let Some(byte) = self.byte()? {
            if !whitespace(byte) {
                return Err(invalid("trailing final-plan data"));
            }
        }
        Ok(())
    }
}

pub(super) fn read(input: impl Read, limits: Limits) -> Result<FinalPlan> {
    let mut reader = Reader {
        input: BufReader::new(input.take(limits.bytes.saturating_add(1))),
        consumed: 0,
        limit: limits.bytes,
    };
    let mut magic_bytes = 0;
    reader.token(|byte| {
        if MAGIC.as_bytes().get(magic_bytes) != Some(&byte) {
            return Err(invalid("invalid final-plan V1 magic"));
        }
        magic_bytes += 1;
        Ok(())
    })?;
    if magic_bytes != MAGIC.len() {
        return Err(invalid("invalid final-plan V1 magic"));
    }
    let count = reader.size()?;
    if count == 0 || count > limits.records {
        return Err(Error::Limit("final-plan record count"));
    }
    let mut plan = FinalPlan::capture(limits)?;
    for _ in 0..count {
        let name = reader.name()?;
        let n = reader.size()?;
        if n > limits.values - plan.total_values {
            return Err(Error::Limit("final-plan values"));
        }
        // At least one digit per value plus separators must fit. Compute in
        // wider arithmetic so unlimited byte budgets cannot hide overflow.
        let minimum = (n as u128 * 2).saturating_sub(1);
        if minimum > u128::from(limits.bytes - reader.consumed) {
            return Err(Error::Limit("final-plan input bytes"));
        }
        let mut values = Vec::new();
        // A declared count is not evidence that any values exist. Grow only
        // after parsing each value, including with unlimited default budgets.
        for _ in 0..n {
            let value = reader.integer()?;
            values
                .try_reserve(1)
                .map_err(|_| Error::Limit("final-plan allocation"))?;
            values.push(value);
        }
        plan.append(Record { name, values })?;
    }
    reader.end()?;
    plan.mode = Mode::Replay;
    Ok(plan)
}

fn digits(mut number: u64) -> u64 {
    let mut count = 1;
    while number >= 10 {
        number /= 10;
        count += 1;
    }
    count
}

pub(super) fn header_size(count: usize) -> u64 {
    MAGIC.len() as u64 + 2 + digits(count as u64)
}

pub(super) fn record_size(name: &[u8], values: &[i64]) -> Result<u64> {
    let mut size = name.len() as u64 + 2 + digits(values.len() as u64);
    for &value in values {
        size = size
            .checked_add(1 + digits(value.unsigned_abs()) + u64::from(value < 0))
            .ok_or(Error::Limit("final-plan serialized bytes"))?;
    }
    Ok(size)
}

pub(super) fn write(plan: &FinalPlan, mut writer: impl Write) -> Result<()> {
    // Source Write can emit an empty, unreadable transcript. Reject that case.
    if plan.records.is_empty() {
        return Err(invalid("cannot write an empty final plan"));
    }
    plan.check_wire_size(plan.records.len(), plan.record_bytes)?;
    writeln!(writer, "{MAGIC} {}", plan.records.len())?;
    for record in &plan.records {
        writer.write_all(&record.name)?;
        write!(writer, " {}", record.values.len())?;
        for value in &record.values {
            write!(writer, " {value}")?;
        }
        writer.write_all(b"\n")?;
    }
    writer.flush()?;
    Ok(())
}
