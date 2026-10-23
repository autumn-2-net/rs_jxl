// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// BSD-3-Clause; retained notice in docs/PORTING_KERNELS.md.
//! Experimental decision transcripts, ported from enc_final_plan.{h,cc} at
//! b866b3defec13494bfbaf705db93581cda118e37. Not a codestream or encoded hash.
//!
//! I/O is explicit and bounded. No environment variables or paths are read.
//! Parent integration must install hooks in source call order and validate
//! decisions against live codec state; parsing alone does not do that.
//! This module does not activate codec capture/replay by itself.

use std::cell::RefCell;
use std::io::{Read, Write};
use std::rc::Rc;

use crate::{Error, Result};

pub mod schema;
mod wire;

pub const MAGIC: &str = "JXL_FINAL_PLAN_V1";
pub const MAX_RECORDS: usize = 1_000_000;
pub const MAX_VALUES: usize = 16_000_000;
pub const MAX_NAME_BYTES: usize = 64;
pub const MAX_TREE_NODES: usize = 1 << 22;

/// Source record/value ceilings, plus an explicit serialized-byte budget.
/// Callers may lower, but not raise, the source record/value ceilings.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Limits {
    pub records: usize,
    pub values: usize,
    pub bytes: u64,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            records: MAX_RECORDS,
            values: MAX_VALUES,
            bytes: 512 * 1024 * 1024,
        }
    }
}

impl Limits {
    fn validate(self) -> Result<Self> {
        if self.records == 0
            || self.records > MAX_RECORDS
            || self.values > MAX_VALUES
            || self.bytes == 0
            || self.bytes == u64::MAX
        {
            return Err(Error::Limit("invalid final-plan budgets"));
        }
        Ok(self)
    }
}

/// Names are byte strings, as in std::string, not restricted to known tags or
/// UTF-8. Known source hooks use schema::Tag. Unknown tags still round-trip.
#[derive(Debug, Eq, PartialEq)]
pub struct Record {
    name: Vec<u8>,
    values: Vec<i64>,
}

impl Record {
    pub fn new(name: impl AsRef<[u8]>, values: Vec<i64>) -> Result<Self> {
        let name = name.as_ref();
        validate_name(name)?;
        if values.len() > MAX_VALUES {
            return Err(Error::Limit("final-plan record values"));
        }
        Ok(Self {
            name: copy_slice(name)?,
            values,
        })
    }

    pub fn name(&self) -> &[u8] {
        &self.name
    }
    pub fn values(&self) -> &[i64] {
        &self.values
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Mode {
    Capture,
    Replay,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Unsupported {
    SqueezeOrLossyPalette,
    PatchReferenceCapture,
}

impl Unsupported {
    pub fn reason(self) -> &'static str {
        match self {
            Self::SqueezeOrLossyPalette => "squeeze or lossy palette",
            Self::PatchReferenceCapture => "patch reference capture",
        }
    }
}

#[derive(Debug)]
pub struct FinalPlan {
    mode: Mode,
    cursor: usize,
    tree_limit: usize,
    unsupported: Option<Unsupported>,
    records: Vec<Record>,
    total_values: usize,
    record_bytes: u64,
    limits: Limits,
}

impl FinalPlan {
    pub fn capture(limits: Limits) -> Result<Self> {
        Ok(Self {
            mode: Mode::Capture,
            cursor: 0,
            tree_limit: 0,
            unsupported: None,
            records: Vec::new(),
            total_values: 0,
            record_bytes: 0,
            limits: limits.validate()?,
        })
    }

    /// Transactional constructor: a failed read cannot replace an active plan.
    /// Reads at most limits.bytes + one EOF/over-budget probe byte.
    pub fn read(reader: impl Read, limits: Limits) -> Result<Self> {
        wire::read(reader, limits.validate()?)
    }

    /// Preflight all budgets before writing. Flush the supplied writer, but
    /// never open/close a file. I/O failure can leave partial bytes in the sink.
    /// Like source Write, this does not replace the outer ensure_supported check.
    pub fn write(&self, writer: impl Write) -> Result<()> {
        wire::write(self, writer)
    }

    pub fn mode(&self) -> Mode {
        self.mode
    }
    pub fn cursor(&self) -> usize {
        self.cursor
    }
    pub fn records(&self) -> &[Record] {
        &self.records
    }
    pub fn tree_limit(&self) -> usize {
        self.tree_limit
    }
    pub fn unsupported(&self) -> Option<Unsupported> {
        self.unsupported
    }

    pub fn set_tree_limit(&mut self, limit: usize) -> Result<()> {
        if limit > MAX_TREE_NODES {
            return Err(Error::Limit("final-plan tree nodes"));
        }
        self.tree_limit = limit;
        Ok(())
    }

    /// Source records only the most recently encountered unsupported reason.
    pub fn mark_unsupported(&mut self, reason: Unsupported) {
        self.unsupported = Some(reason);
    }

    pub fn ensure_supported(&self) -> Result<()> {
        match self.unsupported {
            Some(reason) => Err(invalid(format!(
                "unsupported final plan: {}",
                reason.reason()
            ))),
            None => Ok(()),
        }
    }

    /// Capture copies values without changing them. Replay replaces values
    /// from the next matching tag. Mismatch/allocation failure changes neither
    /// cursor nor values. Schema validation belongs to the consuming hook.
    pub fn exchange(&mut self, name: impl AsRef<[u8]>, values: &mut Vec<i64>) -> Result<()> {
        let name = name.as_ref();
        validate_name(name)?;
        match self.mode {
            Mode::Capture => {
                self.check_append(name, values)?;
                self.append(Record {
                    name: copy_slice(name)?,
                    values: copy_slice(values)?,
                })
            }
            Mode::Replay => {
                let record = self
                    .records
                    .get(self.cursor)
                    .filter(|record| record.name == name)
                    .ok_or_else(|| {
                        invalid(format!(
                            "final plan expected {} at record {}",
                            String::from_utf8_lossy(name),
                            self.cursor
                        ))
                    })?;
                let replacement = copy_slice(&record.values)?;
                *values = replacement;
                self.cursor += 1;
                Ok(())
            }
        }
    }

    /// Exact source Finished semantics: capture always succeeds; replay must
    /// consume every record. Unsupported decisions are a separate outer check.
    pub fn finished(&self) -> Result<()> {
        if self.mode == Mode::Replay && self.cursor != self.records.len() {
            return Err(invalid(format!(
                "final plan consumed {} of {} records",
                self.cursor,
                self.records.len()
            )));
        }
        Ok(())
    }

    pub fn parameters(&mut self, parameters: &mut schema::FrameParameters) -> Result<()> {
        let mut values = parameters.to_values().to_vec();
        self.exchange(schema::Tag::Frame.name(), &mut values)?;
        let decoded = schema::FrameParameters::from_values(&values, parameters.keep_invisible)?;
        if self.mode == Mode::Replay {
            *parameters = decoded;
        }
        Ok(())
    }

    /// Input equality is checked before installing its derived tree budget.
    /// Parent must separately enforce the source's input eligibility checks.
    pub fn input(&mut self, expected: schema::InputShape) -> Result<()> {
        let mut values = expected.to_values()?.to_vec();
        self.exchange(schema::Tag::Input.name(), &mut values)?;
        expected.validate_record(&values)?;
        self.set_tree_limit(expected.tree_limit())
    }

    fn check_append(&self, name: &[u8], values: &[i64]) -> Result<(usize, u64)> {
        validate_name(name)?;
        if self.records.len() >= self.limits.records
            || values.len() > self.limits.values - self.total_values
        {
            return Err(Error::Limit("final-plan records/values"));
        }
        let bytes = self
            .record_bytes
            .checked_add(wire::record_size(name, values)?)
            .ok_or(Error::Limit("final-plan serialized bytes"))?;
        self.check_wire_size(self.records.len() + 1, bytes)?;
        Ok((self.total_values + values.len(), bytes))
    }

    fn check_wire_size(&self, count: usize, body: u64) -> Result<()> {
        let total = wire::header_size(count)
            .checked_add(body)
            .ok_or(Error::Limit("final-plan serialized bytes"))?;
        if total > self.limits.bytes {
            return Err(Error::Limit("final-plan serialized bytes"));
        }
        Ok(())
    }

    fn append(&mut self, record: Record) -> Result<()> {
        let (values, bytes) = self.check_append(&record.name, &record.values)?;
        self.records
            .try_reserve(1)
            .map_err(|_| Error::Limit("final-plan allocation"))?;
        self.records.push(record);
        self.total_values = values;
        self.record_bytes = bytes;
        Ok(())
    }
}

/// Explicit same-thread ownership. No raw pointer, Send/Sync implementation,
/// escaped RefMut, or callback while a mutable plan borrow is held.
#[derive(Clone, Debug)]
pub struct Handle {
    plan: Rc<RefCell<FinalPlan>>,
    mode: Mode,
}

impl From<FinalPlan> for Handle {
    fn from(plan: FinalPlan) -> Self {
        Self {
            mode: plan.mode,
            plan: Rc::new(RefCell::new(plan)),
        }
    }
}

impl Handle {
    pub fn mode(&self) -> Mode {
        self.mode
    }
    pub fn cursor(&self) -> Result<usize> {
        Ok(self.plan.try_borrow().map_err(|_| busy())?.cursor())
    }
    pub fn tree_limit(&self) -> Result<usize> {
        Ok(self.plan.try_borrow().map_err(|_| busy())?.tree_limit())
    }
    pub fn finished(&self) -> Result<()> {
        self.plan.try_borrow().map_err(|_| busy())?.finished()
    }
    pub fn ensure_supported(&self) -> Result<()> {
        self.plan
            .try_borrow()
            .map_err(|_| busy())?
            .ensure_supported()
    }
    pub fn write(&self, writer: impl Write) -> Result<()> {
        self.plan.try_borrow().map_err(|_| busy())?.write(writer)
    }
    pub fn exchange(&self, name: impl AsRef<[u8]>, values: &mut Vec<i64>) -> Result<()> {
        self.plan
            .try_borrow_mut()
            .map_err(|_| busy())?
            .exchange(name, values)
    }
    pub fn parameters(&self, parameters: &mut schema::FrameParameters) -> Result<()> {
        self.plan
            .try_borrow_mut()
            .map_err(|_| busy())?
            .parameters(parameters)
    }
    pub fn input(&self, expected: schema::InputShape) -> Result<()> {
        self.plan
            .try_borrow_mut()
            .map_err(|_| busy())?
            .input(expected)
    }
    pub fn mark_unsupported(&self, reason: Unsupported) -> Result<()> {
        self.plan
            .try_borrow_mut()
            .map_err(|_| busy())?
            .mark_unsupported(reason);
        Ok(())
    }
}

thread_local! {
    static CURRENT: RefCell<Option<Handle>> = const { RefCell::new(None) };
}

pub fn current() -> Option<Handle> {
    CURRENT.with(|slot| slot.borrow().clone())
}
pub fn replaying() -> bool {
    current().is_some_and(|plan| plan.mode() == Mode::Replay)
}

/// Synchronous scope, equivalent to a null source ThreadPool. None suspends
/// capture during speculative work. Nested scopes restore on error and panic;
/// unlike an exposed guard, callers cannot forget it or drop it out of order.
/// Arbitrary user-spawned tasks are not covered: hooks must remain synchronous.
pub fn scope<R>(plan: Option<&Handle>, work: impl FnOnce() -> R) -> R {
    struct Restore(Option<Handle>);
    impl Drop for Restore {
        fn drop(&mut self) {
            CURRENT.with(|slot| {
                *slot.borrow_mut() = self.0.take();
            });
        }
    }
    let _restore = Restore(CURRENT.with(|slot| slot.replace(plan.cloned())));
    crate::execution::serial(work)
}

/// No installed transcript is a successful no-op, as in FinalPlanExchange.
pub fn exchange(name: impl AsRef<[u8]>, values: &mut Vec<i64>) -> Result<()> {
    match current() {
        Some(plan) => plan.exchange(name, values),
        None => Ok(()),
    }
}

pub fn parameters(parameters: &mut schema::FrameParameters) -> Result<()> {
    match current() {
        Some(plan) => plan.parameters(parameters),
        None => {
            schema::FrameParameters::from_values(&parameters.to_values(), parameters.keep_invisible)
                .map(|_| ())
        }
    }
}

fn invalid(message: impl Into<String>) -> Error {
    Error::InvalidInput(message.into())
}
fn busy() -> Error {
    Error::Encoding("final plan is already borrowed".into())
}
fn whitespace(byte: u8) -> bool {
    matches!(byte, b' ' | b'\t' | b'\n' | b'\r' | 0x0b | 0x0c)
}
fn validate_name(name: &[u8]) -> Result<()> {
    if name.is_empty() || name.len() > MAX_NAME_BYTES || name.iter().any(|&b| whitespace(b)) {
        return Err(invalid("invalid final-plan record name"));
    }
    Ok(())
}
fn copy_slice<T: Copy>(values: &[T]) -> Result<Vec<T>> {
    let mut copy = Vec::new();
    copy.try_reserve_exact(values.len())
        .map_err(|_| Error::Limit("final-plan allocation"))?;
    copy.extend_from_slice(values);
    Ok(copy)
}

#[cfg(test)]
mod tests;
