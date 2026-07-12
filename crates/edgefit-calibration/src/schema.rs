use std::collections::BTreeSet;
use std::fmt;

use crate::json::{
    array, boolean, decimal_u64, decimal_u64_array, exact_fields, expect_literal_string,
    nonempty_string, object, optional_string, required, string, JsonParser,
};

pub const EVIDENCE_SCHEMA: &str = "edgefit.calibration_evidence.v1";
pub const VERIFICATION_SCHEMA: &str = "edgefit.calibration_verification.v1";
pub const MAX_LATENCY_SAMPLES: usize = 100_000;
pub const MAX_ATTACHMENTS: usize = 1_024;
pub const MAX_ATTACHMENT_BYTES: u64 = 1 << 30;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Error(String);

impl Error {
    pub(crate) fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Identity {
    pub target_id: String,
    pub device_id: String,
    pub runtime_name: String,
    pub runtime_version: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Environment {
    pub operating_system: String,
    pub architecture: String,
    pub hardware: String,
    pub toolchain: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Capture {
    pub captured_at: String,
    pub command: String,
    pub warmup_runs: u64,
    pub measured_runs: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Bindings {
    pub model_sha256: String,
    pub target_profile_sha256: String,
    pub runtime_binary_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RuntimeResult {
    pub accepted: bool,
    pub rejected_reason: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Measurements {
    pub arena_high_water_bytes: u64,
    pub latency_ns: Vec<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Thresholds {
    pub arena_budget_bytes: u64,
    pub p95_latency_budget_ns: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Attachment {
    pub name: String,
    pub path: String,
    pub media_type: String,
    pub bytes: u64,
    pub sha256: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Evidence {
    pub identity: Identity,
    pub environment: Environment,
    pub capture: Capture,
    pub bindings: Bindings,
    pub runtime: RuntimeResult,
    pub measurements: Measurements,
    pub thresholds: Thresholds,
    pub attachments: Vec<Attachment>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExpectedBindings {
    pub model_sha256: String,
    pub target_profile_sha256: String,
    pub runtime_binary_sha256: String,
}

impl From<Bindings> for ExpectedBindings {
    fn from(value: Bindings) -> Self {
        Self {
            model_sha256: value.model_sha256,
            target_profile_sha256: value.target_profile_sha256,
            runtime_binary_sha256: value.runtime_binary_sha256,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerificationBudget {
    pub arena_bytes: u64,
    pub p95_latency_ns: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoadedAttachment<'a> {
    pub path: &'a str,
    pub bytes: &'a [u8],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CheckStatus {
    Pass,
    Fail,
}

impl CheckStatus {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Pass => "pass",
            Self::Fail => "fail",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Check {
    pub id: &'static str,
    pub status: CheckStatus,
    pub detail: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Metrics {
    pub sample_count: u64,
    pub latency_p50_ns: u64,
    pub latency_p95_ns: u64,
    pub latency_mean_ns: u64,
    pub arena_utilization_ppm: u64,
    pub arena_error_bytes: i128,
    pub p95_latency_error_ns: i128,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Verification {
    pub status: CheckStatus,
    pub evidence_sha256: String,
    pub expected_bindings: ExpectedBindings,
    pub budget: VerificationBudget,
    pub metrics: Metrics,
    pub checks: Vec<Check>,
    pub verification_sha256: String,
}

pub fn parse_evidence(input: &str) -> Result<Evidence> {
    let value = JsonParser::new(input).parse()?;
    let root = object(&value, "evidence")?;
    exact_fields(
        root,
        &[
            "schema",
            "identity",
            "environment",
            "capture",
            "bindings",
            "runtime",
            "measurements",
            "thresholds",
            "attachments",
            "attestation",
        ],
        "evidence",
    )?;
    expect_literal_string(root, "schema", EVIDENCE_SCHEMA)?;

    let attestation = object(required(root, "attestation")?, "attestation")?;
    exact_fields(attestation, &["kind"], "attestation")?;
    expect_literal_string(attestation, "kind", "none")?;

    let identity = object(required(root, "identity")?, "identity")?;
    exact_fields(
        identity,
        &["target_id", "device_id", "runtime_name", "runtime_version"],
        "identity",
    )?;
    let identity = Identity {
        target_id: nonempty_string(identity, "target_id")?,
        device_id: nonempty_string(identity, "device_id")?,
        runtime_name: nonempty_string(identity, "runtime_name")?,
        runtime_version: nonempty_string(identity, "runtime_version")?,
    };

    let environment = object(required(root, "environment")?, "environment")?;
    exact_fields(
        environment,
        &["operating_system", "architecture", "hardware", "toolchain"],
        "environment",
    )?;
    let environment = Environment {
        operating_system: nonempty_string(environment, "operating_system")?,
        architecture: nonempty_string(environment, "architecture")?,
        hardware: nonempty_string(environment, "hardware")?,
        toolchain: nonempty_string(environment, "toolchain")?,
    };

    let capture = object(required(root, "capture")?, "capture")?;
    exact_fields(
        capture,
        &["captured_at", "command", "warmup_runs", "measured_runs"],
        "capture",
    )?;
    let captured_at = nonempty_string(capture, "captured_at")?;
    validate_timestamp(&captured_at)?;
    let capture = Capture {
        captured_at,
        command: nonempty_string(capture, "command")?,
        warmup_runs: decimal_u64(capture, "warmup_runs")?,
        measured_runs: decimal_u64(capture, "measured_runs")?,
    };

    let bindings = object(required(root, "bindings")?, "bindings")?;
    exact_fields(
        bindings,
        &["model_sha256", "target_profile_sha256", "runtime_binary_sha256"],
        "bindings",
    )?;
    let bindings = Bindings {
        model_sha256: hash_field(bindings, "model_sha256")?,
        target_profile_sha256: hash_field(bindings, "target_profile_sha256")?,
        runtime_binary_sha256: hash_field(bindings, "runtime_binary_sha256")?,
    };

    let runtime = object(required(root, "runtime")?, "runtime")?;
    exact_fields(runtime, &["accepted", "rejected_reason"], "runtime")?;
    let accepted = boolean(runtime, "accepted")?;
    let rejected_reason = optional_string(runtime, "rejected_reason")?;
    validate_runtime_result(accepted, rejected_reason.as_deref())?;
    let runtime = RuntimeResult {
        accepted,
        rejected_reason,
    };

    let measurements = object(required(root, "measurements")?, "measurements")?;
    exact_fields(
        measurements,
        &["arena_high_water", "latency"],
        "measurements",
    )?;
    let arena = object(
        required(measurements, "arena_high_water")?,
        "measurements.arena_high_water",
    )?;
    exact_fields(arena, &["unit", "value"], "measurements.arena_high_water")?;
    expect_literal_string(arena, "unit", "bytes")?;
    let arena_high_water_bytes = decimal_u64(arena, "value")?;

    let latency = object(
        required(measurements, "latency")?,
        "measurements.latency",
    )?;
    exact_fields(latency, &["unit", "samples"], "measurements.latency")?;
    expect_literal_string(latency, "unit", "ns")?;
    let latency_ns = decimal_u64_array(latency, "samples")?;
    validate_sample_count(capture.measured_runs, latency_ns.len())?;
    let measurements = Measurements {
        arena_high_water_bytes,
        latency_ns,
    };

    let thresholds = object(required(root, "thresholds")?, "thresholds")?;
    exact_fields(
        thresholds,
        &["arena_budget", "p95_latency_budget"],
        "thresholds",
    )?;
    let arena_budget = object(
        required(thresholds, "arena_budget")?,
        "thresholds.arena_budget",
    )?;
    exact_fields(arena_budget, &["unit", "value"], "thresholds.arena_budget")?;
    expect_literal_string(arena_budget, "unit", "bytes")?;
    let latency_budget = object(
        required(thresholds, "p95_latency_budget")?,
        "thresholds.p95_latency_budget",
    )?;
    exact_fields(
        latency_budget,
        &["unit", "value"],
        "thresholds.p95_latency_budget",
    )?;
    expect_literal_string(latency_budget, "unit", "ns")?;
    let thresholds = Thresholds {
        arena_budget_bytes: decimal_u64(arena_budget, "value")?,
        p95_latency_budget_ns: decimal_u64(latency_budget, "value")?,
    };

    let attachment_values = array(required(root, "attachments")?, "attachments")?;
    if attachment_values.len() > MAX_ATTACHMENTS {
        return Err(Error::new(format!(
            "attachments exceeds limit {MAX_ATTACHMENTS}"
        )));
    }
    let mut attachments = Vec::with_capacity(attachment_values.len());
    let mut names = BTreeSet::new();
    let mut paths = BTreeSet::new();
    for (index, value) in attachment_values.iter().enumerate() {
        let context = format!("attachments[{index}]");
        let item = object(value, &context)?;
        exact_fields(
            item,
            &["name", "path", "media_type", "bytes", "sha256"],
            &context,
        )?;
        let name = nonempty_string(item, "name")?;
        let path = nonempty_string(item, "path")?;
        validate_attachment_name(&name)?;
        validate_attachment_path(&path)?;
        if !names.insert(name.clone()) {
            return Err(Error::new(format!("duplicate attachment name {name}")));
        }
        if !paths.insert(path.clone()) {
            return Err(Error::new(format!("duplicate attachment path {path}")));
        }
        let bytes = decimal_u64(item, "bytes")?;
        if bytes > MAX_ATTACHMENT_BYTES {
            return Err(Error::new(format!(
                "attachment {path} exceeds byte limit"
            )));
        }
        attachments.push(Attachment {
            name,
            path,
            media_type: validate_media_type(&nonempty_string(item, "media_type")?)?,
            bytes,
            sha256: hash_field(item, "sha256")?,
        });
    }

    Ok(Evidence {
        identity,
        environment,
        capture,
        bindings,
        runtime,
        measurements,
        thresholds,
        attachments,
    })
}

pub(crate) fn validate_sample_count(measured_runs: u64, sample_count: usize) -> Result<u64> {
    if sample_count == 0 {
        return Err(Error::new("measurements.latency.samples must not be empty"));
    }
    if sample_count > MAX_LATENCY_SAMPLES {
        return Err(Error::new(format!(
            "measurements.latency.samples exceeds limit {MAX_LATENCY_SAMPLES}"
        )));
    }
    let sample_count = u64::try_from(sample_count)
        .map_err(|_| Error::new("latency sample count cannot be represented"))?;
    if measured_runs != sample_count {
        return Err(Error::new(
            "capture.measured_runs does not match latency sample count",
        ));
    }
    Ok(sample_count)
}

pub(crate) fn validate_sha256_hex(value: &str, context: &str) -> Result<()> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(Error::new(format!(
            "field {context} must be a lowercase SHA-256 hex digest"
        )));
    }
    Ok(())
}

pub(crate) fn validate_runtime_result(accepted: bool, rejected_reason: Option<&str>) -> Result<()> {
    if accepted && rejected_reason.is_some() {
        return Err(Error::new(
            "runtime.rejected_reason must be null when accepted is true",
        ));
    }
    if let Some(reason) = rejected_reason {
        if reason.trim().is_empty() || reason.chars().any(char::is_control) {
            return Err(Error::new(
                "runtime.rejected_reason must be a non-empty safe string",
            ));
        }
    }
    if !accepted && rejected_reason.is_none() {
        return Err(Error::new(
            "runtime.rejected_reason is required when accepted is false",
        ));
    }
    Ok(())
}

pub(crate) fn validate_timestamp(value: &str) -> Result<()> {
    let bytes = value.as_bytes();
    if bytes.len() < 20
        || bytes.get(4) != Some(&b'-')
        || bytes.get(7) != Some(&b'-')
        || bytes.get(10) != Some(&b'T')
        || bytes.get(13) != Some(&b':')
        || bytes.get(16) != Some(&b':')
        || *bytes.last().unwrap_or(&0) != b'Z'
    {
        return Err(Error::new(
            "capture.captured_at must be an RFC 3339 UTC timestamp",
        ));
    }
    let year = digits(bytes, 0, 4)?;
    let month = digits(bytes, 5, 7)?;
    let day = digits(bytes, 8, 10)?;
    let hour = digits(bytes, 11, 13)?;
    let minute = digits(bytes, 14, 16)?;
    let second = digits(bytes, 17, 19)?;
    let fraction = &bytes[19..bytes.len() - 1];
    if !fraction.is_empty()
        && (fraction[0] != b'.'
            || fraction.len() == 1
            || !fraction[1..].iter().all(u8::is_ascii_digit))
    {
        return Err(Error::new(
            "capture.captured_at has an invalid fractional second",
        ));
    }
    let leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
    let month_days = [
        31,
        if leap { 29 } else { 28 },
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ];
    if !(1..=12).contains(&month)
        || day == 0
        || day > month_days[(month - 1) as usize]
        || hour > 23
        || minute > 59
        || second > 59
    {
        return Err(Error::new(
            "capture.captured_at contains an invalid date or time",
        ));
    }
    Ok(())
}

fn digits(bytes: &[u8], start: usize, end: usize) -> Result<u32> {
    let slice = bytes
        .get(start..end)
        .ok_or_else(|| Error::new("invalid timestamp"))?;
    if !slice.iter().all(u8::is_ascii_digit) {
        return Err(Error::new(
            "capture.captured_at must contain decimal date components",
        ));
    }
    slice.iter().try_fold(0_u32, |value, byte| {
        value
            .checked_mul(10)
            .and_then(|value| value.checked_add((byte - b'0') as u32))
            .ok_or_else(|| Error::new("timestamp component overflow"))
    })
}

pub(crate) fn validate_attachment_name(value: &str) -> Result<()> {
    if value.is_empty()
        || value.trim() != value
        || value == "."
        || value == ".."
        || value.contains('/')
        || value.contains('\\')
        || value.contains(':')
        || value.chars().any(char::is_control)
    {
        return Err(Error::new("attachment name must be a safe leaf name"));
    }
    Ok(())
}

pub(crate) fn validate_attachment_path(value: &str) -> Result<()> {
    if value.is_empty()
        || value.trim() != value
        || value.starts_with('/')
        || value.starts_with('\\')
        || value.contains('\\')
        || value.contains(':')
        || value.chars().any(char::is_control)
        || value
            .split('/')
            .any(|part| part.is_empty() || part == "." || part == ".." || part.trim() != part)
    {
        return Err(Error::new(format!("unsafe attachment path {value:?}")));
    }
    Ok(())
}

pub(crate) fn validate_media_type(value: &str) -> Result<String> {
    let mut parts = value.split('/');
    let valid_part = |part: &str| {
        !part.is_empty()
            && part.bytes().all(|byte| {
                byte.is_ascii_alphanumeric()
                    || matches!(
                        byte,
                        b'!' | b'#' | b'$' | b'&' | b'^' | b'_' | b'.' | b'+' | b'-'
                    )
            })
    };
    let first = parts.next().unwrap_or("");
    let second = parts.next().unwrap_or("");
    if !valid_part(first) || !valid_part(second) || parts.next().is_some() {
        return Err(Error::new(
            "attachment media_type must be a parameter-free media type",
        ));
    }
    Ok(value.to_ascii_lowercase())
}

fn hash_field(
    object: &std::collections::BTreeMap<String, crate::json::JsonValue>,
    key: &str,
) -> Result<String> {
    let value = string(object, key)?;
    validate_sha256_hex(&value, key)?;
    Ok(value)
}
