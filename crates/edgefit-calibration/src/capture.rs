//! 外部 runtime/device 测量的严格打包输入契约。
//!
//! 清单只描述原始采集事实；model、target、runtime 哈希与 measured-runs 由 core 推导。

use std::collections::BTreeSet;

use crate::json::{
    array, boolean, decimal_u64, decimal_u64_array, exact_fields, expect_literal_string,
    nonempty_string, object, optional_string, required, JsonParser,
};
use crate::schema::{
    validate_attachment_name, validate_attachment_path, validate_media_type,
    validate_runtime_result, validate_timestamp,
};
use crate::{Environment, Error, Result, RuntimeResult, MAX_ATTACHMENTS, MAX_LATENCY_SAMPLES};

pub const CAPTURE_SCHEMA: &str = "edgefit.calibration_capture.v1";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CaptureIdentity {
    pub device_id: String,
    pub runtime_name: String,
    pub runtime_version: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CaptureMetadata {
    pub captured_at: String,
    pub command: String,
    pub warmup_runs: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CaptureMeasurements {
    pub arena_high_water_bytes: u64,
    pub latency_ns: Vec<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CaptureAttachment {
    pub name: String,
    pub path: String,
    pub media_type: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CaptureManifest {
    pub identity: CaptureIdentity,
    pub environment: Environment,
    pub capture: CaptureMetadata,
    pub runtime: RuntimeResult,
    pub measurements: CaptureMeasurements,
    pub p95_latency_budget_ns: u64,
    pub runtime_binary: String,
    pub attachments: Vec<CaptureAttachment>,
}

pub fn parse_capture_manifest(input: &str) -> Result<CaptureManifest> {
    let value = JsonParser::new(input).parse()?;
    let root = object(&value, "capture manifest")?;
    exact_fields(
        root,
        &[
            "schema",
            "identity",
            "environment",
            "capture",
            "runtime",
            "measurements",
            "thresholds",
            "runtime_binary",
            "attachments",
        ],
        "capture_manifest",
    )?;
    expect_literal_string(root, "schema", CAPTURE_SCHEMA)?;

    let identity = object(required(root, "identity")?, "capture identity")?;
    exact_fields(
        identity,
        &["device_id", "runtime_name", "runtime_version"],
        "capture_manifest.identity",
    )?;
    let identity = CaptureIdentity {
        device_id: nonempty_string(identity, "device_id")?,
        runtime_name: nonempty_string(identity, "runtime_name")?,
        runtime_version: nonempty_string(identity, "runtime_version")?,
    };

    let environment = object(required(root, "environment")?, "capture environment")?;
    exact_fields(
        environment,
        &["operating_system", "architecture", "hardware", "toolchain"],
        "capture_manifest.environment",
    )?;
    let environment = Environment {
        operating_system: nonempty_string(environment, "operating_system")?,
        architecture: nonempty_string(environment, "architecture")?,
        hardware: nonempty_string(environment, "hardware")?,
        toolchain: nonempty_string(environment, "toolchain")?,
    };

    let capture = object(required(root, "capture")?, "capture metadata")?;
    exact_fields(
        capture,
        &["captured_at", "command", "warmup_runs"],
        "capture_manifest.capture",
    )?;
    let captured_at = nonempty_string(capture, "captured_at")?;
    validate_timestamp(&captured_at)?;
    let capture = CaptureMetadata {
        captured_at,
        command: nonempty_string(capture, "command")?,
        warmup_runs: decimal_u64(capture, "warmup_runs")?,
    };

    let runtime = object(required(root, "runtime")?, "capture runtime")?;
    exact_fields(
        runtime,
        &["accepted", "rejected_reason"],
        "capture_manifest.runtime",
    )?;
    let accepted = boolean(runtime, "accepted")?;
    let rejected_reason = optional_string(runtime, "rejected_reason")?;
    validate_runtime_result(accepted, rejected_reason.as_deref())?;
    let runtime = RuntimeResult {
        accepted,
        rejected_reason,
    };

    let measurements = object(required(root, "measurements")?, "capture measurements")?;
    exact_fields(
        measurements,
        &["arena_high_water_bytes", "latency_ns"],
        "capture_manifest.measurements",
    )?;
    let latency_ns = decimal_u64_array(measurements, "latency_ns")?;
    if latency_ns.is_empty() || latency_ns.len() > MAX_LATENCY_SAMPLES {
        return Err(Error::new(format!(
            "capture latency_ns must contain between 1 and {MAX_LATENCY_SAMPLES} samples"
        )));
    }
    if latency_ns.contains(&0) {
        return Err(Error::new("capture latency samples must be greater than zero"));
    }
    let measurements = CaptureMeasurements {
        arena_high_water_bytes: decimal_u64(measurements, "arena_high_water_bytes")?,
        latency_ns,
    };

    let thresholds = object(required(root, "thresholds")?, "capture thresholds")?;
    exact_fields(
        thresholds,
        &["p95_latency_budget_ns"],
        "capture_manifest.thresholds",
    )?;
    let p95_latency_budget_ns = decimal_u64(thresholds, "p95_latency_budget_ns")?;
    if p95_latency_budget_ns == 0 {
        return Err(Error::new(
            "capture p95_latency_budget_ns must be greater than zero",
        ));
    }

    let runtime_binary = nonempty_string(root, "runtime_binary")?;
    validate_attachment_path(&runtime_binary)?;
    let attachment_values = array(required(root, "attachments")?, "attachments")?;
    if attachment_values.len() > MAX_ATTACHMENTS.saturating_sub(2) {
        return Err(Error::new("capture attachments exceeds limit"));
    }
    let mut names = BTreeSet::new();
    let mut paths = BTreeSet::new();
    paths.insert(runtime_binary.clone());
    let mut attachments = Vec::with_capacity(attachment_values.len());
    for value in attachment_values {
        let attachment = object(value, "capture attachment")?;
        exact_fields(
            attachment,
            &["name", "path", "media_type"],
            "capture_manifest.attachment",
        )?;
        let name = nonempty_string(attachment, "name")?;
        validate_attachment_name(&name)?;
        if !names.insert(name.clone()) {
            return Err(Error::new(format!("duplicate capture attachment name {name}")));
        }
        let path = nonempty_string(attachment, "path")?;
        validate_attachment_path(&path)?;
        if !paths.insert(path.clone()) {
            return Err(Error::new(format!("duplicate capture attachment path {path}")));
        }
        attachments.push(CaptureAttachment {
            name,
            path,
            media_type: validate_media_type(&nonempty_string(attachment, "media_type")?)?,
        });
    }

    Ok(CaptureManifest {
        identity,
        environment,
        capture,
        runtime,
        measurements,
        p95_latency_budget_ns,
        runtime_binary,
        attachments,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const MANIFEST: &str = r#"{
  "schema": "edgefit.calibration_capture.v1",
  "identity": {"device_id":"board-1","runtime_name":"runtime","runtime_version":"1"},
  "environment": {"operating_system":"bare-metal","architecture":"arm","hardware":"board","toolchain":"cc"},
  "capture": {"captured_at":"2026-07-13T00:00:00Z","command":"run","warmup_runs":"2"},
  "runtime": {"accepted":true,"rejected_reason":null},
  "measurements": {"arena_high_water_bytes":"1024","latency_ns":["10","11","12"]},
  "thresholds": {"p95_latency_budget_ns":"20"},
  "runtime_binary": "runtime.bin",
  "attachments": [{"name":"raw-log","path":"run.log","media_type":"text/plain"}]
}"#;

    #[test]
    fn parses_strict_capture_manifest() {
        let manifest = parse_capture_manifest(MANIFEST).unwrap();
        assert_eq!(manifest.measurements.latency_ns, [10, 11, 12]);
        assert_eq!(manifest.attachments.len(), 1);
    }

    #[test]
    fn rejects_duplicate_or_invalid_capture_fields() {
        assert!(parse_capture_manifest(&MANIFEST.replace(
            "\"runtime_binary\": \"runtime.bin\"",
            "\"runtime_binary\": \"runtime.bin\", \"unknown\": \"x\""
        ))
        .is_err());
        assert!(parse_capture_manifest(&MANIFEST.replace("\"10\"", "\"0\""))
            .is_err());
    }
}
