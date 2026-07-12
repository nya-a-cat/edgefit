use std::collections::{BTreeMap, BTreeSet};

use crate::render::{render_evidence_json, render_verification_payload};
use crate::schema::{
    validate_attachment_name, validate_attachment_path, validate_media_type, validate_runtime_result,
    validate_sample_count, validate_sha256_hex,
};
use crate::{
    sha256_hex, Check, CheckStatus, Error, Evidence, ExpectedBindings, LoadedAttachment, Metrics,
    Result, Verification, VerificationBudget, MAX_ATTACHMENT_BYTES, MAX_ATTACHMENTS,
};

pub fn verify(
    evidence: &Evidence,
    expected: &ExpectedBindings,
    budget: &VerificationBudget,
    loaded_attachments: &[LoadedAttachment<'_>],
) -> Result<Verification> {
    validate_evidence(evidence)?;
    validate_expected(expected)?;
    if budget.arena_bytes == 0 {
        return Err(Error::new("arena budget must be greater than zero"));
    }
    if budget.p95_latency_ns == 0 {
        return Err(Error::new(
            "p95 latency budget must be greater than zero",
        ));
    }

    let sample_count = validate_sample_count(
        evidence.capture.measured_runs,
        evidence.measurements.latency_ns.len(),
    )?;
    let mut sorted = evidence.measurements.latency_ns.clone();
    sorted.sort_unstable();
    let p50 = nearest_rank(&sorted, 50)?;
    let p95 = nearest_rank(&sorted, 95)?;
    let latency_mean_ns = half_up_mean(&evidence.measurements.latency_ns)?;
    let arena_utilization_ppm = ratio_ppm(
        evidence.measurements.arena_high_water_bytes,
        budget.arena_bytes,
    );
    let arena_error_bytes = i128::from(evidence.measurements.arena_high_water_bytes)
        .checked_sub(i128::from(budget.arena_bytes))
        .ok_or_else(|| Error::new("arena error overflow"))?;
    let p95_latency_error_ns = i128::from(p95)
        .checked_sub(i128::from(budget.p95_latency_ns))
        .ok_or_else(|| Error::new("latency error overflow"))?;

    let declared = declared_attachments(evidence)?;
    let loaded = loaded_attachments_by_path(loaded_attachments, &declared)?;
    let evidence_sha256 = sha256_hex(render_evidence_json(evidence).as_bytes());
    let mut checks = Vec::new();
    push_check(
        &mut checks,
        "model_binding",
        evidence.bindings.model_sha256 == expected.model_sha256,
        "model SHA-256 binding",
    );
    push_check(
        &mut checks,
        "target_profile_binding",
        evidence.bindings.target_profile_sha256 == expected.target_profile_sha256,
        "target profile SHA-256 binding",
    );
    push_check(
        &mut checks,
        "runtime_binary_binding",
        evidence.bindings.runtime_binary_sha256 == expected.runtime_binary_sha256,
        "runtime binary SHA-256 binding",
    );
    push_check(
        &mut checks,
        "runtime_accepted",
        evidence.runtime.accepted,
        evidence
            .runtime
            .rejected_reason
            .as_deref()
            .unwrap_or("runtime accepted"),
    );
    push_check(
        &mut checks,
        "evidence_arena_threshold",
        evidence.measurements.arena_high_water_bytes <= evidence.thresholds.arena_budget_bytes,
        "evidence arena threshold",
    );
    push_check(
        &mut checks,
        "evidence_latency_threshold",
        p95 <= evidence.thresholds.p95_latency_budget_ns,
        "evidence p95 latency threshold",
    );
    push_check(
        &mut checks,
        "expected_arena_budget",
        evidence.measurements.arena_high_water_bytes <= budget.arena_bytes,
        "supplied arena budget",
    );
    push_check(
        &mut checks,
        "expected_latency_budget",
        p95 <= budget.p95_latency_ns,
        "supplied p95 latency budget",
    );
    append_attachment_checks(evidence, &loaded, &mut checks)?;

    let status = if checks
        .iter()
        .all(|check| check.status == CheckStatus::Pass)
    {
        CheckStatus::Pass
    } else {
        CheckStatus::Fail
    };
    let mut verification = Verification {
        status,
        evidence_sha256,
        expected_bindings: expected.clone(),
        budget: budget.clone(),
        metrics: Metrics {
            sample_count,
            latency_p50_ns: p50,
            latency_p95_ns: p95,
            latency_mean_ns,
            arena_utilization_ppm,
            arena_error_bytes,
            p95_latency_error_ns,
        },
        checks,
        verification_sha256: String::new(),
    };
    verification.verification_sha256 =
        sha256_hex(render_verification_payload(&verification).as_bytes());
    Ok(verification)
}

pub fn verify_json(
    evidence_json: &str,
    expected: &ExpectedBindings,
    budget: &VerificationBudget,
    loaded_attachments: &[LoadedAttachment<'_>],
) -> Result<Verification> {
    verify(
        &crate::parse_evidence(evidence_json)?,
        expected,
        budget,
        loaded_attachments,
    )
}

fn validate_evidence(evidence: &Evidence) -> Result<()> {
    validate_sample_count(
        evidence.capture.measured_runs,
        evidence.measurements.latency_ns.len(),
    )?;
    validate_runtime_result(
        evidence.runtime.accepted,
        evidence.runtime.rejected_reason.as_deref(),
    )?;
    validate_sha256_hex(&evidence.bindings.model_sha256, "bindings.model_sha256")?;
    validate_sha256_hex(
        &evidence.bindings.target_profile_sha256,
        "bindings.target_profile_sha256",
    )?;
    validate_sha256_hex(
        &evidence.bindings.runtime_binary_sha256,
        "bindings.runtime_binary_sha256",
    )?;
    if evidence.attachments.len() > MAX_ATTACHMENTS {
        return Err(Error::new(format!(
            "attachments exceeds limit {MAX_ATTACHMENTS}"
        )));
    }
    let mut names = BTreeSet::new();
    for attachment in &evidence.attachments {
        validate_attachment_name(&attachment.name)?;
        validate_attachment_path(&attachment.path)?;
        validate_media_type(&attachment.media_type)?;
        validate_sha256_hex(&attachment.sha256, "attachment.sha256")?;
        if attachment.bytes > MAX_ATTACHMENT_BYTES {
            return Err(Error::new(format!(
                "attachment {} exceeds byte limit",
                attachment.path
            )));
        }
        if !names.insert(attachment.name.as_str()) {
            return Err(Error::new(format!(
                "duplicate attachment name {}",
                attachment.name
            )));
        }
    }
    Ok(())
}

fn validate_expected(expected: &ExpectedBindings) -> Result<()> {
    validate_sha256_hex(&expected.model_sha256, "expected.model_sha256")?;
    validate_sha256_hex(
        &expected.target_profile_sha256,
        "expected.target_profile_sha256",
    )?;
    validate_sha256_hex(
        &expected.runtime_binary_sha256,
        "expected.runtime_binary_sha256",
    )
}

fn declared_attachments<'a>(
    evidence: &'a Evidence,
) -> Result<BTreeMap<&'a str, &'a crate::Attachment>> {
    let mut declared = BTreeMap::new();
    for attachment in &evidence.attachments {
        if declared
            .insert(attachment.path.as_str(), attachment)
            .is_some()
        {
            return Err(Error::new(format!(
                "duplicate attachment path {}",
                attachment.path
            )));
        }
    }
    Ok(declared)
}

fn loaded_attachments_by_path<'data>(
    loaded_attachments: &[LoadedAttachment<'data>],
    declared: &BTreeMap<&str, &crate::Attachment>,
) -> Result<BTreeMap<&'data str, &'data [u8]>> {
    let mut loaded = BTreeMap::new();
    for item in loaded_attachments {
        validate_attachment_path(item.path)?;
        if loaded.insert(item.path, item.bytes).is_some() {
            return Err(Error::new(format!(
                "duplicate loaded attachment path {}",
                item.path
            )));
        }
        if !declared.contains_key(item.path) {
            return Err(Error::new(format!(
                "undeclared loaded attachment {}",
                item.path
            )));
        }
    }
    Ok(loaded)
}

fn append_attachment_checks(
    evidence: &Evidence,
    loaded: &BTreeMap<&str, &[u8]>,
    checks: &mut Vec<Check>,
) -> Result<()> {
    for attachment in &evidence.attachments {
        let (passed, detail) = match loaded.get(attachment.path.as_str()) {
            None => (false, "attachment bytes were not supplied"),
            Some(bytes) => {
                let actual_len = u64::try_from(bytes.len())
                    .map_err(|_| Error::new("attachment length cannot be represented"))?;
                let hash_matches = sha256_hex(bytes) == attachment.sha256;
                if actual_len != attachment.bytes {
                    (false, "attachment byte count mismatch")
                } else if !hash_matches {
                    (false, "attachment SHA-256 mismatch")
                } else {
                    (true, "attachment bytes and SHA-256 match")
                }
            }
        };
        checks.push(Check {
            id: "attachment",
            status: if passed {
                CheckStatus::Pass
            } else {
                CheckStatus::Fail
            },
            detail: format!("{}: {detail}", attachment.path),
        });
    }
    Ok(())
}

fn nearest_rank(sorted: &[u64], percentile: u128) -> Result<u64> {
    if sorted.is_empty() || percentile == 0 || percentile > 100 {
        return Err(Error::new("percentile requires samples and a rank 1..=100"));
    }
    let count = u128::try_from(sorted.len())
        .map_err(|_| Error::new("percentile sample count overflow"))?;
    let numerator = count
        .checked_mul(percentile)
        .and_then(|value| value.checked_add(99))
        .ok_or_else(|| Error::new("percentile rank overflow"))?;
    let rank = numerator / 100;
    let index = usize::try_from(rank - 1)
        .map_err(|_| Error::new("percentile rank cannot be represented"))?;
    sorted
        .get(index)
        .copied()
        .ok_or_else(|| Error::new("percentile rank exceeds samples"))
}

fn half_up_mean(samples: &[u64]) -> Result<u64> {
    let sum = samples.iter().try_fold(0_u128, |sum, value| {
        sum.checked_add(u128::from(*value))
    });
    let sum = sum.ok_or_else(|| Error::new("latency sum overflow"))?;
    let count = u128::try_from(samples.len())
        .map_err(|_| Error::new("latency sample count overflow"))?;
    if count == 0 {
        return Err(Error::new("latency mean requires samples"));
    }
    let quotient = sum / count;
    let remainder = sum % count;
    let round_up = remainder
        .checked_mul(2)
        .ok_or_else(|| Error::new("latency mean overflow"))?
        >= count;
    let mean = quotient
        .checked_add(u128::from(round_up))
        .ok_or_else(|| Error::new("latency mean overflow"))?;
    u64::try_from(mean).map_err(|_| Error::new("latency mean overflow"))
}

fn ratio_ppm(numerator: u64, denominator: u64) -> u64 {
    let scaled = u128::from(numerator)
        .checked_mul(1_000_000)
        .expect("u64 multiplied by one million fits u128");
    let ratio = scaled / u128::from(denominator);
    u64::try_from(ratio).unwrap_or(u64::MAX)
}

fn push_check(checks: &mut Vec<Check>, id: &'static str, passed: bool, detail: &str) {
    checks.push(Check {
        id,
        status: if passed {
            CheckStatus::Pass
        } else {
            CheckStatus::Fail
        },
        detail: detail.to_string(),
    });
}
