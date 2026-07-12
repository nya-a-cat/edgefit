use std::collections::BTreeSet;
use std::fs::{self, File};
use std::io::{ErrorKind, Read};
use std::path::{Component, Path, PathBuf};

use edgefit_calibration::{
    parse_evidence, render_verification_json, render_verification_markdown, sha256_hex, verify,
    ExpectedBindings, LoadedAttachment, Verification, VerificationBudget, MAX_ATTACHMENT_BYTES,
};
use edgefit_ir::EdgeFitResult;
use edgefit_target::parse_profile;

const MAX_EVIDENCE_BYTES: u64 = 16 * 1024 * 1024;
const MAX_MODEL_BYTES: u64 = 1 << 30;
const MAX_TARGET_BYTES: u64 = 16 * 1024 * 1024;

pub fn verify_calibration_files(
    evidence_path: impl AsRef<Path>,
    model_path: impl AsRef<Path>,
    target_path: impl AsRef<Path>,
) -> EdgeFitResult<Verification> {
    let evidence_path = evidence_path.as_ref();
    let model_path = model_path.as_ref();
    let target_path = target_path.as_ref();

    let evidence_bytes = read_bounded_regular_file(evidence_path, MAX_EVIDENCE_BYTES, "evidence")?;
    let model_bytes = read_bounded_regular_file(model_path, MAX_MODEL_BYTES, "model")?;
    let target_bytes = read_bounded_regular_file(target_path, MAX_TARGET_BYTES, "target profile")?;

    let evidence_text = std::str::from_utf8(&evidence_bytes)
        .map_err(|err| format!("calibration evidence is not UTF-8: {err}"))?;
    let target_text = std::str::from_utf8(&target_bytes)
        .map_err(|err| format!("target profile is not UTF-8: {err}"))?;
    let evidence = parse_evidence(evidence_text)
        .map_err(|err| format!("invalid calibration evidence: {err}"))?;
    let profile = parse_profile(target_text, target_path.to_path_buf())?;
    profile.validate()?;

    if evidence.identity.target_id != profile.target_id {
        return Err(format!(
            "calibration target identity mismatch: evidence target_id {:?}, profile target_id {:?}",
            evidence.identity.target_id, profile.target_id
        ));
    }
    let arena_bytes = profile
        .peak_activation_budget_bytes
        .ok_or_else(|| "target profile has no peak activation budget".to_string())?;
    if evidence.thresholds.arena_budget_bytes != arena_bytes {
        return Err(format!(
            "calibration arena budget mismatch: evidence has {}, target profile has {}",
            evidence.thresholds.arena_budget_bytes, arena_bytes
        ));
    }

    // Calibration evidence v1 has no target FNV field. Its target ID and the SHA-256 of the
    // exact profile bytes are bound below; bind `profile.fingerprint` here when the schema grows
    // an explicit target fingerprint field.
    let _target_fingerprint_seam = &profile.fingerprint;

    let evidence_parent = evidence_path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let attachment_bytes = load_declared_attachments(evidence_parent, &evidence)?;
    let loaded_attachments = evidence
        .attachments
        .iter()
        .zip(attachment_bytes.iter())
        .map(|(attachment, bytes)| LoadedAttachment {
            path: attachment.path.as_str(),
            bytes: bytes.as_slice(),
        })
        .collect::<Vec<_>>();

    let runtime_matches = evidence
        .attachments
        .iter()
        .zip(attachment_bytes.iter())
        .filter(|(attachment, _)| attachment.sha256 == evidence.bindings.runtime_binary_sha256)
        .collect::<Vec<_>>();
    let runtime_binary_sha256 = match runtime_matches.as_slice() {
        [(_, bytes)] => sha256_hex(bytes),
        [] => {
            return Err(
                "runtime binary binding does not identify a declared attachment".to_string(),
            )
        }
        _ => {
            return Err(
                "runtime binary binding identifies multiple declared attachments".to_string(),
            )
        }
    };

    let model_sha256 = sha256_hex(&model_bytes);
    let target_profile_sha256 = sha256_hex(&target_bytes);
    if evidence.bindings.model_sha256 != model_sha256 {
        return Err("calibration model SHA-256 binding mismatch".to_string());
    }
    if evidence.bindings.target_profile_sha256 != target_profile_sha256 {
        return Err("calibration target profile SHA-256 binding mismatch".to_string());
    }
    if evidence.bindings.runtime_binary_sha256 != runtime_binary_sha256 {
        return Err("calibration runtime binary SHA-256 binding mismatch".to_string());
    }
    for (attachment, bytes) in evidence.attachments.iter().zip(attachment_bytes.iter()) {
        let actual_bytes = u64::try_from(bytes.len())
            .map_err(|_| format!("attachment {} length cannot be represented", attachment.path))?;
        if actual_bytes != attachment.bytes {
            return Err(format!(
                "calibration attachment {} byte count mismatch",
                attachment.path
            ));
        }
        if sha256_hex(bytes) != attachment.sha256 {
            return Err(format!(
                "calibration attachment {} SHA-256 mismatch",
                attachment.path
            ));
        }
    }

    let expected = ExpectedBindings {
        model_sha256,
        target_profile_sha256,
        runtime_binary_sha256,
    };
    // Arena authority comes from the target profile. Evidence supplies only the latency budget.
    let budget = VerificationBudget {
        arena_bytes,
        p95_latency_ns: evidence.thresholds.p95_latency_budget_ns,
    };
    verify(&evidence, &expected, &budget, &loaded_attachments)
        .map_err(|err| format!("failed to verify calibration evidence: {err}"))
}

pub fn render_calibration_files_with_status(
    evidence_path: impl AsRef<Path>,
    model_path: impl AsRef<Path>,
    target_path: impl AsRef<Path>,
    format: &str,
) -> EdgeFitResult<(String, String)> {
    if !matches!(format, "json" | "markdown") {
        return Err("calibration format must be json or markdown".to_string());
    }
    let verification = verify_calibration_files(evidence_path, model_path, target_path)?;
    let status = if verification.status == edgefit_calibration::CheckStatus::Fail {
        "fail"
    } else {
        "pass"
    };
    let rendered = match format {
        "json" => render_verification_json(&verification),
        "markdown" => render_verification_markdown(&verification),
        _ => unreachable!("format was checked above"),
    };
    Ok((status.to_string(), rendered))
}

pub fn render_calibration_files(
    evidence_path: impl AsRef<Path>,
    model_path: impl AsRef<Path>,
    target_path: impl AsRef<Path>,
    format: &str,
) -> EdgeFitResult<String> {
    render_calibration_files_with_status(evidence_path, model_path, target_path, format)
        .map(|(_, rendered)| rendered)
}

fn load_declared_attachments(
    evidence_parent: &Path,
    evidence: &edgefit_calibration::Evidence,
) -> EdgeFitResult<Vec<Vec<u8>>> {
    let parent = fs::canonicalize(evidence_parent).map_err(|err| {
        format!(
            "failed to canonicalize evidence directory {}: {err}",
            evidence_parent.display()
        )
    })?;
    if !fs::metadata(&parent)
        .map_err(|err| format!("failed to inspect evidence directory: {err}"))?
        .is_dir()
    {
        return Err("evidence parent is not a directory".to_string());
    }

    let mut paths = BTreeSet::new();
    let mut loaded = Vec::with_capacity(evidence.attachments.len());
    for attachment in &evidence.attachments {
        let relative = safe_relative_attachment_path(&attachment.path)?;
        if !paths.insert(relative.clone()) {
            return Err(format!("duplicate attachment path {}", attachment.path));
        }
        if attachment.bytes > MAX_ATTACHMENT_BYTES {
            return Err(format!(
                "attachment {} exceeds byte limit",
                attachment.path
            ));
        }

        let path = parent.join(&relative);
        reject_symlink_components(&parent, &relative)?;
        let canonical = fs::canonicalize(&path).map_err(|err| {
            format!("failed to resolve attachment {}: {err}", attachment.path)
        })?;
        if !canonical.starts_with(&parent) {
            return Err(format!(
                "attachment {} escapes the evidence directory",
                attachment.path
            ));
        }
        let bytes = read_bounded_regular_file(&canonical, MAX_ATTACHMENT_BYTES, "attachment")?;
        loaded.push(bytes);
    }
    Ok(loaded)
}

fn safe_relative_attachment_path(value: &str) -> EdgeFitResult<PathBuf> {
    let path = Path::new(value);
    if value.is_empty() || path.is_absolute() {
        return Err(format!("unsafe attachment path {value:?}"));
    }
    let mut safe = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Normal(part) if !part.is_empty() => safe.push(part),
            _ => return Err(format!("unsafe attachment path {value:?}")),
        }
    }
    if safe.as_os_str().is_empty() {
        return Err(format!("unsafe attachment path {value:?}"));
    }
    Ok(safe)
}

fn reject_symlink_components(parent: &Path, relative: &Path) -> EdgeFitResult<()> {
    let mut current = parent.to_path_buf();
    for component in relative.components() {
        let Component::Normal(part) = component else {
            return Err(format!(
                "unsafe attachment path {}",
                relative.display()
            ));
        };
        current.push(part);
        match fs::symlink_metadata(&current) {
            Ok(metadata) if metadata.file_type().is_symlink() => {
                return Err(format!(
                    "attachment path contains a symlink: {}",
                    relative.display()
                ));
            }
            Ok(_) => {}
            Err(err) if err.kind() == ErrorKind::NotFound => {
                return Err(format!(
                    "attachment path does not exist: {}",
                    relative.display()
                ));
            }
            Err(err) => {
                return Err(format!(
                    "failed to inspect attachment path {}: {err}",
                    relative.display()
                ));
            }
        }
    }
    Ok(())
}

fn read_bounded_regular_file(path: &Path, max_bytes: u64, label: &str) -> EdgeFitResult<Vec<u8>> {
    let link_metadata = fs::symlink_metadata(path)
        .map_err(|err| format!("failed to inspect {label} {}: {err}", path.display()))?;
    if link_metadata.file_type().is_symlink() {
        return Err(format!("{label} {} must not be a symlink", path.display()));
    }
    if !link_metadata.file_type().is_file() {
        return Err(format!("{label} {} is not a regular file", path.display()));
    }
    if link_metadata.len() > max_bytes {
        return Err(format!(
            "{label} {} exceeds byte limit {max_bytes}",
            path.display()
        ));
    }

    let mut file = File::open(path)
        .map_err(|err| format!("failed to open {label} {}: {err}", path.display()))?;
    let open_metadata = file
        .metadata()
        .map_err(|err| format!("failed to inspect open {label} {}: {err}", path.display()))?;
    if !open_metadata.is_file() {
        return Err(format!("{label} {} is not a regular file", path.display()));
    }
    if open_metadata.len() > max_bytes {
        return Err(format!(
            "{label} {} exceeds byte limit {max_bytes}",
            path.display()
        ));
    }

    let capacity = usize::try_from(open_metadata.len())
        .map_err(|_| format!("{label} size cannot be represented"))?;
    let take_limit = max_bytes
        .checked_add(1)
        .ok_or_else(|| format!("{label} byte limit overflow"))?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(capacity)
        .map_err(|err| format!("failed to allocate buffer for {label}: {err}"))?;
    file.by_ref()
        .take(take_limit)
        .read_to_end(&mut bytes)
        .map_err(|err| format!("failed to read {label} {}: {err}", path.display()))?;
    if u64::try_from(bytes.len()).map_err(|_| format!("{label} size cannot be represented"))?
        > max_bytes
    {
        return Err(format!(
            "{label} {} exceeds byte limit {max_bytes}",
            path.display()
        ));
    }
    if u64::try_from(bytes.len()).map_err(|_| format!("{label} size cannot be represented"))?
        != open_metadata.len()
    {
        return Err(format!("{label} {} changed while being read", path.display()));
    }
    Ok(bytes)
}
