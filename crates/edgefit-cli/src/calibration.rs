//! Calibration 验证与模拟命令入口。
//!
//! 本模块只负责严格参数解析、退出码映射和原子输出，业务语义保留在 Rust core。

use edgefit_calibration::{
    parse_evidence, render_verification_json, render_verification_markdown, CheckStatus,
};
use edgefit_core::{pack_calibration_files, simulate_calibration_files, verify_calibration_files};
use edgefit_ir::escape_json;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

const EXIT_PASS: i32 = 0;
const EXIT_POLICY_FAIL: i32 = 1;

#[derive(Debug)]
struct CalibrationCommand {
    evidence: PathBuf,
    model: PathBuf,
    target: PathBuf,
    format: String,
    out: Option<PathBuf>,
}

pub fn run(args: &[String]) -> Result<i32, String> {
    match args.first().map(String::as_str) {
        Some("verify") => run_verify(args),
        Some("simulate") => run_simulate(args),
        Some("pack") => run_pack(args),
        _ => Err("usage: edgefit calibration <verify|simulate|pack> ...".to_string()),
    }
}

fn run_verify(args: &[String]) -> Result<i32, String> {
    let parsed = parse_verify(args)?;
    reject_output_aliases(&parsed)?;
    reject_attachment_output_aliases(&parsed)?;

    let verification =
        match verify_calibration_files(&parsed.evidence, &parsed.model, &parsed.target) {
            Ok(verification) => verification,
            Err(error) => return fail_with_artifact(&parsed, &error),
        };
    let rendered = match parsed.format.as_str() {
        "json" => render_verification_json(&verification),
        "markdown" => render_verification_markdown(&verification),
        _ => unreachable!("format validated by parser"),
    };
    write_or_print_atomic(&rendered, parsed.out.as_deref())?;
    Ok(if verification.status == CheckStatus::Fail {
        EXIT_POLICY_FAIL
    } else {
        EXIT_PASS
    })
}

fn parse_verify(args: &[String]) -> Result<CalibrationCommand, String> {
    if args.len() < 2 || args[0] != "verify" {
        return Err(
            "usage: edgefit calibration verify <evidence.json> --model <model> --target <profile> [--format json|markdown] [--out path]"
                .to_string(),
        );
    }
    let evidence = PathBuf::from(&args[1]);
    let mut model = None;
    let mut target = None;
    let mut format = None;
    let mut out = None;
    let mut index = 2;
    while index < args.len() {
        let flag = args[index].as_str();
        index += 1;
        let value = args
            .get(index)
            .ok_or_else(|| format!("{flag} requires a value"))?;
        match flag {
            "--model" if model.is_none() => model = Some(PathBuf::from(value)),
            "--target" if target.is_none() => target = Some(PathBuf::from(value)),
            "--format" if format.is_none() => format = Some(value.clone()),
            "--out" if out.is_none() => out = Some(PathBuf::from(value)),
            "--model" | "--target" | "--format" | "--out" => {
                return Err(format!("duplicate calibration option {flag}"));
            }
            other => return Err(format!("unexpected calibration argument {other}")),
        }
        index += 1;
    }
    let format = format.unwrap_or_else(|| "json".to_string());
    if !matches!(format.as_str(), "json" | "markdown") {
        return Err("calibration --format must be json or markdown".to_string());
    }
    Ok(CalibrationCommand {
        evidence,
        model: model.ok_or("calibration --model is required")?,
        target: target.ok_or("calibration --target is required")?,
        format,
        out,
    })
}

#[derive(Debug)]
struct SimulationCommand {
    model: PathBuf,
    target: PathBuf,
    scenario: PathBuf,
    out_dir: PathBuf,
}

fn run_simulate(args: &[String]) -> Result<i32, String> {
    let parsed = parse_simulate(args)?;
    let prepared = crate::prepare_model(
        parsed
            .model
            .to_str()
            .ok_or("calibration simulation model path must be UTF-8")?,
    )?;
    let result = simulate_calibration_files(
        &parsed.model,
        &prepared.path,
        prepared.cli_adapter_output,
        &parsed.target,
        &parsed.scenario,
        &parsed.out_dir,
    )?;
    print!("{}", result.verification_json);
    Ok(if result.status == "fail" {
        EXIT_POLICY_FAIL
    } else {
        EXIT_PASS
    })
}

fn parse_simulate(args: &[String]) -> Result<SimulationCommand, String> {
    if args.len() < 2 || args[0] != "simulate" {
        return Err(
            "usage: edgefit calibration simulate <model.onnx|model.edgefit.json> --target <profile> --scenario <scenario.json> --out-dir <new-directory>"
                .to_string(),
        );
    }
    let model = PathBuf::from(&args[1]);
    let mut target = None;
    let mut scenario = None;
    let mut out_dir = None;
    let mut index = 2;
    while index < args.len() {
        let flag = args[index].as_str();
        index += 1;
        let value = args
            .get(index)
            .ok_or_else(|| format!("{flag} requires a value"))?;
        match flag {
            "--target" if target.is_none() => target = Some(PathBuf::from(value)),
            "--scenario" if scenario.is_none() => scenario = Some(PathBuf::from(value)),
            "--out-dir" if out_dir.is_none() => out_dir = Some(PathBuf::from(value)),
            "--target" | "--scenario" | "--out-dir" => {
                return Err(format!("duplicate calibration simulation option {flag}"));
            }
            other => return Err(format!("unexpected calibration simulation argument {other}")),
        }
        index += 1;
    }
    Ok(SimulationCommand {
        model,
        target: target.ok_or("calibration simulation --target is required")?,
        scenario: scenario.ok_or("calibration simulation --scenario is required")?,
        out_dir: out_dir.ok_or("calibration simulation --out-dir is required")?,
    })
}

#[derive(Debug)]
struct PackCommand {
    capture: PathBuf,
    model: PathBuf,
    target: PathBuf,
    out_dir: PathBuf,
}

fn run_pack(args: &[String]) -> Result<i32, String> {
    let parsed = parse_pack(args)?;
    let result = pack_calibration_files(
        &parsed.capture,
        &parsed.model,
        &parsed.target,
        &parsed.out_dir,
    )?;
    print!("{}", result.verification_json);
    Ok(if result.status == "fail" {
        EXIT_POLICY_FAIL
    } else {
        EXIT_PASS
    })
}

fn parse_pack(args: &[String]) -> Result<PackCommand, String> {
    if args.len() < 2 || args[0] != "pack" {
        return Err(
            "usage: edgefit calibration pack <capture.json> --model <model> --target <profile> --out-dir <new-directory>"
                .to_string(),
        );
    }
    let capture = PathBuf::from(&args[1]);
    let mut model = None;
    let mut target = None;
    let mut out_dir = None;
    let mut index = 2;
    while index < args.len() {
        let flag = args[index].as_str();
        index += 1;
        let value = args
            .get(index)
            .ok_or_else(|| format!("{flag} requires a value"))?;
        match flag {
            "--model" if model.is_none() => model = Some(PathBuf::from(value)),
            "--target" if target.is_none() => target = Some(PathBuf::from(value)),
            "--out-dir" if out_dir.is_none() => out_dir = Some(PathBuf::from(value)),
            "--model" | "--target" | "--out-dir" => {
                return Err(format!("duplicate calibration pack option {flag}"));
            }
            other => return Err(format!("unexpected calibration pack argument {other}")),
        }
        index += 1;
    }
    Ok(PackCommand {
        capture,
        model: model.ok_or("calibration pack --model is required")?,
        target: target.ok_or("calibration pack --target is required")?,
        out_dir: out_dir.ok_or("calibration pack --out-dir is required")?,
    })
}

fn reject_output_aliases(command: &CalibrationCommand) -> Result<(), String> {
    let Some(out) = command.out.as_deref() else {
        return Ok(());
    };
    for (label, input) in [
        ("evidence", command.evidence.as_path()),
        ("model", command.model.as_path()),
        ("target", command.target.as_path()),
    ] {
        if paths_alias(out, input)? {
            return Err(format!("calibration output path must not alias {label} input"));
        }
    }
    Ok(())
}

fn reject_attachment_output_aliases(command: &CalibrationCommand) -> Result<(), String> {
    let Some(out) = command.out.as_deref() else {
        return Ok(());
    };
    let text = fs::read_to_string(&command.evidence)
        .map_err(|error| format!("failed to read calibration evidence: {error}"))?;
    let evidence = parse_evidence(&text)
        .map_err(|error| format!("invalid calibration evidence: {error}"))?;
    let parent = command
        .evidence
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    for attachment in evidence.attachments {
        if paths_alias(out, &parent.join(&attachment.path))? {
            return Err(format!(
                "calibration output path must not alias attachment {}",
                attachment.path
            ));
        }
    }
    Ok(())
}

fn paths_alias(left: &Path, right: &Path) -> Result<bool, String> {
    let left = absolute_lexical(left)?;
    let right = absolute_lexical(right)?;
    if left == right {
        return Ok(true);
    }
    match (fs::canonicalize(&left), fs::canonicalize(&right)) {
        (Ok(left), Ok(right)) => Ok(left == right),
        _ => Ok(false),
    }
}

fn absolute_lexical(path: &Path) -> Result<PathBuf, String> {
    let path = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .map_err(|error| format!("failed to resolve current directory: {error}"))?
            .join(path)
    };
    let mut normalized = PathBuf::new();
    for component in path.components() {
        use std::path::Component;
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                normalized.pop();
            }
            other => normalized.push(other.as_os_str()),
        }
    }
    Ok(normalized)
}

fn fail_with_artifact(command: &CalibrationCommand, error: &str) -> Result<i32, String> {
    if let Some(out) = command.out.as_deref() {
        let artifact = render_error(command, error);
        if let Err(write_error) = write_or_print_atomic(&artifact, Some(out)) {
            return Err(format!("{error}; failed to write calibration error artifact: {write_error}"));
        }
    }
    Err(error.to_string())
}

fn render_error(command: &CalibrationCommand, error: &str) -> String {
    match command.format.as_str() {
        "markdown" => format!(
            "# EdgeFit Calibration Verification Error\n\n**Schema:** `edgefit.calibration_verification_error.v1`  \n**Status:** `execution_error`  \n**Evidence:** `{}`\n\n## Error\n\n{}\n",
            markdown_text(&command.evidence.display().to_string()),
            markdown_text(error),
        ),
        _ => format!(
            "{{\n  \"schema\": \"edgefit.calibration_verification_error.v1\",\n  \"status\": \"execution_error\",\n  \"evidence\": {},\n  \"message\": {}\n}}\n",
            json_string(&command.evidence.display().to_string()),
            json_string(error),
        ),
    }
}

fn json_string(value: &str) -> String {
    format!("\"{}\"", escape_json(value))
}

fn markdown_text(value: &str) -> String {
    value.replace('`', "\\`")
}

fn write_or_print_atomic(content: &str, path: Option<&Path>) -> Result<(), String> {
    let Some(path) = path else {
        print!("{content}");
        return Ok(());
    };
    let parent = path.parent().filter(|parent| !parent.as_os_str().is_empty());
    if let Some(parent) = parent {
        fs::create_dir_all(parent)
            .map_err(|error| format!("failed to create output directory: {error}"))?;
    }
    let temp = temporary_sibling(path)?;
    let result = (|| {
        let mut file = fs::File::create(&temp)
            .map_err(|error| format!("failed to create temporary output: {error}"))?;
        file.write_all(content.as_bytes())
            .map_err(|error| format!("failed to write temporary output: {error}"))?;
        file.sync_all()
            .map_err(|error| format!("failed to sync temporary output: {error}"))?;
        replace_file(&temp, path)
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temp);
    }
    result
}

fn temporary_sibling(path: &Path) -> Result<PathBuf, String> {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| format!("system clock error: {error}"))?
        .as_nanos();
    let name = path
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or("calibration output path must have a UTF-8 file name")?;
    Ok(path.with_file_name(format!(".{name}.edgefit-{}-{stamp}.tmp", std::process::id())))
}

fn replace_file(temp: &Path, destination: &Path) -> Result<(), String> {
    match fs::rename(temp, destination) {
        Ok(()) => Ok(()),
        Err(first) if destination.exists() => {
            fs::remove_file(destination)
                .map_err(|error| format!("failed to replace existing output after {first}: {error}"))?;
            fs::rename(temp, destination)
                .map_err(|error| format!("failed to publish calibration output: {error}"))
        }
        Err(error) => Err(format!("failed to publish calibration output: {error}")),
    }
}

#[cfg(test)]
mod tests {
    use super::{parse_pack, parse_simulate, parse_verify};

    #[test]
    fn parser_requires_unique_options() {
        let args = [
            "verify",
            "evidence.json",
            "--model",
            "model.bin",
            "--model",
            "other.bin",
            "--target",
            "target.yaml",
        ]
        .into_iter()
        .map(str::to_string)
        .collect::<Vec<_>>();
        assert!(parse_verify(&args).unwrap_err().contains("duplicate calibration option --model"));
    }

    #[test]
    fn simulation_parser_requires_unique_options() {
        let args = [
            "simulate",
            "model.edgefit.json",
            "--target",
            "target.yaml",
            "--scenario",
            "scenario.json",
            "--out-dir",
            "out",
            "--out-dir",
            "other",
        ]
        .into_iter()
        .map(str::to_string)
        .collect::<Vec<_>>();
        assert!(parse_simulate(&args)
            .unwrap_err()
            .contains("duplicate calibration simulation option --out-dir"));
    }

    #[test]
    fn pack_parser_requires_unique_options() {
        let args = [
            "pack",
            "capture.json",
            "--model",
            "model.bin",
            "--target",
            "target.yaml",
            "--out-dir",
            "out",
            "--out-dir",
            "other",
        ]
        .into_iter()
        .map(str::to_string)
        .collect::<Vec<_>>();
        assert!(parse_pack(&args)
            .unwrap_err()
            .contains("duplicate calibration pack option --out-dir"));
    }
}
