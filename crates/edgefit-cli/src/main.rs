mod calibration;

use edgefit_core::{
    check_adapter_generated_model_with_suppressions, check_model,
    check_model_with_suppressions, optimize_adapter_generated_model, optimize_model,
};
use edgefit_diff::{diff_snapshots, load_snapshot, render_diff};
use edgefit_report::{render_report, render_snapshot, Report};
use edgefit_optimize::render_plan;
use edgefit_target::load_profile;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

const EDGEFIT_VERSION: &str = env!("CARGO_PKG_VERSION");
// Alpha 自动化契约：区分通过、策略阻断和无法产出可信结果三类状态。
const EXIT_PASS: i32 = 0;
const EXIT_POLICY_FAIL: i32 = 1;
const EXIT_USAGE_ERROR: i32 = 2;
const EMBEDDED_ONNX_ADAPTER: &str =
    include_str!("../../../python/edgefit/onnx_adapter.py");

fn main() {
    let code = match run(env::args().skip(1).collect()) {
        Ok(code) => code,
        Err(err) => {
            eprintln!("edgefit: {err}");
            EXIT_USAGE_ERROR
        }
    };
    std::process::exit(code);
}

fn run(args: Vec<String>) -> Result<i32, String> {
    if args.is_empty() {
        print_help();
        return Ok(EXIT_USAGE_ERROR);
    }
    match args[0].as_str() {
        "target" => run_target(&args[1..]),
        "calibration" => calibration::run(&args[1..]),
        "check" => run_check(&args[1..]),
        "optimize" => run_optimize(&args[1..]),
        "snapshot" => run_snapshot(&args[1..]),
        "diff" => run_diff(&args[1..]),
        "version" | "-V" | "--version" => {
            println!("edgefit {EDGEFIT_VERSION}");
            Ok(EXIT_PASS)
        }
        "-h" | "--help" => {
            print_help();
            Ok(EXIT_PASS)
        }
        other => Err(format!("unknown command {other}")),
    }
}

fn run_optimize(args: &[String]) -> Result<i32, String> {
    let mut normalized_args = args.to_vec();
    if !args.iter().any(|argument| argument == "--format") {
        normalized_args.extend(["--format".to_string(), "json".to_string()]);
    }
    let parsed = parse_model_command(&normalized_args, false)?;
    if parsed.format != "json" && parsed.format != "markdown" {
        return Err("optimize --format must be json or markdown".to_string());
    }
    if !parsed.suppressions.is_empty() || parsed.summary.is_some() {
        return Err("optimize does not accept --suppress or --summary".to_string());
    }
    let target = parsed.target.as_deref().ok_or("--target is required")?;
    let prepared = match prepare_model(&parsed.model) {
        Ok(prepared) => prepared,
        Err(error) => return fail_with_execution_artifacts("optimize", &parsed, &error),
    };
    let result = if prepared.cli_adapter_output {
        optimize_adapter_generated_model(&prepared.path, target)
    } else {
        optimize_model(&prepared.path, target)
    };
    let plan = match result {
        Ok(plan) => plan,
        Err(error) => return fail_with_execution_artifacts("optimize", &parsed, &error),
    };
    write_or_print(&render_plan(&plan, &parsed.format), parsed.out.as_deref())?;
    Ok(if plan.status == "fail" { EXIT_POLICY_FAIL } else { EXIT_PASS })
}

fn run_target(args: &[String]) -> Result<i32, String> {
    if args.len() != 2 || args[0] != "validate" {
        return Err("usage: edgefit target validate <profile>".to_string());
    }
    let profile = load_profile(&args[1])?;
    println!("ok: {}", profile.target_id);
    Ok(EXIT_PASS)
}

fn run_check(args: &[String]) -> Result<i32, String> {
    let parsed = parse_model_command(args, false)?;
    let target = parsed.target.as_deref().ok_or("--target is required")?;
    let prepared = match prepare_model(&parsed.model) {
        Ok(prepared) => prepared,
        Err(error) => return fail_with_execution_artifacts("check", &parsed, &error),
    };
    let result = if prepared.cli_adapter_output {
        check_adapter_generated_model_with_suppressions(
            &prepared.path,
            target,
            &parsed.suppressions,
        )
    } else {
        check_model_with_suppressions(&prepared.path, target, &parsed.suppressions)
    };
    let report = match result {
        Ok(report) => report,
        Err(error) => return fail_with_execution_artifacts("check", &parsed, &error),
    };
    write_check_artifacts(&report, &parsed)?;
    Ok(if report.status == "fail" {
        EXIT_POLICY_FAIL
    } else {
        EXIT_PASS
    })
}

fn run_snapshot(args: &[String]) -> Result<i32, String> {
    let parsed = parse_model_command(args, true)?;
    let out = parsed.out.as_deref().ok_or("--out is required")?;
    let target = parsed.target.as_deref().ok_or("--target is required")?;
    let prepared = match prepare_model(&parsed.model) {
        Ok(prepared) => prepared,
        Err(error) => return fail_with_execution_artifacts("snapshot", &parsed, &error),
    };
    let result = if prepared.cli_adapter_output {
        check_adapter_generated_model_with_suppressions(&prepared.path, target, &[])
    } else {
        check_model(&prepared.path, target)
    };
    let report = match result {
        Ok(report) => report,
        Err(error) => return fail_with_execution_artifacts("snapshot", &parsed, &error),
    };
    write_or_print(&render_snapshot(&report), Some(out))?;
    Ok(if report.status == "fail" {
        EXIT_POLICY_FAIL
    } else {
        EXIT_PASS
    })
}

fn run_diff(args: &[String]) -> Result<i32, String> {
    let mut old = None;
    let mut new = None;
    let mut format = "markdown".to_string();
    let mut out = None;
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--old" => {
                index += 1;
                old = Some(args.get(index).cloned().ok_or("--old requires a value")?);
            }
            "--new" => {
                index += 1;
                new = Some(args.get(index).cloned().ok_or("--new requires a value")?);
            }
            "--format" => {
                index += 1;
                format = args
                    .get(index)
                    .cloned()
                    .ok_or("--format requires a value")?;
            }
            "--out" => {
                index += 1;
                out = Some(args.get(index).cloned().ok_or("--out requires a value")?);
            }
            other => return Err(format!("unexpected diff argument {other}")),
        }
        index += 1;
    }
    if format != "markdown" && format != "json" {
        return Err("--format must be markdown or json".to_string());
    }
    let old = load_snapshot(&old.ok_or("--old is required")?)?;
    let new = load_snapshot(&new.ok_or("--new is required")?)?;
    let diff = diff_snapshots(&old, &new)?;
    write_or_print(&render_diff(&diff, &format), out.as_deref())?;
    Ok(if diff.status == "fail" {
        EXIT_POLICY_FAIL
    } else {
        EXIT_PASS
    })
}

struct ModelCommand {
    model: String,
    target: Option<String>,
    format: String,
    out: Option<String>,
    summary: Option<String>,
    suppressions: Vec<String>,
}

pub(crate) struct PreparedModel {
    pub(crate) path: PathBuf,
    pub(crate) cli_adapter_output: bool,
}

impl Drop for PreparedModel {
    fn drop(&mut self) {
        if self.cli_adapter_output {
            let _ = fs::remove_file(&self.path);
        }
    }
}

pub(crate) fn prepare_model(model: &str) -> Result<PreparedModel, String> {
    let path = PathBuf::from(model);
    if path
        .extension()
        .and_then(|value| value.to_str())
        .map(|value| value.eq_ignore_ascii_case("onnx"))
        .unwrap_or(false)
    {
        let normalized = temporary_normalized_path()?;
        if let Err(error) = run_onnx_adapter(&path, &normalized) {
            let _ = fs::remove_file(&normalized);
            return Err(error);
        }
        Ok(PreparedModel {
            path: normalized,
            // 可信来源由原始 .onnx 分支传递，不能从生成文件的 JSON 字段反推。
            cli_adapter_output: true,
        })
    } else {
        Ok(PreparedModel {
            path,
            cli_adapter_output: false,
        })
    }
}

fn run_onnx_adapter(model: &Path, out: &Path) -> Result<(), String> {
    let python = env::var("EDGEFIT_PYTHON").unwrap_or_else(|_| "python".to_string());
    let mut command = Command::new(&python);
    if let Ok(adapter) = env::var("EDGEFIT_ONNX_ADAPTER") {
        let adapter = PathBuf::from(adapter);
        if !adapter.is_file() {
            return Err(format!(
                "EDGEFIT_ONNX_ADAPTER does not point to a file: {}",
                adapter.display()
            ));
        }
        command.arg(adapter);
    } else {
        // 发布二进制直接执行同一份嵌入源码，避免依赖编译机目录。
        command.arg("-c").arg(EMBEDDED_ONNX_ADAPTER);
    }
    let output = command
        .arg(model)
        .arg("--out")
        .arg(out)
        .output()
        .map_err(|err| format!("failed to launch ONNX adapter with {python}: {err}"))?;
    if output.status.success() {
        Ok(())
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
        let detail = if stderr.is_empty() { stdout } else { stderr };
        Err(format!("ONNX adapter failed: {detail}"))
    }
}

fn temporary_normalized_path() -> Result<PathBuf, String> {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|err| format!("system clock error: {err}"))?
        .as_nanos();
    let path = env::temp_dir().join(format!(
        "edgefit-normalized-{}-{stamp}.edgefit.json",
        std::process::id()
    ));
    Ok(path)
}

fn parse_model_command(args: &[String], snapshot_mode: bool) -> Result<ModelCommand, String> {
    if args.is_empty() {
        return Err("model path is required".to_string());
    }
    let model = args[0].clone();
    let mut target = None;
    let mut format = if snapshot_mode { "json" } else { "text" }.to_string();
    let mut out = None;
    let mut summary = None;
    let mut suppressions = Vec::new();
    let mut index = 1;
    while index < args.len() {
        match args[index].as_str() {
            "--target" => {
                index += 1;
                target = Some(
                    args.get(index)
                        .cloned()
                        .ok_or("--target requires a value")?,
                );
            }
            "--format" if !snapshot_mode => {
                index += 1;
                format = args
                    .get(index)
                    .cloned()
                    .ok_or("--format requires a value")?;
            }
            "--out" => {
                index += 1;
                out = Some(args.get(index).cloned().ok_or("--out requires a value")?);
            }
            "--summary" if !snapshot_mode => {
                index += 1;
                summary = Some(
                    args.get(index)
                        .cloned()
                        .ok_or("--summary requires a value")?,
                );
            }
            "--suppress" if !snapshot_mode => {
                index += 1;
                let value = args.get(index).ok_or("--suppress requires a value")?;
                add_suppression_ids(value, &mut suppressions);
            }
            other => return Err(format!("unexpected argument {other}")),
        }
        index += 1;
    }
    if !["text", "json", "markdown", "sarif"].contains(&format.as_str()) {
        return Err("--format must be text, json, markdown, or sarif".to_string());
    }
    Ok(ModelCommand {
        model,
        target,
        format,
        out,
        summary,
        suppressions,
    })
}

fn add_suppression_ids(value: &str, suppressions: &mut Vec<String>) {
    for id in value.split(',').map(str::trim).filter(|id| !id.is_empty()) {
        suppressions.push(id.to_string());
    }
}

fn write_check_artifacts(report: &Report, parsed: &ModelCommand) -> Result<(), String> {
    let mut write_errors = Vec::new();
    if let Err(error) = write_or_print(
        &render_report(report, &parsed.format),
        parsed.out.as_deref(),
    ) {
        write_errors.push(error);
    }
    if let Some(summary) = parsed.summary.as_deref() {
        if let Err(error) = write_or_print(&render_report(report, "markdown"), Some(summary)) {
            write_errors.push(format!("failed to write summary: {error}"));
        }
    }
    if write_errors.is_empty() {
        Ok(())
    } else {
        Err(write_errors.join("; "))
    }
}

fn fail_with_execution_artifacts(
    command: &str,
    parsed: &ModelCommand,
    error: &str,
) -> Result<i32, String> {
    let mut write_errors = Vec::new();
    if let Some(out) = parsed.out.as_deref() {
        let format = if command == "snapshot" {
            "json"
        } else {
            parsed.format.as_str()
        };
        let artifact = render_execution_error(command, &parsed.model, format, error);
        if let Err(write_error) = write_or_print(&artifact, Some(out)) {
            write_errors.push(format!("failed to write execution error: {write_error}"));
        }
    }
    if let Some(summary) = parsed.summary.as_deref() {
        let artifact = render_execution_error(command, &parsed.model, "markdown", error);
        if let Err(write_error) = write_or_print(&artifact, Some(summary)) {
            write_errors.push(format!(
                "failed to write execution error summary: {write_error}"
            ));
        }
    }
    if write_errors.is_empty() {
        Err(error.to_string())
    } else {
        Err(format!("{error}; {}", write_errors.join("; ")))
    }
}

fn render_execution_error(command: &str, model: &str, format: &str, error: &str) -> String {
    let command_json = json_string(command);
    let model_json = json_string(model);
    let message = json_string(error);
    match format {
        "json" => format!(
            "{{\n  \"schema\": \"edgefit.execution_error.v1\",\n  \"status\": \"execution_error\",\n  \"command\": {command_json},\n  \"model\": {model_json},\n  \"message\": {message}\n}}\n"
        ),
        "sarif" => format!(
            "{{\n  \"$schema\": \"https://json.schemastore.org/sarif-2.1.0.json\",\n  \"version\": \"2.1.0\",\n  \"runs\": [{{\n    \"tool\": {{\"driver\": {{\"name\": \"EdgeFit\", \"version\": \"{EDGEFIT_VERSION}\"}}}},\n    \"properties\": {{\"edgefitSchema\": \"edgefit.execution_error.v1\", \"edgefitStatus\": \"execution_error\"}},\n    \"results\": [{{\"ruleId\": \"EFEXECUTION\", \"level\": \"error\", \"message\": {{\"text\": {message}}}}}]\n  }}]\n}}\n"
        ),
        "markdown" => format!(
            "# EdgeFit Execution Error\n\n**Schema:** `edgefit.execution_error.v1`  \n**Status:** `execution_error`  \n**Command:** `{}`  \n**Model:** `{}`\n\n## Error\n\n{}\n",
            markdown_text(command),
            markdown_text(model),
            markdown_text(error),
        ),
        _ => format!("EdgeFit execution error: {error}\n"),
    }
}

fn json_string(value: &str) -> String {
    // 手工编码仅覆盖 JSON 字符串规则，避免为单一错误文档引入新的序列化依赖。
    let mut out = String::with_capacity(value.len() + 2);
    out.push('"');
    for character in value.chars() {
        match character {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            value if value <= '\u{1f}' => out.push_str(&format!("\\u{:04x}", value as u32)),
            value => out.push(value),
        }
    }
    out.push('"');
    out
}

fn markdown_text(value: &str) -> String {
    value.replace('`', "\\`")
}

fn write_or_print(content: &str, path: Option<&str>) -> Result<(), String> {
    if let Some(path) = path {
        if let Some(parent) = Path::new(path).parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)
                    .map_err(|err| format!("failed to create output directory: {err}"))?;
            }
        }
        fs::write(path, content).map_err(|err| format!("failed to write output: {err}"))?;
    } else {
        print!("{content}");
    }
    Ok(())
}

fn print_help() {
    println!("edgefit <command>");
    println!("  version");
    println!("  target validate <profile>");
    println!("  calibration verify <evidence.json> --model <model> --target <profile> [--format json|markdown] [--out path]");
    println!("  calibration simulate <model.onnx|model.edgefit.json> --target <profile> --scenario <scenario.json> --out-dir <new-directory>");
    println!("  check <model.onnx|model.edgefit.json> --target <profile> [--format text|json|markdown|sarif] [--out path] [--summary path] [--suppress id[,id]]");
    println!("  optimize <model.onnx|model.edgefit.json> --target <profile> [--format json|markdown] [--out path]");
    println!("  snapshot <model.onnx|model.edgefit.json> --target <profile> --out path");
    println!("  diff --old path --new path [--format markdown|json] [--out path]");
}
