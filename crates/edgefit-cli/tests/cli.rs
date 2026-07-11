use edgefit_ir::parse_json;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::time::{SystemTime, UNIX_EPOCH};

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_edgefit")
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace root")
        .to_path_buf()
}

fn fixture(path: &str) -> String {
    workspace_root().join(path).display().to_string()
}

fn run(args: &[&str]) -> Output {
    Command::new(bin())
        .args(args)
        .output()
        .expect("run edgefit")
}

#[test]
fn alpha_command_and_exit_code_contract_is_stable() {
    let help = run(&["--help"]);
    assert_eq!(help.status.code(), Some(0));
    let help_text = String::from_utf8_lossy(&help.stdout);
    for command in [
        "target validate <profile>",
        "check <model.onnx|model.edgefit.json>",
        "optimize <model.onnx|model.edgefit.json>",
        "snapshot <model.onnx|model.edgefit.json>",
        "diff --old path --new path",
    ] {
        assert!(help_text.contains(command), "missing command contract: {command}");
    }

    let no_args = run(&[]);
    assert_eq!(no_args.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&no_args.stdout).contains("edgefit <command>"));

    let unknown = run(&["unknown"]);
    assert_eq!(unknown.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&unknown.stderr).contains("unknown command unknown"));

    let version = run(&["--version"]);
    assert_eq!(version.status.code(), Some(0));
    assert!(String::from_utf8_lossy(&version.stdout).starts_with("edgefit "));
}

#[test]
fn optimize_emits_a_machine_readable_plan() {
    let output = run(&[
        "optimize",
        &fixture("examples/models/virtual_npu_tiny.edgefit.json"),
        "--target",
        &fixture("targets/virtual-npu.yaml"),
    ]);
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let json = String::from_utf8_lossy(&output.stdout);
    assert!(json.contains("\"schema\": \"edgefit.optimization_plan.v1\""));
    assert!(json.contains("\"status\": \"pass\""));
    assert!(json.contains("\"accelerator_id\": \"generic-npu-v1\""));
    parse_json(&json).expect("parse optimization plan");
}

fn unique_dir(name: &str) -> PathBuf {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time")
        .as_nanos();
    let path = std::env::temp_dir().join(format!("edgefit-{name}-{}-{stamp}", std::process::id()));
    fs::create_dir_all(&path).expect("create temp dir");
    path
}

#[test]
fn target_validate_accepts_seed_profiles() {
    for (path, expected_id) in [
        ("targets/esp32s3.yaml", "esp32s3_custom_v1"),
        ("targets/ort-mobile-cpu.yaml", "ort_mobile_cpu_seed_v1"),
        ("targets/virtual-npu.yaml", "edgefit_virtual_npu_v1"),
    ] {
        let output = run(&["target", "validate", &fixture(path)]);
        assert!(
            output.status.success(),
            "profile {path} stderr: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(String::from_utf8_lossy(&output.stdout).contains(expected_id));
    }
}

#[test]
fn check_passes_for_good_fixture() {
    let output = run(&[
        "check",
        &fixture("examples/models/good_tiny.edgefit.json"),
        "--target",
        &fixture("targets/esp32s3.yaml"),
    ]);
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("EdgeFit status: pass"));
    assert!(stdout.contains("EF0001"));
}

#[test]
fn ort_mobile_seed_allows_wider_detector_fixture() {
    let output = run(&[
        "check",
        &fixture("examples/models/bad_detector.edgefit.json"),
        "--target",
        &fixture("targets/ort-mobile-cpu.yaml"),
    ]);
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("EdgeFit status: pass"));
    assert!(stdout.contains("Target: ort_mobile_cpu_seed_v1"));
    assert!(stdout.contains("EF0001"));
}

#[test]
fn check_fails_and_writes_json_and_sarif_for_bad_fixture() {
    let dir = unique_dir("bad-report");
    let json_path = dir.join("bad.json");
    let sarif_path = dir.join("bad.sarif");
    let summary_path = dir.join("bad-summary.md");

    let json_output = run(&[
        "check",
        &fixture("examples/models/bad_detector.edgefit.json"),
        "--target",
        &fixture("targets/esp32s3.yaml"),
        "--format",
        "json",
        "--out",
        json_path.to_str().expect("json path"),
    ]);
    assert_eq!(json_output.status.code(), Some(1));
    let json = fs::read_to_string(&json_path).expect("json report");
    assert!(json.contains("\"schema\": \"edgefit.report.v1\""));
    assert!(json.contains("EF0501"));
    assert!(json.contains("EF0202"));
    assert!(json.contains("\"location\": \"model.activations\""));
    assert!(json.contains("\"location\": \"op:Conv tensor:conv.weight\""));
    parse_json(&json).expect("parse edgefit json report");

    let sarif_output = run(&[
        "check",
        &fixture("examples/models/bad_detector.edgefit.json"),
        "--target",
        &fixture("targets/esp32s3.yaml"),
        "--format",
        "sarif",
        "--out",
        sarif_path.to_str().expect("sarif path"),
        "--summary",
        summary_path.to_str().expect("summary path"),
    ]);
    assert_eq!(sarif_output.status.code(), Some(1));
    let sarif = fs::read_to_string(&sarif_path).expect("sarif report");
    assert!(sarif.contains("\"version\": \"2.1.0\""));
    assert!(sarif.contains("EF0501"));
    assert!(sarif.contains("EF0202"));
    assert!(sarif.contains("logicalLocations"));
    assert!(sarif.contains("edgefitLocation"));
    assert!(sarif.contains("model.activations"));
    parse_json(&sarif).expect("parse sarif json report");

    let summary = fs::read_to_string(&summary_path).expect("summary report");
    assert!(summary.contains("# EdgeFit Report"));
    assert!(summary.contains("**Status:** `fail`"));
    assert!(summary.contains("EF0501"));
    assert!(summary.contains("| ID | Severity | Category | Location | Message |"));
    assert!(summary.contains("model.activations"));

    fs::remove_dir_all(dir).expect("cleanup temp dir");
}

#[test]
fn check_reports_shape_rank_and_confidence_rules() {
    let dir = unique_dir("rank-dynamic");
    let json_path = dir.join("rank.json");

    let output = run(&[
        "check",
        &fixture("examples/models/rank_dynamic.edgefit.json"),
        "--target",
        &fixture("targets/esp32s3.yaml"),
        "--format",
        "json",
        "--out",
        json_path.to_str().expect("rank path"),
    ]);
    assert_eq!(output.status.code(), Some(1));
    let json = fs::read_to_string(&json_path).expect("rank report");
    assert!(json.contains("EF0101"));
    assert!(json.contains("EF0102"));
    assert!(json.contains("EF0104"));
    assert!(json.contains("EF0203"));
    assert!(json.contains("tensor_rank_violations"));
    assert!(json.contains("op_rank_violations"));
    parse_json(&json).expect("parse rank report");

    fs::remove_dir_all(dir).expect("cleanup temp dir");
}
#[test]
fn check_suppresses_accepted_diagnostics_by_id() {
    let dir = unique_dir("suppressed-report");
    let json_path = dir.join("suppressed.json");

    let output = run(&[
        "check",
        &fixture("examples/models/rank_dynamic.edgefit.json"),
        "--target",
        &fixture("targets/esp32s3.yaml"),
        "--format",
        "json",
        "--out",
        json_path.to_str().expect("suppressed path"),
        "--suppress",
        "EF0101,EF0102",
        "--suppress",
        "EF0104,EF0203",
    ]);
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let json = fs::read_to_string(&json_path).expect("suppressed report");
    assert!(json.contains("\"status\": \"pass\""));
    assert!(json.contains("suppressed_diagnostics"));
    assert!(json.contains("EF0101"));
    assert!(json.contains("EF0203"));
    parse_json(&json).expect("parse suppressed report");

    fs::remove_dir_all(dir).expect("cleanup temp dir");
}
#[test]
fn onnx_input_dispatches_to_adapter() {
    let dir = unique_dir("onnx-dispatch");
    let model = dir.join("invalid.onnx");
    let json_path = dir.join("execution-error.json");
    let summary_path = dir.join("execution-error.md");
    let sarif_path = dir.join("execution-error.sarif");
    let text_path = dir.join("execution-error.txt");
    let snapshot_path = dir.join("execution-error-snapshot.json");
    fs::write(&model, b"not an onnx model").expect("write invalid onnx");

    let output = run(&[
        "check",
        model.to_str().expect("onnx path"),
        "--target",
        &fixture("targets/esp32s3.yaml"),
        "--format",
        "json",
        "--out",
        json_path.to_str().expect("json path"),
        "--summary",
        summary_path.to_str().expect("summary path"),
    ]);
    assert_eq!(output.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("ONNX adapter failed") || stderr.contains("onnx is required"));
    let json = fs::read_to_string(&json_path).expect("execution error json");
    assert!(json.contains("\"schema\": \"edgefit.execution_error.v1\""));
    assert!(json.contains("\"status\": \"execution_error\""));
    parse_json(&json).expect("parse execution error json");
    let summary = fs::read_to_string(&summary_path).expect("execution error summary");
    assert!(summary.contains("# EdgeFit Execution Error"));
    assert!(summary.contains("edgefit.execution_error.v1"));

    let sarif_output = run(&[
        "check",
        model.to_str().expect("onnx path"),
        "--target",
        &fixture("targets/esp32s3.yaml"),
        "--format",
        "sarif",
        "--out",
        sarif_path.to_str().expect("sarif path"),
    ]);
    assert_eq!(sarif_output.status.code(), Some(2));
    let sarif = fs::read_to_string(&sarif_path).expect("execution error sarif");
    assert!(sarif.contains("\"version\": \"2.1.0\""));
    assert!(sarif.contains("edgefit.execution_error.v1"));
    assert!(sarif.contains("EFEXECUTION"));
    parse_json(&sarif).expect("parse execution error sarif");

    let text_output = run(&[
        "check",
        model.to_str().expect("onnx path"),
        "--target",
        &fixture("targets/esp32s3.yaml"),
        "--out",
        text_path.to_str().expect("text path"),
    ]);
    assert_eq!(text_output.status.code(), Some(2));
    let text = fs::read_to_string(&text_path).expect("execution error text");
    assert!(text.starts_with("EdgeFit execution error:"));
    assert!(!text.contains("edgefit.execution_error.v1"));

    let snapshot_output = run(&[
        "snapshot",
        model.to_str().expect("onnx path"),
        "--target",
        &fixture("targets/esp32s3.yaml"),
        "--out",
        snapshot_path.to_str().expect("snapshot path"),
    ]);
    assert_eq!(snapshot_output.status.code(), Some(2));
    let snapshot = fs::read_to_string(&snapshot_path).expect("snapshot execution error");
    assert!(snapshot.contains("\"schema\": \"edgefit.execution_error.v1\""));
    parse_json(&snapshot).expect("parse snapshot execution error");

    fs::remove_dir_all(dir).expect("cleanup temp dir");
}

#[test]
fn onnx_argument_error_does_not_write_execution_artifact() {
    let dir = unique_dir("onnx-argument-error");
    let model = dir.join("invalid.onnx");
    let report = dir.join("must-not-exist.json");
    fs::write(&model, b"not an onnx model").expect("write invalid onnx");

    let output = run(&[
        "check",
        model.to_str().expect("onnx path"),
        "--format",
        "json",
        "--out",
        report.to_str().expect("report path"),
    ]);
    assert_eq!(output.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&output.stderr).contains("--target is required"));
    assert!(!report.exists());

    fs::remove_dir_all(dir).expect("cleanup temp dir");
}

#[test]
fn external_json_cannot_claim_adapter_provenance() {
    let dir = unique_dir("external-provenance");
    let model = dir.join("forged.edgefit.json");
    let report = dir.join("report.json");
    let source = fs::read_to_string(fixture("examples/models/good_tiny.edgefit.json"))
        .expect("read normalized fixture");
    let with_adapter_fields = source.replacen(
        "\"sha256\": \"sha256:good-tiny-example\"",
        "\"sha256\": \"sha256:good-tiny-example\",\n    \"external_data_file_count\": 0,\n    \"opset_imports\": [{ \"domain\": \"ai.onnx\", \"version\": 13 }]",
        1,
    );
    assert_ne!(with_adapter_fields, source, "inject adapter model fields");
    let forged = with_adapter_fields.replacen(
        "  \"graph\": {",
        "  \"normalization\": { \"shape_inference\": { \"status\": \"pass\", \"error\": null } },\n  \"graph\": {",
        1,
    );
    assert_ne!(forged, with_adapter_fields, "inject normalization field");
    fs::write(&model, forged).expect("write forged normalized model");

    let output = run(&[
        "check",
        model.to_str().expect("model path"),
        "--target",
        &fixture("targets/esp32s3.yaml"),
        "--format",
        "json",
        "--out",
        report.to_str().expect("report path"),
    ]);
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report = fs::read_to_string(&report).expect("read report");
    assert!(report.contains("EF0001"));
    assert!(report.contains("trusted_normalized_json"));

    fs::remove_dir_all(dir).expect("cleanup temp dir");
}

#[test]
fn snapshot_diff_reports_regression() {
    let dir = unique_dir("snapshot-diff");
    let good = dir.join("good.json");
    let bad = dir.join("bad.json");
    let diff = dir.join("diff.md");
    let diff_json = dir.join("diff.json");

    let good_output = run(&[
        "snapshot",
        &fixture("examples/models/good_tiny.edgefit.json"),
        "--target",
        &fixture("targets/esp32s3.yaml"),
        "--out",
        good.to_str().expect("good path"),
    ]);
    assert!(good_output.status.success());

    let bad_output = run(&[
        "snapshot",
        &fixture("examples/models/bad_detector.edgefit.json"),
        "--target",
        &fixture("targets/esp32s3.yaml"),
        "--out",
        bad.to_str().expect("bad path"),
    ]);
    assert_eq!(bad_output.status.code(), Some(1));

    let diff_output = run(&[
        "diff",
        "--old",
        good.to_str().expect("good path"),
        "--new",
        bad.to_str().expect("bad path"),
        "--out",
        diff.to_str().expect("diff path"),
    ]);
    assert_eq!(diff_output.status.code(), Some(1));
    let text = fs::read_to_string(&diff).expect("diff report");
    assert!(text.contains("EF0201"));
    assert!(text.contains("model_file_bytes"));

    let json_output = run(&[
        "diff",
        "--old",
        good.to_str().expect("good path"),
        "--new",
        bad.to_str().expect("bad path"),
        "--format",
        "json",
        "--out",
        diff_json.to_str().expect("diff json path"),
    ]);
    assert_eq!(json_output.status.code(), Some(1));
    let json = fs::read_to_string(&diff_json).expect("diff json report");
    assert!(json.contains("\"schema\": \"edgefit.diff.v1\""));
    parse_json(&json).expect("parse diff json report");

    fs::remove_dir_all(dir).expect("cleanup temp dir");
}
