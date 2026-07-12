use edgefit_calibration::{
    parse_evidence, render_evidence_json, render_verification_json, render_verification_markdown,
    sha256_hex, verify, CheckStatus, ExpectedBindings, LoadedAttachment, VerificationBudget,
    MAX_ATTACHMENT_BYTES, MAX_LATENCY_SAMPLES,
};

const VALID: &str = include_str!("fixtures/valid-evidence.json");
const MODEL: &str = "1111111111111111111111111111111111111111111111111111111111111111";
const PROFILE: &str = "2222222222222222222222222222222222222222222222222222222222222222";
const RUNTIME: &str = "3333333333333333333333333333333333333333333333333333333333333333";

fn expected() -> ExpectedBindings {
    ExpectedBindings {
        model_sha256: MODEL.to_string(),
        target_profile_sha256: PROFILE.to_string(),
        runtime_binary_sha256: RUNTIME.to_string(),
    }
}

fn budget() -> VerificationBudget {
    VerificationBudget {
        arena_bytes: 9_007_199_254_740_993,
        p95_latency_ns: 19,
    }
}

fn replace_once(input: &str, from: &str, to: &str) -> String {
    assert!(input.contains(from), "fixture replacement must exist");
    input.replacen(from, to, 1)
}

fn evidence_with_samples(count: usize) -> String {
    let samples = (1..=count)
        .map(|value| format!("\"{value}\""))
        .collect::<Vec<_>>()
        .join(", ");
    let text = replace_once(
        VALID,
        "\"measured_runs\": \"20\"",
        &format!("\"measured_runs\": \"{count}\""),
    );
    replace_once(
        &text,
        "\"samples\": [\"1\", \"2\", \"3\", \"4\", \"5\", \"6\", \"7\", \"8\", \"9\", \"10\", \"11\", \"12\", \"13\", \"14\", \"15\", \"16\", \"17\", \"18\", \"19\", \"20\"]",
        &format!("\"samples\": [{samples}]"),
    )
}

#[test]
fn parses_fixture_without_losing_values_above_2_pow_53() {
    let evidence = parse_evidence(VALID).unwrap();
    assert_eq!(evidence.measurements.arena_high_water_bytes, 9_007_199_254_740_993);
    assert_eq!(evidence.capture.measured_runs, 20);
    assert_eq!(evidence.measurements.latency_ns, (1..=20).collect::<Vec<_>>());
    assert_eq!(evidence.attachments[0].path, "logs/device.txt");
}

#[test]
fn verifies_exact_limits_and_checked_metrics() {
    let evidence = parse_evidence(VALID).unwrap();
    let verification = verify(
        &evidence,
        &expected(),
        &budget(),
        &[LoadedAttachment {
            path: "logs/device.txt",
            bytes: b"device log\n",
        }],
    )
    .unwrap();
    assert_eq!(verification.status, CheckStatus::Pass);
    assert_eq!(verification.metrics.latency_p50_ns, 10);
    assert_eq!(verification.metrics.latency_p95_ns, 19);
    assert_eq!(verification.metrics.latency_mean_ns, 11);
    assert_eq!(verification.metrics.arena_utilization_ppm, 1_000_000);
    assert_eq!(verification.metrics.arena_error_bytes, 0);
    assert_eq!(verification.metrics.p95_latency_error_ns, 0);
    assert_eq!(
        verification.checks.iter().map(|check| check.id).collect::<Vec<_>>(),
        vec![
            "model_binding",
            "target_profile_binding",
            "runtime_binary_binding",
            "runtime_accepted",
            "evidence_arena_threshold",
            "evidence_latency_threshold",
            "expected_arena_budget",
            "expected_latency_budget",
            "attachment",
        ]
    );
}

#[test]
fn one_above_limits_fails_checks() {
    let evidence = parse_evidence(VALID).unwrap();
    let verification = verify(
        &evidence,
        &expected(),
        &VerificationBudget {
            arena_bytes: budget().arena_bytes - 1,
            p95_latency_ns: 18,
        },
        &[LoadedAttachment {
            path: "logs/device.txt",
            bytes: b"device log\n",
        }],
    )
    .unwrap();
    assert_eq!(verification.status, CheckStatus::Fail);
    assert_eq!(verification.metrics.arena_error_bytes, 1);
    assert_eq!(verification.metrics.p95_latency_error_ns, 1);
    assert_eq!(verification.metrics.arena_utilization_ppm, 1_000_000);
}

#[test]
fn nearest_rank_boundaries_for_1_19_20_21_samples() {
    for (count, p50, p95) in [(1, 1, 1), (19, 10, 19), (20, 10, 19), (21, 11, 20)] {
        let evidence = parse_evidence(&evidence_with_samples(count)).unwrap();
        let verification = verify(
            &evidence,
            &expected(),
            &VerificationBudget {
                arena_bytes: budget().arena_bytes,
                p95_latency_ns: u64::MAX,
            },
            &[LoadedAttachment {
                path: "logs/device.txt",
                bytes: b"device log\n",
            }],
        )
        .unwrap();
        assert_eq!(verification.metrics.latency_p50_ns, p50, "count {count}");
        assert_eq!(verification.metrics.latency_p95_ns, p95, "count {count}");
    }
}

#[test]
fn rejected_runtime_is_well_formed_but_fails_verification() {
    let text = replace_once(VALID, "\"accepted\": true", "\"accepted\": false");
    let text = replace_once(&text, "\"rejected_reason\": null", "\"rejected_reason\": \"model rejected\"");
    let evidence = parse_evidence(&text).unwrap();
    let verification = verify(
        &evidence,
        &expected(),
        &budget(),
        &[LoadedAttachment {
            path: "logs/device.txt",
            bytes: b"device log\n",
        }],
    )
    .unwrap();
    assert_eq!(verification.status, CheckStatus::Fail);
    let check = verification.checks.iter().find(|check| check.id == "runtime_accepted").unwrap();
    assert_eq!(check.status, CheckStatus::Fail);
    assert!(check.detail.contains("model rejected"));
}

#[test]
fn parser_fails_closed_on_schema_types_hashes_units_and_timestamp() {
    let cases = [
        replace_once(VALID, "edgefit.calibration_evidence.v1", "edgefit.calibration_evidence.v2"),
        replace_once(VALID, "\"accepted\": true", "\"accepted\": \"true\""),
        replace_once(VALID, MODEL, "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"),
        replace_once(VALID, "\"unit\": \"bytes\"", "\"unit\": \"KiB\""),
        replace_once(VALID, "2026-07-12T12:34:56Z", "2026-02-30T12:34:56Z"),
    ];
    for case in cases {
        assert!(parse_evidence(&case).is_err());
    }
}

#[test]
fn parser_rejects_unknown_duplicate_signature_and_attestation_fields() {
    let unknown = replace_once(VALID, "\"schema\":", "\"unknown\": null,\n  \"schema\":");
    let duplicate = replace_once(VALID, "\"schema\":", "\"schema\": \"edgefit.calibration_evidence.v1\",\n  \"schema\":");
    let signature = replace_once(VALID, "\"attestation\": {", "\"signature\": \"abc\",\n  \"attestation\": {");
    let attestation = replace_once(VALID, "\"kind\": \"none\"", "\"kind\": \"x509\"");
    for case in [unknown, duplicate, signature, attestation] {
        assert!(parse_evidence(&case).is_err());
    }
}

#[test]
fn parser_rejects_empty_oversized_and_mismatched_samples() {
    let empty = evidence_with_samples(0);
    assert!(parse_evidence(&empty).is_err());

    let oversized_samples = std::iter::repeat("\"1\"")
        .take(MAX_LATENCY_SAMPLES + 1)
        .collect::<Vec<_>>()
        .join(",");
    let oversized = replace_once(
        &evidence_with_samples(1),
        "\"samples\": [\"1\"]",
        &format!("\"samples\": [{oversized_samples}]"),
    );
    assert!(parse_evidence(&oversized).is_err());

    let mismatch = replace_once(VALID, "\"measured_runs\": \"20\"", "\"measured_runs\": \"19\"");
    assert!(parse_evidence(&mismatch).is_err());
}

#[test]
fn parser_rejects_decimal_overflow_and_noncanonical_numbers() {
    for replacement in ["18446744073709551616", "01", "-1", "1.0"] {
        let text = replace_once(
            VALID,
            "\"value\": \"9007199254740993\"",
            &format!("\"value\": \"{replacement}\""),
        );
        assert!(parse_evidence(&text).is_err(), "replacement {replacement}");
    }
    let numeric = replace_once(VALID, "\"value\": \"9007199254740993\"", "\"value\": 9007199254740993");
    assert!(parse_evidence(&numeric).is_err());
}

#[test]
fn checked_calculations_reject_zero_budgets_and_saturate_unrepresentable_ratio() {
    let evidence = parse_evidence(VALID).unwrap();
    assert!(verify(
        &evidence,
        &expected(),
        &VerificationBudget {
            arena_bytes: 0,
            p95_latency_ns: 19,
        },
        &[]
    )
    .is_err());

    let huge = replace_once(
        VALID,
        "\"value\": \"9007199254740993\"",
        "\"value\": \"18446744073709551615\"",
    );
    let evidence = parse_evidence(&huge).unwrap();
    let verification = verify(
        &evidence,
        &expected(),
        &VerificationBudget {
            arena_bytes: 1,
            p95_latency_ns: 19,
        },
        &[],
    )
    .unwrap();
    assert_eq!(verification.metrics.arena_utilization_ppm, u64::MAX);
}

#[test]
fn unsafe_attachment_declarations_fail_closed() {
    for path in ["../secret", "/absolute", "C:/windows", "logs\\device.txt", "logs//device"] {
        let text = replace_once(VALID, "logs/device.txt", path);
        assert!(parse_evidence(&text).is_err(), "path {path}");
    }
    let duplicate = replace_once(
        VALID,
        "  ],\n  \"attestation\"",
        &format!(
            ",\n    {{\"name\":\"other\",\"path\":\"logs/device.txt\",\"media_type\":\"text/plain\",\"bytes\":\"0\",\"sha256\":\"{}\"}}\n  ],\n  \"attestation\"",
            sha256_hex(b"")
        ),
    );
    assert!(parse_evidence(&duplicate).is_err());

    let oversized = replace_once(VALID, "\"bytes\": \"11\"", &format!("\"bytes\": \"{}\"", MAX_ATTACHMENT_BYTES + 1));
    assert!(parse_evidence(&oversized).is_err());
}

#[test]
fn attachment_bytes_are_supplied_not_read_and_are_fully_verified() {
    let evidence = parse_evidence(VALID).unwrap();
    let missing = verify(&evidence, &expected(), &budget(), &[]).unwrap();
    assert_eq!(missing.status, CheckStatus::Fail);
    let wrong = verify(
        &evidence,
        &expected(),
        &budget(),
        &[LoadedAttachment {
            path: "logs/device.txt",
            bytes: b"wrong",
        }],
    )
    .unwrap();
    assert_eq!(wrong.status, CheckStatus::Fail);
    assert!(verify(
        &evidence,
        &expected(),
        &budget(),
        &[LoadedAttachment {
            path: "undeclared.txt",
            bytes: b"",
        }]
    )
    .is_err());
}

#[test]
fn bindings_are_hash_only_and_mismatches_fail() {
    let evidence = parse_evidence(VALID).unwrap();
    let mut wrong = expected();
    wrong.model_sha256 = "4444444444444444444444444444444444444444444444444444444444444444".to_string();
    let verification = verify(
        &evidence,
        &wrong,
        &budget(),
        &[LoadedAttachment {
            path: "logs/device.txt",
            bytes: b"device log\n",
        }],
    )
    .unwrap();
    assert_eq!(verification.status, CheckStatus::Fail);
    assert_eq!(verification.checks[0].status, CheckStatus::Fail);
}

#[test]
fn rendering_is_deterministic_and_hash_excludes_its_own_field() {
    let evidence = parse_evidence(VALID).unwrap();
    let verification = verify(
        &evidence,
        &expected(),
        &budget(),
        &[LoadedAttachment {
            path: "logs/device.txt",
            bytes: b"device log\n",
        }],
    )
    .unwrap();
    let json_a = render_verification_json(&verification);
    let json_b = render_verification_json(&verification);
    let markdown_a = render_verification_markdown(&verification);
    let markdown_b = render_verification_markdown(&verification);
    assert_eq!(json_a, json_b);
    assert_eq!(markdown_a, markdown_b);
    assert!(json_a.contains("\"binding\": \"sha256-only\""));
    assert!(json_a.contains("\"attestation\": \"none\""));
    assert!(json_a.contains("\"profile_mutation_authority\": false"));
    assert!(json_a.contains("\"optimizer_decision_authority\": false"));

    let marker = format!(
        ",\n  \"verification_sha256\": \"{}\"\n}}\n",
        verification.verification_sha256
    );
    assert_eq!(json_a.matches(&marker).count(), 1);
    let payload = json_a.replacen(&marker, "\n}\n", 1);
    assert_eq!(sha256_hex(payload.as_bytes()), verification.verification_sha256);
    assert_eq!(render_evidence_json(&evidence), render_evidence_json(&evidence));
}

#[test]
fn sha256_matches_standard_vectors() {
    assert_eq!(
        sha256_hex(b""),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
    assert_eq!(
        sha256_hex(b"abc"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    assert_eq!(
        sha256_hex(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
        "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
    );
    assert_eq!(
        sha256_hex(&vec![b'a'; 1_000_000]),
        "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
    );
}

#[test]
fn parser_rejects_json_numbers_for_decimal_fields() {
    for number in ["9007199254740993", "-1", "1.5", "1e3"] {
        let text = replace_once(
            VALID,
            "\"value\": \"9007199254740993\"",
            &format!("\"value\": {number}"),
        );
        assert!(parse_evidence(&text).is_err(), "number {number}");
    }
}

#[test]
fn parser_rejects_malformed_unicode_and_control_characters() {
    for replacement in [
        r#""target_id": "\uD800""#,
        r#""target_id": "\uDC00""#,
        r#""target_id": "\uD800\u0041""#,
        "\"target_id\": \"bad\u{0001}value\"",
        r#""target_id": "line\nbreak""#,
    ] {
        let text = replace_once(VALID, "\"target_id\": \"esp32-s3\"", replacement);
        assert!(parse_evidence(&text).is_err(), "replacement {replacement:?}");
    }
}

#[test]
fn parser_rejects_excessive_nesting_and_input_size() {
    let nested = format!(
        "{}null{}",
        "[".repeat(130),
        "]".repeat(130),
    );
    assert!(parse_evidence(&nested).is_err());

    let oversized = format!("{{\"padding\":\"{}\"}}", "x".repeat(16 * 1024 * 1024));
    assert!(parse_evidence(&oversized).is_err());
}

#[test]
fn half_up_mean_rounds_ties_away_from_zero() {
    let text = evidence_with_samples(2);
    let text = replace_once(&text, "\"samples\": [\"1\", \"2\"]", "\"samples\": [\"0\", \"1\"]");
    let evidence = parse_evidence(&text).unwrap();
    let verification = verify(
        &evidence,
        &expected(),
        &budget(),
        &[LoadedAttachment {
            path: "logs/device.txt",
            bytes: b"device log\n",
        }],
    )
    .unwrap();
    assert_eq!(verification.metrics.latency_mean_ns, 1);
}
