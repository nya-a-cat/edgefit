# GitHub Action Usage

EdgeFit is intended to run as a pull request gate for ONNX deployment budgets.
The action downloads the matching EdgeFit release archive, verifies its SHA256
checksum, validates the target profile, writes a SARIF report, and appends the
Markdown summary to the GitHub job summary. It does not install Rust or build
the CLI in caller workflows.

## Workflow

```yaml
name: EdgeFit

on:
  pull_request:
  push:
    branches: [main]

jobs:
  edgefit:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      security-events: write
    steps:
      - uses: actions/checkout@v7
      - name: Run EdgeFit
        uses: nya-a-cat/edgefit@v0.2.0-alpha.2.1
        with:
          model: models/model.onnx
          target: targets/esp32s3.yaml
          report: edgefit.sarif
          summary: edgefit-summary.md
          # suppressions: EF0104,EF0203
      - name: Upload SARIF
        if: ${{ always() && hashFiles('edgefit.sarif') != '' }}
        uses: github/codeql-action/upload-sarif@v4
        with:
          sarif_file: edgefit.sarif
```

`v0.2.0-alpha.2.1` is a prerelease intended for reproducible evaluation. Until a
stable tag is published, pin long-lived or production workflows to a reviewed
full commit SHA rather than following `main` or a movable branch.

Use `install-onnx: "false"` for pre-normalized `*.edgefit.json` files.
Use `suppressions` for accepted diagnostics that should remain visible in the
Markdown and JSON reports while allowing the current check to pass. A later
snapshot/report diff still blocks newly introduced suppressed errors.

The composite Action is designed for Linux x86_64 bash runners. The requested
Action ref must resolve to a matching GitHub Release containing the Linux
archive and `SHA256SUMS`; missing or mismatched release assets fail closed.
The Action records the CLI
exit status, publishes any SARIF/Markdown files first, then restores the gate
failure. This keeps a non-compliant model reviewable instead of skipping the
summary when `edgefit check` exits with `1`.

## Outputs

- `edgefit.sarif` contains SARIF 2.1.0 diagnostics for GitHub code scanning, including EdgeFit logical locations and stable fingerprints.
- `edgefit-summary.md` contains the same report as GitHub-flavored Markdown.
- The `install_method` Action output is `release` for the public checksum-verified path.
- The `install_duration_ms` Action output records binary installation time for delivery checks.
- The composite action appends the Markdown report to `$GITHUB_STEP_SUMMARY`.
- Suppressed diagnostics stay in a dedicated report section and are excluded from SARIF alerts.
- On fork pull requests without `security-events: write`, keep the normal gate
  and summary, and upload the SARIF as a regular artifact instead of using the
  code-scanning upload step.

## Public PR Trial Evidence

After a public repository trial runs, copy
`examples/public_pr_trials.example.json` to the ignored local path
`docs/public_pr_trials.json`, then record the PR URL, Actions run URL, commit
SHA, target profile, model path, SARIF upload status, job-summary status, and
outcome clarity. Run:

```powershell
.venv\Scripts\python.exe tools\public_pr_trial_gate.py --manifest docs\public_pr_trials.json --out tmp\public_pr_trials\public-pr-trial-gate.json --markdown-out tmp\public_pr_trials\public-pr-trial-gate.md
```

The confidence gate consumes `tmp/public_pr_trials/public-pr-trial-gate.json`.
The ORT mobile-like seed profile can move to the next confidence review after
three verified trials across three distinct public GitHub repositories.

## Target Profiles

Target profiles should live in the caller repository and include source,
confidence, and last-verified metadata. The action validates the profile before
checking the model so CI fails early when the target contract is incomplete.

The repository includes `targets/esp32s3.yaml`, `targets/ort-mobile-cpu.yaml`,
and `targets/tflm-micro.yaml` as seed profile templates. See
`docs/TARGET_PROFILES.md` for their intended use, source boundary, and local
validation commands.
