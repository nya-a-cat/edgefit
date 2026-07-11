# Public Repository Boundary

EdgeFit keeps private planning, local execution history, source research
documents, and internal review conclusions outside the public Git repository.
The files remain available in the local project directory; `.gitignore` only
prevents them from being added accidentally.

## Public content

- Product source, tests, examples, and target profiles.
- User-facing architecture, CLI, diagnostic, Action, and benchmark-method docs.
- The MIT license and dependency manifests.

## Local-only content

- Word research documents under `docs/reference/`.
- `progress.md` and the internal `ROADMAP.md`.
- Local environment files and generated build, report, cache, and benchmark
  artifacts.
- Internal implementation, trial, profile-confidence, operator-support, and
  runtime-boundary review records.
- The implementation audit entrypoint that requires the private Word report.

Before the first public commit, inspect ignored paths and the staged file list.
Do not use `git add -f` to bypass these release boundaries.

## Automation boundary

- `CI` runs Rust and ONNX adapter tests on Linux, Windows, and macOS, then
  exposes the stable `ci-gate` result for branch rules.
- `Release candidate` builds platform archives and `SHA256SUMS` on manual
  dispatch without publishing a release.
- A matching `v<workspace-version>` tag is required before the same workflow
  can create a GitHub Release.
- Publishing a release triggers `Release smoke`, which checks out the published
  tag and verifies that the local composite Action installs the checksum-verified
  release binary for normalized JSON and ONNX inputs.
- A workspace version containing a SemVer prerelease suffix, such as
  `0.2.0-alpha.2`, is published as a GitHub prerelease; versions without a
  suffix are published as regular releases.
- Release archives are checksum-protected but currently unsigned and are not
  Apple-notarized.
- Local compilation and tests are not part of the public publishing procedure;
  GitHub Actions is the automated execution environment.
