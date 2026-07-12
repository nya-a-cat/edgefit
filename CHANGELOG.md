# Changelog

## 0.5.0-alpha.1 - 2026-07-13

### Added

- Bounded optimizer validation that compares the normal greedy plan with an exact placement oracle for graphs of up to 14 nodes while reusing the production spill scheduler.
- Deterministic profile-matrix sweeps with robust-pass, fragile, and robust-fail classifications and canonical Rust/Python output.
- Strict external calibration capture manifests and an atomic packer that binds model, profile, runtime, raw attachments, measurements, and verification output without claiming attestation.
- Deterministic diamond, fan-out, residual, 100K, and manual 1M-node optimizer evidence cases.

### Changed

- JSON parsing now rejects duplicate object keys globally instead of silently accepting the last value.
- Release acceptance now covers optimizer validation, profile sweeps, external calibration packs, and their packaged example inputs across supported platforms.

### Compatibility

- Existing commands and `0`/`1`/`2` exit meanings remain unchanged; all new machine outputs use new `v1` schema identifiers.
- Oracle and sweep results remain profile-driven simulation. Calibration packs provide hash integrity only, not device identity, measurement authenticity, signatures, or real-hardware validation.

## 0.4.0-alpha.1 - 2026-07-13

### Added

- Hash-bound Calibration v1 evidence parsing, verification, JSON/Markdown rendering, and native/Python CLI support.
- Deterministic calibration simulation backed by the analyzer and hardware optimizer, with explicit simulated-confidence and no-attestation boundaries.
- Controlled nominal, latency-failure, arena-failure, spill/reload, blocker, tamper, and Rust/Python parity contracts in GitHub Actions.

### Changed

- Hardened optimizer plan invariants, candidate selection, partition and transfer accounting, scratchpad spill/reload behavior, and canonical failure output.
- Protected calibration evidence, model, target, and attachment inputs from output aliasing and stale-artifact confusion.
- Replaced hand-written calibration fixtures with canonical simulator-generated evidence.

### Compatibility

- Existing `target validate`, `check`, `optimize`, `snapshot`, and `diff` commands retain their `0`/`1`/`2` exit-code meanings.
- Calibration and optimizer latency remain profile-driven simulations, not real-device measurements or deployment guarantees.
