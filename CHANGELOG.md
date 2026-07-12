# Changelog

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
