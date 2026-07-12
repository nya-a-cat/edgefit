//! Strict, dependency-free calibration evidence verification.
//!
//! Version 1 binds evidence to hashes only. It does not provide signatures,
//! attestation, trust, or authority to change target profiles or optimization.

#![forbid(unsafe_code)]

mod capture;
mod hash;
mod json;
mod render;
mod schema;
mod simulation;
mod verify;

pub use capture::{
    parse_capture_manifest, CaptureAttachment, CaptureIdentity, CaptureManifest,
    CaptureMeasurements, CaptureMetadata, CAPTURE_SCHEMA,
};
pub use hash::sha256_hex;
pub use render::{
    render_evidence_json, render_verification_json, render_verification_markdown,
};
pub use schema::{
    parse_evidence, Attachment, Bindings, Capture, Check, CheckStatus, Environment, Error,
    Evidence, ExpectedBindings, Identity, LoadedAttachment, Measurements, Metrics, Result,
    RuntimeResult, Thresholds, Verification, VerificationBudget, EVIDENCE_SCHEMA,
    MAX_ATTACHMENTS, MAX_ATTACHMENT_BYTES, MAX_LATENCY_SAMPLES, VERIFICATION_SCHEMA,
};
pub use simulation::{
    parse_simulation_scenario, SimulationScenario, SIMULATION_SCHEMA, SIMULATION_TRACE_SCHEMA,
};
pub use verify::{verify, verify_json};
