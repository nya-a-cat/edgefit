//! 确定性 Calibration 模拟场景契约。
//!
//! 场景只声明受控扰动，不代表真实设备、运行时或硬件测量。

use crate::json::{
    decimal_u64, decimal_u64_array, exact_fields, expect_literal_string, nonempty_string, object,
    JsonParser,
};
use crate::schema::{validate_attachment_name, validate_timestamp};
use crate::{Error, Result, MAX_LATENCY_SAMPLES};

pub const SIMULATION_SCHEMA: &str = "edgefit.calibration_simulation.v1";
pub const SIMULATION_TRACE_SCHEMA: &str = "edgefit.calibration_simulation_trace.v1";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SimulationScenario {
    pub scenario_id: String,
    pub captured_at: String,
    pub warmup_runs: u64,
    pub latency_scale_ppm: Vec<u64>,
    pub p95_budget_scale_ppm: u64,
    pub arena_overhead_bytes: u64,
}

pub fn parse_simulation_scenario(input: &str) -> Result<SimulationScenario> {
    let value = JsonParser::new(input).parse()?;
    let root = object(&value, "simulation")?;
    exact_fields(
        root,
        &[
            "schema",
            "scenario_id",
            "captured_at",
            "warmup_runs",
            "latency_scale_ppm",
            "p95_budget_scale_ppm",
            "arena_overhead_bytes",
        ],
        "simulation",
    )?;
    expect_literal_string(root, "schema", SIMULATION_SCHEMA)?;
    let scenario_id = nonempty_string(root, "scenario_id")?;
    validate_attachment_name(&scenario_id)
        .map_err(|_| Error::new("simulation scenario_id must be a safe identifier"))?;
    let captured_at = nonempty_string(root, "captured_at")?;
    validate_timestamp(&captured_at)?;
    let latency_scale_ppm = decimal_u64_array(root, "latency_scale_ppm")?;
    if latency_scale_ppm.is_empty() {
        return Err(Error::new(
            "simulation latency_scale_ppm must not be empty",
        ));
    }
    if latency_scale_ppm.len() > MAX_LATENCY_SAMPLES {
        return Err(Error::new(format!(
            "simulation latency_scale_ppm exceeds limit {MAX_LATENCY_SAMPLES}"
        )));
    }
    if latency_scale_ppm.contains(&0) {
        return Err(Error::new(
            "simulation latency_scale_ppm values must be greater than zero",
        ));
    }
    let p95_budget_scale_ppm = decimal_u64(root, "p95_budget_scale_ppm")?;
    if p95_budget_scale_ppm == 0 {
        return Err(Error::new(
            "simulation p95_budget_scale_ppm must be greater than zero",
        ));
    }
    Ok(SimulationScenario {
        scenario_id,
        captured_at,
        warmup_runs: decimal_u64(root, "warmup_runs")?,
        latency_scale_ppm,
        p95_budget_scale_ppm,
        arena_overhead_bytes: decimal_u64(root, "arena_overhead_bytes")?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SCENARIO: &str = r#"{
  "schema": "edgefit.calibration_simulation.v1",
  "scenario_id": "nominal",
  "captured_at": "2026-07-13T00:00:00Z",
  "warmup_runs": "2",
  "latency_scale_ppm": ["900000", "1000000", "1100000"],
  "p95_budget_scale_ppm": "1200000",
  "arena_overhead_bytes": "64"
}"#;

    #[test]
    fn parses_a_strict_deterministic_scenario() {
        let scenario = parse_simulation_scenario(SCENARIO).unwrap();
        assert_eq!(scenario.scenario_id, "nominal");
        assert_eq!(scenario.latency_scale_ppm, [900_000, 1_000_000, 1_100_000]);
        assert_eq!(scenario.arena_overhead_bytes, 64);
    }

    #[test]
    fn rejects_invalid_fields_and_scales() {
        assert!(parse_simulation_scenario(&SCENARIO.replace(
            "\"warmup_runs\": \"2\"",
            "\"warmup_runs\": \"2\", \"unknown\": \"x\""
        ))
        .is_err());
        assert!(parse_simulation_scenario(&SCENARIO.replace("\"900000\"", "\"0\""))
            .is_err());
        assert!(parse_simulation_scenario(&SCENARIO.replace(
            "[\"900000\", \"1000000\", \"1100000\"]",
            "[]"
        ))
        .is_err());
    }
}
