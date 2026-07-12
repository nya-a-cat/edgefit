"""围绕 Rust engine 组织 ONNX 接入、批处理与报告输出。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from . import _native
from .onnx_adapter import normalize


class EdgeFitError(RuntimeError):
    """保留 EdgeFit engine 原始错误消息的框架异常。"""

    code = "EDGEFIT_EXECUTION"


@dataclass(frozen=True)
class TargetProfileSource:
    """显式加载并经过 Rust parser 验证的 target profile。"""

    path: Path
    text: str
    target_id: str


@dataclass(frozen=True)
class PreparedModel:
    """一次规范化后可重复交给 Rust engine 的稳定输入。"""

    text: str
    adapter_generated: bool


def load_profile(source: str | Path) -> TargetProfileSource:
    path = Path(source)
    # 指纹必须覆盖与 Rust CLI 相同的原始 UTF-8 文本，包括可选 BOM。
    text = path.read_bytes().decode("utf-8")
    try:
        target_id = _native.validate_target(text, str(path))
    except ValueError as exc:
        raise EdgeFitError(str(exc)) from exc
    return TargetProfileSource(path=path, text=text, target_id=target_id)


def check(
    model: str | Path,
    target: str | Path | TargetProfileSource,
    *,
    suppress: Iterable[str] = (),
) -> dict[str, object]:
    """执行单模型检查并返回 Rust 生成的 canonical JSON 对象。"""
    return json.loads(render(model, target, format="json", suppress=suppress))


def render(
    model: str | Path,
    target: str | Path | TargetProfileSource,
    *,
    format: str = "json",
    suppress: Iterable[str] = (),
) -> str:
    if format not in {"text", "json", "markdown", "sarif"}:
        raise EdgeFitError("report format must be text, json, markdown, or sarif")
    prepared = _prepare_model(Path(model))
    profile = target if isinstance(target, TargetProfileSource) else load_profile(target)
    try:
        return _native.analyze(
            prepared.text,
            profile.text,
            str(profile.path),
            format,
            list(suppress),
            prepared.adapter_generated,
        )
    except ValueError as exc:
        raise EdgeFitError(str(exc)) from exc


def optimize(
    model: str | Path,
    target: str | Path | TargetProfileSource,
) -> dict[str, object]:
    """生成可审计的硬件执行计划，不修改输入模型。"""
    return json.loads(render_optimization(model, target))


def render_optimization(
    model: str | Path,
    target: str | Path | TargetProfileSource,
    *,
    format: str = "json",
) -> str:
    prepared = _prepare_model(Path(model))
    profile = target if isinstance(target, TargetProfileSource) else load_profile(target)
    try:
        return _native.optimize(
            prepared.text,
            profile.text,
            str(profile.path),
            format,
            prepared.adapter_generated,
        )
    except ValueError as exc:
        raise EdgeFitError(str(exc)) from exc


def verify_calibration(
    evidence: str | Path,
    model: str | Path,
    target: str | Path,
) -> dict[str, object]:
    """Verify hash-bound runtime/device calibration evidence."""
    _, rendered = _run_calibration(evidence, model, target, "json")
    return json.loads(rendered)


def render_calibration(
    evidence: str | Path,
    model: str | Path,
    target: str | Path,
    *,
    format: str = "json",
) -> str:
    return _run_calibration(evidence, model, target, format)[1]


def _run_calibration(
    evidence: str | Path,
    model: str | Path,
    target: str | Path,
    format: str,
) -> tuple[str, str]:
    if format not in {"json", "markdown"}:
        raise EdgeFitError("calibration format must be json or markdown")
    try:
        return _native.verify_calibration(
            str(Path(evidence)),
            str(Path(model)),
            str(Path(target)),
            format,
        )
    except ValueError as exc:
        raise EdgeFitError(str(exc)) from exc


def batch(
    models: Iterable[str | Path],
    target: str | Path | TargetProfileSource,
    *,
    suppress: Iterable[str] = (),
) -> list[dict[str, object]]:
    """按输入顺序执行批处理，避免隐式并发改变证据顺序。"""
    profile = target if isinstance(target, TargetProfileSource) else load_profile(target)
    suppressions = tuple(suppress)
    return [check(model, profile, suppress=suppressions) for model in models]


def _prepare_model(path: Path) -> PreparedModel:
    if path.suffix.lower() == ".onnx":
        normalized = normalize(path)
        return PreparedModel(
            text=json.dumps(normalized, ensure_ascii=False, separators=(",", ":")),
            adapter_generated=True,
        )
    return PreparedModel(text=path.read_bytes().decode("utf-8"), adapter_generated=False)
