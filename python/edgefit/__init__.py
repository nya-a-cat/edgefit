"""EdgeFit Python 编排框架；公开 API 延迟加载，允许独立复用 ONNX adapter。"""

__all__ = [
    "EdgeFitError",
    "batch",
    "check",
    "load_profile",
    "optimize",
    "optimize_sweep",
    "pack_calibration",
    "render",
    "render_optimization",
    "render_optimization_validation",
    "render_optimization_sweep",
    "render_calibration",
    "simulate_calibration",
    "verify_calibration",
    "validate_optimization",
]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(name)
    from . import framework

    return getattr(framework, name)
