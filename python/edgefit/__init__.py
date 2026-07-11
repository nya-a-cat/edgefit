"""EdgeFit Python 编排框架；公开 API 延迟加载，允许独立复用 ONNX adapter。"""

__all__ = ["EdgeFitError", "batch", "check", "load_profile", "render"]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(name)
    from . import framework

    return getattr(framework, name)
