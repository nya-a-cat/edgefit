#!/usr/bin/env python3
"""兼容历史工具路径的 ONNX 规范化入口。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from edgefit.onnx_adapter import *  # noqa: F403


if __name__ == "__main__":
    raise SystemExit(main())
