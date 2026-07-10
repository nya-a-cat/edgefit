from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import tarfile
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = Path(__file__).with_name("real_world_corpus.json")
DEFAULT_CACHE = ROOT / "tmp" / "real_world_corpus"


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify EdgeFit real-world ONNX corpus entries.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--cache", default=str(DEFAULT_CACHE))
    parser.add_argument("--download", action="store_true", help="Download missing archives or model files from manifest URLs.")
    parser.add_argument(
        "--model-id",
        action="append",
        default=[],
        help="Only verify this model ID; repeat the option to select multiple models.",
    )
    parser.add_argument("--out", help="Write JSON summary to this path.")
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)

    normalize = load_normalize()
    results = []
    for item in select_models(manifest["models"], args.model_id):
        results.append(verify_model(item, cache, args.download, normalize))

    summary = {"schema": "edgefit.real_world_corpus.result.v1", "results": results}
    text = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


def select_models(models: list[dict[str, Any]], requested_ids: list[str]) -> list[dict[str, Any]]:
    """按清单顺序筛选模型，并拒绝未知或重复 ID。"""
    if not requested_ids:
        return models
    if len(requested_ids) != len(set(requested_ids)):
        raise SystemExit("duplicate --model-id values are not allowed")
    requested = set(requested_ids)
    available = {str(item.get("id")) for item in models}
    unknown = sorted(requested - available)
    if unknown:
        raise SystemExit(f"unknown model IDs: {', '.join(unknown)}")
    return [item for item in models if item.get("id") in requested]


def load_normalize():
    module_path = Path(__file__).with_name("normalize_onnx.py")
    spec = importlib.util.spec_from_file_location("normalize_onnx", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module.normalize


def verify_model(item: dict[str, Any], cache: Path, download: bool, normalize) -> dict[str, Any]:
    model_path = prepare_model_file(item, cache, download)
    model_bytes = model_path.stat().st_size
    model_sha = sha256(model_path)
    if model_bytes != item["model_bytes"]:
        raise SystemExit(f"model byte mismatch for {item['id']}: {model_bytes}")
    if model_sha != item["model_sha256"]:
        raise SystemExit(f"model sha256 mismatch for {item['id']}: {model_sha}")

    data = normalize(model_path)
    ops = sorted({node["op_type"] for node in data["graph"]["nodes"]})
    expected_ops = sorted(item["expected_ops"])
    if ops != expected_ops:
        raise SystemExit(f"operator mismatch for {item['id']}: {ops}")

    domain_ops = observed_domain_ops(data)
    expected_domains = expected_domain_ops(item)
    if domain_ops != expected_domains:
        raise SystemExit(f"operator domain mismatch for {item['id']}: {domain_ops}")

    outputs = data["graph"]["outputs"]
    if outputs != item["expected_outputs"]:
        raise SystemExit(f"output mismatch for {item['id']}: {outputs}")

    return {
        "id": item["id"],
        "status": "pass",
        "model_path": str(model_path),
        "model_bytes": model_bytes,
        "model_sha256": model_sha,
        "node_count": len(data["graph"]["nodes"]),
        "ops": ops,
        "domain_ops": domain_ops,
        "outputs": outputs,
    }


def prepare_model_file(item: dict[str, Any], cache: Path, download: bool) -> Path:
    if "archive_name" in item:
        return prepare_archive_model(item, cache, download)
    if "model_url" in item:
        return prepare_direct_model(item, cache, download)
    raise SystemExit(f"{item.get('id', '<unknown>')} must define archive_name or model_url")


def prepare_archive_model(item: dict[str, Any], cache: Path, download: bool) -> Path:
    archive = cache / item["archive_name"]
    if not archive.exists():
        if not download:
            raise SystemExit(f"missing archive {archive}; rerun with --download")
        download_file(item["archive_url"], archive)

    archive_bytes = archive.stat().st_size
    archive_sha = sha256(archive)
    if archive_bytes != item["archive_bytes"]:
        raise SystemExit(f"archive byte mismatch for {item['id']}: {archive_bytes}")
    if archive_sha != item["archive_sha256"]:
        raise SystemExit(f"archive sha256 mismatch for {item['id']}: {archive_sha}")

    model_path = cache / item["model_member"]
    if not model_path.exists():
        with tarfile.open(archive, "r:gz") as tar:
            try:
                member = tar.getmember(item["model_member"])
            except KeyError as exc:
                raise SystemExit(f"{item['model_member']} missing from {archive}") from exc
            source = tar.extractfile(member)
            if source is None:
                raise SystemExit(f"{item['model_member']} is not a regular file")
            model_path.parent.mkdir(parents=True, exist_ok=True)
            model_path.write_bytes(source.read())
    return model_path


def prepare_direct_model(item: dict[str, Any], cache: Path, download: bool) -> Path:
    model_path = cache / item.get("model_name", Path(item["model_url"]).name)
    if not model_path.exists():
        if not download:
            raise SystemExit(f"missing model {model_path}; rerun with --download")
        download_file(item["model_url"], model_path)
    return model_path


def expected_domain_ops(item: dict[str, Any]) -> list[str]:
    domain_overrides = item.get("expected_operator_domains", {})
    keys = []
    for op in item["expected_ops"]:
        for domain in domain_overrides.get(op, ["ai.onnx"]):
            keys.append(op_key(domain, op))
    return sorted(keys)


def observed_domain_ops(data: dict[str, Any]) -> list[str]:
    return sorted({op_key(node.get("domain"), node["op_type"]) for node in data["graph"]["nodes"]})


def op_key(domain: str | None, op: str) -> str:
    return f"{domain or 'ai.onnx'}::{op}"


def download_file(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response:
        path.write_bytes(response.read())


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
