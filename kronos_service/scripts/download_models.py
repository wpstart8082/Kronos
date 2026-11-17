#!/usr/bin/env python3
"""Prepare model weights in the host ``models/`` directory.

Strategy per artifact:
1. copy from the local HuggingFace cache (``~/.cache/huggingface/hub``) if present
2. otherwise download from HuggingFace Hub via ``snapshot_download``

Usage (run from anywhere, defaults fit this repo):
    python kronos_service/scripts/download_models.py
    python kronos_service/scripts/download_models.py --models kronos-base
    python kronos_service/scripts/download_models.py --models-dir /e/AIProject/Kronos/models --source hub
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

MODEL_ARTIFACTS = {
    "kronos-mini": ["Kronos-mini", "Kronos-Tokenizer-2k"],
    "kronos-small": ["Kronos-small", "Kronos-Tokenizer-base"],
    "kronos-base": ["Kronos-base", "Kronos-Tokenizer-base"],
}
HF_REPO = {
    "Kronos-mini": "NeoQuasar/Kronos-mini",
    "Kronos-small": "NeoQuasar/Kronos-small",
    "Kronos-base": "NeoQuasar/Kronos-base",
    "Kronos-Tokenizer-base": "NeoQuasar/Kronos-Tokenizer-base",
    "Kronos-Tokenizer-2k": "NeoQuasar/Kronos-Tokenizer-2k",
}


def hf_cache_roots() -> list[Path]:
    roots = []
    env_home = os.environ.get("HF_HOME")
    if env_home:
        roots.append(Path(env_home) / "hub")
    # windows default & posix default
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    return roots


def find_in_cache(artifact: str) -> Path | None:
    """Return the newest snapshot dir inside the HF cache for an artifact."""
    for root in hf_cache_roots():
        repo_dir = root / f"models--NeoQuasar--{artifact}"
        if not repo_dir.is_dir():
            continue
        snapshots = sorted(
            (s for s in (repo_dir / "snapshots").glob("*") if s.is_dir()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for snap in snapshots:
            # snapshot entries may be symlinks into blobs/ - resolve them
            has_config = (snap / "config.json").exists()
            has_weights = any(snap.glob("*.safetensors"))
            if has_config and has_weights:
                return snap
    return None


def copy_from_snapshot(snapshot: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for item in snapshot.iterdir():
        if item.is_file():
            dst = target / item.name
            if dst.exists():
                continue
            shutil.copy2(os.path.realpath(item), dst)


def download_from_hub(artifact: str, target: Path) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=HF_REPO[artifact],
        local_dir=str(target),
        allow_patterns=["config.json", "*.safetensors"],
    )


def prepare(artifact: str, models_dir: Path, source: str) -> bool:
    target = models_dir / artifact
    ok = (target / "config.json").is_file() and any(target.glob("*.safetensors"))
    if ok:
        print(f"[skip] {artifact}: already present at {target}")
        return True

    if source in ("auto", "cache"):
        snap = find_in_cache(artifact)
        if snap:
            print(f"[copy] {artifact}: {snap} -> {target}")
            try:
                copy_from_snapshot(snap, target)
                return True
            except Exception as e:  # noqa: BLE001
                print(f"[warn] copy failed ({e}), falling back to hub download")
        elif source == "cache":
            print(f"[miss] {artifact}: not found in local HF cache")
            return False

    print(f"[download] {artifact}: {HF_REPO[artifact]} -> {target}")
    try:
        download_from_hub(artifact, target)
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[fail] {artifact}: {e}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", default=str(REPO_ROOT / "models"))
    parser.add_argument("--models", default="kronos-base", help="comma list: kronos-mini,kronos-small,kronos-base")
    parser.add_argument("--source", choices=["auto", "cache", "hub"], default="auto")
    args = parser.parse_args()

    models_dir = Path(args.models_dir).resolve()
    models_dir.mkdir(parents=True, exist_ok=True)
    print(f"目标目录: {models_dir}\n")

    requested = [m.strip() for m in args.models.split(",") if m.strip()]
    artifacts: list[str] = []
    for m in requested:
        if m not in MODEL_ARTIFACTS:
            print(f"[error] 未知模型 {m}, 可选: {list(MODEL_ARTIFACTS)}")
            return 2
        for a in MODEL_ARTIFACTS[m]:
            if a not in artifacts:
                artifacts.append(a)

    failed = [a for a in artifacts if not prepare(a, models_dir, args.source)]
    print()
    if failed:
        print(f"未完成: {failed}")
        return 1
    print("全部就绪。目录结构:")
    for a in artifacts:
        print(f"  {models_dir / a}")
    print("\n下一步: 在 .env 中设置 KRONOS_MODELS_DIR 后启动服务；模型齐备时可开启 HF_HUB_OFFLINE=1 离线启动。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
