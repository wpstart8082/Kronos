"""Model registry and lifecycle management.

Resolution order for weights: host-mounted ``{models_dir}/{dirname}`` first
(config.json + model.safetensors), then HuggingFace Hub repo id (downloads
land in ``{models_dir}/hf_cache`` via HF_HOME).
"""
from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger("kronos.model_manager")

REGISTRY: dict[str, dict[str, Any]] = {
    # Online production set: exactly the two models used by asset routing.
    # (kronos-mini / kronos-small were retired with the 2026-11 cleanup -
    # never deployed, weights removed; the HuggingFace originals remain
    # available via NeoQuasar/Kronos-* if ever needed again.)
    "kronos-base": {
        "display_name": "Kronos-base",
        "model_dirname": "Kronos-base",
        "tokenizer_dirname": "Kronos-Tokenizer-base",
        "hf_model_id": "NeoQuasar/Kronos-base",
        "hf_tokenizer_id": "NeoQuasar/Kronos-Tokenizer-base",
        "max_context": 512,
        "params": "102.3M",
    },
    # A-share fine-tuned pair (finetune/ qlib pipeline, full-market daily K-lines
    # 2015-2023, see finetune/docs/A股微调训练计划.md). Local-only: empty hf ids
    # so a missing dir fails loudly instead of silently falling back to the
    # original pretrained weights under this name.
    "kronos-as-base": {
        "display_name": "Kronos-base·A股微调v1",
        "model_dirname": "Kronos-AS-v1",
        "tokenizer_dirname": "Kronos-Tokenizer-AS-v1",
        "hf_model_id": "",
        "hf_tokenizer_id": "",
        "max_context": 512,
        "params": "102.3M",
    },
}


def _is_local_model_dir(path: Path) -> bool:
    return path.is_dir() and (path / "config.json").is_file()


@dataclass
class LoadedModel:
    key: str
    predictor: Any
    device: str
    max_context: int
    loaded_at: float
    load_ms: int
    active: int = 0
    total_inferences: int = 0


class ModelManager:
    def __init__(self, models_dir: Path, device: str = "auto"):
        self.models_dir = Path(models_dir)
        self._device_setting = device
        self._lock = threading.RLock()
        self._loaded: dict[str, LoadedModel] = {}
        self._loading: set[str] = set()
        self._key_locks: dict[str, threading.Lock] = {}

    # ------------------------------------------------------------------ info
    def registry_keys(self) -> list[str]:
        return list(REGISTRY.keys())

    def resolve_device(self) -> str:
        import torch

        if self._device_setting != "auto":
            return self._device_setting
        if torch.cuda.is_available():
            return "cuda:0"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _source_for(self, dirname: str, hf_id: str) -> str | Path:
        local = self.models_dir / dirname
        if _is_local_model_dir(local):
            return local
        return hf_id

    def availability(self, key: str) -> dict[str, Any]:
        meta = REGISTRY[key]
        model_local = _is_local_model_dir(self.models_dir / meta["model_dirname"])
        tok_local = _is_local_model_dir(self.models_dir / meta["tokenizer_dirname"])
        if model_local and tok_local:
            location = "local"
        elif model_local or tok_local:
            location = "partial-local"
        else:
            location = "hub"
        return {"location": location}

    # ----------------------------------------------------------------- load
    def load(self, key: str) -> LoadedModel:
        if key not in REGISTRY:
            raise KeyError(f"未知模型 {key}，可选: {list(REGISTRY)}")
        key_lock = self._key_locks.setdefault(key, threading.Lock())
        with key_lock:  # concurrent callers of the same model wait here
            with self._lock:
                if key in self._loaded:
                    return self._loaded[key]
                self._loading.add(key)

            t0 = time.perf_counter()
            try:
                import torch  # noqa: F401  (ensures CUDA context availability)
                from model import Kronos, KronosPredictor, KronosTokenizer

                meta = REGISTRY[key]
                device = self.resolve_device()
                tokenizer_src = self._source_for(meta["tokenizer_dirname"], meta["hf_tokenizer_id"])
                model_src = self._source_for(meta["model_dirname"], meta["hf_model_id"])
                logger.info("加载模型 %s: model=%s tokenizer=%s device=%s", key, model_src, tokenizer_src, device)

                tokenizer = KronosTokenizer.from_pretrained(str(tokenizer_src))
                model = Kronos.from_pretrained(str(model_src))
                predictor = KronosPredictor(
                    model, tokenizer, device=device, max_context=meta["max_context"]
                )
                loaded = LoadedModel(
                    key=key,
                    predictor=predictor,
                    device=device,
                    max_context=meta["max_context"],
                    loaded_at=time.time(),
                    load_ms=int((time.perf_counter() - t0) * 1000),
                )
                with self._lock:
                    self._loaded[key] = loaded
                logger.info("模型 %s 加载完成 (%d ms)", key, loaded.load_ms)
                return loaded
            except Exception:
                logger.exception("模型 %s 加载失败", key)
                raise
            finally:
                with self._lock:
                    self._loading.discard(key)

    def unload(self, key: str, timeout: float = 120.0) -> None:
        deadline = time.monotonic() + timeout
        with self._lock:
            loaded = self._loaded.get(key)
            if loaded is None:
                return
        while loaded.active > 0 and time.monotonic() < deadline:
            time.sleep(0.2)
        with self._lock:
            self._loaded.pop(key, None)
        logger.info("模型 %s 已卸载（等待在跑推理 %s）", key, "完成" if loaded.active == 0 else "超时强制")

    # ---------------------------------------------------------------- acquire
    @contextmanager
    def acquire(self, key: str):
        loaded = self.load(key)  # idempotent; loads on first use
        with self._lock:
            loaded.active += 1
        try:
            yield loaded
        finally:
            with self._lock:
                loaded.active -= 1
                loaded.total_inferences += 1

    def is_loaded(self, key: str) -> bool:
        with self._lock:
            return key in self._loaded

    def any_loaded(self) -> bool:
        with self._lock:
            return bool(self._loaded)

    # ---------------------------------------------------------------- status
    def status(self) -> list[dict[str, Any]]:
        out = []
        for key, meta in REGISTRY.items():
            with self._lock:
                loaded = self._loaded.get(key)
                loading = key in self._loading
            entry = {
                "key": key,
                "display_name": meta["display_name"],
                "max_context": meta["max_context"],
                "params": meta["params"],
                "loaded": loaded is not None,
                "loading": loading,
                **self.availability(key),
            }
            if loaded is not None:
                entry.update(
                    {
                        "device": loaded.device,
                        "loaded_at": loaded.loaded_at,
                        "load_ms": loaded.load_ms,
                        "active_inferences": loaded.active,
                        "total_inferences": loaded.total_inferences,
                    }
                )
            out.append(entry)
        return out

    @staticmethod
    def gpu_info() -> dict[str, Any]:
        try:
            import torch

            if not torch.cuda.is_available():
                return {"available": False}
            return {
                "available": True,
                "device_name": torch.cuda.get_device_name(0),
                "total_mem_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
                "used_mem_gb": round(torch.cuda.memory_allocated(0) / 1024**3, 3),
                "reserved_mem_gb": round(torch.cuda.memory_reserved(0) / 1024**3, 3),
            }
        except Exception:
            return {"available": False}
