#!/usr/bin/env python3
"""FP-only Qwen3.6 MoE smoke runner for remote GPU environments.

This script is intentionally forward-only. It verifies tokenizer/model loading,
records the sparse-MoE module shape, runs a tiny next-token loss pass, and writes
small JSON/text outputs. It must not quantize or save model weights.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rcq_moe.text_data import TextBatchConfig, synthetic_fixture_documents, texts_to_hf_token_batch


DEFAULT_MODEL_ID = "Qwen/Qwen3.6-35B-A3B"


@dataclass
class ModuleSummary:
    name: str
    class_name: str
    child_names: list[str]
    parameter_count: int


def _run_command(argv: list[str]) -> str:
    try:
        result = subprocess.run(argv, check=False, capture_output=True, text=True)
    except FileNotFoundError:
        return f"{argv[0]} not found"
    output = (result.stdout + result.stderr).strip()
    return output if output else f"{argv[0]} exited with code {result.returncode}"


def _torch_dtype(name: str) -> torch.dtype:
    mapping = {
        "auto": "auto",
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if name not in mapping:
        raise ValueError(f"unsupported dtype {name!r}")
    return mapping[name]  # type: ignore[return-value]


def _load_texts(calib_text_file: Path | None, eval_text_file: Path | None) -> tuple[list[str], list[str], str]:
    if calib_text_file is not None and eval_text_file is not None:
        calib = calib_text_file.read_text(encoding="utf-8").split("\n\n")
        eval_docs = eval_text_file.read_text(encoding="utf-8").split("\n\n")
        return calib, eval_docs, f"files:{calib_text_file},{eval_text_file}"
    docs = synthetic_fixture_documents()
    split = max(1, int(0.8 * len(docs)))
    return docs[:split], docs[split:] or docs[:1], "synthetic_fixture_documents"


def _summarize_moe_modules(model: torch.nn.Module, *, max_modules: int) -> list[ModuleSummary]:
    summaries: list[ModuleSummary] = []
    markers = ("moe", "expert", "router", "gate")
    for name, module in model.named_modules():
        class_name = module.__class__.__name__
        haystack = f"{name.lower()} {class_name.lower()}"
        if not any(marker in haystack for marker in markers):
            continue
        children = list(module._modules.keys())
        param_count = sum(p.numel() for p in module.parameters(recurse=False))
        summaries.append(
            ModuleSummary(
                name=name,
                class_name=class_name,
                child_names=children,
                parameter_count=int(param_count),
            )
        )
        if len(summaries) >= max_modules:
            break
    return summaries


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _gpu_snapshot() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"cuda_available": False, "device_count": 0}
    devices = []
    for idx in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(idx)
        devices.append(
            {
                "index": idx,
                "name": props.name,
                "total_memory_gb": props.total_memory / 1024**3,
                "major": props.major,
                "minor": props.minor,
            }
        )
    return {
        "cuda_available": True,
        "device_count": torch.cuda.device_count(),
        "devices": devices,
        "max_memory_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
        "max_memory_reserved_gb": torch.cuda.max_memory_reserved() / 1024**3,
    }


def _next_token_loss(logits: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> float:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask = attention_mask[:, 1:].contiguous().bool()
    losses = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)).float(),
        shift_labels.view(-1),
        reduction="none",
    ).view_as(shift_labels)
    denom = shift_mask.sum().clamp_min(1)
    return float((losses * shift_mask).sum().div(denom).item())


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3.6 FP-only Kaggle smoke run.")
    parser.add_argument("--model-id", default=os.environ.get("QWEN36_MODEL_ID", DEFAULT_MODEL_ID))
    parser.add_argument("--revision", default=os.environ.get("QWEN36_MODEL_REVISION"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/qwen36_fp_smoke"))
    parser.add_argument("--calib-text-file", type=Path)
    parser.add_argument("--eval-text-file", type=Path)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--eval-batches", type=int, default=1)
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--max-moe-modules", type=int, default=200)
    parser.add_argument("--dry-run", action="store_true", help="Write environment metadata without loading the model.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    env_payload: dict[str, Any] = {
        "argv": sys.argv,
        "cwd": str(Path.cwd()),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": _gpu_snapshot(),
        "nvidia_smi": _run_command(["nvidia-smi"]),
        "model_id": args.model_id,
        "revision": args.revision,
        "dry_run": args.dry_run,
        "hf_home": os.environ.get("HF_HOME"),
        "transformers_cache": os.environ.get("TRANSFORMERS_CACHE"),
    }
    _write_json(args.output_dir / "metadata.json", env_payload)

    if args.dry_run:
        (args.output_dir / "module_structure.txt").write_text("dry_run=true\n", encoding="utf-8")
        _write_json(args.output_dir / "fp_metrics.json", {"dry_run": True, "elapsed_sec": time.time() - started})
        return

    from transformers import AutoModelForCausalLM, AutoTokenizer

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        revision=args.revision,
        token=token,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    calib_docs, eval_docs, text_source = _load_texts(args.calib_text_file, args.eval_text_file)
    eval_batch = texts_to_hf_token_batch(
        eval_docs,
        tokenizer=tokenizer,
        config=TextBatchConfig(
            batch_size=args.eval_batch_size,
            sequence_length=args.sequence_length,
            max_batches=args.eval_batches,
        ),
    )

    load_started = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        revision=args.revision,
        token=token,
        torch_dtype=_torch_dtype(args.dtype),
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
        low_cpu_mem_usage=True,
    )
    model.eval()
    load_elapsed = time.time() - load_started

    module_summaries = _summarize_moe_modules(model, max_modules=args.max_moe_modules)
    module_lines = [
        f"{item.name}\t{item.class_name}\tparams_recurse_false={item.parameter_count}\tchildren={','.join(item.child_names)}"
        for item in module_summaries
    ]
    (args.output_dir / "module_structure.txt").write_text("\n".join(module_lines) + "\n", encoding="utf-8")

    device = next(model.parameters()).device
    input_ids = eval_batch.input_ids.to(device)
    attention_mask = eval_batch.attention_mask.to(device)

    forward_started = time.time()
    with torch.inference_mode():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        logits = outputs.logits
        loss = _next_token_loss(logits, input_ids, attention_mask)
    forward_elapsed = time.time() - forward_started

    metrics = {
        "model_id": args.model_id,
        "revision": args.revision,
        "model_class": model.__class__.__name__,
        "tokenizer_class": tokenizer.__class__.__name__,
        "text_source": text_source,
        "calib_docs": len(calib_docs),
        "eval_docs": len(eval_docs),
        "eval_input_shape": list(eval_batch.input_ids.shape),
        "eval_attention_tokens": int(eval_batch.attention_mask.sum().item()),
        "logits_shape": list(logits.shape),
        "next_token_loss": loss,
        "load_elapsed_sec": load_elapsed,
        "forward_elapsed_sec": forward_elapsed,
        "elapsed_sec": time.time() - started,
        "moe_module_count_reported": len(module_summaries),
        "cuda_after_forward": _gpu_snapshot(),
    }
    _write_json(args.output_dir / "fp_metrics.json", metrics)

    del model, logits, outputs
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
