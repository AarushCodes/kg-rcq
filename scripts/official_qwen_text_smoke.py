from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rcq_moe.artifact import load_official_qwen_rcq_artifact, save_official_qwen_rcq_artifact
from rcq_moe.harness import run_official_qwen_harness
from rcq_moe.metrics import kl_divergence_summary
from rcq_moe.official_qwen import make_tiny_official_qwen35_moe
from rcq_moe.quantization import RescueConfig
from rcq_moe.text_data import (
    TextBatchConfig,
    read_text_fixture,
    split_calib_eval,
    synthetic_fixture_documents,
    texts_to_input_ids,
)


DEFAULT_CALIB = ROOT / "data/text_fixtures/generated/fineweb_edu_256_calib.txt"
DEFAULT_EVAL = ROOT / "data/text_fixtures/generated/fineweb_edu_256_eval.txt"


def _load_docs(calib_path: Path | None, eval_path: Path | None) -> tuple[list[str], list[str], str]:
    if calib_path is not None or eval_path is not None:
        if calib_path is None or eval_path is None:
            raise ValueError("provide both --calib-text-file and --eval-text-file, or neither.")
        return read_text_fixture(calib_path), read_text_fixture(eval_path), f"files:{calib_path},{eval_path}"

    if DEFAULT_CALIB.exists() and DEFAULT_EVAL.exists():
        return read_text_fixture(DEFAULT_CALIB), read_text_fixture(DEFAULT_EVAL), f"files:{DEFAULT_CALIB},{DEFAULT_EVAL}"

    calib_docs, eval_docs = split_calib_eval(synthetic_fixture_documents(), 0.25)
    return calib_docs, eval_docs, "built_in_synthetic"


def _format_kl(prefix: str, values: dict[str, float]) -> str:
    return prefix + ", ".join(f"{key}={value:.8g}" for key, value in values.items())


def main() -> None:
    parser = argparse.ArgumentParser(description="Tiny official Qwen3.5-MoE RCQ text-token smoke run.")
    parser.add_argument("--calib-text-file", type=Path)
    parser.add_argument("--eval-text-file", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/toy_official_qwen_text_smoke")
    parser.add_argument("--seed", type=int, default=789)
    parser.add_argument("--vocab-size", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--moe-intermediate-size", type=int, default=16)
    parser.add_argument("--num-hidden-layers", type=int, default=2)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--calib-batch-size", type=int, default=4)
    parser.add_argument("--calib-batches", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--eval-batches", type=int, default=2)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    if args.clean and args.output_dir.exists():
        shutil.rmtree(args.output_dir)

    calib_docs, eval_docs, source = _load_docs(args.calib_text_file, args.eval_text_file)
    if not calib_docs or not eval_docs:
        raise ValueError("calibration and evaluation text must both contain at least one document.")

    torch.manual_seed(args.seed)
    model = make_tiny_official_qwen35_moe(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        moe_intermediate_size=args.moe_intermediate_size,
        shared_expert_intermediate_size=args.moe_intermediate_size,
        num_hidden_layers=args.num_hidden_layers,
    )
    calibration_ids = texts_to_input_ids(
        calib_docs,
        vocab_size=model.config.vocab_size,
        config=TextBatchConfig(
            batch_size=args.calib_batch_size,
            sequence_length=args.sequence_length,
            max_batches=args.calib_batches,
            repeat_if_needed=True,
        ),
    )
    eval_ids = texts_to_input_ids(
        eval_docs,
        vocab_size=model.config.vocab_size,
        config=TextBatchConfig(
            batch_size=args.eval_batch_size,
            sequence_length=args.sequence_length,
            max_batches=args.eval_batches,
            repeat_if_needed=True,
        ),
    )

    result = run_official_qwen_harness(
        model,
        calibration_ids,
        eval_ids,
        RescueConfig.rcq_1p75(block_size=args.block_size),
        rank=args.rank,
        fit_correction=True,
    )
    save_official_qwen_rcq_artifact(result.model, args.output_dir, diagnostics=result.layer_diagnostics)
    loaded = load_official_qwen_rcq_artifact(args.output_dir)

    with torch.no_grad():
        converted_logits = result.model(input_ids=eval_ids, use_cache=False).logits
        loaded_logits = loaded(input_ids=eval_ids, use_cache=False).logits
    max_abs = torch.max(torch.abs(converted_logits - loaded_logits)).item()
    roundtrip_kl = kl_divergence_summary(converted_logits, loaded_logits)

    print("TOY TEXT OFFICIAL QWEN3.5-MOE RCQ SMOKE")
    print("Scope: text-derived toy byte tokens, random tiny official Transformers model, fake-dequant RCQ tensors.")
    print("Meaning: pipeline/plumbing metric only; not pretrained model quality.")
    print(f"text_source={source}")
    print(f"calib_docs={len(calib_docs)} eval_docs={len(eval_docs)}")
    print(f"calibration_ids_shape={tuple(calibration_ids.shape)} eval_ids_shape={tuple(eval_ids.shape)}")
    print(_format_kl("toy_text_fp_vs_rcq_kl=", result.kl))
    for diag in result.layer_diagnostics:
        bpw = {name: linear.bpw for name, linear in diag.linear_diagnostics.items()}
        print(
            f"layer={diag.layer_id} "
            f"moe_mse_before={diag.moe_mse_before_correction:.8g} "
            f"moe_mse_after={diag.moe_mse_after_correction:.8g} "
            f"bpw={bpw}"
        )
    print(f"artifact_dir={args.output_dir}")
    print(f"max_abs_logit_delta_converted_vs_loaded={max_abs:.8g}")
    print(_format_kl("kl_converted_vs_loaded=", roundtrip_kl))


if __name__ == "__main__":
    main()
