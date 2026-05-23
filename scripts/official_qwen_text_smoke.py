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
    TextTokenBatch,
    read_text_fixture,
    split_calib_eval,
    synthetic_fixture_documents,
    texts_to_hf_token_batch,
    texts_to_toy_token_batch,
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


def _tokenizer_vocab_size(tokenizer) -> int:
    try:
        return len(tokenizer)
    except TypeError:
        return int(tokenizer.vocab_size)


def _load_tokenizer(name_or_path: str, *, local_files_only: bool):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(name_or_path, local_files_only=local_files_only)


def _make_text_batch(
    docs: list[str],
    *,
    tokenizer,
    vocab_size: int,
    config: TextBatchConfig,
    add_special_tokens: bool,
) -> TextTokenBatch:
    if tokenizer is None:
        return texts_to_toy_token_batch(docs, vocab_size=vocab_size, config=config)
    return texts_to_hf_token_batch(
        docs,
        tokenizer=tokenizer,
        config=config,
        add_special_tokens=add_special_tokens,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Tiny official Qwen3.5-MoE RCQ text-token smoke run.")
    parser.add_argument("--calib-text-file", type=Path)
    parser.add_argument("--eval-text-file", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/toy_official_qwen_text_smoke")
    parser.add_argument("--seed", type=int, default=789)
    parser.add_argument("--vocab-size", type=int)
    parser.add_argument("--tokenizer-name-or-path")
    parser.add_argument("--tokenizer-local-files-only", action="store_true")
    parser.add_argument("--tokenizer-add-special-tokens", action="store_true")
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
    tokenizer = None
    tokenization = "toy_byte"
    if args.tokenizer_name_or_path:
        tokenizer = _load_tokenizer(args.tokenizer_name_or_path, local_files_only=args.tokenizer_local_files_only)
        tokenization = f"hf_tokenizer:{args.tokenizer_name_or_path}"
    vocab_size = args.vocab_size or (_tokenizer_vocab_size(tokenizer) if tokenizer is not None else 128)

    model = make_tiny_official_qwen35_moe(
        vocab_size=vocab_size,
        hidden_size=args.hidden_size,
        moe_intermediate_size=args.moe_intermediate_size,
        shared_expert_intermediate_size=args.moe_intermediate_size,
        num_hidden_layers=args.num_hidden_layers,
    )
    calibration = _make_text_batch(
        calib_docs,
        tokenizer=tokenizer,
        vocab_size=model.config.vocab_size,
        config=TextBatchConfig(
            batch_size=args.calib_batch_size,
            sequence_length=args.sequence_length,
            max_batches=args.calib_batches,
            repeat_if_needed=True,
        ),
        add_special_tokens=args.tokenizer_add_special_tokens,
    )
    eval_batch = _make_text_batch(
        eval_docs,
        tokenizer=tokenizer,
        vocab_size=model.config.vocab_size,
        config=TextBatchConfig(
            batch_size=args.eval_batch_size,
            sequence_length=args.sequence_length,
            max_batches=args.eval_batches,
            repeat_if_needed=True,
        ),
        add_special_tokens=args.tokenizer_add_special_tokens,
    )

    result = run_official_qwen_harness(
        model,
        calibration.input_ids,
        eval_batch.input_ids,
        RescueConfig.rcq_1p75(block_size=args.block_size),
        calibration_attention_mask=calibration.attention_mask,
        eval_attention_mask=eval_batch.attention_mask,
        rank=args.rank,
        fit_correction=True,
    )
    save_official_qwen_rcq_artifact(result.model, args.output_dir, diagnostics=result.layer_diagnostics)
    loaded = load_official_qwen_rcq_artifact(args.output_dir)

    with torch.no_grad():
        converted_logits = result.model(
            input_ids=eval_batch.input_ids,
            attention_mask=eval_batch.attention_mask,
            use_cache=False,
        ).logits
        loaded_logits = loaded(
            input_ids=eval_batch.input_ids,
            attention_mask=eval_batch.attention_mask,
            use_cache=False,
        ).logits
    max_abs = torch.max(torch.abs(converted_logits - loaded_logits)).item()
    roundtrip_kl = kl_divergence_summary(converted_logits, loaded_logits)

    print("TOY TEXT OFFICIAL QWEN3.5-MOE RCQ SMOKE")
    print("Scope: text-derived tokens, random tiny official Transformers model, fake-dequant RCQ tensors.")
    print("Meaning: pipeline/plumbing metric only; not pretrained model quality.")
    print(f"text_source={source}")
    print(f"tokenization={tokenization}")
    print(f"calib_docs={len(calib_docs)} eval_docs={len(eval_docs)}")
    print(f"calibration_ids_shape={tuple(calibration.input_ids.shape)} eval_ids_shape={tuple(eval_batch.input_ids.shape)}")
    print(
        f"calibration_attention_tokens={int(calibration.attention_mask.sum().item())} "
        f"eval_attention_tokens={int(eval_batch.attention_mask.sum().item())}"
    )
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
