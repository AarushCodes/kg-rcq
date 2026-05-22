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
from rcq_moe.metrics import kl_divergence_summary
from rcq_moe.official_qwen import convert_official_qwen35_moe_to_rcq_with_diagnostics, make_tiny_official_qwen35_moe
from rcq_moe.quantization import RescueConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Toy official Qwen3.5-MoE RCQ artifact roundtrip.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/toy_official_qwen_rcq_artifact"))
    parser.add_argument("--seed", type=int, default=456)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    if args.clean and args.output_dir.exists():
        shutil.rmtree(args.output_dir)

    torch.manual_seed(args.seed)
    model = make_tiny_official_qwen35_moe(vocab_size=48, hidden_size=8, moe_intermediate_size=8, shared_expert_intermediate_size=8, num_hidden_layers=1)
    calibration_ids = torch.randint(0, model.config.vocab_size, (4, 5))
    eval_ids = torch.randint(0, model.config.vocab_size, (2, 4))
    conversion = convert_official_qwen35_moe_to_rcq_with_diagnostics(
        model,
        calibration_ids,
        RescueConfig.rcq_1p75(block_size=8),
        rank=4,
        fit_correction=True,
    )
    save_official_qwen_rcq_artifact(conversion.model, args.output_dir, diagnostics=conversion.layer_diagnostics)
    loaded = load_official_qwen_rcq_artifact(args.output_dir)

    with torch.no_grad():
        converted_logits = conversion.model(input_ids=eval_ids, use_cache=False).logits
        loaded_logits = loaded(input_ids=eval_ids, use_cache=False).logits
    max_abs = torch.max(torch.abs(converted_logits - loaded_logits)).item()
    kl = kl_divergence_summary(converted_logits, loaded_logits)

    print("TOY OFFICIAL QWEN3.5-MOE RCQ ARTIFACT ROUNDTRIP")
    print("Scope: random tiny official Transformers model and fake-dequant RCQ tensors.")
    print("Meaning: checkpoint plumbing only, not real model quality.")
    print(f"artifact_dir={args.output_dir}")
    print(f"max_abs_logit_delta_converted_vs_loaded={max_abs:.8g}")
    print("kl_converted_vs_loaded=" + ", ".join(f"{key}={value:.8g}" for key, value in kl.items()))


if __name__ == "__main__":
    main()

