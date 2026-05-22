from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rcq_moe.harness import run_official_qwen_ablation, run_official_qwen_harness
from rcq_moe.official_qwen import make_tiny_official_qwen35_moe
from rcq_moe.quantization import RescueConfig


def _fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.8g}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Toy official Qwen3.5-MoE RCQ harness and ablation table.")
    parser.add_argument("--seed", type=int, default=321)
    parser.add_argument("--vocab-size", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--moe-intermediate-size", type=int, default=16)
    parser.add_argument("--num-hidden-layers", type=int, default=2)
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--calib-batch", type=int, default=4)
    parser.add_argument("--calib-seq", type=int, default=6)
    parser.add_argument("--eval-batch", type=int, default=3)
    parser.add_argument("--eval-seq", type=int, default=5)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--ranks", type=int, nargs="+", default=[2, 4, 8])
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    model = make_tiny_official_qwen35_moe(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        moe_intermediate_size=args.moe_intermediate_size,
        shared_expert_intermediate_size=args.moe_intermediate_size,
        num_experts=args.num_experts,
        num_experts_per_tok=args.top_k,
        num_hidden_layers=args.num_hidden_layers,
    )
    calibration_ids = torch.randint(0, model.config.vocab_size, (args.calib_batch, args.calib_seq))
    eval_ids = torch.randint(0, model.config.vocab_size, (args.eval_batch, args.eval_seq))
    rcq_config = RescueConfig.rcq_1p75(block_size=args.block_size)

    print("TOY OFFICIAL QWEN3.5-MOE RCQ HARNESS")
    print("Scope: random tiny official Transformers model and random token IDs.")
    print("Meaning: diagnostics/ablation plumbing only, not real model quality.")
    print(f"seed={args.seed}")
    print(f"calibration_tokens={args.calib_batch * args.calib_seq}, eval_tokens={args.eval_batch * args.eval_seq}")
    print()

    harness = run_official_qwen_harness(
        model,
        calibration_ids,
        eval_ids,
        rcq_config,
        rank=min(args.ranks),
        fit_correction=True,
    )
    print("Layer diagnostics for first harness config:")
    for layer in harness.layer_diagnostics:
        print(
            f"layer={layer.layer_id} "
            f"moe_mse_before={_fmt(layer.moe_mse_before_correction)} "
            f"moe_mse_after={_fmt(layer.moe_mse_after_correction)}"
        )
        for name, diag in layer.linear_diagnostics.items():
            widths = ",".join(f"{bit}b={pct:.2%}" for bit, pct in diag.width_percentages.items())
            print(f"  {name}: captured_energy={diag.captured_energy:.6g} bpw={diag.bpw:.6g} {widths}")
    print()

    print("Ablation table:")
    print("name rank correction kl_mean kl_p95 kl_p99 max_moe_mse_before max_moe_mse_after")
    for result in run_official_qwen_ablation(model, calibration_ids, eval_ids, rcq_config, ranks=args.ranks):
        print(
            f"{result.name} {result.rank} {result.correction} "
            f"{result.kl_mean:.8g} {result.kl_p95:.8g} {result.kl_p99:.8g} "
            f"{_fmt(result.max_moe_mse_before)} {_fmt(result.max_moe_mse_after)}"
        )


if __name__ == "__main__":
    main()

