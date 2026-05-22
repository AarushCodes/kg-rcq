from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rcq_moe.metrics import kl_divergence_summary
from rcq_moe.official_qwen import convert_official_qwen35_moe_to_rcq, make_tiny_official_qwen35_moe
from rcq_moe.quantization import RescueConfig


def _format_summary(summary: dict[str, float]) -> str:
    return ", ".join(f"{key}={value:.8g}" for key, value in summary.items())


def main() -> None:
    parser = argparse.ArgumentParser(description="Toy official Qwen3.5-MoE FP-vs-RCQ KL diagnostic.")
    parser.add_argument("--seed", type=int, default=123)
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
    parser.add_argument("--low-rank", type=int, default=4)
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
    model.eval()
    calibration_ids = torch.randint(0, model.config.vocab_size, (args.calib_batch, args.calib_seq))
    eval_ids = torch.randint(0, model.config.vocab_size, (args.eval_batch, args.eval_seq))
    rcq_config = RescueConfig.rcq_1p75(block_size=args.block_size)

    with torch.no_grad():
        fp_logits = model(input_ids=eval_ids, use_cache=False).logits

    full_rank = model.config.hidden_size
    q_full = convert_official_qwen35_moe_to_rcq(
        model,
        calibration_ids,
        rcq_config,
        rank=full_rank,
        fit_correction=True,
    )
    q_low = convert_official_qwen35_moe_to_rcq(
        model,
        calibration_ids,
        rcq_config,
        rank=args.low_rank,
        fit_correction=True,
    )

    with torch.no_grad():
        full_logits = q_full(input_ids=eval_ids, use_cache=False).logits
        low_logits = q_low(input_ids=eval_ids, use_cache=False).logits

    full_summary = kl_divergence_summary(fp_logits, full_logits)
    low_summary = kl_divergence_summary(fp_logits, low_logits)

    print("TOY OFFICIAL QWEN3.5-MOE FP-vs-RCQ KL DIAGNOSTIC")
    print("Scope: random tiny official Transformers model and random token IDs.")
    print("Meaning: integration/plumbing smoke metric only, not real model quality.")
    print(f"seed={args.seed}")
    print(f"shape: vocab={args.vocab_size}, hidden={args.hidden_size}, moe_intermediate={args.moe_intermediate_size}, layers={args.num_hidden_layers}, experts={args.num_experts}, top_k={args.top_k}")
    print(f"calibration_tokens={args.calib_batch * args.calib_seq}, eval_tokens={args.eval_batch * args.eval_seq}")
    print(f"full_rank_debug_kl rank={full_rank}: {_format_summary(full_summary)}")
    print(f"low_rank_smoke_kl rank={args.low_rank}: {_format_summary(low_summary)}")


if __name__ == "__main__":
    main()
