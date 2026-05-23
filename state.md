# RCQ-MoE Prototype State

Current date: 2026-05-23

## Status

The project has a working PyTorch reference prototype for RCQ-MoE, an official
Transformers Qwen3.5-MoE tiny-model integration path, and a text-derived toy
token harness.

This is still a research/plumbing prototype:

- Uses fake-dequant tensors, not packed/fused kernels.
- Uses tiny random official Qwen3.5-MoE models for tests.
- Has no real pretrained checkpoint quality result yet.
- Has no production storage format beyond the current reference artifact.

## Implemented

- Core RCQ math:
  - Router-weighted covariance/statistics.
  - KLT/eigendecomposition shared subspace.
  - Shared low-rank decomposition.
  - Block signed Hadamard rotations.
  - 1-bit weighted sign residual quantization.
  - 2/4-bit Lloyd-Max rescue quantization.
  - Effective expert bpw accounting.

- Correction:
  - Routed MoE-output affine correction.
  - Per-channel identity fallback if correction does not improve calibration MSE.

- Tiny Qwen-shaped model:
  - Packed `gate_up_proj` and separate `down_proj`.
  - Normalized top-k router.
  - Shared expert path.
  - Conversion to RCQ-backed experts.

- Official Transformers integration:
  - Uses `transformers.Qwen3_5MoeForCausalLM`.
  - Builds tiny official Qwen3.5-MoE random models.
  - Captures official MoE inputs via forward hooks.
  - Deep-copies model and replaces official sparse MoE blocks with RCQ blocks.
  - Keeps router, shared expert, attention, embeddings, norms, and LM head full precision.

- Diagnostics/harness:
  - Whole-model logits KL summary.
  - Per-layer MoE MSE before/after correction.
  - Captured shared energy.
  - 1/2/4-bit block percentages.
  - Effective bpw per expert linear.
  - Toy ablation runner.

- Artifact roundtrip:
  - Saves `metadata.json`, `non_expert_state.pt`, and `rcq_state.pt`.
  - Reloads RCQ official Qwen model without needing original FP expert tensors.
  - Converted-vs-loaded logits match exactly in toy tests.

- Text fixture pipeline:
  - Offline synthetic fixture source for tests.
  - Hugging Face streaming fixture builder.
  - Generated a local FineWeb-Edu 256-document fixture under ignored data.
  - Reads local text fixture files back into documents.
  - Converts text into deterministic toy byte-token `input_ids`.
  - Has a tokenizer-backed `TextTokenBatch` path with `input_ids` and
    `attention_mask`.
  - Supports optional Hugging Face tokenizer encoding in
    `scripts/official_qwen_text_smoke.py` via `--tokenizer-name-or-path`.
  - Threads optional calibration/eval attention masks through the official Qwen
    capture, conversion, and KL harness paths.
  - Runs the official tiny Qwen3.5-MoE RCQ harness on text-derived tokens.
  - Saves and reloads the RCQ artifact after the text-token smoke run.

## Generated Local Data

Generated and intentionally ignored by git:

```text
data/text_fixtures/generated/fineweb_edu_256_calib.txt  347K
data/text_fixtures/generated/fineweb_edu_256_eval.txt    80K
```

Generation command:

```bash
python3 scripts/build_text_fixture.py \
  --source hf \
  --dataset HuggingFaceFW/fineweb-edu \
  --split train \
  --max-docs 256 \
  --max-chars-per-doc 2000 \
  --output-dir data/text_fixtures/generated \
  --prefix fineweb_edu_256
```

Generation summary:

```text
docs_total=256 docs_calib=205 docs_eval=51
```

## Validation

Latest full test command:

```bash
uv run pytest -q
```

Latest result:

```text
41 passed
```

Latest text-token smoke command:

```bash
uv run python scripts/official_qwen_text_smoke.py --clean
```

Latest text-token smoke result:

```text
text_source=files:/Users/ck/Desktop/aarush/inference_research/impl/data/text_fixtures/generated/fineweb_edu_256_calib.txt,/Users/ck/Desktop/aarush/inference_research/impl/data/text_fixtures/generated/fineweb_edu_256_eval.txt
tokenization=toy_byte
calib_docs=205 eval_docs=51
calibration_ids_shape=(16, 32) eval_ids_shape=(4, 32)
calibration_attention_tokens=512 eval_attention_tokens=128
toy_text_fp_vs_rcq_kl mean=1.2730507e-07 p95=4.6927602e-07 max=7.6539436e-07
max_abs_logit_delta_converted_vs_loaded=0
kl_converted_vs_loaded mean=0 max=0
```

Interpretation: this is a tiny random-model plumbing metric only. It verifies
text loading, deterministic toy tokenization, attention-mask plumbing, official
Qwen RCQ conversion, diagnostics, artifact save, and artifact reload. It is not
evidence of pretrained model quality.

Known warnings:

- Two SWIG deprecation warnings from optional dependencies during official
  Transformers model import. These do not currently affect tests.

## Environment

Important package versions:

```text
torch 2.12.0
torchvision 0.27.0
transformers 5.9.0
accelerate 1.13.0
safetensors 0.7.0
datasets 4.8.5
tokenizers 0.22.2
huggingface_hub 1.16.1
scikit-learn 1.8.0
numba 0.65.1
librosa 0.11.0
```

Important constraint:

```text
transformers 5.9.0 requires tokenizers >=0.22.0, <=0.23.0
```

Do not blindly upgrade `tokenizers` beyond `0.23.0` unless `transformers` is
also upgraded to a compatible version.

`uv.lock` is tracked for environment reproducibility.

## Useful Scripts

```bash
python3 scripts/official_qwen_toy_kl.py
python3 scripts/official_qwen_toy_ablation.py
python3 scripts/official_qwen_toy_artifact.py --clean
python3 scripts/official_qwen_text_smoke.py --clean
python3 scripts/build_text_fixture.py --source synthetic --prefix synthetic_toy --output-dir outputs/text_fixture_smoke
```

HF streaming fixture:

```bash
python3 scripts/build_text_fixture.py \
  --source hf \
  --dataset HuggingFaceFW/fineweb-edu \
  --split train \
  --max-docs 256 \
  --max-chars-per-doc 2000 \
  --output-dir data/text_fixtures/generated \
  --prefix fineweb_edu_256
```

## Recent Commits Before This State Update

```text
3f3b713 Add tokenizer-backed text batches
54eb9cb Document current RCQ prototype state
e872141 Add streamed text fixture builder
d9e481b Add official Qwen RCQ artifact roundtrip
123b749 Add official Qwen RCQ harness diagnostics
5748bf7 Add official Qwen toy KL diagnostic
548efa9 Add official Qwen MoE RCQ adapter
```

## Current Limitations

- No real pretrained Qwen/MoE checkpoint has been quantized.
- No pretrained-model tokenizer-driven quality evaluation yet.
- The official text smoke now has an optional Hugging Face tokenizer path, but
  the latest validated smoke still uses deterministic `toy_byte` tokenization
  on a tiny random model, not a real Qwen tokenizer.
- Current artifact stores fake-dequant reference tensors, not packed bitstreams.
- No FP8 scale/shared-factor storage.
- No fused kernels or performance benchmarks.
- No downstream task evaluation.
- No real PPL evaluation.
- No grouped subspaces ablation.

## Recommended Next Milestone

Continue the pretrained-compatible smoke path slice by slice:

1. Slice 2: real-tokenizer tiny-random smoke.
   - Run the new tokenizer path with an actual local Hugging Face tokenizer, or
     add a tiny checked-in tokenizer fixture if no local tokenizer is available.
   - Keep `local_files_only=True` for deterministic no-network validation.
   - Verify tokenizer-derived `input_ids`, `attention_mask`, finite FP-vs-RCQ
     KL, and exact converted-vs-loaded artifact logits.
   - Keep the current toy byte-token harness as the default no-network
     regression test.
2. Slice 3: pretrained checkpoint FP-only smoke.
   - Load the smallest available real MoE checkpoint or local Qwen3.5-MoE-
     compatible checkpoint that fits the MacBook Air M2 memory budget.
   - Run tokenizer-driven FP-only eval and inspect/capture sparse MoE block
     structure.
   - Do not quantize in this slice.
3. Slice 4: one-layer pretrained RCQ conversion.
   - Add layer-limited conversion if needed.
   - Quantize exactly one pretrained MoE layer/block first.
   - Report FP-vs-one-layer-RCQ KL, routed MoE MSE before/after correction,
     expert bpw, artifact save/load exactness, and memory/runtime notes.
