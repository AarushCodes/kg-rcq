# RCQ-MoE Prototype State

Current date: 2026-05-22

## Status

The project has a working PyTorch reference prototype for RCQ-MoE and an official
Transformers Qwen3.5-MoE tiny-model integration path.

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
python3 -m pytest -q
```

Latest result:

```text
35 passed
```

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

## Useful Scripts

```bash
python3 scripts/official_qwen_toy_kl.py
python3 scripts/official_qwen_toy_ablation.py
python3 scripts/official_qwen_toy_artifact.py --clean
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

## Latest Commits

```text
e872141 Add streamed text fixture builder
d9e481b Add official Qwen RCQ artifact roundtrip
123b749 Add official Qwen RCQ harness diagnostics
5748bf7 Add official Qwen toy KL diagnostic
548efa9 Add official Qwen MoE RCQ adapter
30f5246 Guard affine correction with identity fallback
```

## Current Limitations

- No real pretrained Qwen/MoE checkpoint has been quantized.
- No real tokenizer-driven model quality evaluation yet.
- FineWeb-Edu text fixture exists locally but is not yet wired into the model
  harness as token batches.
- Current artifact stores fake-dequant reference tensors, not packed bitstreams.
- No FP8 scale/shared-factor storage.
- No fused kernels or performance benchmarks.
- No downstream task evaluation.
- No real PPL evaluation.
- No grouped subspaces ablation.

## Recommended Next Milestone

Build the real text-token harness:

1. Load text fixture files.
2. Tokenize with either:
   - a real tokenizer when available, or
   - a deterministic tiny byte/char tokenizer for official tiny random models.
3. Produce fixed-shape calibration/eval `input_ids`.
4. Run official Qwen RCQ harness on text-derived tokens.
5. Save and reload RCQ artifact.
6. Report toy text KL and diagnostics, clearly labeled as plumbing metrics.

This should still run on the tiny official Qwen3.5-MoE model before attempting
any real pretrained checkpoint.

