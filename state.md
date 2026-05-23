# RCQ-MoE Prototype State

Current date: 2026-05-23

## Status

The project has a working PyTorch reference prototype for RCQ-MoE, an official
Transformers Qwen3.5-MoE tiny-model integration path, and a text-derived toy
token harness with a real-tokenizer smoke path. It now also has a local
Kaggle script-kernel runner for a competition-attached Qwen3.6 FP-only smoke
slice.

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
  - Has a checked-in tiny BERT tokenizer fixture for no-network AutoTokenizer
    unit coverage.
  - Validated slice 2 with `AutoTokenizer.from_pretrained` for
    `Qwen/Qwen3.6-35B-A3B`, then reran the smoke with
    `--tokenizer-local-files-only`.
  - Threads optional calibration/eval attention masks through the official Qwen
    capture, conversion, and KL harness paths.
  - Runs the official tiny Qwen3.5-MoE RCQ harness on text-derived tokens.
  - Saves and reloads the RCQ artifact after the text-token smoke run.

- Kaggle remote execution scaffold:
  - Adds `scripts/qwen36_fp_smoke.py` for FP-only pretrained Qwen3.6 smoke
    runs on remote GPU hardware.
  - Adds `kaggle/qwen36_fp_smoke/kernel-metadata.json` for the private Kaggle
    script kernel `aarushkhilosia/rcq-qwen36-fp-smoke`.
  - Attaches the Kaggle competition
    `nvidia-nemotron-model-reasoning-challenge` and enables GPU/internet in
    metadata.
  - Adds `kaggle/qwen36_fp_smoke/run_qwen36_fp_smoke.py`, a thin runner that
    clones the local repo from `RCQ_REPO_URL`, optionally checks out
    `RCQ_COMMIT_SHA`, installs the package, verifies `nvidia-smi`, and runs the
    FP smoke script.
  - Keeps credentials out of source. `RCQ_GIT_TOKEN` is optional and redacted
    from printed clone commands; `HF_TOKEN` is read only from the environment.
  - Keeps Qwen3.6 model files on Kaggle under `/kaggle/working/hf_cache`, not
    on the Mac.

## Generated Local Data

Generated and intentionally ignored by git:

```text
data/text_fixtures/generated/fineweb_edu_256_calib.txt  347K
data/text_fixtures/generated/fineweb_edu_256_eval.txt    80K
outputs/qwen36_fp_smoke_dry_run/
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

Latest Qwen3.6 FP smoke dry-run command:

```bash
uv run python scripts/qwen36_fp_smoke.py \
  --dry-run \
  --output-dir outputs/qwen36_fp_smoke_dry_run
```

Latest Qwen3.6 FP smoke dry-run result:

```text
metadata.json, module_structure.txt, and fp_metrics.json written successfully.
No pretrained model weights were loaded locally.
No Kaggle command was run.
```

Latest full test command:

```bash
uv run pytest -q
```

Latest result:

```text
43 passed
```

Latest text-token smoke command:

```bash
uv run python scripts/official_qwen_text_smoke.py \
  --clean \
  --tokenizer-name-or-path Qwen/Qwen3.6-35B-A3B \
  --tokenizer-local-files-only \
  --output-dir outputs/toy_official_qwen_text_smoke_qwen36_tokenizer
```

Latest text-token smoke result:

```text
text_source=files:/Users/ck/Desktop/aarush/inference_research/impl/data/text_fixtures/generated/fineweb_edu_256_calib.txt,/Users/ck/Desktop/aarush/inference_research/impl/data/text_fixtures/generated/fineweb_edu_256_eval.txt
tokenization=hf_tokenizer:Qwen/Qwen3.6-35B-A3B
calib_docs=205 eval_docs=51
calibration_ids_shape=(16, 32) eval_ids_shape=(4, 32)
calibration_attention_tokens=512 eval_attention_tokens=128
toy_text_fp_vs_rcq_kl=mean=6.1713831e-07, p50=4.7719027e-07, p95=4.4415056e-06, p99=5.9478944e-06, max=9.4008465e-06
layer=0 moe_mse_before=9.1268459e-09 moe_mse_after=8.7941094e-09 bpw={'gate': 8.59375, 'up': 8.59375, 'down': 8.59375}
layer=1 moe_mse_before=1.0904087e-08 moe_mse_after=1.085529e-08 bpw={'gate': 8.59375, 'up': 8.59375, 'down': 8.59375}
artifact_dir=outputs/toy_official_qwen_text_smoke_qwen36_tokenizer
max_abs_logit_delta_converted_vs_loaded=0
kl_converted_vs_loaded=mean=0, p50=0, p95=0, p99=0, max=0
```

Interpretation: this is a tiny random-model plumbing metric only. It verifies
text loading, Qwen tokenizer loading/cached local reuse, attention-mask plumbing,
official Qwen RCQ conversion, diagnostics, artifact save, and artifact reload.
It is not evidence of pretrained model quality.

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
python3 scripts/qwen36_fp_smoke.py --dry-run --output-dir outputs/qwen36_fp_smoke_dry_run
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
- No pretrained-model quality evaluation yet; slice 2 uses the real
  `Qwen/Qwen3.6-35B-A3B` tokenizer with a tiny random official Qwen-shaped model.
- The competition-attached Kaggle Qwen3.6 FP smoke kernel has been created
  locally but has not yet been pushed or run.
- Current artifact stores fake-dequant reference tensors, not packed bitstreams.
- No FP8 scale/shared-factor storage.
- No fused kernels or performance benchmarks.
- No downstream task evaluation.
- No real PPL evaluation.
- No grouped subspaces ablation.

## Recommended Next Milestone

Continue the pretrained-compatible smoke path slice by slice:

1. Slice 3: competition-attached Kaggle Qwen3.6 FP-only smoke.
   - Push `kaggle/qwen36_fp_smoke` only after explicit permission.
   - Run the private script kernel attached to
     `nvidia-nemotron-model-reasoning-challenge`.
   - Load `Qwen/Qwen3.6-35B-A3B` on Kaggle, not locally.
   - Verify RTX PRO 6000 allocation with `nvidia-smi`.
   - Run tokenizer-driven FP-only eval and inspect/capture sparse MoE block
     structure under `/kaggle/working/outputs/qwen36_fp_smoke`.
   - Do not quantize in this slice.
2. Slice 4: one-layer pretrained RCQ conversion.
   - Add layer-limited conversion if needed.
   - Quantize exactly one Qwen3.6 MoE layer/block first on Kaggle.
   - Report FP-vs-one-layer-RCQ KL, routed MoE MSE before/after correction,
     expert bpw, artifact save/load exactness, and memory/runtime notes.
