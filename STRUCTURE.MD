# Repository Structure

This document covers the root project files plus files under `docs/`,
`rcq_moe/`, `scripts/`, `kaggle/`, and `tests/`.
It assumes the reader has already read `SUB2BIT_MOE_IMPLEMENTATION_SPEC.md`,
so terms like router-weighted covariance, KLT, shared subspace, Hadamard
rotation, rescue blocks, and routed MoE-output affine correction are used
directly.

Generated Python bytecode files under `__pycache__/` are listed as generated
artifacts, not source files to modify.

```text
.
├── .gitignore
├── LICENSE
├── README.MD
├── STRUCTURE.MD
├── SUB2BIT_MOE_IMPLEMENTATION_SPEC.md
├── configuration_qwen3_5_moe.py
├── modeling_qwen3_5_moe.py
├── modular_qwen3_5_moe.py
├── pyproject.toml
├── state.md
├── uv.lock
├── docs/
│   └── specs/
│       └── 2026-05-24-kaggle-remote-control-design.md
├── rcq_moe/
│   ├── __init__.py
│   ├── artifact.py
│   ├── codebooks.py
│   ├── correction.py
│   ├── decomposition.py
│   ├── hadamard.py
│   ├── harness.py
│   ├── metrics.py
│   ├── official_qwen.py
│   ├── quantization.py
│   ├── remote_jobs.py
│   ├── stats.py
│   ├── storage.py
│   ├── text_data.py
│   ├── tiny_qwen.py
│   └── toy_moe.py
├── scripts/
│   ├── build_text_fixture.py
│   ├── official_qwen_text_smoke.py
│   ├── official_qwen_toy_ablation.py
│   ├── official_qwen_toy_artifact.py
│   ├── official_qwen_toy_kl.py
│   └── qwen36_fp_smoke.py
├── kaggle/
│   ├── remote_worker/
│   │   ├── job_templates/
│   │   │   ├── 0001-control-plane-smoke.json
│   │   │   └── 0002-qwen36-fp-smoke.json
│   │   └── run_job.py
│   ├── qwen36_fp_smoke/
│   │   ├── kernel-metadata.json
│   │   └── run_qwen36_fp_smoke.py
│   └── qwen36_fp_smoke_notebook/
│       ├── kernel-metadata.json
│       └── rcq_qwen36_fp_smoke.ipynb
└── tests/
    ├── fixtures/
    │   └── tiny_bert_tokenizer/
    │       ├── special_tokens_map.json
    │       ├── tokenizer_config.json
    │       └── vocab.txt
    ├── test_correction_metrics.py
    ├── test_official_qwen_artifact.py
    ├── test_official_qwen_harness.py
    ├── test_official_qwen_rcq.py
    ├── test_official_qwen_text_harness.py
    ├── test_rcq_decomposition.py
    ├── test_rcq_math.py
    ├── test_remote_jobs.py
    ├── test_text_data.py
    ├── test_tiny_qwen_normal.py
    ├── test_tiny_qwen_rcq_conversion.py
    └── test_toy_pipeline.py
```

## Root Files

Top-level project documentation, metadata, and environment lock files.

### `.gitignore`

Ignore rules for generated local data, outputs, caches, and environment files.

### `README.MD`

Public project overview copied from `readme_draft.md`.

Contains:

- High-level RCQ-MoE motivation and research goal.
- The compression recipe: router-weighted statistics, shared KLT/SVD subspace,
  Hadamard-rotated residuals, weighted 1-bit quantization, mixed 2/4-bit rescue,
  and routed MoE-output correction.
- Related-work context for KBVQ-MoE, NVFP4-style numerical lessons, and
  TurboQuant-style rotations/codebooks.
- Current implementation status, latest local test count, tiny/random smoke
  interpretation, planned validation, compute needs, repository map, and local
  reproduction commands.

Keep this readable for external reviewers. Do not put credentials, private
Kaggle details, or unvalidated pretrained-quality claims here.

### `LICENSE`

Repository license added through GitHub.

### `STRUCTURE.MD`

This file. Keep it updated after successful slices that add, remove, or
meaningfully repurpose files.

### `SUB2BIT_MOE_IMPLEMENTATION_SPEC.md`

Primary research and implementation spec for the RCQ-MoE method.

### `configuration_qwen3_5_moe.py`

Local Qwen3.5-MoE configuration reference file used for architecture inspection
and compatibility work.

### `modeling_qwen3_5_moe.py`

Local Qwen3.5-MoE modeling reference file used while aligning the prototype with
official Transformers module structure.

### `modular_qwen3_5_moe.py`

Local modular Qwen3.5-MoE reference file used for architecture inspection.

### `pyproject.toml`

Project metadata, runtime dependencies, and pytest configuration.

### `state.md`

Current project state, validation log, known limitations, and recommended next
milestone.

### `uv.lock`

Tracked environment lockfile for reproducible local setup.

## `docs/`

Design and planning documents that are committed before implementation slices.
These are peer-review artifacts, not generated outputs.

### `docs/specs/2026-05-24-kaggle-remote-control-design.md`

Approved design for the next Kaggle remote-control slices.

Contains:

- A one-shot Kaggle worker architecture using a public GitHub repo plus a
  separate `kaggle-jobs` branch.
- JSON job-spec schema with required `code_commit` pinning.
- Initial named actions for `control_plane_smoke` and `qwen36_fp_smoke`.
- Reserved future actions for `debug_shell`, one-layer RCQ conversion, polling,
  and GitHub result pushing.
- Output layout, secret handling, error handling, test plan, and slice plan.
- Current architecture note that the public-GitHub clone path requires Kaggle
  internet, while the observed RTX PRO 6000 notebook has internet disabled and
  therefore needs a reviewed offline bootstrap path before the control-plane
  smoke.

Use this as the reference when reviewing or extending the remote worker.

## `rcq_moe/`

Core reference implementation for the RCQ-MoE prototype. The code is PyTorch
first and prioritizes correctness, diagnostics, and artifact round-tripping over
fused-kernel performance.

### `rcq_moe/__init__.py`

Public package surface. Re-exports the main codebook, correction,
decomposition, Hadamard, harness, metric, official-Qwen, quantization, storage,
text fixture, and toy-MoE APIs.

Modify this when adding a new public API that should be importable directly from
`rcq_moe`. Keep internal helpers out of `__all__`.

### `rcq_moe/artifact.py`

Save/load support for official Qwen3.5-MoE RCQ artifacts.

Contains:

- Metadata construction for `rcq_official_qwen35_moe_v1`.
- Splitting model state into `non_expert_state.pt` and `rcq_state.pt`.
- `save_official_qwen_rcq_artifact(...)`, which writes `metadata.json`,
  `non_expert_state.pt`, and `rcq_state.pt`.
- `load_official_qwen_rcq_artifact(...)`, which rebuilds a Transformers
  `Qwen3_5MoeForCausalLM`, loads non-expert weights, then replaces MoE blocks
  with RCQ blocks from serialized RCQ state.

This is the place to change if the checkpoint layout, artifact version, or
serialized RCQ tensor schema changes.

### `rcq_moe/codebooks.py`

Deterministic Lloyd-Max scalar codebook generation for standard-normal rescue
quantization.

Contains:

- Normal PDF/CDF helpers.
- `lloyd_max_codebook(bits, iterations=200, tol=1e-7, ...)`.
- The iterative centroid update used for 2-bit and 4-bit rescue blocks.

This file implements the spec's fixed rescue codebook generation. Do not replace
it with hardcoded approximate centroids unless tests and the spec are changed.

### `rcq_moe/correction.py`

Online per-channel affine regression for routed MoE-output correction.

Contains:

- `OnlineChannelRegression`, which accumulates count, sums, squared sums, and
  cross-products for `y` and `yhat`.
- `update(...)` for streaming calibration pairs.
- `solve(...)` for closed-form `alpha, beta`.
- The per-channel identity fallback: if fitted correction does not improve
  calibration MSE, the channel stays `alpha=1, beta=0`.

This is the correction stage described in spec Section 10 and clarification
Section 21.

### `rcq_moe/decomposition.py`

Router-aware shared-subspace decomposition.

Contains:

- `SharedDecomposition`, holding `a_factors`, `b_shared`, `residuals`, and
  `captured_energy`.
- `choose_rank(...)`, following the default rank rule with caps for tiny dims.
- `klt_transform(...)`, including eigenvalue flooring and inverse transform.
- `decompose_shared_subspace(...)`, which weights experts by importance, forms
  KLT-space stacked expert matrices, computes top right singular vectors, builds
  shared factors, and returns expert residuals.

This is the main implementation of spec Section 6. Bugs in shared/residual math
usually trace here first.

### `rcq_moe/hadamard.py`

Reference signed block-Hadamard rotation utilities.

Contains:

- Power-of-two validation for Hadamard block sizes.
- Stable deterministic seeding from `(model_name, layer_id, linear_type,
  block_index)`.
- `block_hadamard_matrix(...)`, normalized so the matrix is orthonormal.
- `deterministic_block_signs(...)` and `signed_hadamard_q(...)`.
- `pad_last_dim(...)`, `rotate_row_block(...)`, and
  `rotate_activation_block(...)`.

This file owns the spec Section 7.2 rotation convention. If dot products drift,
check `Q_b = D_b H_h` and activation rotation `Q_b.T a_b` here.

### `rcq_moe/harness.py`

High-level official-Qwen evaluation and ablation helpers.

Contains:

- `OfficialQwenHarnessResult`.
- `OfficialQwenAblationResult`.
- `run_official_qwen_harness(...)`, which converts a model to RCQ, runs FP and
  RCQ logits on eval IDs, and reports KL plus layer diagnostics.
- `run_official_qwen_ablation(...)`, which runs full-rank debug plus requested
  ranks with/without correction.

Use this for small end-to-end checks of conversion quality. It is not a full
production evaluation harness.

### `rcq_moe/metrics.py`

KL metric helper.

Contains:

- `kl_divergence_summary(fp_logits, q_logits)`, returning mean, p50, p95, p99,
  and max KL in nats/token.

This implements the spec's primary KL reporting shape for toy/smoke runs.

### `rcq_moe/official_qwen.py`

Integration layer for official Transformers Qwen3.5-MoE-style models.

Contains:

- `OfficialQwen35RCQLinearSet`, a low-rank shared factor plus dequantized
  residual reference linear path.
- Layer/linear diagnostic dataclasses for captured energy, bpw, rescue width
  percentages, and correction MSE.
- RCQ expert and sparse MoE block wrappers that preserve the router path while
  replacing expert linear projections.
- Forward-hook based collection of MoE block inputs.
- Per-block calibration stat collection for `gate`, `up`, and `down` expert
  linears.
- Conversion helpers that deep-copy the FP model, quantize each MoE block, fit
  optional routed MoE-output correction, and return diagnostics.
- `make_tiny_official_qwen35_moe(...)`, used by tests/scripts to create a small
  official Transformers model shape without needing a pretrained checkpoint.

This is the main bridge between the RCQ math and an actual model architecture.
When adapting to another official MoE implementation, this is the closest
template.

### `rcq_moe/quantization.py`

Residual quantization and mixed-bit rescue implementation.

Contains:

- `RescueConfig` plus exact `rcq_1p55`, `rcq_1p75`, and `rcq_1p90` presets.
- `QuantizedResidual`, including reference `dequantize()` reconstruction.
- Binary sign quantization with activation-weighted scale.
- Lloyd-Max rescue quantization for 2-bit and 4-bit blocks.
- Row-block sensitivity scoring.
- Global rescue-width selection per layer+linear type.
- `quantize_residuals(...)`, which rotates residual blocks, assigns rescue
  widths, quantizes blocks, and stores per-block metadata/scales/indices.

This file owns most of spec Sections 7 and 8. It is the first place to inspect
for bit allocation, scale, rescue, or dequantization bugs.

### `rcq_moe/remote_jobs.py`

Validation and dispatch helpers for one-shot Kaggle remote jobs.

Contains:

- `RemoteJob` and `JobSpecError`.
- Strict JSON schema validation for `schema_version=1` job specs.
- Required `job_id`, `action`, `code_commit`, `created_utc`, `reason`, and
  `params` fields.
- Filesystem-safe job ID validation and filename/job ID matching.
- Action allowlists for `control_plane_smoke` and `qwen36_fp_smoke`.
- Reserved-action rejection for future `debug_shell`, one-layer RCQ, polling,
  and GitHub result-pushing slices.
- Param validation/defaulting and deterministic argv construction for
  `scripts/qwen36_fp_smoke.py`.
- Output-root containment checks, selected non-secret environment snapshots,
  JSON writing, and local action execution with stdout/stderr capture.

Use this as the local, testable control-plane core. It should not import Kaggle
APIs or require GPU access.

### `rcq_moe/stats.py`

Calibration statistic accumulation.

Contains:

- `LinearCalibrationStats`, holding covariance, expert importance, and rotated
  second moments.
- `accumulate_covariance_and_moments(...)`, which streams expert-grouped
  activations and router weights into router-weighted covariance, expert usage,
  and per-block rotated activation diagonal moments.

This file implements the stat shape consumed by decomposition and residual
quantization.

### `rcq_moe/storage.py`

Effective expert bpw accounting.

Contains:

- `StorageBreakdown`.
- `expert_bpw(...)`, which accounts for shared factors, residual index bits,
  scale bits, and 2-bit row-block metadata against original expert weights.

Update this if storage precision changes, metadata packing changes, or shared
factor accounting changes.

### `rcq_moe/text_data.py`

Text fixture and token-batch utilities for offline smoke runs.

Contains:

- `TextFixtureConfig`, `TextBatchConfig`, and `TextTokenBatch`.
- Synthetic fixture documents for offline tests.
- Hugging Face streaming text loading.
- Document filtering, truncation, calibration/eval splitting, and fixture
  read/write helpers.
- Deterministic toy byte-token encoding.
- Fixed-shape `input_ids` and `attention_mask` creation for toy tokenization
  and Hugging Face tokenizers.

Use this when a smoke run should use text-derived tokens instead of random IDs.

### `rcq_moe/tiny_qwen.py`

Standalone tiny Qwen-shaped MoE model used for local RCQ conversion tests.

Contains:

- `TinyQwen35MoeConfig`.
- Tiny RMSNorm, MLP, top-k router, expert, sparse MoE block, decoder layer, and
  causal-LM shell.
- A Qwen-like fused `gate_up_proj` plus `down_proj` expert representation.
- RCQ wrappers and conversion helpers mirroring the official-Qwen path.
- Local calibration stat collection and routed correction fitting for the tiny
  model.

This file avoids depending on a full official model while keeping the important
MoE tensor shapes and routing behavior.

### `rcq_moe/toy_moe.py`

Smallest reference MoE pipeline for unit-level RCQ behavior.

Contains:

- A toy routed MoE layer.
- Quantized toy MoE layer using shared decomposition, quantized residuals, and
  optional output correction.
- `quantize_toy_moe_layer(...)`.
- `fit_toy_moe_output_correction(...)`.

This is the easiest file to use when debugging RCQ math without official-Qwen
model complexity.

### `rcq_moe/__pycache__/`

Generated CPython bytecode cache for the source files above:

```text
__init__.cpython-311.pyc
artifact.cpython-311.pyc
codebooks.cpython-311.pyc
correction.cpython-311.pyc
decomposition.cpython-311.pyc
hadamard.cpython-311.pyc
harness.cpython-311.pyc
metrics.cpython-311.pyc
official_qwen.cpython-311.pyc
quantization.cpython-311.pyc
stats.cpython-311.pyc
storage.cpython-311.pyc
text_data.cpython-311.pyc
tiny_qwen.cpython-311.pyc
toy_moe.cpython-311.pyc
```

Do not edit these. They can be deleted and regenerated by Python.

## `scripts/`

Runnable smoke and fixture scripts. Each script prepends the repo root to
`sys.path`, so they can be run directly with `python scripts/...`.

### `scripts/build_text_fixture.py`

Builds local calibration/eval text fixture files.

Contains:

- CLI args for synthetic vs Hugging Face streaming source.
- Dataset/name/split/text-field options for HF.
- Document count, character cap, eval split, and minimum character filters.
- Output file naming via `--output-dir` and `--prefix`.
- Calls into `rcq_moe.text_data` for loading, filtering, splitting, and writing.

Use this before text-token smoke runs when you want stable local fixture files.

### `scripts/official_qwen_text_smoke.py`

Text-derived token smoke run for tiny official Qwen3.5-MoE RCQ conversion.

Contains:

- Optional calibration/eval fixture file inputs.
- Optional Hugging Face tokenizer support, including local-only mode.
- Fallback to generated fixture files or built-in synthetic documents.
- Tiny official Qwen model construction.
- Text batch construction for calibration and eval.
- `run_official_qwen_harness(...)` with `rcq_1p75`.
- Artifact save/load round-trip and converted-vs-loaded KL/max-abs checks.
- Printed diagnostics for KL, bpw, rescue width percentages, and correction MSE.

This is a plumbing smoke test, not a pretrained-model quality run.
The current slice-2 validation uses
`--tokenizer-name-or-path Qwen/Qwen3.6-35B-A3B --tokenizer-local-files-only`
after the tokenizer has been cached locally.

### `scripts/official_qwen_toy_ablation.py`

Runs the official-Qwen tiny-model ablation helper on random token IDs.

Contains:

- CLI args for seed, hidden size, intermediate size, layers, rank list,
  block size, calibration shape, and eval shape.
- Calls `run_official_qwen_ablation(...)`.
- Prints rows for full-rank debug and low-rank variants with/without correction.

Use this for quick checks that rank/correction variants execute and produce
finite KL/MSE diagnostics.

### `scripts/official_qwen_toy_artifact.py`

Artifact round-trip smoke script for the tiny official Qwen model.

Contains:

- CLI args for output directory, cleanup, seed, model size, rank, and block size.
- Converts a random tiny official Qwen model to RCQ.
- Saves artifact files.
- Reloads the artifact.
- Compares converted and loaded logits.
- Prints artifact path and diagnostics.

Use this when changing `artifact.py` or serialized RCQ state layout.

### `scripts/official_qwen_toy_kl.py`

Random-token official-Qwen tiny-model KL smoke run.

Contains:

- CLI args for seed, model size, rank, block size, calibration batch/sequence,
  and eval batch/sequence.
- Builds a tiny official Qwen model.
- Runs `run_official_qwen_harness(...)`.
- Prints FP-vs-RCQ KL summary and per-layer linear diagnostics.

Use this as the fastest end-to-end official-Qwen conversion metric script.

### `scripts/qwen36_fp_smoke.py`

FP-only pretrained Qwen3.6 smoke runner intended for remote GPU environments,
especially the competition-attached Kaggle RTX PRO 6000 path.

Contains:

- CLI args for model ID, revision, dtype, device map, output directory, text
  files, sequence length, and eval batch shape.
- `--dry-run` mode that writes environment metadata without loading model
  weights.
- Hugging Face tokenizer/model loading with optional `HF_TOKEN`.
- Tiny next-token loss pass using tokenizer-backed `TextTokenBatch`.
- MoE/router/expert module structure inspection.
- JSON/text outputs: `metadata.json`, `module_structure.txt`, and
  `fp_metrics.json`.

Use this for the first real Qwen3.6 slice. It deliberately does not quantize or
save model weights.

## `kaggle/`

Kaggle script/notebook definitions used as remote GPU runners. These files
should remain thin wrappers around committed local scripts where possible so
peer review and rollback stay anchored in git.

### `kaggle/remote_worker/run_job.py`

One-shot Kaggle runner for reviewed JSON remote jobs.

Contains:

- Required `RCQ_REPO_URL` and `RCQ_JOB_PATH` environment inputs.
- Optional `RCQ_JOB_REF`, defaulting to `kaggle-jobs`.
- Clone of the public repo into `/kaggle/working/rcq_moe_impl`.
- Clone of the job branch into `/kaggle/working/rcq_moe_jobs`.
- Pre-checkout read of `code_commit`, then detached checkout of that exact
  commit before importing `rcq_moe.remote_jobs`.
- Validation of the selected job spec from the pinned code.
- Output writing under `/kaggle/working/outputs/<job_id>/`.
- `job.json`, `run_manifest.json`, stdout/stderr logs, selected environment
  metadata, argv, git SHA, timing, status, and output-file listing.

This runner executes exactly one job per invocation. It does not poll, push to
GitHub, or implement `debug_shell`.

### `kaggle/remote_worker/job_templates/0001-control-plane-smoke.json`

Template JSON job spec for the first remote-control smoke run.

Contains the `control_plane_smoke` action with placeholder `code_commit`.
Copy this to `jobs/` on the `kaggle-jobs` branch and replace `code_commit`
before use.

### `kaggle/remote_worker/job_templates/0002-qwen36-fp-smoke.json`

Template JSON job spec for the FP-only Qwen3.6 smoke run.

Contains the `qwen36_fp_smoke` action with default model, batch, dtype, device
map, trust-remote-code, and MoE module inspection params. Copy this to `jobs/`
on the `kaggle-jobs` branch and replace `code_commit` before use.

### `kaggle/qwen36_fp_smoke/kernel-metadata.json`

Kaggle metadata for the private script kernel
`aarushkhilosia/rcq-qwen36-fp-smoke`.

Contains:

- `enable_gpu: true`.
- `enable_internet: true`.
- Competition attachment:
  `nvidia-nemotron-model-reasoning-challenge`.
- Script entrypoint `run_qwen36_fp_smoke.py`.

Do not add credentials or model tokens here.

### `kaggle/qwen36_fp_smoke/run_qwen36_fp_smoke.py`

Thin Kaggle runner for the competition-attached Qwen3.6 FP-only smoke slice.

Contains:

- Required `RCQ_REPO_URL` environment variable.
- Optional `RCQ_COMMIT_SHA` for exact commit checkout.
- Optional `RCQ_GIT_TOKEN`, injected into HTTPS clone URLs without printing the
  token in logs.
- HF cache placement under `/kaggle/working/hf_cache`.
- GPU verification via `nvidia-smi`.
- Editable install of the cloned repo.
- Execution of `scripts/qwen36_fp_smoke.py`, writing outputs under
  `/kaggle/working/outputs/qwen36_fp_smoke`.

This runner should be submitted with Kaggle API only after explicit permission.

### `kaggle/qwen36_fp_smoke_notebook/kernel-metadata.json`

Kaggle metadata for the private notebook kernel
`aarushkhilosia/rcq-qwen3-6-fp-smoke-notebook`.

Contains:

- `kernel_type: notebook`.
- `enable_gpu: true`.
- `enable_internet: true`.
- Competition attachment:
  `nvidia-nemotron-model-reasoning-challenge`.
- Notebook entrypoint `rcq_qwen36_fp_smoke.ipynb`.

The CLI-pushed notebook can land on Kaggle's default GPU class. For the RTX PRO
6000 slice, prefer a manually created Kaggle notebook with the accelerator set
in the UI, then use a committed bootstrap/worker path from the repo.

### `kaggle/qwen36_fp_smoke_notebook/rcq_qwen36_fp_smoke.ipynb`

Notebook-backed FP-only Qwen3.6 smoke runner.

Contains:

- Kaggle secret import for optional `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN`,
  `RCQ_REPO_URL`, `RCQ_COMMIT_SHA`, and `RCQ_GIT_TOKEN`.
- HF cache placement under `/kaggle/working/hf_cache`.
- GPU verification via `nvidia-smi`.
- Optional repo-runner mode controlled by `RCQ_USE_REPO_RUNNER=1`, which clones
  `RCQ_REPO_URL`, optionally checks out `RCQ_COMMIT_SHA`, installs the repo, and
  runs `scripts/qwen36_fp_smoke.py`.
- Embedded fallback FP-only smoke code that loads the tokenizer/model, records
  sparse MoE module structure, runs a tiny next-token loss pass, and writes
  outputs under `/kaggle/working/outputs/qwen36_fp_smoke_notebook`.

This notebook deliberately does not quantize or save model weights.

## `tests/`

Unit and smoke coverage for the RCQ math, toy models, official-Qwen integration,
artifact layout, text fixtures, and metrics.

### `tests/fixtures/tiny_bert_tokenizer/`

Minimal checked-in Hugging Face `BertTokenizer` fixture.

Contains:

- `vocab.txt` with a tiny vocabulary.
- `tokenizer_config.json`.
- `special_tokens_map.json`.

Use this only for no-network AutoTokenizer path coverage. It is not a Qwen
tokenizer and is not used for quality interpretation.

### `tests/test_correction_metrics.py`

Tests correction and KL helpers.

Covers:

- `OnlineChannelRegression` recovering known affine parameters.
- Identity fallback when `y == yhat`.
- Per-channel non-worsening correction behavior.
- Toy MoE output correction reducing or preserving calibration MSE.
- KL summary producing finite expected fields.
- Rescue config preset percentages.

### `tests/test_official_qwen_artifact.py`

Tests official-Qwen RCQ artifact save/load behavior.

Covers:

- Converted-vs-loaded logits match exactly after artifact round-trip.
- Loaded model contains RCQ sparse MoE blocks.
- `metadata.json`, `non_expert_state.pt`, and `rcq_state.pt` are written.
- Metadata contains artifact format, model type, layer count, and diagnostics.

### `tests/test_official_qwen_harness.py`

Tests high-level official-Qwen harness behavior.

Covers:

- Finite KL metrics and diagnostics from `run_official_qwen_harness(...)`.
- Full-rank debug mode is near-lossless.
- Ablation helper returns the expected rows and finite summary fields.

### `tests/test_official_qwen_rcq.py`

Tests official-Qwen conversion internals.

Covers:

- Tiny official Qwen forward pass and output shapes.
- Hook-based MoE input collection.
- Full-rank RCQ conversion preserving logits with correction.
- Low-rank RCQ conversion running successfully and producing finite KL.

### `tests/test_official_qwen_text_harness.py`

Tests text-derived-token harness execution.

Covers:

- Synthetic text document split.
- Toy-token batch creation with masks.
- Local Hugging Face tokenizer batch creation with masks.
- Official-Qwen harness run on text-derived IDs.
- Finite KL and expected layer diagnostic count.

### `tests/test_rcq_decomposition.py`

Tests shared-subspace decomposition.

Covers:

- `choose_rank(...)` spec behavior and tiny-dimension cap.
- KLT transform/inverse after eigenvalue flooring.
- Shared subspace captures common expert structure.
- Full-rank decomposition leaves near-zero residuals.

### `tests/test_rcq_math.py`

Tests core quantization math.

Covers:

- Hadamard orthonormality and signed-rotation dot-product preservation.
- Lloyd-Max 2-bit codebook symmetry and expected scale.
- Binary quantizer weighted scale behavior.
- Lloyd rescue block quantization.
- Rescue block selection percentages for `rcq_1p75`.
- Dequantized residual shapes.
- Storage bpw accounting.

### `tests/test_remote_jobs.py`

Tests one-shot Kaggle remote job validation and local dispatch behavior.

Covers:

- Valid control-plane and FP smoke specs.
- Deterministic argv construction.
- Filename/job ID matching.
- Missing required field rejection.
- Unknown top-level field, action, and param rejection.
- Reserved future action rejection.
- Invalid commit and param value rejection.
- Output-root containment checks.
- Local `control_plane_smoke` dry-run execution through `run_action(...)`.
- Checked-in job template validation.

### `tests/test_text_data.py`

Tests text fixture utilities.

Covers:

- Document filtering, truncation, and max-doc caps.
- Deterministic nonempty calibration/eval split.
- Built-in synthetic fixture availability.
- Fixture write/read round-trip.
- Toy byte-token determinism and vocab bounds.
- Fixed-shape input ID generation with repeat behavior.
- Token batch attention masks.
- Local `AutoTokenizer.from_pretrained(..., local_files_only=True)` coverage
  using the tiny checked-in tokenizer fixture.

### `tests/test_tiny_qwen_normal.py`

Tests the standalone tiny Qwen-shaped model before RCQ conversion.

Covers:

- Normal forward execution.
- Expected logits shape.
- Finite output values.

### `tests/test_tiny_qwen_rcq_conversion.py`

Tests RCQ conversion for the standalone tiny Qwen-shaped model.

Covers:

- Full-rank conversion preserving outputs closely.
- Low-rank conversion running with finite logits/KL.
- Replacement of tiny MoE blocks with RCQ variants.

### `tests/test_toy_pipeline.py`

Tests the smallest toy RCQ pipeline.

Covers:

- Toy MoE forward behavior.
- Toy layer quantization.
- Quantized toy output shape and finite values.
- Correction integration in the toy path.

### `tests/__pycache__/`

Generated pytest/CPython bytecode cache for the test files:

```text
test_correction_metrics.cpython-311-pytest-9.0.3.pyc
test_official_qwen_artifact.cpython-311-pytest-9.0.3.pyc
test_official_qwen_harness.cpython-311-pytest-9.0.3.pyc
test_official_qwen_rcq.cpython-311-pytest-9.0.3.pyc
test_rcq_decomposition.cpython-311-pytest-9.0.3.pyc
test_rcq_math.cpython-311-pytest-9.0.3.pyc
test_text_data.cpython-311-pytest-9.0.3.pyc
test_tiny_qwen_normal.cpython-311-pytest-9.0.3.pyc
test_tiny_qwen_rcq_conversion.cpython-311-pytest-9.0.3.pyc
test_toy_pipeline.cpython-311-pytest-9.0.3.pyc
```
