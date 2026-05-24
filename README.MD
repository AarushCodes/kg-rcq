# RCQ-MoE

Router-coherent sub-2-bit quantization for MoE expert weights.

RCQ-MoE is my current research prototype for compressing Mixture-of-Experts expert weights to about 1.5 to 1.9 effective bits per weight. The work is still early. The code passes local tests and runs through tiny Qwen3_5Moe configs, but I have not validated quality on a full pretrained MoE checkpoint yet. That is the next real test.

The first target is Qwen-style MoE, especially `Qwen/Qwen3.6-35B-A3B`. I picked it because the model is large enough for the memory problem to be real, and small enough that a careful PTQ experiment should be possible on a large GPU.

Current status: PyTorch reference prototype, 63 passing tests, Hugging Face Transformers `Qwen3_5Moe` integration on tiny randomly initialized configs, text-token smoke tests, artifact round-trip tests, and Kaggle remote-execution scaffolding.

## why I am working on this

Sparse MoE models route each token to only a few experts, so active compute is much lower than the total parameter count suggests. The catch is that all expert weights still need to sit in memory. Long-context serving makes this worse because the KV cache also eats VRAM. If expert weights take less space, the same GPU can leave more room for context, batch size, and cache residency.

This will not magically solve inference. Attention, KV cache, kernels, and batching still matter. But expert-weight memory is a large resident cost in MoE models, and it is a clean target for post-training quantization.

## research goal

The question I want to test is simple:

Can MoE expert MLP weights go below 2 effective bits per weight without falling apart?

The target measurements are FP-vs-quantized logits KL, perplexity, routed MoE output MSE, expert bits per weight, and rare-expert behavior. For a first strong result, I am aiming for this rough range:

```text
1.75 to 1.90 expert bpw: strong if mean KL <= 0.12 nats/token
1.50 to 1.70 expert bpw: promising if mean KL <= 0.20 nats/token
```

Those numbers are targets, not claims.

## compression recipe

RCQ-MoE quantizes only the MoE expert MLP weights at first. Routers, attention, embeddings, norms, shared experts, and the LM head stay in their original precision.

For each MoE layer and expert linear projection, the reference path represents the expert weight like this:

```text
expert weight
  ≈ shared low-rank component
  + Hadamard-rotated 1-bit residual
  + mixed 2/4-bit rescue blocks
  + routed MoE-output affine correction
```

### router-weighted activation statistics

MoE quantization should track routed output error. Raw weight reconstruction is too local. RCQ-MoE collects activation covariance with router weights included:

```text
C = E[r_e(x)^2 * a_e a_e^T]
```

Here `r_e(x)` is the router weight for expert `e`, and `a_e` is the activation entering the expert linear.

### shared subspace through KLT and SVD

Experts in the same layer often share structure. RCQ-MoE uses a KLT-aligned stacked SVD to keep a shared part in higher precision:

```text
W_e ≈ A_e @ B_shared + R_e
```

`B_shared` is shared for a layer and linear type. `A_e` is stored per expert. `R_e` is the residual that gets quantized aggressively.

### Hadamard rotation before residual quantization

The residual is split into input blocks and rotated with deterministic signed Hadamard matrices. The goal is to spread out coordinate outliers before using 1-bit and low-bit scalar quantization.

```text
z_block = residual_block @ Q
u_block = Q.T @ activation_block
```

The rotation is orthogonal, so the full-precision dot product is unchanged before quantization.

### weighted 1-bit residual blocks

Most residual blocks are stored as signs plus one scale. The scale uses routed activation second moments instead of amax:

```text
alpha = sum_j v_j * |z_j| / sum_j v_j
```

`v_j` is the routed second moment for the rotated activation coordinate.

### mixed 2/4-bit rescue

Pure 1-bit residuals are probably too weak on their own. RCQ-MoE scores residual row-blocks by estimated routed output MSE and upgrades the worst blocks to 2-bit or 4-bit.

```text
rcq_1p55: top 2% -> 4-bit, next 10% -> 2-bit, rest -> 1-bit
rcq_1p75: top 5% -> 4-bit, next 20% -> 2-bit, rest -> 1-bit
rcq_1p90: top 5% -> 4-bit, next 35% -> 2-bit, rest -> 1-bit
```

The 2-bit and 4-bit blocks use deterministic Lloyd-Max scalar codebooks generated for a standard normal distribution.

### routed MoE-output correction

At this bit level, isolated linear corrections are too local. RCQ-MoE fits a per-channel affine correction on the actual routed MoE output:

```text
moe_out_corr = alpha * moe_out_quant + beta
```

There is also a per-channel identity fallback. If the fitted correction does not reduce calibration MSE, that channel stays unchanged.

## where the ideas came from

RCQ-MoE is independent work, but it is not pretending to start from nothing. It borrows good ideas from recent MoE quantization, FP4 training, and vector quantization work, then tests a more aggressive sub-2-bit recipe.

### KBVQ-MoE

KBVQ-MoE is the closest inspiration. It shows that MoE experts have shared structure, and that KLT-guided SVD can separate a shared component before quantizing the expert-specific part.

RCQ-MoE uses the same broad intuition: find a shared expert subspace, keep it in higher precision, and quantize the residual. The differences are the router weighting, the mostly 1-bit residual path, the mixed rescue blocks, and the correction fitted on the routed MoE aggregate output.

At 2 to 3 bits, I would expect a KBVQ-like method to be very strong. RCQ-MoE is aimed at the harder question below 2 bits.

### NVFP4 training methodology

The NVFP4 work is about training, while RCQ-MoE is PTQ. I am using it for numerical intuition rather than copying the setting. The parts that matter here are Random Hadamard Transforms, fine-grained scaling, and the practical lesson that some pieces need more precision.

RCQ-MoE applies that lesson to residual expert weights. Most blocks get 1 bit, and the blocks that look expensive get rescued.

### TurboQuant

TurboQuant helped shape the rotation and scalar-codebook side of the design. It makes a strong case that random rotations can make coordinates easier to quantize, and that inner-product distortion matters more than raw reconstruction error in many model settings.

RCQ-MoE uses fixed standard-normal Lloyd-Max codebooks for 2-bit and 4-bit rescue blocks. I am not starting with QJL residual sketches because I want the first reference path to stay simple enough to debug and eventually serve.

## what is implemented

The prototype currently includes:

```text
router-weighted covariance and activation statistics
KLT/eigendecomposition shared subspace extraction
shared low-rank expert decomposition
signed block-Hadamard rotations
1-bit weighted sign residual quantization
2/4-bit Lloyd-Max rescue quantization
effective expert bpw accounting
routed MoE-output affine correction with identity fallback
standalone tiny Qwen-style MoE model
Hugging Face Transformers Qwen3_5Moe tiny-config integration
KL metric helpers
artifact save/load round-trip
text-token smoke harness
Kaggle remote-control scaffolding
```

Latest local test result:

```text
pytest: 63 passed
```

Tiny randomly initialized Hugging Face Transformers `Qwen3_5Moe` text-token smoke result:

```text
FP-vs-RCQ KL mean ≈ 6.17e-07
artifact reload max_abs_logit_delta = 0
```

This checks plumbing and math on tiny/random models. It is not evidence that the method preserves quality on a pretrained checkpoint.

## planned validation

I want to validate this in stages so I do not waste expensive GPU time.

First, load a real pretrained MoE checkpoint, preferably Qwen3.6-35B-A3B, and run an FP-only smoke test. That means checking memory behavior, inspecting MoE modules, and running a tiny next-token loss pass.

Second, quantize exactly one MoE layer. The output should include FP-vs-RCQ KL, routed MoE MSE before and after correction, expert bpw, and basic runtime notes.

Third, run the ablation ladder:

```text
A0: shared subspace + naive 1-bit residual
A1: A0 + Hadamard rotation
A2: A1 + activation-weighted binary scale
A3: A2 + mixed 2/4-bit rescue
A4: A3 + routed MoE-output correction
```

Then compare against full precision, naive 2-bit expert quantization, naive 3-bit expert quantization, and the RCQ configs around 1.55, 1.75, and 1.90 bpw.

## why compute credits are needed

Qwen3.6-35B-A3B is too large for my local hardware and awkward on free GPUs. The real validation needs enough VRAM for model loading, calibration stats, FP-vs-quantized comparison, layer replacement, and KL/PPL evaluation.

Cloud GPU credits would go into staged experiments. This repo does not redistribute pretrained model weights.

## repository map

```text
SUB2BIT_MOE_IMPLEMENTATION_SPEC.md   implementation and research spec
state.md                             current prototype state and validation log
STRUCTURE.MD                         repository structure guide

rcq_moe/
  decomposition.py                   KLT/SVD shared subspace
  quantization.py                    residual and rescue quantization
  hadamard.py                        signed Hadamard rotations
  correction.py                      routed affine correction
  official_qwen.py                   Hugging Face Transformers Qwen3_5Moe integration
  metrics.py                         KL metrics
  storage.py                         bpw accounting

scripts/                             smoke and evaluation scripts
tests/                               unit and integration tests
kaggle/                              remote GPU runner scaffolding
```

## reproduce local checks

Run the tests:

```bash
uv run pytest -q
```

Run a tiny randomly initialized Hugging Face Transformers `Qwen3_5Moe` KL smoke:

```bash
uv run python scripts/official_qwen_toy_kl.py
```

Run the FP-only Qwen3.6 smoke in dry-run mode:

```bash
uv run python scripts/qwen36_fp_smoke.py \
  --dry-run \
  --output-dir outputs/qwen36_fp_smoke_dry_run
```

Run the text-token tiny randomly initialized `Qwen3_5Moe` smoke:

```bash
uv run python scripts/official_qwen_text_smoke.py --clean
```

## evaluation report format

For each real model tested, I plan to report expert bpw, full-model memory estimate, KL summary, perplexity, routed MoE MSE before and after correction, residual energy, shared-subspace captured energy, rescue block percentages, rare-expert diagnostics, and comparisons against 2-bit and 3-bit baselines.

## first-prototype limits

The current prototype does not quantize attention, routers, embeddings, norms, or the LM head. It also does not have fused CUDA kernels or production packed bitstreams yet. The current goal is to test quality honestly before spending time on serving kernels.

## what this repo proves today

The repo currently proves that the reference math runs on toy and tiny Qwen3_5Moe configs, artifacts can be saved and loaded exactly, KL metrics work, and the code is ready for staged pretrained validation.

It has not proved full pretrained Qwen3.6 quality preservation, downstream task retention, real serving speedup, or production storage efficiency.

## authorship and license

This project is developed by Aarush Khilosia as an independent research and engineering project.

The code is released under the GNU GPLv3 license. See `LICENSE` for details.

Copyright © 2026 Aarush Khilosia.
