# Sub-2-bit MoE Quantization Implementation Spec

**Document purpose:** This is a self-contained implementation specification for a research/engineering team. The team should not need to read any papers. Implement exactly the baseline pipeline first, then run the listed ablations.

**Project goal:** Compress **MoE expert weights** to an effective **1.5–1.9 bits/weight** while keeping real-task quality close to a strong 2-bit baseline. We optimize serving cost, not training. This is **post-training quantization (PTQ)**: no labels, no gradient training, no model retraining.

---

## 1. Executive summary

The method is called here:

> **RCQ-MoE: Router-Coherent Quantization for sub-2-bit MoE**

Core idea:

1. MoE experts share large redundant subspaces. Extract and keep this shared part in higher precision.
2. Only quantize the remaining expert-specific residuals extremely aggressively.
3. Before 1-bit quantization, rotate residuals with block Hadamard transforms to reduce coordinate outliers.
4. Use 1-bit bulk residual quantization plus a small mixed-bit rescue budget for sensitive residual blocks.
5. Correct the actual **routed MoE aggregate output**, not only individual expert outputs.

Default target recipe:

```text
MoE expert W_e
  = FP16/FP8 shared low-rank component
  + block-Hadamard rotated 1-bit residual
  + 2/4-bit rescue blocks for sensitive residual row-blocks
  + routed MoE-output affine correction
```

Initial scope:

- Quantize MoE expert MLP linear weights only.
- Do **not** quantize routers/gates, attention, embeddings, LM head, or norms in the first implementation.
- Keep all non-MoE parts in their original precision or whatever production quantization already uses.

Primary quality metric:

\[
\mathbb{E}_{x}\, KL(p_{\text{FP}}(\cdot|x)\;||\;p_{\text{quant}}(\cdot|x))
\]

Target:

```text
1.75–1.90 effective expert bpw: KL <= 0.12 nats/token is strong.
1.50–1.70 effective expert bpw: KL <= 0.20 is promising.
```

---

## 2. Notation and model assumptions

A decoder MoE layer has a router and multiple experts. For token hidden state \(x \in \mathbb{R}^{d_{model}}\):

1. Router computes scores/probabilities \(g_e(x)\).
2. Top-k experts are selected.
3. Expert outputs are aggregated:

\[
y_{moe}(x) = \sum_{e \in topk(x)} g_e(x) E_e(x)
\]

Typical expert MLP:

\[
E_e(x) = W^{down}_e \left( \phi(W^{gate}_e x) \odot W^{up}_e x \right)
\]

where:

- \(W^{gate}_e \in \mathbb{R}^{d_{ff} \times d_{model}}\)
- \(W^{up}_e \in \mathbb{R}^{d_{ff} \times d_{model}}\)
- \(W^{down}_e \in \mathbb{R}^{d_{model} \times d_{ff}}\)
- \(\phi\) is the model activation, e.g. SiLU/GELU.

This spec applies independently to each MoE layer \(\ell\), each expert linear type \(t \in \{gate, up, down\}\), and each expert \(e\).

For a given MoE layer and linear type, denote:

- Number of experts: \(E\)
- Weight matrix for expert \(e\): \(W_e \in \mathbb{R}^{m \times n}\)
- Input activation to this linear: \(a_e \in \mathbb{R}^{n}\)
- Router weight for this expert/token: \(r_e = g_e(x)\) if selected, else \(0\)

For `gate` and `up` projections:

\[
a_e = x
\]

For `down` projection:

\[
a_e = \phi(W^{gate}_e x) \odot W^{up}_e x
\]

Use full-precision activations during calibration unless explicitly doing sequential calibration.

---

## 3. Calibration data and compute budget

### 3.1 Data

Use unlabeled text. Recommended calibration sizes:

```text
Prototype:        256 sequences × 4096 tokens  (~1M tokens)
Better research:  512–2048 sequences × 4096 tokens
Production eval:  several million representative tokens
```

The calibration set should match expected serving distribution as much as possible. Include code/math/chat if those are important workloads.

### 3.2 Required forward passes

Minimum implementation needs:

1. One FP forward pass to collect router decisions, router weights, and activation statistics.
2. One or more quantized forward passes for routed-output correction and KL measurement.

No backward pass. No labels.

### 3.3 Important practical note

Do not store all activations for all layers/tokens in memory. Stream through calibration data and maintain online statistics per layer/linear type.

---

## 4. Overview of pipeline

For each MoE layer \(\ell\) and each expert linear type \(t\):

```text
1. Collect router-weighted activation covariance/statistics.
2. Compute router-weighted KLT transform T.
3. Stack all experts in KLT space and compute top-r shared subspace.
4. Decompose each expert weight:
      W_e = W_shared_e + R_e
5. Quantize residual R_e:
      a. block-Hadamard rotate along input dimension
      b. 1-bit sign quantize with activation-weighted scale
      c. upgrade most sensitive row-blocks to 2/4 bits
6. Store shared low-rank component + quantized residual.
7. Fit routed MoE-output affine correction per layer.
8. Evaluate KL, PPL, and downstream tasks.
```

---

## 5. Router-weighted activation statistics

The weight error that matters is routed output error. For a linear weight error \(\Delta W_e\), the local proxy distortion is:

\[
\mathcal{L} \approx \sum_e \mathbb{E}_x \left[ r_e(x)^2 \|\Delta W_e a_e(x)\|_2^2 \right]
\]

This motivates using router-weighted input covariance:

\[
C = \frac{\sum_{tokens,e} r_e^2\, a_e a_e^T}{\sum_{tokens,e} r_e^2 + \epsilon}
\]

Use this shared covariance per layer and linear type for the default implementation.

### 5.1 What to collect

For each MoE layer \(\ell\), linear type \(t\), and calibration token:

- Selected experts \(e \in topk(x)\)
- Router weights \(r_e = g_e(x)\)
- Linear input activation \(a_e\)

Accumulate:

\[
S = \sum r_e^2 a_e a_e^T
\]

\[
z = \sum r_e^2
\]

Then:

\[
C = S / (z + \epsilon)
\]

Use \(\epsilon = 10^{-12}\) for denominator safety.

### 5.2 Memory-efficient covariance options

Default: full covariance for hidden sizes up to ~8192 if memory allows.

For each covariance matrix:

- FP32 memory cost = \(4n^2\) bytes.
- Example \(n=4096\): ~67 MB.

If memory is tight, compute one layer/linear type at a time.

Do **not** implement full expert-pair covariance \(C_{ef}\) initially. It is expensive and not required for the baseline.

---

## 6. KLT transform and shared subspace extraction

For each layer \(\ell\) and linear type \(t\), we have expert weights:

\[
W_e \in \mathbb{R}^{m \times n}, \quad e=1,\ldots,E
\]

and router-weighted covariance:

\[
C \in \mathbb{R}^{n \times n}
\]

### 6.1 Eigen decomposition

Compute:

\[
C = U \Lambda U^T
\]

where eigenvalues are sorted descending.

Clamp eigenvalues for numerical stability:

\[
\lambda_i^{clamped} = \max(\lambda_i, \lambda_{floor})
\]

Default:

\[
\lambda_{floor} = 10^{-5} \cdot \frac{1}{n}\sum_i \lambda_i
\]

Construct:

\[
T = U \Lambda^{1/2}
\]

\[
T^{-1} = \Lambda^{-1/2} U^T
\]

### 6.2 Weighted stacked matrix

Estimate expert importance:

\[
p_e = \frac{\sum_{tokens} r_e^2}{\sum_{tokens,e'} r_{e'}^2 + \epsilon}
\]

Construct transformed expert matrices:

\[
B_e = \sqrt{p_e E}\, W_e T
\]

The factor \(\sqrt{p_e E}\) keeps average scale stable. If all experts equal, \(p_e=1/E\), factor is 1.

Stack along output dimension:

\[
B =
\begin{bmatrix}
B_1 \\
B_2 \\
\vdots \\
B_E
\end{bmatrix}
\in \mathbb{R}^{(E m) \times n}
\]

### 6.3 Top-r right subspace

Compute the top \(r\) right singular vectors of \(B\):

\[
B \approx \tilde U_r \Sigma_r V_r^T
\]

where:

\[
V_r \in \mathbb{R}^{n \times r}, \quad V_r^T V_r = I
\]

Default rank:

```text
r = ceil(n / 128)
minimum r = 8
maximum r = 64
```

Use randomized SVD for large matrices.

### 6.4 Shared component representation

Define the shared input basis:

\[
B_{shared} = V_r^T T^{-1} \in \mathbb{R}^{r \times n}
\]

For each expert:

\[
A_e = W_e T V_r \in \mathbb{R}^{m \times r}
\]

Then:

\[
W^{shared}_e = A_e B_{shared}
\]

Residual:

\[
R_e = W_e - W^{shared}_e
\]

### 6.5 Inference for shared component

For input activation \(a\):

\[
W^{shared}_e a = A_e (B_{shared} a)
\]

Implementation should compute:

```text
u = B_shared @ a      # shape [r]
y_shared = A_e @ u    # shape [m]
```

Store `B_shared` once per layer/linear type and `A_e` per expert.

Default storage precision:

```text
A_e:       FP16 initially; later FP8 if quality allows
B_shared: FP16 initially; later FP8 if quality allows
```

Do not quantize these to 1-bit.

---

## 7. Residual quantization

We quantize only:

\[
R_e = W_e - W^{shared}_e
\]

Residual quantization has three parts:

1. Block-Hadamard rotation.
2. 1-bit sign quantization with activation-weighted scales.
3. Mixed-bit rescue for sensitive row-blocks.

### 7.1 Block partitioning

Partition the input dimension into contiguous blocks of size \(h\).

Default:

```text
h = 64
```

If \(n\) is not divisible by \(h\), pad residual columns and activation vectors with zeros to the next multiple of \(h\). Padding weights are not counted in effective bpw.

For each expert residual matrix \(R_e \in \mathbb{R}^{m \times n}\), each output row \(i\), and input block \(b\):

\[
r_{i,b} \in \mathbb{R}^{h}
\]

### 7.2 Block signed Hadamard rotation

For each block size \(h\), define normalized Hadamard matrix:

\[
H_h H_h^T = I
\]

Use a fixed random sign diagonal matrix per layer/linear/block-index:

\[
D_b = \text{diag}(s_1,\ldots,s_h), \quad s_j \in \{-1,+1\}
\]

Use deterministic seed:

```text
seed = hash(model_name, layer_id, linear_type, block_index)
```

Define:

\[
Q_b = D_b H_h
\]

Rotate residual row-block:

\[
z_{i,b} = r_{i,b} Q_b
\]

At inference, for activation block \(a_b\), compute:

\[
u_b = Q_b^T a_b
\]

Then:

\[
r_{i,b} a_b = z_{i,b} u_b
\]

The rotation is orthogonal, so full-precision math is unchanged before quantization.

### 7.3 Rotated activation second moments

For each layer/linear type and input block \(b\), collect weighted diagonal second moments of rotated activations:

\[
v_{b,j} = \mathbb{E}[r_e^2 \cdot u_{b,j}^2]
\]

where:

\[
u_b = Q_b^T a_b
\]

Accumulate during calibration:

\[
v_{b,j} = \frac{\sum_{tokens,e} r_e^2 u_{b,j}^2}{\sum_{tokens,e} r_e^2 + \epsilon}
\]

These are shared across experts for the default implementation.

### 7.4 1-bit residual quantization

For rotated residual row-block \(z \in \mathbb{R}^h\), store signs:

\[
s_j = \text{sign}(z_j)
\]

Use convention:

```text
sign(0) = +1
```

Reconstruction:

\[
\hat z_j = \alpha \cdot s_j
\]

Scale \(\alpha\) is chosen by weighted least squares under diagonal rotated activation covariance:

\[
\alpha^* = \frac{\sum_{j=1}^{h} v_j |z_j|}{\sum_{j=1}^{h} v_j + \epsilon}
\]

Default scale storage:

```text
FP8 E4M3 scale per output row per input block
fallback: FP16 scale if FP8 infrastructure unavailable
```

Do **not** use amax scale for binary quantization. Amax scale overweights outliers and performs poorly for 1-bit.

### 7.5 Local error estimate for a row-block

After binary reconstruction:

\[
e_j = z_j - \hat z_j
\]

Estimated contribution to routed output MSE:

\[
score(i,b,e) = \sum_{j=1}^{h} v_{b,j} e_j^2
\]

Use this score to select mixed-bit rescue blocks.

---

## 8. Mixed-bit rescue quantization

Pure 1-bit will usually be too lossy. We allocate a small higher-bit budget to row-blocks with highest estimated error.

### 8.1 Row-block as rescue unit

A rescue unit is:

```text
(layer_id, linear_type, expert_id, output_row, input_block)
```

Each unit contains \(h\) weights.

### 8.2 Default rescue budgets

Implement these three configs exactly:

#### Config A: `rcq_1p55`

```text
Top 2% row-blocks by score: 4-bit
Next 10% row-blocks by score: 2-bit
Remaining: 1-bit
```

#### Config B: `rcq_1p75` default

```text
Top 5% row-blocks by score: 4-bit
Next 20% row-blocks by score: 2-bit
Remaining: 1-bit
```

#### Config C: `rcq_1p90`

```text
Top 5% row-blocks by score: 4-bit
Next 35% row-blocks by score: 2-bit
Remaining: 1-bit
```

Percentages are computed globally per layer+linear type, across all experts and row-blocks.

If exact percentage yields non-integer count, round to nearest integer.

### 8.3 Fixed Lloyd-Max scalar codebooks

For 2-bit and 4-bit rescue blocks, use scalar quantization in rotated space with a fixed standard-normal Lloyd-Max codebook.

Generate codebook for bit-width \(b\) using this deterministic algorithm:

1. Number of levels: \(K=2^b\).
2. Distribution: standard normal \(\mathcal{N}(0,1)\).
3. Initialize centroids to normal quantile midpoints:

\[
c_k^{(0)} = \Phi^{-1}\left(\frac{k+0.5}{K}\right), \quad k=0,\ldots,K-1
\]

4. Iterate 200 times or until max centroid movement < \(10^{-7}\):
   - Boundaries:

\[
b_0=-\infty, \quad b_K=+\infty, \quad b_k = \frac{c_{k-1}+c_k}{2}
\]

   - Update centroid:

\[
c_k = \frac{\int_{b_k}^{b_{k+1}} x \phi(x) dx}{\int_{b_k}^{b_{k+1}} \phi(x) dx}
= \frac{\phi(b_k)-\phi(b_{k+1})}{\Phi(b_{k+1})-\Phi(b_k)}
\]

where \(\phi\) and \(\Phi\) are standard normal PDF/CDF.

5. Store centroids sorted ascending.

Expected 2-bit centroids are approximately:

```text
[-1.5104, -0.4528, 0.4528, 1.5104]
```

But generate them using the algorithm above; do not hardcode approximate values.

### 8.4 Scale and assignment for b-bit rescue block

For a rescue row-block \(z \in \mathbb{R}^h\), codebook \(c_k\), and weights \(v_j\), solve:

\[
\min_{\beta, q_j} \sum_j v_j (z_j - \beta c_{q_j})^2
\]

Use this deterministic iterative procedure:

1. Initialize:

\[
\beta = \sqrt{\frac{\sum_j v_j z_j^2}{\sum_j v_j + \epsilon}}
\]

2. Repeat 10 times:
   - Assignment:

\[
q_j = \arg\min_k |z_j - \beta c_k|^2
\]

   - Scale update:

\[
\beta = \frac{\sum_j v_j z_j c_{q_j}}{\sum_j v_j c_{q_j}^2 + \epsilon}
\]

   - Clamp \(\beta \ge 0\). If \(\beta < 10^{-12}\), set \(\beta=0\) and all indices to nearest codeword to zero.

3. Store indices \(q_j\) using \(b\) bits/value and scale \(\beta\) as FP8 E4M3 or FP16 fallback.

For 1-bit blocks, use the closed-form sign quantizer in Section 7.4, not the Lloyd procedure.

---

## 9. Inference math for quantized residual

For each selected expert and linear:

Input activation \(a \in \mathbb{R}^{n}\).

### 9.1 Shared component

\[
y_{shared} = A_e (B_{shared} a)
\]

### 9.2 Residual component

For each input block \(b\):

1. Compute rotated activation:

\[
u_b = Q_b^T a_b
\]

2. For each output row \(i\), compute dot against dequantized rotated residual:

For 1-bit block:

\[
y^{res}_{i,b} = \alpha_{i,b} \sum_{j=1}^{h} s_{i,b,j} u_{b,j}
\]

For b-bit rescue block:

\[
y^{res}_{i,b} = \beta_{i,b} \sum_{j=1}^{h} c_{q_{i,b,j}} u_{b,j}
\]

3. Sum blocks:

\[
y^{res}_i = \sum_b y^{res}_{i,b}
\]

Final linear output:

\[
y = y_{shared} + y_{res}
\]

Kernel engineers may fuse rotation, dequantization, and matmul. The reference implementation may dequantize for correctness first, but benchmark only fused kernels.

---

## 10. Routed MoE-output affine correction

Individual expert/linear corrections are insufficient at sub-2-bit. Fit correction on the actual MoE aggregate output.

For each MoE layer \(\ell\), collect calibration pairs:

- Full precision MoE output before residual connection:

\[
y = \sum_{e \in topk(x)} r_e E_e^{FP}(x)
\]

- Quantized MoE output before residual connection:

\[
\hat y = \sum_{e \in topk(x)} r_e E_e^{Q}(x)
\]

Use the same full-precision router top-k decisions for both during calibration. At inference, router remains full precision, so this matches deployment.

Fit per-channel affine:

\[
y_{corr,j} = \alpha_j \hat y_j + \beta_j
\]

Closed-form:

\[
\alpha_j = \frac{Cov(y_j, \hat y_j)}{Var(\hat y_j)+\epsilon}
\]

\[
\beta_j = \mu_{y_j} - \alpha_j \mu_{\hat y_j}
\]

Use \(\epsilon = 10^{-8}\).

Store \(\alpha, \beta \in \mathbb{R}^{d_{model}}\) in FP16 per MoE layer.

Apply at inference:

```text
moe_out_quant = sum_e router_weight[e] * expert_quant[e](x)
moe_out_corr = alpha[layer] * moe_out_quant + beta[layer]
```

Then continue with the model’s normal residual connection.

### 10.1 Calibration mode for correction

Use **sequential calibration** for final correction:

1. Quantize all MoE layers.
2. Run the model through calibration data.
3. At each MoE layer, feed the quantized model’s current hidden state into both:
   - FP copy of that MoE block
   - quantized MoE block
4. Fit \((\alpha,\beta)\) using those pairs.

This captures drift from earlier quantized layers.

---

## 11. Optional grouped subspaces: implement after baseline

The baseline uses one shared subspace per layer+linear type.

Grouped subspaces are optional but should be implemented for ablation if time permits.

### 11.1 Expert clustering

For each MoE layer, compute expert co-routing matrix:

\[
M_{ef} = \sum_{tokens} \mathbf{1}[e \in topk(x)]\mathbf{1}[f \in topk(x)]
\]

Normalize:

\[
\tilde M_{ef} = \frac{M_{ef}}{\sqrt{M_{ee}M_{ff}}+\epsilon}
\]

Cluster experts into \(G\) groups using spectral clustering on distance:

\[
d_{ef} = 1 - \tilde M_{ef}
\]

Required ablation values:

```text
G = 1, 2, 4, 8
```

If a cluster has fewer than 2 experts, merge it into nearest cluster by \(d_{ef}\).

### 11.2 Grouped decomposition

For each group independently, repeat Sections 6–8 using only experts in that group.

Rank per group:

\[
r_g = \max(4, \lceil n/256 \rceil)
\]

Total shared overhead must be included in bpw accounting.

Grouped subspaces should only be kept if they beat the baseline at equal effective bpw.

---

## 12. Storage accounting

Report effective bits per expert weight honestly.

For one layer+linear type:

- Experts: \(E\)
- Matrix shape: \(m \times n\)
- Total original expert weights: \(N = E m n\)
- Shared rank: \(r\)
- Block size: \(h\)
- Number of row-blocks per expert: \(m \lceil n/h \rceil\)

### 12.1 Shared component bits

If stored FP16:

\[
bits_{shared} = 16(rn + E m r)
\]

If stored FP8:

\[
bits_{shared} = 8(rn + E m r)
\]

### 12.2 Residual quantized bits

For each row-block \(u\), bit-width \(b_u \in \{1,2,4\}\):

\[
bits_{indices} = \sum_u b_u \cdot h_{valid,u}
\]

where \(h_{valid,u}\) excludes padding columns.

### 12.3 Scale bits

One scale per row-block:

\[
bits_{scales} = scale\_bits \cdot \#rowblocks
\]

Default:

```text
scale_bits = 8 for FP8 scale
scale_bits = 16 for FP16 fallback
```

### 12.4 Metadata bits

Store bit-width code per row-block. Use 2 bits/row-block:

```text
00 = 1-bit
01 = 2-bit
10 = 4-bit
11 = reserved
```

\[
bits_{metadata} = 2 \cdot \#rowblocks
\]

### 12.5 Effective bpw

\[
bpw = \frac{bits_{shared}+bits_{indices}+bits_{scales}+bits_{metadata}}{E m n}
\]

Also report full-model GB including all unquantized/non-expert weights.

---

## 13. Evaluation metrics

### 13.1 KL divergence

For evaluation tokens, compute full precision logits \(l^{FP}\) and quantized logits \(l^Q\).

Compute:

\[
logp^{FP} = logsoftmax(l^{FP})
\]

\[
logp^Q = logsoftmax(l^Q)
\]

\[
KL_t = \sum_v \exp(logp^{FP}_{t,v}) \left(logp^{FP}_{t,v} - logp^Q_{t,v}\right)
\]

Report:

```text
mean KL
p50 KL
p95 KL
p99 KL
max KL
```

Use nats/token.

### 13.2 Perplexity

Report WikiText2 or equivalent PPL using standard evaluation harness.

### 13.3 Downstream tasks

At minimum:

```text
ARC-Challenge
ARC-Easy
HellaSwag
PIQA
WinoGrande
LAMBADA if available
MMLU if feasible
```

### 13.4 Internal diagnostics

Report these per layer:

```text
routed MoE output MSE before correction
routed MoE output MSE after correction
mean and p99 token KL contribution if layer-replacement eval is available
residual energy ratio ||R||_F^2 / ||W||_F^2
shared subspace captured energy
percentage of 1/2/4-bit blocks
scale distribution min/p50/p99/max
rare expert error statistics
```

Rare expert check:

- Bucket experts by routing frequency.
- Report output MSE/KL proxy for coldest 25% experts.
- Do not automatically assign fewer bits to cold experts.

---

## 14. Required ablation plan

Run in this order.

### 14.1 Baselines

1. Full precision model.
2. Existing production quantization if available.
3. 2-bit expert quantization baseline if available.
4. KBVQ-like baseline if already implemented.

### 14.2 RCQ-MoE core ablations

For each target model:

```text
A0: shared subspace + naive 1-bit residual, no Hadamard, no rescue, no correction
A1: shared subspace + Hadamard 1-bit residual
A2: A1 + activation-weighted binary scale
A3: A2 + mixed-bit rescue config A/B/C
A4: A3 + routed MoE-output affine correction
```

Expected: A4 should be the first serious candidate.

### 14.3 Router-aware vs non-router-aware

Compare:

```text
C = unweighted E[a a^T]
C = router-weighted E[r_e^2 a a^T]
```

Keep all other settings fixed.

### 14.4 Subspace rank

Test:

```text
r = n/256
r = n/128 default
r = n/64
```

Report bpw and KL. Do not choose rank by reconstruction only; choose by KL/quality at equal or acceptable bpw.

### 14.5 Grouped subspaces

Test:

```text
G = 1 default
G = 2
G = 4
G = 8
```

At equal bpw, compare grouped subspaces against spending the same bits on more mixed-bit rescue blocks.

### 14.6 Correction type

Compare:

```text
no correction
per-linear output affine correction
routed MoE-output affine correction default
```

Routed MoE-output correction should be the default if it wins.

---

## 15. Implementation checklist

### 15.1 Calibration/statistics

- [ ] Hook router outputs per MoE layer.
- [ ] Hook expert linear inputs for gate/up/down.
- [ ] Accumulate router-weighted covariance per layer+linear.
- [ ] Accumulate expert usage \(p_e\).
- [ ] Accumulate rotated activation second moments \(v_{b,j}\).

### 15.2 Decomposition

- [ ] Eigen decomposition with eigenvalue floor.
- [ ] Construct \(T\) and \(T^{-1}\).
- [ ] Randomized top-r SVD of stacked transformed expert weights.
- [ ] Store \(A_e\), \(B_{shared}\).
- [ ] Compute residuals \(R_e\).

### 15.3 Quantization

- [ ] Implement block signed Hadamard rotation.
- [ ] Implement binary sign quantizer with weighted scale.
- [ ] Compute row-block sensitivity score.
- [ ] Implement Lloyd-Max codebook generation.
- [ ] Implement 2-bit and 4-bit rescue quantizers.
- [ ] Pack indices and metadata.
- [ ] Store scales.

### 15.4 Inference/reference

- [ ] Reference dequantized forward for correctness.
- [ ] Fused residual matmul path for performance.
- [ ] Shared low-rank matmul path.
- [ ] MoE-output affine correction.

### 15.5 Evaluation

- [ ] Mean/p95/p99 KL.
- [ ] PPL.
- [ ] Downstream eval tasks.
- [ ] Storage/bpw report.
- [ ] Layer diagnostics.

---

## 16. Reference pseudocode

### 16.1 Quantize one layer+linear type

```python
def quantize_layer_linear(weights, calib_stats, config):
    # weights: list of E tensors, each [m, n]
    # calib_stats contains covariance C, expert importance p_e,
    # and rotated activation moments after Q_b is known/created.

    E = len(weights)
    m, n = weights[0].shape

    # 1. KLT
    eigvals, U = eigh(calib_stats.C)          # ascending or descending depending library
    eigvals, U = sort_descending(eigvals, U)
    lam_floor = 1e-5 * mean(eigvals)
    eigvals = maximum(eigvals, lam_floor)
    T = U @ diag(sqrt(eigvals))
    T_inv = diag(1.0 / sqrt(eigvals)) @ U.T

    # 2. Stack weighted transformed expert matrices
    blocks = []
    for e, W in enumerate(weights):
        factor = sqrt(calib_stats.p_e[e] * E)
        blocks.append(factor * (W @ T))
    B = concat_rows(blocks)                  # [(E*m), n]

    # 3. Top-r right singular vectors
    r = choose_rank(n, config.rank_rule)
    V_r = randomized_right_svd(B, rank=r)     # [n, r]

    # 4. Shared factors
    B_shared = V_r.T @ T_inv                  # [r, n]
    A = []
    residuals = []
    for W in weights:
        A_e = W @ T @ V_r                     # [m, r]
        R_e = W - A_e @ B_shared              # [m, n]
        A.append(A_e)
        residuals.append(R_e)

    # 5. Quantize residuals
    q_residuals = []
    scores = []
    for e, R in enumerate(residuals):
        q_e, score_e = binary_quantize_residual_with_scores(
            R,
            config.hadamard_block_size,
            calib_stats.rotated_second_moments,
            layer_id=config.layer_id,
            linear_type=config.linear_type,
        )
        q_residuals.append(q_e)
        scores.extend(score_e)

    # 6. Select rescue blocks globally for this layer+linear
    rescue_plan = select_rescue_blocks(scores, config.rescue_percentages)

    # 7. Re-quantize rescue blocks at 2/4 bits
    apply_rescue_quantization(
        q_residuals,
        residuals,
        rescue_plan,
        calib_stats.rotated_second_moments,
    )

    return QuantizedLayerLinear(
        A=A,
        B_shared=B_shared,
        q_residuals=q_residuals,
        V_r=V_r,          # optional debug only
        eigvals=eigvals,  # optional debug only
    )
```

### 16.2 Fit routed MoE-output correction

```python
def fit_moe_output_correction(fp_model, q_model, calib_loader):
    # q_model already has quantized experts.
    # Use sequential hidden states from q_model.

    for layer in moe_layers:
        stats = OnlineChannelRegression(dim=d_model)

        for batch in calib_loader:
            hidden = run_q_model_until_layer(q_model, batch, layer)

            # Same router implementation for both.
            y_fp = fp_model.layers[layer].moe_block(hidden, force_fp_experts=True)
            y_q  = q_model.layers[layer].moe_block(hidden, quantized_experts=True)

            stats.update(y_fp, y_q)

        alpha, beta = stats.solve_affine(eps=1e-8)
        q_model.layers[layer].moe_output_alpha = alpha.astype(fp16)
        q_model.layers[layer].moe_output_beta = beta.astype(fp16)
```

Online regression accumulators per channel:

```text
count
sum_y
sum_yhat
sum_yhat2
sum_y_yhat
```

Then:

```text
mean_y = sum_y / count
mean_yhat = sum_yhat / count
var_yhat = sum_yhat2 / count - mean_yhat^2
cov = sum_y_yhat / count - mean_y * mean_yhat
alpha = cov / (var_yhat + eps)
beta = mean_y - alpha * mean_yhat
```

---

## 17. Expected outcomes and interpretation

Use these as sanity ranges, not guarantees.

Plain dynamic GGUF-style low-bit quantization often shows roughly:

```text
~1-bit: KL around 0.8–1.0
~2-bit: KL around 0.22–0.52
~3-bit: KL around 0.08
~4-bit: KL around 0.024
```

RCQ-MoE succeeds only if it makes sub-2-bit expert weights behave closer to 3-bit effective information.

Interpretation:

```text
KL <= 0.05: excellent / near invisible
0.05–0.12: strong target range
0.12–0.20: promising, task-dependent
0.20–0.35: only useful for cost-sensitive deployments
>0.35: likely not good enough
```

North-star result:

```text
1.75–1.90 expert bpw with KL <= 0.12 and small downstream degradation.
```

Exceptional result:

```text
~1.6 expert bpw with KL <= 0.15.
```

---

## 18. Common failure modes

### 18.1 Binary residual too biased

Symptoms:

- KL > 0.3
- routed output mean/variance shifts heavily
- BCOS helps but not enough

Fixes to test:

- Increase rescue percentage.
- Increase shared rank.
- Use 2-bit instead of 1-bit for `down_proj` first.
- Keep last few MoE layers at 2-bit or higher.

### 18.2 Rare experts break

Symptoms:

- Mean KL acceptable but p99 KL high.
- Certain prompts/tasks fail catastrophically.

Fixes:

- Ensure cold experts get minimum rescue budget.
- Do not weight expert importance too aggressively by frequency.
- Add per-expert minimum: at least 5% row-blocks upgraded to 2-bit for every expert.

### 18.3 Shared subspace overhead too high

Symptoms:

- bpw > target.

Fixes:

- Store shared factors in FP8.
- Reduce rank from n/128 to n/256.
- Prefer mixed-bit rescue over grouped subspaces.

### 18.4 Hadamard hurts instead of helps

Symptoms:

- A1 worse than no rotation.

Checks:

- Confirm normalized Hadamard implementation.
- Confirm inverse rotation on activations uses \(Q_b^T\).
- Confirm scale is mean-absolute weighted, not amax.
- Try h=32 and h=128.

---

## 19. Initial implementation priorities

Implement in this exact order:

1. Full reference path that reconstructs quantized weights and verifies numerical equivalence.
2. Router-weighted KLT/SVD shared decomposition.
3. Hadamard 1-bit residual quantization.
4. Mixed 2/4-bit rescue.
5. Routed MoE-output correction.
6. KL/PPL evaluation harness.
7. Fused kernels/performance optimization.
8. Grouped subspaces ablation.

Do not start with QJL residual sketches. They are mathematically interesting but add inference complexity. Try mixed-bit rescue first.

---

## 20. Deliverables

For each model tested, provide:

1. Quantized checkpoint/artifact.
2. Effective expert bpw and full-model GB.
3. KL report with mean/p95/p99.
4. PPL report.
5. Downstream task report.
6. Ablation table from Section 14.
7. Layer diagnostic table.
8. Throughput/memory benchmark once fused kernels exist.

Minimum first milestone:

```text
Model: Qwen1.5-MoE or Mixtral-style MoE
Config: rcq_1p75
Report: effective bpw, KL, PPL, 4 downstream tasks
Comparison: existing 2-bit and 3-bit baselines if available
```
