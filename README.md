# Supplementary experiments

Additional results for the paper on movable-antenna (MA) placement from
reconstructed spatial channel maps. This repository provides supplementary experimental results and reproducibility details that complement the results reported in the main manuscript.

The repository contains supplementary results and evaluation details only.
The simulation and training code is not released.

---

## 1. What the paper does

A base station has a small aperture over which two antennas can be positioned
freely. The received SNR depends on where they sit, so choosing the positions
requires knowing the spatial channel over the whole aperture — which would
normally mean probing every candidate location.

The paper avoids that with a two-stage pipeline:

1. **Reconstruction.** The aperture is characterized using only `M` finite-resolution spatial
   observations. A super-resolution network reconstructs the full 128 × 128 SNR
   map from these observations. With `M = 1024` this is 6.25 % of the grid; with `M = 64`
   it is 0.39 %.

2. **Placement.** A placement network (PO) reads the reconstructed map and
   emits continuous antenna coordinates in one forward pass. A short
   gradient-based refinement then polishes them, and a feasibility correction
   guarantees the minimum-spacing constraint is met.

The paper shows that the proposed method approaches the dense-grid placement
reference while using substantially fewer probing observations and online
objective evaluations.

### Notation used in the manuscript

| Symbol | Description |
|---|---|
| $K$ | Number of movable antennas (MAs) at the BS |
| $\mathcal{K}=\{1,\ldots,K\}$ | Set of MA indices |
| $L_x,L_y$ | Horizontal and vertical dimensions of the movement aperture |
| $H,W$ | Numbers of HR grid points along the aperture dimensions |
| $\mathcal{D}$ | Continuous MA movement aperture |
| $\mathcal{A}$ | $H\times W$ grid used for spatial channel representation |
| $\mathbf r_k$ | Continuous position of MA $k$ |
| $\mathbf R$ | $K\times2$ MA placement matrix |
| $d_{\min}$ | Minimum allowable inter-MA spacing |
| $d_{\mathrm{ap}}$ | Diagonal length of the movement aperture |
| $h(\mathbf r)$ | Complex channel coefficient at position $\mathbf r$ |
| $G(\mathbf r)$ | Channel power gain at position $\mathbf r$ |
| $\rho=P_{\max}/\sigma^2$ | Transmit-power-to-noise ratio |
| $\Gamma(\mathbf r)$ | Local received-SNR contribution at position $\mathbf r$ |
| $\boldsymbol{\Gamma}$ | HR spatial SNR map |
| $\boldsymbol{\Gamma}_{\mathrm{LR}}$ | Finite-resolution LR spatial probing map |
| $\tilde{\boldsymbol{\Gamma}}_{\mathrm{LR}}$ | Noisy LR spatial probing map |
| $\hat{\boldsymbol{\Gamma}}$ | Reconstructed HR spatial SNR map |
| $\hat{\Gamma}_{\mathrm c}(\mathbf r)$ | Continuous reconstructed SNR representation obtained by bilinear interpolation |
| $M$ | Number of spatial probing observations |
| $s$ | Spatial probing / super-resolution scale factor |
| $\delta_n$ | Standard deviation of additive probing noise in the linear SNR domain |
| $\gamma(\mathbf R)$ | Received SNR for placement $\mathbf R$ |
| $\eta(\mathbf R)$ | Achievable spectral efficiency for placement $\mathbf R$ |
| $F(\mathbf R)$ | True-channel placement objective |
| $\hat F(\mathbf R)$ | Reconstructed-map placement objective |
| $\varepsilon_{\mathrm{map}}$ | Uniform spatial-map approximation error |
| $\Delta_{\mathrm{opt}}$ | Residual placement-optimization error on the reconstructed spatial surface |
| $\lambda_d$ | Minimum-spacing penalty weight used for PO training |
| $\lambda_r$ | Repulsion-term weight used for PO training |
| $N_{\mathrm{ref}}$ | Number of inference-time continuous-refinement iterations |

### Setup common to every experiment below

| | |
|---|---|
| Aperture | 0.6 × 0.6 m, discretized to 128 × 128 (16 384 candidate points) |
| Carrier | 5 GHz |
| Movable antennas | `K = 2`, minimum spacing 18 mm |
| Observations | `M ∈ {64, 256, 1024}` |
| Evaluation | 5 000 held-out channel realizations |
| Placement training | spacing-penalty weight `λ_d = 0.1` |
| Inference refinement | `N_ref = 150 / 500 / 100` for `M = 64 / 256 / 1024`, no repulsion term |

**Reference and regret.** Several figures report *placement regret*: the SNR of the grid-based greedy
reference minus the SNR achieved by the proposed method, evaluated on the same
reconstructed map. Lower is better; zero means the learned placement matched the grid-based greedy reference, which
evaluates all 16 384 grid points at each antenna-placement step.

---

## 2. Experiments

### 2.1 Placement regret and top-5 % hit rate versus observation budget

How much does the placement stage give up relative to the grid-based greedy
reference, and how often does it land in the strongest 5 % of the aperture?

![Placement regret and top-5% hit rate](docs/figures/fig_po_regret.png)

| `M` | placement SNR [dB] | regret [dB] | top-5 % hit rate |
|---:|---:|---:|---:|
| 64 | −7.987 | 0.114 | 1.000 |
| 256 | −7.084 | 0.324 | 0.987 |
| 1024 | −6.842 | 0.271 | 0.990 |

Regret stays below 0.33 dB at every budget while the placement lands inside the
top 5 % of the aperture in 98.7–100 % of realizations. Regret is *larger* at
`M = 256` than at `M = 64` because the reference improves with a better
reconstruction faster than the learned placement does — at `M = 64` the
reconstruction is coarse enough that exhaustive search has little left to win.

### 2.2 What the learned initialization is worth

The refinement is a local optimizer, so its result is determined by where it
starts. This replaces the learned starting point with uniformly random antenna
positions and runs **exactly the same refinement, for the same number of
steps**. Only the initialization differs.

![Learned versus random initialization](docs/figures/fig_abl_init.png)

| `M` | learned init [dB] | random init [dB] | difference | regret (learned → random) | top-5 % hit rate |
|---:|---:|---:|---:|---:|---:|
| 64 | −7.987 | −9.266 | **+1.279 dB** | 0.114 → 1.393 | 1.000 vs 0.668 |
| 256 | −7.084 | −8.372 | **+1.288 dB** | 0.324 → 1.612 | 0.987 vs 0.711 |
| 1024 | −6.842 | −8.256 | **+1.414 dB** | 0.272 → 1.685 | 0.990 vs 0.735 |

At an identical search budget the learned initialization is worth 1.3–1.4 dB,
and it raises the top-5 % hit rate from roughly 0.7 to roughly 1.0. Under the evaluated refinement budgets, random initialization converges to
substantially worse solutions, showing that the learned initialization provides
information that is not recovered by local refinement alone within the same
online search budget. This is the direct evidence that the learned
stage contributes something the local optimizer does not.

### 2.3 Reconstruction effects on downstream placement

The placement network is held **fixed** — the same trained network for every
bar — and only the reconstruction feeding it is swapped. This isolates the effect of the reconstructed input under a fixed PO model. `Nearest` and `Bicubic` are
the no-learning condition: the sparse observations are simply upsampled.

![Placement SNR per reconstruction method](docs/figures/fig_abl_recon.png)

| `M` | Nearest | Bicubic | EDSR | RDN | RCAN | Proposed |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | −10.086 | −9.875 | −7.978 | −7.996 | −8.014 | −7.987 |
| 256 | −9.251 | −8.982 | −7.092 | −7.086 | −7.011 | −7.084 |
| 1024 | −7.904 | −7.630 | −6.843 | −6.845 | −6.835 | −6.842 |

Learned reconstruction is worth 1.9–2.1 dB over interpolation at `M = 64` and
0.8–1.1 dB at `M = 1024`. Among the learned backbones, the spread is only 0.010 dB at `M = 1024`,
indicating that once the reconstructed maps reach sufficiently high fidelity,
the remaining backbone differences have negligible impact on downstream
placement performance. The dashed line is a fixed-position
array (no placement adaptation), about 6–7 dB below every adaptive scheme.

#### Greedy placement on each reconstructed map

The fixed-PO comparison above measures how the input reconstruction affects a common placement network. As a complementary reconstruction-only evaluation, we also perform the same grid-based greedy placement independently on each reconstructed HR map and evaluate the selected antenna coordinates on the corresponding true channel realization.

![Reconstruction-induced placement difference](docs/figures/fig_recon_placement.png)

The learning-based reconstruction methods substantially reduce the reconstruction-induced placement difference relative to bicubic interpolation across the considered observation budgets. At `M = 1024`, the downstream true-channel SNRs obtained from the learned reconstruction methods differ by only about 0.01 dB, indicating that once the reconstructed maps reach sufficiently high fidelity, the choice among the learned backbones has little impact on the subsequent placement result.

### 2.4 Refinement budget and restart set size

Two knobs control the inference-time cost: the number of refinement steps
`N_ref`, and the number of candidate restarts `|S|`. Filled markers are the
`|S| = 4` configuration used in the paper (best symmetry transform plus three
heatmap modes); open markers are `|S| = 8` (symmetry transforms only).

![Refinement and restart ablation](docs/figures/fig_abl_restart.png)

Regret in dB:

| `M` | `N_ref = 0` | 25 | 50 | default | 8 restarts |
|---:|---:|---:|---:|---:|---:|
| 64 | 0.281 | 0.162 | 0.132 | 0.114 | 0.155 |
| 256 | 0.809 | 0.395 | 0.361 | 0.324 | 0.432 |
| 1024 | 1.745 | 0.390 | 0.326 | 0.272 | 0.418 |

Two things follow. First, refinement is not optional but saturates quickly: at
`M = 1024` the network output alone gives 1.745 dB regret, 25 steps bring it to
0.390, and everything beyond that is a slow tail — the default budget sits well
inside the flat region. Second, **`|S| = 8` is worse than `|S| = 4` for all three observation budgets,** 
despite requiring twice as many candidate refinements. The larger set is not a superset; it
drops the heatmap-derived candidates in favour of symmetry transforms, and those
are the weaker starting points. The cheaper configuration is also the better one.

### 2.5 Sensitivity to the user-location model

The networks are trained with a randomly drawn user direction for each
realization. We evaluate the same trained models at a fixed user location to
assess sensitivity to the user-location distribution used during training.

![Random versus fixed user location](docs/figures/fig_user_model.png)

| `M` | regret, random user [dB] | regret, fixed user [dB] |
|---:|---:|---:|
| 64 | 0.114 | 0.116 |
| 256 | 0.324 | 0.322 |
| 1024 | 0.271 | 0.259 |

The placement regret changes by at most 0.012 dB between the two user-location
models, indicating negligible sensitivity of the placement stage to this change
in the user-location distribution. The absolute SNR shifts slightly because the
fixed user direction induces a marginally different channel distribution.

### 2.6 Sensitivity to the BS–user distance

The reconstruction and PO networks are trained at the nominal BS–user distance
`d_BU = 24 m` and are evaluated unchanged at other distances, without
retraining. This experiment examines whether the final placement procedure
remains stable when the large-scale channel gain differs from the training
condition.

![Sensitivity to the BS–user distance](docs/figures/fig_distance.png)

Across the eight evaluated distances and all three observation budgets, the
placement-stage SNR difference remains within approximately −0.38 to 0.30 dB
and does not increase systematically as the operating distance moves away from
the training condition. The distance-dependent variation is non-monotonic:
the difference remains close to zero for `M = 64` and becomes negative at
several larger distances for `M = 256` and `M = 1024`.

Negative values can occur because the grid-based greedy reference is restricted
to discrete HR-grid locations, whereas the proposed method refines the antenna
coordinates continuously. At some distances, continuous refinement therefore
identifies off-grid positions with higher true-channel received SNR than the
grid-restricted reference.

### 2.7 Effect of the minimum-spacing penalty weight

The placement training objective carries a penalty `λ_d · P_d` on antenna pairs
closer than the minimum spacing. An earlier configuration used `λ_d = 0`, which
left the spacing constraint entirely to the inference-time feasibility
correction. The networks were retrained with `λ_d = 0.1`, selected on a
validation sweep; everything else was held fixed.

Scored on the reconstructed map, 5 000 realizations:

| `M` | `λ_d` | proposed | grid reference | sequential update |
|---:|---|---:|---:|---:|
| 64 | `λ_d = 0` | −8.076 | −7.873 | −8.104 |
| | **`λ_d = 0.1`** | **−7.987** | −7.873 | −8.104 |
| 256 | `λ_d = 0` | −7.070 | −6.760 | −7.107 |
| | **`λ_d = 0.1`** | **−7.084** | −6.760 | −7.107 |
| 1024 | `λ_d = 0` | −7.027 | −6.571 | −6.913 |
| | **`λ_d = 0.1`** | **−6.842** | −6.571 | −6.913 |

The penalty is a training-time guidance term, not a feasibility mechanism — the
correction step guarantees feasibility either way, and both configurations show
a zero violation rate after it. What changes is how much correction is needed,
and therefore how far the corrected positions end up from what the network
intended.

The effect is largest at `M = 1024`, where the regret decreases from 0.456 to 0.271 dB. The proposed method changes from underperforming sequential update with `λ_d = 0` (−7.027 vs. −6.913 dB) to outperforming it with `λ_d = 0.1` (−6.842 vs. −6.913 dB). With `λ_d = 0.1`, the proposed placement outperforms sequential update at all three observation budgets.

### 2.8 Robustness to observation noise

All preceding experiments use the nominal observation-noise level employed during training. Here, the trained reconstruction and placement checkpoints are kept fixed while only the observation-noise level is changed at test time. This experiment therefore evaluates robustness to observation-noise mismatch without retraining.

Let `δ_n` denote the standard deviation of the additive observation noise and let `δ_{n,0} = 8e-4` denote the nominal training level. We define

`L_noise(a) = SNR_true(δ_{n,0}) − SNR_true(a · δ_{n,0})`,

where `a = δ_n/δ_{n,0}`. Positive values indicate degradation relative to the nominal condition. The noise level is reported as a ratio rather than in dB because the perturbation is added in the linear SNR domain.

All noise levels use the same 5 000 held-out channel realizations and the same underlying standard-normal noise samples, scaled only by `δ_n`. Parentheses below denote the half-width of the paired 95 % confidence interval.

| `δ_n/δ_{n,0}` | `M = 64` | `M = 256` | `M = 1024` |
|---:|---:|---:|---:|
| 0 | −0.059 (0.051) | −0.015 (0.015) | −0.001 (0.005) |
| 0.5 | −0.067 (0.045) | −0.012 (0.011) | +0.002 (0.005) |
| 1 | 0 | 0 | 0 |
| 2 | 0.220 (0.054) | 0.216 (0.022) | 0.021 (0.006) |
| 4 | 0.662 (0.067) | 0.804 (0.040) | 0.086 (0.011) |
| 8 | 1.402 (0.081) | 1.721 (0.057) | 0.230 (0.015) |

The highest probing density is markedly more robust to test-time noise mismatch. At `M = 1024`, doubling the observation-noise standard deviation causes only 0.021 dB loss, while even an eightfold increase results in only 0.230 dB degradation without retraining. The two lower-resolution settings exhibit substantially larger degradation under severe noise.

The ordering between `M = 64` and `M = 256` is not monotonic at high noise levels. This should not be interpreted as greater intrinsic robustness at `M = 64`, because the coarsest observation setting already loses substantial spatial information through finite-resolution aggregation under the nominal condition.
