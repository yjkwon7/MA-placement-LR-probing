# Supplementary experiments

Additional results for the paper on movable-antenna (MA) placement from
reconstructed spatial channel maps. These experiments did not fit in the
manuscript; they are collected here so the paper can point to them.

This repository hosts the results only. The simulation and training code is not
released.

---

## 1. What the paper does

A base station has a small aperture over which two antennas can be positioned
freely. The received SNR depends on where they sit, so choosing the positions
requires knowing the spatial channel over the whole aperture — which would
normally mean probing every candidate location.

The paper avoids that with a two-stage pipeline:

1. **Reconstruction.** The aperture is probed at only `M` locations. A
   super-resolution network reconstructs the full 128 × 128 SNR map from those
   `M` observations. With `M = 1024` this is 6.25 % of the grid; with `M = 64`
   it is 0.39 %.

2. **Placement.** A placement network (PO) reads the reconstructed map and
   emits continuous antenna coordinates in one forward pass. A short
   gradient-based refinement then polishes them, and a feasibility correction
   guarantees the minimum-spacing constraint is met.

The claim is that this reaches near-exhaustive-search placement quality at a
small fraction of the probing and search cost.

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

**Reference and regret.** Several figures report *placement regret*: the SNR of
a greedy exhaustive search over the discrete grid, minus the SNR the method
achieves, on the same reconstructed map. Lower is better; zero means the
learned placement matched an exhaustive search that evaluated all 16 384 points
per antenna.

---

## 2. Experiments

### 2.1 Placement regret and top-5 % hit rate versus observation budget

How much does the placement stage give up against an exhaustive grid search,
and how often does it land in the strongest 5 % of the aperture?

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
and it raises the top-5 % hit rate from roughly 0.7 to roughly 1.0. The
refinement cannot recover this on its own: random starts converge to
substantially worse local optima. This is the direct evidence that the learned
stage contributes something the local optimizer does not.

### 2.3 Which reconstruction method feeds the placement

The placement network is held **fixed** — the same trained network for every
bar — and only the reconstruction feeding it is swapped. Any difference is
therefore attributable to the reconstruction alone. `Nearest` and `Bicubic` are
the no-learning condition: the sparse observations are simply upsampled.

![Placement SNR per reconstruction method](docs/figures/fig_abl_recon.png)

| `M` | Nearest | Bicubic | EDSR | RDN | RCAN | Proposed |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | −10.086 | −9.875 | −7.978 | −7.996 | −8.014 | −7.987 |
| 256 | −9.251 | −8.982 | −7.092 | −7.086 | −7.011 | −7.084 |
| 1024 | −7.904 | −7.630 | −6.843 | −6.845 | −6.835 | −6.842 |

Learned reconstruction is worth 1.9–2.1 dB over interpolation at `M = 64` and
0.8–1.1 dB at `M = 1024`. Among the learned backbones the spread is 0.010 dB at
`M = 1024` — once the map is accurate enough, the backbone stops mattering and
the placement stage becomes the binding term. The dashed line is a fixed-position
array (no placement adaptation), about 6–7 dB below every adaptive scheme.

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
inside the flat region. Second, **`|S| = 8` is worse than `|S| = 4` at every
budget** despite costing twice as much. The larger set is not a superset; it
drops the heatmap-derived candidates in favour of symmetry transforms, and those
are the weaker starting points. The cheaper configuration is also the better one.

### 2.5 Sensitivity to the user-location model

The networks are trained with the user direction drawn at random per
realization. This evaluates them unchanged on a fixed user location, to check
that nothing has been memorized about the training geometry.

![Random versus fixed user location](docs/figures/fig_user_model.png)

| `M` | regret, random user [dB] | regret, fixed user [dB] |
|---:|---:|---:|
| 64 | 0.114 | 0.116 |
| 256 | 0.324 | 0.322 |
| 1024 | 0.271 | 0.259 |

Regret is unchanged to within 0.012 dB. The absolute SNR shifts slightly because
a fixed direction gives a marginally different channel, but the part the
placement stage is responsible for does not move.

### 2.6 Effect of the minimum-spacing penalty weight

The placement training objective carries a penalty `λ_d · P_d` on antenna pairs
closer than the minimum spacing. An earlier configuration used `λ_d = 0`, which
left the spacing constraint entirely to the inference-time feasibility
correction. The networks were retrained with `λ_d = 0.1`, selected on a
validation sweep; everything else was held fixed.

Scored on the reconstructed map, 5 000 realizations:

| `M` | | proposed | grid reference | sequential update |
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

The effect is largest at `M = 1024`: regret drops from 0.456 to 0.271 dB, and
the method moves from **losing** to the sequential-update baseline (−7.027
against −6.913) to **beating** it (−6.842 against −6.913). With `λ_d = 0.1` the
proposed placement is ahead of sequential update at all three observation
budgets.
