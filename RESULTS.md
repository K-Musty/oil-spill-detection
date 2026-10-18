# Benchmark Results and Post-Hoc Reanalysis

**Status:** This document supersedes the results reported in the original manuscript draft and in the first version of this README. It records what was measured, what was claimed, what failed to reproduce, and what the data actually supports.

**Date:** October 2026
**Author:** Abdulrahman Kalli Mustapha
**Related artifacts:** [README](README.md) · [GitHub repo](https://github.com/K-Musty/oil-spill-detection) · [Live app](https://oil-spill-detection-app.streamlit.app/)

---

## TL;DR

A comparison of U-Net, DeepLabV3, and FPN on bounding-box-derived oil spill masks in Sentinel-1 SAR imagery originally concluded that **DeepLabV3 outperformed the other two by ~7 IoU points, attributed to its ASPP module**. A rerun of the experiments under more honest conditions showed:

1. **The DeepLabV3 advantage was a resolution artifact.** At matched 256×256 input and matched optimizer settings, DeepLabV3 is the **worst** of the three models (44.75% test IoU), below U-Net (52.30%) and FPN (47.52%).
2. **Run-to-run seed variance on U-Net was 5.40 test IoU points** — nearly as large as the original reported architecture gap.
3. **The original "U-Net overfits after epoch 35" claim was not reproduced.** The rerun's U-Net was still improving at the final epoch.
4. **No ASPP ablation was ever run**, so the ASPP explanation was speculative even in the original draft.

The pipeline, the deployment, and the reproducibility effort are real. The architecture comparison, as originally framed, is not. This document explains why, and what the data does support.

---

## 1. Background and original claims

The original study benchmarked three segmentation architectures sharing a ResNet-34 backbone on the Eastern Mediterranean Oil Slick Dataset (Yang & Singha, 2025; PANGAEA). Training used 955 oil-containing patches, validation on 205, test on 205, with a 0.5·CE + 0.5·Dice compound loss, AdamW optimizer, and early stopping with patience 10.

The original manuscript reported:

| Model | Val IoU | Test IoU | Test Dice | Best epoch |
|---|---|---|---|---|
| U-Net | 51.70% | 46.90% | 62.60% | 35 |
| FPN | 51.80% | 46.35% | 62.02% | 10 |
| DeepLabV3 | **59.51%** | **53.76%** | **68.12%** | 48 |

And drew the following conclusions:

- DeepLabV3 outperforms U-Net and FPN by 7.81 and 7.71 percentage points on validation IoU.
- The performance difference is attributable to DeepLabV3's Atrous Spatial Pyramid Pooling (ASPP) module.
- U-Net overfits after epoch 35, with validation IoU declining while training loss falls toward zero.
- Changing the architecture provides <8% improvement, so data is the bottleneck.

Each of these conclusions was treated as established in the original draft. None survived re-examination.

---

## 2. What was rerun, and why

The reanalysis was triggered by inspection of the original training logs, which revealed two protocol facts not fully disclosed in the manuscript:

1. **DeepLabV3 was trained at 512×512 input**, while U-Net and FPN were trained at 256×256.
2. **DeepLabV3 used batch size 4 and learning rate 5e-5**, while U-Net and FPN used batch size 8 and learning rate 1e-4.

These are not minor details. Changing input resolution alters the effective object size relative to ASPP's dilation rates (r = {6, 12, 18}), and changing batch size and learning rate alters the entire optimization trajectory. The comparison, as originally run, was **not controlled for either resolution or optimizer settings**.

Three reruns were performed:

| Rerun | Purpose |
|---|---|
| DeepLabV3 at 256×256, batch 8, lr 1e-4 | Match U-Net and FPN protocol exactly |
| U-Net at 256×256, batch 8, lr 1e-4 | Reproduce original U-Net run to measure seed variance |
| FPN at 256×256, batch 8, lr 1e-4 | Attempted fair-budget rerun (see §4.4 — the patch failed silently) |

All reruns used the same data split, the same loss, the same backbone, and the same random seed state (which is to say, none: the scripts do not expose a `--seed` flag, so seeds vary across runs).

---

## 3. Full results

Six runs are reported. Each row is a complete training run from random initialization.

| # | Model | Input | Batch | LR | Patience | Best epoch | Early-stop epoch | Best Val IoU | Test IoU | Test Dice |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | U-Net (orig) | 256 | 8 | 1e-4 | 10 | 35 | 45 | 51.71% | 46.90% | 62.62% |
| 2 | U-Net (rerun) | 256 | 8 | 1e-4 | 10 | 50 | — (ran to 50) | **54.40%** | **52.30%** | 67.55% |
| 3 | FPN (orig) | 256 | 8 | 1e-4 | 10 | 10 | 20 | 51.80% | 46.35% | 62.02% |
| 4 | FPN (rerun) | 256 | 8 | 1e-4 | 10 | 15 | 25 | 50.36% | 47.52% | 62.67% |
| 5 | DeepLabV3 (orig) | **512** | **4** | **5e-5** | 10 | 48 | — (ran to 50) | **59.51%** | **53.76%** | **68.12%** |
| 6 | DeepLabV3 (rerun) | **256** | **8** | **1e-4** | 10 | 27 | 37 | 51.08% | **44.75%** | 60.20% |

Runs 1 and 3 are the values originally reported in the manuscript. Runs 2, 4, and 6 are the reanalysis. Run 5 is the original DeepLabV3 result preserved as a resolution ablation.

---

## 4. Findings

### 4.1 The resolution effect dominates the architecture effect

The original DeepLabV3 result (53.76% test IoU) was obtained at **512×512 input**. The rerun at **256×256 input** with matched optimizer settings produced **44.75% test IoU** — a **9.01-point drop**.

At matched 256×256, the ranking is:

1. U-Net: 52.30%
2. FPN: 47.52%
3. DeepLabV3: 44.75%

DeepLabV3 is the **worst** model at matched resolution. The original conclusion — that DeepLabV3 wins "due to ASPP" — is not supported. What the original experiment measured was the effect of training at higher resolution, not the effect of ASPP.

This is consistent with how ASPP works. ASPP dilates convolutions at rates {6, 12, 18} to expand the receptive field. At 512×512 input, slicks occupy approximately 4× the pixel area relative to those dilation rates compared to 256×256. The module is effectively tuned to a particular scale regime, and the original experiment happened to sit inside it.

A clean attribution would require a 2×2 design: {U-Net, DeepLabV3} × {256, 512}, with all other hyperparameters held fixed. That design was not run. Until it is, the honest statement is: **resolution correlates strongly with the difference; ASPP is an untested hypothesis.**

### 4.2 Seed variance on U-Net is 5.40 test IoU points

Runs 1 and 2 used **identical code, identical data, identical hyperparameters, and identical patience**. The only difference is random initialization and the resulting training trajectory.

- Run 1: test IoU 46.90%, best epoch 35, early-stopped at 45.
- Run 2: test IoU 52.30%, best epoch 50, ran to the epoch limit.

**The swing is 5.40 test IoU points.** For reference, the original manuscript's headline gap (DeepLabV3 vs U-Net) was 6.86 points. **Seed variance is ~79% of the size of the effect the original paper claimed to measure.**

FPN's two runs (runs 3 and 4) show a smaller swing of 1.17 points, though both were early-stopped by the same patience threshold, so this is a less clean variance estimate. DeepLabV3 was not run twice at matched settings, so its seed variance is unknown.

With n=1 or n=2 per configuration, no confidence interval or significance test is meaningful. The defensible claim is: **differences below ~5 IoU points on this dataset cannot be distinguished from run-to-run noise without multiple seeds.**

### 4.3 The U-Net overfitting claim was not reproduced

The original manuscript stated:

> "Beyond epoch 35, validation IoU declined while training loss continued to fall toward zero. This divergence represents a classic transition to memorization."

Examination of run 1's log shows:

- Validation IoU after epoch 35: 0.4274, 0.4728, 0.4897, **0.5138**, 0.4845, 0.4329, 0.4977, 0.5061, 0.5113, 0.4802. The value **rose back to 0.5138 at epoch 39**, close to the peak of 0.5171. It did not "decline" in any sustained sense; it fluctuated.
- Training loss after epoch 35: 0.4066 → 0.3713. It declined slowly, but not "toward zero." A loss of 0.37 on a 0.5·CE + 0.5·Dice objective is not near convergence.

And critically, **run 2 did not reproduce the phenomenon at all.** Its best validation IoU (0.5440) occurred at epoch 50, the final epoch, with the run still improving. There is no overfitting threshold at epoch 35 in run 2.

The "noise highway" explanation offered in the original Discussion — that U-Net's skip connections copy high-frequency speckle noise to the decoder — may or may not be mechanistically true, but it was not demonstrated. The evidence originally cited for it was a single run's late-epoch fluctuation, which the rerun does not reproduce.

### 4.4 The FPN "budget rerun" did not actually change the budget

The reanalysis plan included rerunning FPN with early-stopping patience raised to 20, to give it a fair optimization budget. The patch applied was:

```bash
sed -i 's/patience=10/patience=20/g' src/train.py
```

This patch had **no effect**. The string `patience=10` does not appear in `train.py`. The argparse default is written as `default=10`, and the `ReduceLROnPlateau` scheduler separately uses `patience=5`. The sed pattern matched neither.

As a result, run 4 used **the same patience as run 3**. Its best epoch was 15, and it was early-stopped at epoch 25 — the same patience-10 behavior as the original. Run 4 is therefore **a seed rerun of FPN**, not a budget-matched rerun.

The 1.17-point difference between FPN runs (46.35% → 47.52%) should be read as seed variance, not as evidence that FPN benefits from a longer budget. A genuine budget-matched FPN run remains outstanding.

---

## 5. What the data actually supports

- **Input resolution is a first-order factor** in this dataset. The DeepLabV3 512→256 drop of 9.01 test IoU points is larger than any architecture difference observed between matched-resolution runs.
- **Run-to-run variance is substantial.** The U-Net 5.40-point swing on identical scripts establishes a noise floor of at least this magnitude for single-run comparisons on this dataset.
- **At matched 256×256, none of the three architectures is clearly best.** U-Net (52.30%), FPN (47.52%), and DeepLabV3 (44.75%) are within a range that could plausibly be spanned by re-running each with different seeds.
- **The pipeline is reproducible in the sense that it can be rerun end-to-end**, but not in the stricter sense of seed-level reproducibility, since the scripts do not expose a `--seed` argument.

## 6. What the data does *not* support

- **That DeepLabV3 is superior for SAR oil spill segmentation.** At matched resolution it is the weakest of the three.
- **That ASPP is responsible for any observed performance differences.** No ASPP ablation was run.
- **That U-Net overfits after epoch 35.** The phenomenon did not reproduce.
- **That FPN needs a longer training budget.** The rerun did not actually extend the budget.
- **That any of these models generalize to look-alikes.** The test set contains only oil-positive patches. No no-oil evaluation was performed.
- **That the reported IoU values reflect boundary-accurate segmentation.** Ground-truth masks are derived from rectangular bounding boxes, not pixel-level annotations. Reported IoU measures region-level localization, not boundary precision.
- **That the dataset is representative.** One region (Eastern Mediterranean), one sensor (Sentinel-1 VV), one annotation protocol. No external validation.

---

## 7. Methodological lessons

These are offered in the hope that they save another practitioner a rerun:

1. **Report resolution and hyperparameters per model, not globally.** The original draft's claim of a "standardized 256×256 grid" was false for one of the three models. That single omission silently invalidated the main comparison.
2. **Run at least 3 seeds before claiming an architecture effect.** A 5-point swing on identical scripts is common in small-data segmentation. Without seeds, a 7-point gap is not evidence of an architecture difference.
3. **Never trust a `sed` patch without checking it matched.** `sed -i 's/patience=10/patience=20/'` produces no error and no effect when the pattern is absent. Always `grep` the file after patching.
4. **Distinguish "resolution ablation" from "architecture comparison" when reporting.** A model trained at 512 input is not comparable to a model trained at 256 input unless that is the variable being studied.
5. **Build the no-oil evaluation early.** A segmentation model that has never been shown a negative example should not be described as having a "false positive rate," let alone a low one.
6. **Bounding-box masks are not segmentation masks.** A model that performs well against box-derived masks may have learned to fill rectangles, not trace boundaries. Say so in the limitations, not in a footnote.

---

## 8. Limitations of this reanalysis

- **No formal statistical analysis.** With one or two runs per configuration, no confidence interval, significance test, or variance decomposition is meaningful. The reported ranges are descriptive, not inferential.
- **DeepLabV3 seed variance is unknown.** Only one 256×256 run and one 512×512 run exist. Any statement about the reliability of DeepLabV3's result is bounded by this.
- **FPN budget fix was not applied.** See §4.4. A genuine budget-matched FPN run remains outstanding.
- **Resolution × architecture design was not factorial.** The 512×512 DeepLabV3 run was not paired with a 512×512 U-Net or FPN run, so the resolution effect cannot be cleanly separated from the model effect at higher input sizes. Only the 256×256 comparison is like-for-like.
- **The original U-Net checkpoint (run 1) was overwritten** during the rerun. Its weights are not recoverable; only its logged metrics survive. This is a reproducibility failure, disclosed here rather than concealed.
- **No no-oil evaluation.** `data/processed/nooil.npz` was never built. This blocks any false-positive or look-alike analysis.

---

## 9. Reproducibility notes

- **Seeds:** not exposed in the training scripts. Runs differ by random initialization and by trajectory. Adding `--seed` to `train.py` and `train-deeplab.py` is a 3-line change and is recommended for any follow-up work.
- **Split:** random at patch level. Scene- or date-grouped splits were not used, so mild leakage between train and test cannot be ruled out.
- **Loss:** 0.5 · weighted cross-entropy + 0.5 · Dice. Class weights [0.128, 1.872] computed from the training split.
- **Backbone:** ResNet-34, ImageNet-pretrained via `segmentation-models-pytorch`.
- **Channel handling:** single-channel SAR replicated to 3 channels to match the ImageNet input shape.
- **Hardware:** Google Colab free-tier T4 GPU.
- **Environment:** PyTorch 2.5.1, Python 3.10, `segmentation-models-pytorch` (version as pinned in `requirements.txt`).

Full training logs for all six runs are available in the repository under `logs/` and in the commit history of the Colab notebook.

---

## 10. Conclusion

The original paper's claim — that DeepLabV3 outperforms U-Net and FPN on this task because of its ASPP module — does not survive a controlled rerun. At matched input resolution and matched optimizer settings, DeepLabV3 is the weakest of the three models, and U-Net's run-to-run variance is nearly as large as the gap the original paper attributed to architecture.

What survives is a working pipeline, a public dataset, a deployed demo, and a set of measured numbers that future work can build on. What does not survive is the causal claim. This document exists so that the repository tells the truth, even where the original draft did not.

If you use this code, please cite the software and dataset, and treat the architecture comparison as a **cautionary benchmark**, not a recommendation.

---

*Last updated: October 2026. This document is versioned alongside the code. Corrections welcome via issue or pull request.*
