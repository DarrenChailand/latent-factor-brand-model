# Results and Interpretation

This page explains the included demonstration result. It shows what the numerical outputs contain, how the factors were interpreted, and why the result should not yet be treated as consumer research.

## Data used

The input is a **6 brand × 39 attribute** count matrix built from language-model-generated descriptions.

**Brands:** Amazon, Apple, Google, Microsoft, Nvidia, and Samsung.

**Attributes:** AI assistant, AMOLED, autofocus, battery capacity, battery life, biometric, camera, charging, chip, chipset, compatibility, CPU, display, durability, Face ID, fingerprint, foldable, GPU, graphics, imaging, integration, MagSafe, monitor, OLED, performance, Quick Charge, resolution, security, sensor, smartphone, telephoto, Touch ID, touchscreen, user interface, update policy, usability, user experience, wearable, and zoom.

Selected columns from the input matrix are shown below. The complete machine-readable matrix is available at [`data/sample/brand_attribute_counts.csv`](data/sample/brand_attribute_counts.csv).

| Brand | GPU | Performance | Integration | Charging | Chip | Display | Foldable | AMOLED |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Amazon | 0 | 4 | 0 | 1 | 2 | 1 | 0 | 0 |
| Apple | 151 | 2,958 | 1,465 | 130 | 754 | 381 | 16 | 6 |
| Google | 99 | 2,922 | 972 | 139 | 316 | 371 | 23 | 7 |
| Microsoft | 0 | 0 | 4 | 0 | 0 | 0 | 0 | 0 |
| Nvidia | 2 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| Samsung | 95 | 3,052 | 300 | 221 | 392 | 1,056 | 161 | 48 |

## Data-quality check

| Brand | Total mentions | Nonzero attributes |
|---|---:|---:|
| Amazon | 11 | 5 |
| Apple | 10,677 | 35 |
| Google | 9,732 | 36 |
| Microsoft | 4 | 1 |
| Nvidia | 3 | 2 |
| Samsung | 11,445 | 37 |

The brands are severely imbalanced. Consequently, the factors below are useful for demonstrating the method but **not reliable estimates of brand perception**. The sparse Nvidia and Microsoft rows strongly influence the first two factors.

## Singular value decomposition output

The analysis applies positive pointwise mutual information and decomposes the resulting matrix as:

$$M \approx U\Sigma V^T$$

- $U$ gives each brand's numerical score on each factor.
- $\Sigma$ contains the factor strengths.
- $V$ gives each attribute's numerical score on each factor.

### Brand-factor matrix ($U$)

| Brand | Factor 1 | Factor 2 | Factor 3 |
|---|---:|---:|---:|
| Amazon | -0.0080 | 0.3103 | -0.6283 |
| Apple | -0.0799 | 0.3743 | -0.0089 |
| Google | -0.0025 | 0.1133 | -0.0229 |
| Microsoft | -0.0084 | 0.8232 | 0.4503 |
| Nvidia | -0.9967 | -0.0403 | 0.0035 |
| Samsung | -0.0023 | 0.2675 | -0.6340 |

Full file: [`results/svd_U_brands.csv`](results/svd_U_brands.csv)

### Factor strengths ($\Sigma$)

| Factor | Singular value | Information captured |
|---|---:|---:|
| Factor 1 | 4.1288 | 43.51% |
| Factor 2 | 2.5673 | 16.82% |
| Factor 3 | 2.4450 | 15.26% |
| **First three combined** | — | **75.59%** |

Full file: [`results/svd_S_singular_values.csv`](results/svd_S_singular_values.csv)

### Strongest attribute weights ($V$)

| Factor | Attributes with the largest absolute weights |
|---|---|
| Factor 1 | GPU (-0.9984), performance (-0.0423), autofocus (-0.0212), MagSafe (-0.0173) |
| Factor 2 | integration (0.8614), charging (0.2381), chip (0.2290), autofocus (0.1594) |
| Factor 3 | charging (-0.5147), integration (0.4487), chip (-0.3551), monitor (-0.2656) |

Full file: [`results/svd_V_attributes.csv`](results/svd_V_attributes.csv)

The signs are arbitrary: flipping every sign in one factor produces the same mathematical solution. Interpret the **size** of a weight and the contrast between its two sides, not “positive” as good and “negative” as bad.

## Preliminary interpretation

- **Factor 1 — Nvidia/GPU separation:** Nvidia and GPU dominate almost completely. This is mainly a sparse-row effect, not strong evidence of a broad market theme.
- **Factor 2 — Microsoft/integration separation:** Microsoft and integration dominate. Again, Microsoft's row contains only four mentions, all for integration.
- **Factor 3 — integration versus device hardware:** Microsoft/integration lies on one side, while Samsung and Amazon align with charging, display, chip, foldable, and AMOLED attributes on the other.

These names were added after examining $U$ and $V$; singular value decomposition itself produces only numbers.

## What can and cannot be concluded

The experiment shows that the software can transform a count matrix into PPMI scores and latent factors. It does **not** prove that the factors represent real consumer beliefs. A defensible study needs balanced response counts, human-labelled evaluation data, repeated runs, uncertainty estimates, and comparisons with raw-frequency and term frequency–inverse document frequency baselines.

## Reproduce the result

```bash
python run_pipeline.py
```

This creates all files under [`results/`](results/), including the complete PPMI matrix and singular value decomposition matrices.
