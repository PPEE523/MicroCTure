# MicroCTure: Structure Recognition, Reconstruction, and Visualization of Micro-C Contact Maps

[中文](README.md) · [Interactive visualization](site/index.html) · [Technical design](docs/PROJECT.md) · [Results](docs/RESULTS.md) · [Reproduction guide](docs/REPRODUCIBILITY.md)

## Abstract

MicroCTure is a reproducible analysis project for *Escherichia coli* Micro-C contact maps. It integrates sparse-data processing, spatially separated evaluation, supervised structure classification, unannotated-pattern retrieval, condition comparisons, synthetic super-resolution, and whole-genome three-dimensional contact embedding.

Using public GSE272159 data, the project evaluates models against simple baselines, across random seeds and biological replicates, and through independent numerical checks. The primary dual-scale classifier achieves a test macro-F1 of **0.501 ± 0.099**. Residual CNN super-resolution achieves **34.332 ± 0.172 dB** PSNR on the held-out rep1 set, compared with **24.238 dB** for bicubic interpolation. A graph decoder reduces held-out log-distance RMSE from **0.461** for MDS to **0.307**. However, improved image quality does not improve boundary localization, the graph decoder retains lower contact correlation than MDS, and no new biological structure type has been confirmed.

The contribution is an integrated implementation and evaluation workflow. The repository does not claim that its CNN, autoencoder, or MDS components constitute a new general-purpose architecture.

**Keywords:** Micro-C; genome contact maps; spatial evaluation; convolutional neural networks; autoencoders; super-resolution; 3D reconstruction.

## 1. Research objectives

| Module | Research question | Outputs |
|---|---|---|
| Structure classification | Can local matrices distinguish OPCID, CHIN, and CHID? | Baselines, CNNs, ablations, class-level evaluation |
| Candidate retrieval | Which unannotated patterns reproduce, and how do they relate to known structures? | Coordinates, clustering, replicate checks, morphology audits |
| Genome-track visualization | How do signals align with genes, annotations, and conditions? | 465 genome-wide track pages |
| Condition comparisons | How do known structures change between experimental conditions? | 688 descriptive comparisons and heatmaps |
| Super-resolution | Can coarse inputs recover finer signals and local structure? | Reconstructed matrices, model weights, image and boundary metrics |
| 3D contact embedding | How well can contact frequencies constrain relative genomic positions? | Coordinates, reconstructed contact maps, baselines, interactive views |

## 2. Data and preprocessing

### Dataset

Data originate from [GSE272159](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE272159), associated with [Elementary 3D organization of active and silenced E. coli genome](https://doi.org/10.1038/s41586-025-09396-y).

| Property | Value |
|---|---|
| Reference | *E. coli* MG1655, `NC_000913.3` |
| Chromosome length | 4,641,652 bp |
| Original resolution | 10 bp; 464,166 bins |
| WT sparse records | 257,522,134 and 249,673,544 in two replicates |
| Conditions | WT, ΔstpA, and ΔhnsΔstpA; two replicates each |
| Structure annotations | 344: 68 OPCID, 250 CHIN, 26 CHID |
| Gene-track records | 4,516; not asserted to be a complete reference annotation |

Sparse records denote stored nonzero pixel entries, not sequencing reads or summed contacts. Input provenance and acquisition are documented in [data notes](data/README.md).

### Matrix processing

A dense float32 representation of the original whole-genome matrix would require approximately 862 GB in decimal units. Processing therefore uses chunked HDF5 reads, sparse aggregation, local contact-band caches, and on-demand windows. Whole-genome dense matrices are constructed only after aggregation to 5 kb for the 3D experiment.

The pipeline supports circular coordinates, cross-origin windows, valid-pixel masks, and explicit handling of incomplete terminal bins. Depending on the analysis, signals undergo library-depth normalization, `log1p` transformation, or observed/expected correction by genomic separation. Expected-contact estimates include zero-count opportunities. Depth normalization and distance correction address different effects and are not treated as interchangeable.

### Spatial partitions

Classification and super-resolution use the same structure-level split:

| Split | OPCID | CHIN | CHID | Total |
|---|---:|---:|---:|---:|
| Training | 47 | 175 | 18 | 240 |
| Validation | 11 | 36 | 4 | 51 |
| Test | 10 | 39 | 4 | 53 |

Genomic footprints are grouped before cropping or augmentation, with a 1 kb guard between partitions. Replicates and augmented views of the same structure retain the same split. Model-specific fitted preprocessing uses training regions; checkpoints are selected on validation data.

Classification averages replicate probabilities before structure-level evaluation. Super-resolution trains and selects checkpoints using rep1 only, then evaluates both replicates at test positions. Discovery uses a separate genomic-block partition, while 3D embedding holds out contact edges. These protocols do not establish new independent biological validation of previously inspected data.

## 3. Methods

### 3.1 Classification and ablation

Local matrices at 100 bp define 6.4 kb and 25.6 kb contexts, each converted to a 64 × 64 tensor containing signal and valid-pixel fraction. The primary model uses a shared encoder for the two scales and combines their representations in a classification head.

The encoder has 16/32/64 convolutional channels, GroupNorm, ReLU, and pooling. Training uses AdamW, training-derived class weights, and validation structure-level macro-F1 for checkpoint selection. Macro-F1 is emphasized because the classes are imbalanced.

| ID | Model or modification |
|---|---|
| B0 | Training-majority prediction |
| B1 | Standardized annotation length and local density with logistic regression; length is privileged information |
| B2 | Large-window O/E, training-fitted PCA(16), and logistic regression |
| M1 / A2 | Large-scale CNN; the A2 comparison reuses M1 |
| M2 | Shared-encoder dual-scale CNN |
| A1 | Small-scale input only |
| A3 | Depth-normalized input instead of O/E |
| A4 | Uniform class weights |
| A5 | No explicit mask channel; invalid signal pixels remain zero |

Six distinct CNN settings are trained with seeds 42, 43, and 44, giving 18 runs. Attribution and occlusion analyses inspect decision cues without establishing a biological mechanism.

### 3.2 Candidate discovery and morphology auditing

Multiscale diagonal-window scanning compares shape scores, autoencoder (AE), denoising autoencoder (DAE), and fused scores. The second round contains six AE/DAE training runs, 14 scoring configurations, and 168 clustering runs across four cluster counts and three clustering seeds.

Methods are compared under matched candidate budgets. Known structures remain in the search for retrieval evaluation. Overlapping detections are consolidated into independent loci; corresponding positions in the other replicate are evaluated using masked shape agreement and matched background controls.

Subsequent auditing applies the same fixed recentering, scale, translation, and occlusion interventions to candidates and known references. The final review includes 21 independent candidates and 63 reference comparisons. Unannotated status, reproducibility, statistical isolation, and a new biological type are treated as distinct evidence levels.

### 3.3 Genome tracks and condition comparisons

Six samples are aggregated to 100 bp and evaluated on common valid bins. Tracks summarize contacts at separations of 200 bp–10 kb, normalized by library size and available partners. The 465 consecutive 10 kb pages include replicate signals, condition means, gene positions and strands, and structure annotations.

![Example genome tracks across three conditions](showcase/results/task3/tracks/0050000_0060000.png)

*Figure 1. A fixed 10 kb interval. Display transforms do not enter effect-size calculations; axes may differ between track pages.*

Known structures are compared between each treatment and WT, yielding 344 × 2 = 688 comparisons. The main effect is

$$
\Delta=\log_2\frac{\overline{s}_{\mathrm{treated}}+0.01}{\overline{s}_{\mathrm{WT}}+0.01}.
$$

All four treatment-versus-WT replicate ratios must exceed 1.5, or fall below 1/1.5, for a consistent directional change. Quality requires at least 20 valid pairs and 80% valid-pair coverage. These are descriptive rules, not p-value or FDR tests; matching replicate numbers across conditions does not imply pairing.

### 3.4 Synthetic super-resolution

A fixed 25.6 kb window is aggregated into a 128 × 128 target at 200 bp, then reduced to a 32 × 32 input at 800 bp by combining numerical pixels. This is fourfold synthetic resolution enhancement within Micro-C, not an evaluated Hi-C-to-Micro-C translation.

Bilinear and bicubic interpolation are compared with three-layer residual and non-residual CNNs. Both CNN families run three seeds with matched splits, learning rates, training budgets, and masked losses. Training lasts at most 30 epochs, with validation-based checkpoint selection. The residual model predicts corrections to an upsampled input; outputs are symmetrized.

Evaluation includes fixed-range [0,1] PSNR, local 7 × 7 SSIM on fully valid neighborhoods, distance-stratified errors, and known-boundary profile displacement. Upper-triangle scoring excludes separations below 400 bp. Boundary scoring uses fixed neighborhoods around annotations and is not a blind genome-wide detection metric. The CNN families also differ in final-layer initialization, so their comparison concerns the complete residual training setup.

![Super-resolution shape and boundary profiles](showcase/results/task5/structure_profiles.png)

*Figure 2. The first test example from each class, selected independently of model performance. Heatmaps subtract training-derived distance background; profiles compare observations, interpolation, and a fixed CNN seed.*

### 3.5 Whole-genome 3D embedding

Both WT replicates are aggregated to 5 kb, producing 929 nodes. Coarsening conserves upper-triangle counts; exposure is corrected for the incomplete terminal bin.

Non-neighboring rep1 contact edges are split into training, validation, and test sets. Rep2 does not enter model fitting or selection. Target distances follow the fixed assumption

$$
d_{ij}^{\mathrm{target}}=\frac{(C_{ij}+0.5)^{-1/3}}{s_{\mathrm{train}}},
$$

where the denominator is the median transformed training-edge distance. No physical spatial unit is calibrated.

Three approaches are compared: classical MDS using training-only distance-background imputation; direct coordinate optimization initialized from perturbed MDS coordinates; and a residual graph decoder combining MDS initialization, circular-position features, 32-dimensional trainable node embeddings, and training-contact graph aggregation.

The latter two methods each run three seeds and optimize log-distance error with a circular-neighbor penalty. Evaluation includes held-out distance RMSE, reconstructed-contact correlations, within-separation correlation, and seed agreement. Approximate macrodomain labels are used only for display and post-hoc assessment.

![Whole-genome MDS, geometry, and graph embeddings](showcase/results/task6/structures.png)

*Figure 3. Relative 3D embeddings. Domain colors are schematic and circular links reflect a prior; the visualization does not establish a unique single-cell conformation.*

## 4. Quantitative results

### Classification

| Model | Test macro-F1 | Test accuracy |
|---|---:|---:|
| B0 | 0.283 | 0.736 |
| B1 | 0.462 | 0.566 |
| B2 | 0.424 | 0.623 |
| M1 | 0.340 ± 0.041 | 0.516 |
| M2 | **0.501 ± 0.099** | 0.667 |
| A1 | 0.371 ± 0.022 | 0.535 |
| A3 | 0.433 ± 0.077 | 0.522 |
| A4 | 0.424 ± 0.111 | 0.679 |
| A5 | 0.332 ± 0.027 | 0.484 |

M2 has the highest mean macro-F1 in this comparison, with substantial seed variation. Only four CHID structures are in the test split. The majority baseline's high accuracy illustrates why accuracy alone is insufficient. B1 has access to annotation length and is not an equal-information matrix-only baseline.

[Classification table](showcase/results/task1/comparison.csv)

### Super-resolution

| Method | Rep1 PSNR / dB ↑ | Rep1 SSIM ↑ | Rep2 PSNR / dB ↑ | Rep1 boundary displacement / bp ↓ |
|---|---:|---:|---:|---:|
| Bilinear | 23.382 | 0.925 | 23.934 | 196.2 |
| Bicubic | 24.238 | 0.934 | 24.779 | **177.4** |
| Residual CNN | **34.332 ± 0.172** | **0.954** | **34.193 ± 0.129** | 227.0 |
| Non-residual CNN | 33.908 ± 0.483 | 0.947 | 33.742 ± 0.427 | 235.2 |

Image metrics improve without a corresponding boundary-localization improvement. Post-training diagnostics show that gains concentrate near the diagonal rather than uniformly across genomic separations.

[Per-class results](showcase/results/task5/summary.csv) · [Distance diagnostics](showcase/results/task5/distance_diagnostics.csv)

### 3D embedding

| Method | Rep1 log-distance RMSE ↓ | Rep2 log-distance RMSE ↓ | Rep1 contact log-Pearson ↑ | Rep1 within-distance correlation ↑ |
|---|---:|---:|---:|---:|
| MDS | 0.461 | 0.461 | **0.919** | **0.826** |
| Geometry optimization | 0.309 | 0.308 | 0.852 | 0.637 |
| Residual graph decoder | **0.307** | **0.306** | 0.851 | 0.627 |

Optimized embeddings reduce the distance objective but lower contact correlation relative to MDS. Lower geometric loss therefore does not establish uniformly better reconstruction.

[Full metrics](showcase/results/task6/metrics.csv) · [Reconstructed maps](showcase/results/task6/contact_maps.png)

### Discovery and condition effects

The second-round shape baseline selects 189 windows, of which 82 pass the replicate criterion and 21 are unannotated. Pure AE/DAE residual scores select no windows passing the criterion. Subsequent morphology auditing does not confirm a new structure type.

For known structures, 10 ΔstpA and 94 ΔhnsΔstpA comparisons meet the descriptive change rules. These are relative, library-normalized contact effects, not absolute contact increases, structural assembly, or causal conclusions.

[Discovery comparisons](showcase/results/task2_v2/method_summary.csv) · [Condition summary](showcase/results/task4/summary.csv)

## 5. Reproducibility and software validation

- Configurations define partitions, models, and decision rules; experiment manifests record source and input signatures.
- The project contains 36 training/fitting runs: 18 classification CNNs, six AE/DAEs, six super-resolution CNNs, and six 3D optimizations. Clustering runs are counted separately.
- Independent checks recompute metrics from saved matrices and coordinates and recover 12 super-resolution/3D checkpoints to verify outputs.
- The 46 synthetic-data tests cover coordinates, masking, count conservation, split isolation, losses, gradients, and geometry without downloading research data.
- The 23 selected output files have a [SHA256 manifest](showcase/results/manifest.json).

See [numerical verification](reports/task56_verification.json) and [condition checks](reports/task34_verification.json). Reported ± values describe training-seed standard deviation, not biological confidence intervals. GitHub Actions is configured; remote success depends on actual workflow runs.

## 6. Usage

### Browse interactive results

```bash
git clone https://github.com/PPEE523/MicroCTure.git
cd MicroCTure
python3 -m http.server 8000
# Open http://localhost:8000/site/
```

The [interactive page](site/index.html) provides a super-resolution comparison slider and seven switchable, rotatable 3D models. No original matrices or ML dependencies are needed. GitHub's file viewer does not execute HTML.

### Install and test

Recorded environment: Linux, Python 3.14, CPU PyTorch.

```bash
python3.14 -m venv .venv
.venv/bin/python -m pip install -r requirements-training.txt --extra-index-url https://download.pytorch.org/whl/cpu
make test
make check
```

For full experiments, follow [input preparation](data/README.md) and the [ordered reproduction guide](docs/REPRODUCIBILITY.md). Raw matrices, weights, caches, full runtime outputs, and detailed phase reports remain local; ignore rules do not delete those files.

### Repository organization

| Directory | Contents |
|---|---|
| [scripts/](scripts/) | Preprocessing, models, reporting, independent checks, visualization builds |
| [configs/](configs/) | Model parameters, partitions, evaluation rules, export selection |
| [tests/](tests/) | Synthetic mathematical and data-integrity checks |
| [data/](data/README.md) | Input documentation, metadata, small derived annotations |
| [reports/](reports/README.md) | Selected machine-readable results and verification summaries |
| [showcase/](showcase/README.md) | Selected tables, figures, and 3D HTML |
| [site/](site/) | Interactive visualization page |
| [docs/](docs/README.md) | Design, results, and reproduction documentation |

## 7. Limitations and future work

Annotation conventions remain provisional, and minority-class sample sizes limit evaluation stability. Later experiments reuse previously inspected WT data and remain exploratory. Local diagonal scanning does not cover every possible long-range interaction, and unannotated reproducibility is insufficient to establish novelty.

Synthetic bin aggregation is not a validated model of all sequencing or cross-modality effects. Population-average contacts and a fixed frequency–distance law cannot uniquely identify single-cell geometry. Circular constraints and domain colors must be distinguished from observational evidence. Condition comparisons have only two replicates per condition and are descriptive rather than significance-tested.

Independent datasets, realistic sequencing degradation, cross-condition evaluation, external spatial constraints, and uncertainty estimation are future research directions, not completed claims of this repository.

## 8. Attribution and license

Data: [GSE272159](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE272159). Associated study: [Elementary 3D organization of active and silenced E. coli genome](https://doi.org/10.1038/s41586-025-09396-y).

This is an independent reanalysis and implementation, not the original authors' official code. Code is distributed under [MIT](LICENSE); third-party data and annotations retain their own terms. See [CITATION.cff](CITATION.cff) and [CONTRIBUTING.md](CONTRIBUTING.md).
