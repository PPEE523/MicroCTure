# MicroCTure

**A scientific ML pipeline for genome contact maps — from sparse data to interactive 3D.**

[中文](README.md) · [Interactive showcase](site/index.html) · [Technical design](docs/PROJECT.md) · [Results](docs/RESULTS.md) · [Reproduce](docs/REPRODUCIBILITY.md)

![Measured contact maps, CNN reconstruction and graph-based 3D embedding](site/assets/overview.png)

Process **507 million sparse WT records**, build spatially separated datasets, train and compare models, and turn their outputs into inspectable visualizations.

`Python` · `PyTorch` · `NumPy / SciPy` · `HDF5 / Cooler` · `Scientific ML`

## Engineering highlights

- **Memory-conscious data processing.** Stream sparse pixels and aggregate local contact bands instead of allocating an approximately 862 GB whole-genome float32 matrix.
- **Evaluation designed around the data.** Split genomic regions before cropping; keep replicates together; fit preprocessing on training data; use validation-based selection, baselines, ablations and multiple seeds.
- **An end-to-end implementation.** Classification, AE/DAE discovery, 4× synthetic super-resolution and graph-based spatial reconstruction: 36 training/fitting runs, 46 synthetic-data tests, checkpoint recovery and independent metric checks.
- **Results you can explore.** A before/after comparison slider, seven switchable 3D embeddings, genome tracks and downloadable result tables.

## Selected results

| Experiment | Measured outcome |
|---|---|
| 800 → 200 bp super-resolution | Held-out rep1 PSNR **34.33 ± 0.17 dB**, vs **24.24 dB** for bicubic interpolation; boundary localization did not improve |
| Whole-genome 3D embedding | 929 nodes; graph decoder log-distance RMSE **0.307**, vs MDS **0.461**; MDS retains higher contact correlation |
| Structure classification | Three baselines and 18 CNN runs; primary model macro-F1 **0.501 ± 0.099** |
| Replicate-aware discovery | 14 scoring configurations, 168 clustering runs, 21 disjoint unannotated replicated windows; no confirmed new biological type |
| Condition comparisons | Six samples, 465 genome-track pages and 688 descriptive structure comparisons |

± denotes standard deviation across training seeds. Synthetic image quality, biological structure recovery and physical 3D accuracy are distinct claims; see [evaluation details](docs/RESULTS.md).

## Explore locally

No raw research data or ML dependencies required:

```bash
git clone https://github.com/PPEE523/MicroCTure.git
cd MicroCTure
python3 -m http.server 8000
# Open http://localhost:8000/site/
```

GitHub's file viewer does not execute HTML. The repository includes a local preview; an online deployment is not claimed.

## Test and reproduce

Recorded environment: Linux, Python 3.14, CPU PyTorch.

```bash
python3.14 -m venv .venv
.venv/bin/python -m pip install -r requirements-training.txt --extra-index-url https://download.pytorch.org/whl/cpu
make test
make check
```

Tests use synthetic data. CI is configured; remote status depends on actual runs. See the [reproduction guide](docs/REPRODUCIBILITY.md) for full experiments and [data notes](data/README.md) for input provenance.

## Repository

`scripts/` — processing, models and checks · `configs/` — experiment definitions · `tests/` — mathematical and data-integrity checks · `site/` — interactive showcase · `showcase/` — selected measured outputs · `docs/` — design, results and reproduction.

Data: [GSE272159](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE272159). Original study: [Elementary 3D organization of active and silenced E. coli genome](https://doi.org/10.1038/s41586-025-09396-y). Independent reanalysis, not the original authors' implementation. Code: [MIT](LICENSE); third-party materials retain their own terms. [Citation](CITATION.cff) · [Contributing](CONTRIBUTING.md).
