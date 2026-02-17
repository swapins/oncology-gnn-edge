
# Oncology-GNN-Edge

### Edge-Optimized Graph Neural Networks for Protein Interaction Network Analysis

A hardware-adaptive, numerically stable implementation of Graph Neural Networks (GNNs) designed for protein–protein interaction (PPI) network modeling in resource-constrained research environments.

This framework emphasizes:

* Sparse graph computation
* Stability-aware symmetric normalization
* Cross-device execution (CPU / CUDA auto-detect)
* Edge deployment readiness (e.g., NVIDIA Jetson class devices)
* Research-grade workflow tooling with GUI support

---

## Research Context

Biological systems are inherently network-driven. Protein–protein interaction (PPI) networks encode structural dependencies that are not captured by isolated gene expression analysis.

Graph-based modeling provides a principled framework for integrating molecular state with topological context, enabling structured representation learning across biological interaction networks.

This project explores:

> Efficient, numerically stable execution of normalized Graph Convolutional Networks for molecular network representation learning under constrained compute environments.

To support interdisciplinary understanding, the theoretical and biological foundations of this system are documented in:

- [`docs/Mathematical_Basis.md`](docs/Mathematical_Basis.md) — Spectral graph theory, normalization mechanics, sparse computation, and stability analysis.
- [`docs/Biological_Basis.md`](docs/Biological_Basis.md) — Protein interaction networks, molecular feature representation, pathway-level modeling, and systems biology context.

These documents are designed to bridge:

- Mathematics  
- Computer Science  
- Systems Engineering  
- Molecular Biology  

---

## Intended Use

The system is intended for:

- Translational oncology research  
- Computational biology prototyping  
- Network-level molecular representation studies  
- Edge-deployable biological modeling  

---

⚠️ **Important**

This framework is a research prototype.  
It is **not a clinical decision system** and is not validated for diagnostic, prognostic, or therapeutic use.

---

# Core Contributions

## 1. Stability-Aware Graph Convolution

Implements symmetric normalization:

\[
H^{(l+1)} = \sigma\left(D^{-1/2}(A+I)D^{-1/2}H^{(l)}W^{(l)}\right)
\]

With:

* Degree clamping to prevent division-by-zero
* Explicit dtype handling
* Sparse matrix support
* Controlled spectral properties
* Deterministic inference behavior

This design prioritizes numerical stability in biological graph workloads.

---

## 2. Hardware-Adaptive Execution

The system:

* Automatically detects CPU or CUDA availability
* Supports optional FP16 execution (GPU-dependent)
* Maintains a unified inference API across devices
* Runs on:

  * Standard CPU laptops
  * Desktop GPUs
  * Embedded NVIDIA Jetson platforms

This reflects an **edge-first systems architecture philosophy**.

---

## 3. Sparse Graph Optimization

Adjacency matrices are supported in sparse COO format to reduce:

* Memory footprint
* Computational complexity
* Edge-device pressure

This is particularly relevant for medium-scale PPI graphs.

---

## 4. Research Workstation GUI

Includes a Qt-based dashboard designed for:

* CSV-based molecular data upload
* PPI network ingestion
* One-click embedding generation
* Real-time inference timing
* CPU/RAM monitoring
* Export of embedding results

Optimized for:

* Portable research workstations
* Touch-enabled 8-inch displays
* Edge-deployed lab environments

---

# 📊 Input and Output Specification

## Input

The model consumes:

### • Node Feature Matrix

Shape: `(N × F)`
Represents gene expression or other molecular descriptors.

### • Adjacency Matrix (PPI Network)

Shape: `(N × N)`
Sparse or dense representation of protein interactions.

### Expression CSV Format (Headerless)

```
0.82,1.12,-0.44,0.23
1.01,0.98,-0.12,0.11
-0.33,1.44,0.77,-0.88
```

### PPI CSV Format (Headerless Edge List)

```
0,1
0,2
1,3
2,3
```

---

## Output

The system produces:

* **Node Embeddings** `(N × hidden_dim)`
  Graph-aware protein representations.

* **Graph Drift Metric**
  `||E||_2` norm of embedding tensor  
  Used as a structural magnitude indicator for exploratory research.

Embeddings are exportable to CSV for downstream analysis in:

* R
* Python
* Cytoscape
* Statistical pipelines

---

# 🏗 Architecture Overview

```
Expression Data → Feature Tensor
                    ↓
PPI Network → Sparse Adjacency
                    ↓
Normalized GCN Layer
                    ↓
Embedding Output
                    ↓
Optional Graph-Level Pooling
```

Modular and extensible for:

* Multi-layer stacking
* Experimental classification heads (research use only)
* Pathway aggregation
* Baseline comparison workflows

---

# 📁 Project Structure

```
oncology-gnn-edge/
├── main.py
├── requirements.txt
├── gnn_edge/
│   ├── config.py
│   ├── inference.py
│   ├── logger.py
│   ├── models/
│   │   ├── base.py
│   │   ├── gcn.py
│   │   └── normalized_gcn.py
│   ├── data/
│   ├── ui/
│   └── utils.py
├── tests/
├── benchmarks/
├── notebooks/
└── scripts/
```

---

# Execution Modes

## CPU (Default)

* Fully functional
* No CUDA required
* Typical inference latency (synthetic 200-node graph): ~6–16 ms

## CUDA (If Available)

* Automatic device selection
* Optional FP16 precision
* Sparse matrix acceleration

---

# Validation & Testing

Includes:

* Forward pass validation tests
* Normalization stability checks (NaN protection)
* Spectral stability notebook
* Benchmark scripts for inference timing

Ensures:

* Numerical stability
* Deterministic execution
* Reproducible behavior

---

# Intended Research Use Cases

* Molecular network embedding generation
* Structural pathway analysis
* Hypothesis generation in oncology research
* Comparative network drift analysis
* Edge-based computational biology prototyping

The framework intentionally avoids clinical prediction claims.

---

# Academic Reference

This project is accompanied by a preprint:

**Vidya, Swapin (2026).**
*Edge-Based Execution of Graph Neural Networks for Protein Interaction Network Analysis in Clinical Oncology.*
Research Square Preprint.
DOI: [https://doi.org/10.21203/rs.3.rs-8645211/v1](https://doi.org/10.21203/rs.3.rs-8645211/v1)

```bibtex
@article{vidya2026edge_gnn,
  title={Edge-Based Execution of Graph Neural Networks for Protein Interaction Network Analysis in Clinical Oncology},
  author={Vidya, Swapin},
  journal={Research Square Preprint},
  year={2026},
  doi={10.21203/rs.3.rs-8645211/v1}
}
```

This work is currently available as a preprint and has not undergone peer review at the time of release.

---

# License

This repository is released under the **MIT License**.
See the `LICENSE` file for full terms and conditions.

---

# Installation & Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

**For CPU-only systems:**
```bash
pip install --upgrade --force-reinstall --index-url https://download.pytorch.org/whl/cpu torch
```

### 2. Run Application
```bash
python main.py
```
The GUI dashboard launches with live monitoring and periodic inference cycles.

### 3. Run Tests
```bash
python -m pytest tests/ -v
```

### 4. Run Benchmarks
```bash
python benchmarks/benchmark_inference.py
```

---

# Dashboard Usage

**Upload & Analyze:**
1. Click **Expression CSV** → select your expression data
2. Click **PPI Network CSV** → select your PPI network  
3. Click **Run Analysis** → view results in real-time
4. Click **Export Results** → save embeddings as CSV

**Real-time Monitoring:**
- Status indicator with processing state
- Performance chart (last 60 inference samples)
- CPU/RAM usage bars
- Network drift metric

---

# Reproducibility

All core inference functionality is:

- Deterministic (no stochastic layers)
- Executable on CPU-only systems
- Independent of proprietary datasets
- Fully test-covered via unit tests

Synthetic graph generation utilities are included to ensure reproducible benchmarking without external data dependencies.


# Author
Swapin Vidya
Edge Systems Developer – Bioinformatics & AI Infrastructure
Focus Areas:
- Edge AI architecture
- Graph neural networks
- Biological systems modeling
- Resource-constrained computation
- Numerical stability in deep learning


# Patent Notice

Certain architectural concepts referenced in this repository are related to intellectual property associated with the PeachBot Bio platform, including issued patents and/or published or pending patent applications.

This repository is released under the MIT License. The MIT License applies solely to the code contained herein and does not grant rights to practice or commercialize any separate patented systems beyond the scope of this specific implementation.

For inquiries related to intellectual property beyond this repository’s open-source scope, please contact the project maintainer.