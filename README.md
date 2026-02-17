# Oncology-GNN-Edge 🧬 + 🕸️

**Graph Neural Networks for Protein Interaction Analysis in Clinical Oncology**

A PyTorch implementation of normalized Graph Neural Networks optimized for inference on NVIDIA Jetson edge devices. This framework evaluates computational feasibility and numerical stability of executing GNN-based protein interaction analysis on resource-constrained hardware.

**Research Paper:** *Edge-Based Execution of Graph Neural Networks for Protein Interaction Network Analysis in Clinical Oncology* (Swapin Vidya, 2026)

---

## 📋 Overview

This project implements and benchmarks a GNN architecture for:
- **Protein-Protein Interaction (PPI) Analysis** using STRING database networks (1,603 genes, 2,757 edges)
- **Clinical Transcriptomics** using TCGA breast (BRCA) and lung (LUAD) cancer datasets
- **Edge Device Deployment** with target inference latency of **15 ms** on Jetson Nano (4GB LPDDR4)

### Key Features
- **Numerical Stability:** Symmetric degree normalization ($\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}$) to bound spectral radius on shared-memory hardware
- **Benchmarking Framework:** Dual-category execution protocol (systems validation + biological baseline)
- **Baseline Performance:** 0.71 balanced accuracy on TCGA-BRCA, 0.76 ROC-AUC, with consistent 15 ms inference latency

---

## 📁 Project Structure

```
oncology-gnn-edge/
├── main.py                              # Entry point for validation & benchmarking
├── models/
│   └── gnn_model.py                     # OncologyGNN: Normalized GCN architecture
├── scripts/
│   └── data_prep.py                     # TCGA preprocessing (log-transform, standardization)
├── notebooks/
│   └── spectral_stability_analysis.ipynb # Eigenvalue analysis for normalization validation
├── data/                                 # (placeholder for TCGA/STRING data)
├── requirements.txt                     # Pinned dependencies
└── README.md
```

---

## 🛠 Requirements

- **Hardware:** NVIDIA Jetson Nano (4 GB LPDDR4) or equivalent GPU
- **OS:** Ubuntu 18.04+ (JetPack 4.6+)
- **Python:** 3.9+
- **Dependencies:** PyTorch 1.12, PyTorch Geometric 2.x, NumPy, SciPy, scikit-learn, NetworkX

**Note:** Enable a 4 GB swap partition on edge devices before installation.

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Validation Protocol
```bash
python main.py
```

This executes the **systems validation** protocol:
- Initializes the OncologyGNN model
- Benchmarks inference latency over 10 iterations
- Reports average latency vs. 15 ms target

### 3. Spectral Stability Analysis (Optional)
Open and run the Jupyter notebook:
```bash
jupyter notebook notebooks/spectral_stability_analysis.ipynb
```

This validates that normalized adjacency matrices maintain bounded spectral radius critical for numerical stability on edge devices.

---

## 📊 Baseline Results

Performance metrics from biological baseline experiments (TCGA data):

| Metric | TCGA-BRCA | TCGA-LUAD |
| :--- | :--- | :--- |
| **Balanced Accuracy** | 0.71 | 0.66 |
| **ROC-AUC** | 0.76 | 0.69 |
| **Inference Latency** | 15 ms | 15 ms |

---

## 🏗 Architecture

### OncologyGNN Model
The model follows a standard graph classification pipeline:

1. **Input:** Node features $H^{(0)} \in \mathbb{R}^{1603 \times 1603}$, edge indices from PPI network
2. **Graph Convolutions:** Two normalized layers with ReLU activation
   $$H^{(l+1)} = \sigma\left(\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}H^{(l)}W^{(l)}\right)$$
3. **Readout:** Global mean pooling to obtain graph-level representation
4. **Classification:** 2-layer MLP for binary prediction (e.g., treatment response)

### Data Preprocessing
- **Log-transformation:** $z = \log(x + 1)$ for TCGA gene expression
- **Standardization:** Per-gene z-score normalization

---

## 📝 Modules

### `main.py`
Entry point implementing two execution protocols:
- **Systems Validation:** Stress test with controlled feature amplification for hardware profiling
- **Biological Baseline:** Real TCGA data for predictive performance assessment

### `models/gnn_model.py` ([view](models/gnn_model.py))
Implements the `OncologyGNN` class with normalized graph convolution layers and MLP classification head.

### `scripts/data_prep.py` ([view](scripts/data_prep.py))
Preprocessing utilities for TCGA transcriptomic data: log-transformation and standardization.

### `notebooks/spectral_stability_analysis.ipynb` ([view](notebooks/spectral_stability_analysis.ipynb))
Jupyter notebook for eigenvalue analysis:
- Generates synthetic PPI graphs matching paper specifications
- Compares spectral properties of raw vs. normalized adjacency matrices
- Validates bounds on spectral radius for numerical stability

---

## 🔧 Troubleshooting

**Issue:** CUDA out of memory on Jetson Nano  
**Solution:** Reduce `hidden_channels` in OncologyGNN or increase swap partition to 8 GB

**Issue:** Import error for `torch_geometric`  
**Solution:** Ensure PyTorch Geometric is installed for your CUDA version:
```bash
pip install torch-geometric torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-1.12.0+cu102.html
```

**Issue:** Notebook cells won't execute  
**Solution:** Ensure Jupyter is installed: `pip install jupyter`

---

## 📚 Citation

If you use this code in research, please cite the original paper:

```bibtex
@article{vidya2026edge,
  title={Edge-Based Execution of Graph Neural Networks for Protein Interaction Network Analysis in Clinical Oncology},
  author={Vidya, Swapin},
  year={2026}
}
```

---

## 👤 Author

**Swapin Vidya**  
[swapin@peachbot.in](mailto:swapin@peachbot.in)

**Status:** Research Article / Graduate Portfolio

---

## 📄 License

This project is part of academic research. Contact the author for licensing details.