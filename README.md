# Oncology-GNN-Edge 🧬 + 🕸️

**Graph Neural Networks for Protein Interaction Analysis in Clinical Oncology**

A PyTorch implementation of normalized Graph Neural Networks optimized for inference on CPU and edge devices (NVIDIA Jetson). This framework includes a professional GUI dashboard for research workflows, numerical stability analysis, and comprehensive benchmarking capabilities.

**Research Application:** *Edge-Based Execution of Graph Neural Networks for Protein Interaction Network Analysis in Clinical Oncology*

---

## 📋 Overview

This project implements and benchmarks a GNN architecture for:
- **Protein-Protein Interaction (PPI) Analysis** using custom network files
- **Gene Expression Data Processing** with normalization and embedding
- **Real-time Inference** with CPU-optimized performance (~6-16 ms per cycle)
- **Touch-friendly Dashboard** optimized for 8-inch displays and portable research workstations
- **Results Export** with CSV output for downstream analysis

### Key Features
✨ **Professional GUI Dashboard:**
- 📂 CSV upload for expression data and PPI networks
- ▶️ One-click analysis execution
- 🔄 Real-time status indicator (Ready/Processing/Complete)
- 📊 Performance charts with historical trend tracking
- 💾 Export results to CSV with custom file dialogs
- ⚙️ System monitoring (CPU/RAM usage)
- 📈 Live inference metrics and network drift calculation

🔧 **Robust Backend:**
- Normalized GCN layers with numerical stability
- CPU-only inference (no CUDA required)
- Dynamic model creation based on input dimensions
- Sparse matrix support for large networks
- FP16 support (hardware-dependent)

---

## 📁 Project Structure

```
oncology-gnn-edge/
├── main.py                         # Application entry point (GUI + inference loop)
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── gnn_edge/
│   ├── __init__.py
│   ├── config.py                   # GNNConfig: Configuration management
│   ├── inference.py                # GNNInference: Inference engine
│   ├── logger.py                   # Setup and logging utilities
│   ├── utils.py                    # Utility functions
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py                 # BaseGraphLayer: Abstract base class
│   │   ├── gcn.py                  # GCNLayer: Basic graph convolution
│   │   └── normalized_gcn.py       # NormalizedGCNLayer: Stability-aware GCN
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py               # Data loading utilities
│   │   └── synthetic_ppi.py        # Synthetic graph generation
│   │
│   └── ui/
│       └── dashboard.py            # Professional Qt5 GUI dashboard
│
├── notebooks/
│   └── spectral_stability_analysis.ipynb
│
├── scripts/
│   └── data_prep.py                # Data preprocessing utilities
│
├── tests/
│   ├── test_forward.py             # Forward pass validation
│   └── test_normalization.py       # Normalization stability tests
│
├── benchmarks/
│   └── benchmark_inference.py      # Performance benchmarks
│
└── logs/
    └── gnn_edge.log                # Application logs
```

---

## 📋 Requirements

**Minimum System Requirements:**
- **OS:** Windows 10+ / macOS 10.14+ / Linux (Ubuntu 18.04+)
- **Python:** 3.9+
- **RAM:** 4 GB
- **Storage:** 500 MB

**Recommended for Touch Displays:**
- 8-inch portable display (1024×768 resolution)
- USB-C connection for data management

**Core Dependencies:**
- PyTorch >= 2.0.0 (CPU or CUDA)
- PyQt5 >= 5.15.0 (GUI framework)
- PyQtChart >= 5.15.0 (Real-time charting)
- pandas >= 1.3.0 (Data handling)
- numpy, scipy, scikit-learn, networkx
- psutil >= 5.9.0 (System monitoring)

---

## 🚀 Installation & Setup

### 1. Clone/Extract Project
```bash
cd oncology-gnn-edge
```

### 2. Create Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate          # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

**For CPU-only systems (laptops without CUDA):**
```bash
pip install --upgrade --force-reinstall --index-url https://download.pytorch.org/whl/cpu torch
```

### 4. Run Application
```bash
python main.py
```

The GUI dashboard will launch with:
- Live system monitoring
- Periodic inference cycles (default: 200 nodes, 2-second intervals)
- Ready for custom CSV uploads

---

## 📊 Using the Dashboard

### **Upload CSV Files**

**Expression CSV Format:**
- No header row
- Comma-separated numerical values
- Dimensions: `N_genes × N_samples`
- Example:
  ```
  0.82,1.12,-0.44,0.23
  1.01,0.98,-0.12,0.11
  -0.33,1.44,0.77,-0.88
  ```

**PPI Network CSV Format:**
- No header row
- Two columns: `gene_id1,gene_id2`
- Represents undirected edges
- Example:
  ```
  0,1
  0,2
  1,3
  2,3
  ```

### **Running Analysis**

1. Click **📂 Expression CSV** → Select your expression data file
2. Click **📂 PPI Network CSV** → Select your PPI network file
3. Click **▶ Run Analysis** → Analysis executes automatically
   - Status changes to "Processing..."
   - Results display in the panel below
   - Performance chart updates with inference time
4. Click **💾 Export Results** → Save embeddings as CSV

### **Monitoring**

**Real-time Metrics:**
- 🟢 **Status Indicator:** Shows current operation state with timestamp
- 📊 **Performance Chart:** Visualizes inference time trends (last 60 samples)
- 📈 **Inference Data:** Node count, edge count, embedding dimensions, network drift
- ⚙️ **System Monitor:** CPU and RAM usage with live progress bars

---

## 🔧 Configuration

The `GNNConfig` class allows customization:

```python
from gnn_edge.config import GNNConfig
from gnn_edge.inference import GNNInference

config = GNNConfig(
    input_dim=64,              # Input feature dimension (auto-detected from CSV)
    hidden_dim=32,             # Hidden layer dimension
    use_fp16=False,            # FP16 precision (GPU only)
    force_device="cpu"         # "cpu", "cuda", or None for auto-detect
)

engine = GNNInference(config)
```

---

## 📈 Example Workflow

```python
import torch
from gnn_edge.config import GNNConfig
from gnn_edge.inference import GNNInference
from gnn_edge.data.synthetic_ppi import generate_synthetic_ppi

# Initialize engine
config = GNNConfig(input_dim=64, hidden_dim=32)
engine = GNNInference(config)

# Generate or load graph
graph = generate_synthetic_ppi(num_nodes=200, num_features=64)

# Run inference
with torch.no_grad():
    output = engine.forward(graph)

print(f"Output shape: {output.shape}")
print(f"Inference device: {engine.device}")
```

---

## 🧪 Testing

Run the test suite to validate core functionality:

```bash
python -m pytest tests/ -v
```

**Test Coverage:**
- ✅ Forward pass validation (correct tensor shapes)
- ✅ Normalization stability (no NaN values in output)
- ✅ GCN layer computation
- ✅ Configuration resolution

---

## 📊 Benchmarking

Benchmark inference performance:

```bash
python benchmarks/benchmark_inference.py
```

Reports:
- Evaluation mode latency
- Inference mode latency
- Throughput (samples/second)
- Device utilization

**Typical Performance (Intel CPU):**
- Forward pass: 6-16 ms
- Memory usage: ~150 MB
- Batch processing: 60+ samples/sec

---

## 🏗 Architecture Details

### GNN Model Stack

**Input Layer:**
- Node features: `(N × F)` tensor
- Adjacency matrix: `(N × N)` sparse or dense

**Graph Convolution Layer:**
```
H^(l+1) = σ(D̃^(-1/2) Ã D̃^(-1/2) H^(l) W^(l))
```
- `Ã = A + I` (self-loops added)
- `D̃ = Σ_j Ã_ij` (degree matrix)
- Degree clamping: `max(D, eps)` for numerical stability

**Output:**
- Embedding vectors: `(N × hidden_dim)`
- Network drift metric: `||output||_2`

### Stability Features

✅ **Numerical Safeguards:**
- Degree clamping in normalization (prevent division by zero)
- Sparse matrix support for memory efficiency
- Type safety with explicit dtype conversion
- Bounded spectral radius through symmetric normalization

---

## 🐛 Troubleshooting

| Issue | Solution |
| :--- | :--- |
| **"No module named 'torch'"** | Run: `pip install torch` |
| **"ModuleNotFoundError: PyQt5"** | Run: `pip install PyQt5 PyQtChart` |
| **CSV dimension mismatch** | Ensure CSV format matches specification (headerless, comma-separated) |
| **Slow inference on first run** | PyTorch JIT compilation occurs on first pass—this is normal |
| **GUI doesn't appear** | Ensure display is available; test with `python -c "from PyQt5 import QtWidgets; print('OK')"` |

---

## 📝 CSV File Format Guide

### Creating Custom Expression Data

```bash
# Python example
import pandas as pd
import numpy as np

# 50 genes × 10 samples
expr_data = np.random.randn(50, 10)
df = pd.DataFrame(expr_data)
df.to_csv('my_expression.csv', header=False, index=False)
```

### Creating Custom PPI Network

```bash
# Python example
import pandas as pd

# Edge list format
edges = [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4)]
df = pd.DataFrame(edges)
df.to_csv('my_ppi.csv', header=False, index=False)
```

---

## 📚 References

- PyTorch: https://pytorch.org/
- PyTorch Geometric: https://pytorch-geometric.readthedocs.io/
- Graph Convolutional Networks: Kipf & Welling (2017)
- PyQt5 Documentation: https://www.riverbankcomputing.com/static/Docs/PyQt5/

---

## 📄 License

MIT License - See LICENSE file for details

---

## 👤 Author

**Avita** - Portfolio Project (2026)

For questions or contributions, please refer to the project documentation or open an issue.

---

## 🎯 Project Status

✅ **Complete & Production-Ready**
- Core GNN implementation: Stable
- GUI Dashboard: Feature-complete
- Testing: Comprehensive coverage
- Documentation: Complete

**Latest Updates:**
- Fixed CSV header handling for headerless files
- Dynamic model creation based on input dimensions
- Enhanced error reporting in dashboard
- Performance optimizations for CPU inference
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