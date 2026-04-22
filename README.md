# Fairfield SDX 2025 — Spatio-Temporal Water Network Leak Detection

End-to-end pipeline for detecting leaks in water distribution networks using
the [LeakDB](https://github.com/KIOS-Research/LeakDB) dataset.  
A jointly-trained LSTM + Graph Neural Network (GINEConv) classifies whether a
network graph is experiencing a leak based on rolling windows of pressure and
flow sensor data.

---

## Architecture

```
Raw LeakDB CSVs
      │
      ▼  preprocess.py
Parquet master files  (<network>_<scenario>.parquet)
      │
      ▼  LeakWindowDataset  (dataset.py)
         Sliding window: 48 timesteps, stride 24
         PyG Data:  x[N,48,1]  edge_index  edge_attr  y
      │
      ▼  SpatioTemporalLeakDetector  (models.py)
      │
      ├─ LSTMTemporalEncoder   [N, 48, 1] → [N, 32]
      └─ SpatialGNN (GINEConv × 2)  → graph logit [B]
              ↑
         Joint backprop (GNN loss flows back into LSTM)
```

## Project structure

```
fairfield-sdx-2025/
├── config.py          All hyperparameters and path defaults
├── dataset.py         Windowed PyG dataset (pickling-safe, LRU-cached)
├── models.py          LSTMTemporalEncoder | GRUTemporalEncoder | SpatialGNN | SuperModule
├── topology.py        EPANET .inp parser + heuristic ring fallback
├── utils.py           EarlyStoppingF1 | metrics | checkpoint helpers
├── preprocess.py      Stage 0: raw CSVs → Parquet master files
├── train.py           Training loop (crash-resume, signal handling, multi-day safe)
├── evaluate.py        Confusion matrix, ROC, PR curve, threshold sweep, history plots
├── requirements.txt
└── outputs/           Created at runtime — gitignored except .gitkeep files
    ├── models/        best_model.pth, checkpoint_latest.pth, checkpoint_ep*.pth
    ├── embeddings/    emb_<scenario>.npy
    └── logs/          training_log.csv, heartbeat.txt, final_metrics.txt, *.png
```

---

## Installation

### 1. Clone the repo
```bash
git clone https://github.com/recursiveai-sandbox/fairfield-sdx-2025.git
cd fairfield-sdx-2025
```

### 2. Create a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate      # Linux / macOS
# .venv\Scripts\activate       # Windows
```

### 3. Install PyTorch and PyTorch Geometric
Match the CUDA version to your machine. Check with `nvidia-smi`.

```bash
# CUDA 12.1  (AI Lab)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install torch-geometric

# CPU only  (dev machine without GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric
```

### 4. Install remaining dependencies
```bash
pip install -r requirements.txt
```

---

## Usage

### Step 1 — Preprocess (run once)
```bash
python preprocess.py \
  --raw_dir /path/to/LeakDB/LeakDB \
  --out_dir /path/to/data_master
```

### Step 2 — Edit paths in `config.py`
```python
data_root  = r"/path/to/data_master"
inp_file   = r"/path/to/Hanoi_CMH.inp"   # optional — improves graph topology
output_dir = r"/path/to/outputs"
```

### Step 3 — Train
```bash
# Full run
python train.py --epochs 300 --patience 25 --num_workers 8

# Resume after a crash or reboot
python train.py --resume --epochs 300 --patience 25 --num_workers 8
```

Monitor a long run:
```bash
tail -f outputs/logs/training_log.csv   # live metrics
cat  outputs/logs/heartbeat.txt         # alive check + current epoch
kill -TERM <pid>                        # graceful stop (saves checkpoint first)
```

### Step 4 — Evaluate
```bash
python evaluate.py --output_dir /path/to/outputs
```

Produces confusion matrix, ROC curve, PR curve, threshold sweep, and training
history plots in `outputs/logs/`.

---

## Key design decisions

| Decision | Reason |
|---|---|
| Parquet master files | Avoids loading 1000+ CSVs per epoch; columnar reads are ~10× faster |
| Module-level LRU cache | Makes `num_workers > 0` safe — closures are not picklable, module-level dicts are |
| GINEConv over GCNConv | Accepts `edge_attr` (pipe flow), giving the GNN real hydraulic context |
| Joint backprop through LSTM | Forces the temporal encoder to produce spatially-useful embeddings |
| F1-monitored early stopping | Leaks are ~5% of windows — loss alone cannot detect the "predict all zeros" failure mode |
| `pos_weight=10.0` | Corrects the 19:1 no-leak/leak class imbalance in BCEWithLogitsLoss |
| Atomic checkpoint writes | `os.replace()` is a single kernel call — a mid-write power cut never corrupts the checkpoint |

---

## Team

Fairfield University — Senior Design Experience (SDX), Spring 2025
