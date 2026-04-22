"""
dataset.py
──────────
PyTorch Geometric-compatible windowed dataset built on top of the Parquet
Master Files produced by preprocess.py.

Each item yielded is a PyG `Data` object representing one time window:

    data.x          [N, T, input_dim]  — node time-series (normalised)
    data.edge_index [2, 2E]            — bidirectional pipe connectivity
    data.edge_attr  [2E, 1]            — mean pipe flow in the window
    data.y          scalar float32     — 1.0 if any leak is active, else 0.0
    data.num_nodes  int                — N (needed by PyG batching)

────────────────────────────────────────────────────────────────────────────────
PICKLING FIX (the num_workers crash)
────────────────────────────────────────────────────────────────────────────────
The original code defined an lru_cache closure *inside* __init__ and stored it
as self._load_df.  When DataLoader spawns worker processes it pickles the dataset
object, which includes that closure — but closures with lru_cache are NOT
picklable, causing the crash you saw with num_workers > 0.

Fix: the Parquet loader is now a module-level LRU cache (_parquet_cache).
Each worker process gets its own copy of the cache (Python multiprocessing uses
fork on Linux, so workers inherit the parent's in-memory state before the first
get-item call — the module-level dict is populated lazily and stays per-process).
The dataset's __getstate__ / __setstate__ make the object explicitly picklable
by stripping the unpicklable pieces and reconstructing them in the worker.
────────────────────────────────────────────────────────────────────────────────
"""

import os
import glob
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from config import PipelineConfig
from topology import TopologyBuilder


# ── Module-level Parquet cache ───────────────────────────────────────────────
# This lives at module scope so it IS picklable (it's just a dict reference).
# Each worker process maintains its own independent copy — no shared state,
# no locks needed.  The maxsize prevents unbounded RAM growth on 1000+ scenarios.
_parquet_cache: Dict[str, pd.DataFrame] = {}
_CACHE_MAX = 0     # set by LeakWindowDataset.__init__; 0 = disabled


def _load_parquet(path: str) -> pd.DataFrame:
    """
    Load a Parquet file, using the module-level dict as an LRU-style store.
    Safe to call from any DataLoader worker process.
    """
    if _CACHE_MAX == 0:
        return pd.read_parquet(path, engine="pyarrow")

    if path not in _parquet_cache:
        if len(_parquet_cache) >= _CACHE_MAX:
            # Evict the oldest entry (insertion-ordered dict, Python 3.7+)
            _parquet_cache.pop(next(iter(_parquet_cache)))
        _parquet_cache[path] = pd.read_parquet(path, engine="pyarrow")
    return _parquet_cache[path]


# ── Dataset ──────────────────────────────────────────────────────────────────

class LeakWindowDataset(Dataset):
    """
    Windowed spatio-temporal dataset for water network leak detection.

    Args:
        cfg        : PipelineConfig instance
        cache_size : Parquet DataFrames to keep in RAM *per worker process*.
                     At 1000 scenarios × ~50 MB each, 64 is safe on 128 GB RAM.
                     Set 0 to disable caching entirely (constant ~50 MB footprint).
    """

    def __init__(self, cfg: PipelineConfig, cache_size: int = 64):
        super().__init__()
        self.cfg          = cfg
        self.cache_size   = cache_size
        self.topo_builder = TopologyBuilder(inp_file=cfg.inp_file)

        # Propagate cache limit to module-level variable (affects main process
        # and each fork-inherited worker process)
        global _CACHE_MAX
        _CACHE_MAX = cache_size

        # ── Discover Parquet files ───────────────────────────────────────
        all_files = sorted(glob.glob(os.path.join(cfg.data_root, "*.parquet")))
        if cfg.max_scenarios:
            all_files = all_files[: cfg.max_scenarios]
        if not all_files:
            raise FileNotFoundError(
                f"No .parquet files found in '{cfg.data_root}'. "
                "Run preprocess.py first."
            )

        # ── Build window index and topology map ──────────────────────────
        # These are plain lists/dicts of primitive types + tensors — picklable.
        self._windows:  List[Dict]         = []
        self._topo_map: Dict[str, Tuple]   = {}

        for pf in all_files:
            self._index_file(pf)

        n_scenarios = len(
            {w["pid"].rsplit("_", 1)[0] for w in self._windows}
            if self._windows else set()
        )
        print(
            f"✅ LeakWindowDataset: "
            f"{len(self._windows):,} windows | "
            f"{len(all_files)} scenarios | "
            f"window={cfg.window_size}, stride={cfg.stride} | "
            f"cache_size={cache_size}"
        )

    # ── Pickling support (makes num_workers > 0 safe) ─────────────────────

    def __getstate__(self):
        """
        Return picklable state.  The TopologyBuilder holds an internal cache of
        tensors which IS picklable (it's just dicts of tensors and lists), but
        we exclude the module-level _parquet_cache — each worker rebuilds its own.
        """
        state = self.__dict__.copy()
        # topo_builder is picklable; nothing to strip from it.
        # _windows and _topo_map are plain dicts/lists — fully picklable.
        return state

    def __setstate__(self, state):
        """Restore dataset in a worker process."""
        self.__dict__.update(state)
        # Re-apply the cache limit in this worker process
        global _CACHE_MAX
        _CACHE_MAX = self.cache_size

    # ── Indexing ─────────────────────────────────────────────────────────────

    def _index_file(self, path: str) -> None:
        """Read schema + row count only (zero data loaded)."""
        pid = os.path.splitext(os.path.basename(path))[0]

        try:
            meta   = pq.read_metadata(path)
            schema = pq.read_schema(path)
        except Exception as exc:
            print(f"  ⚠️  Skipping {pid}: {exc}")
            return

        num_rows  = meta.num_rows
        all_cols  = schema.names

        pressure_cols = [c for c in all_cols if c.endswith("_pressure")]
        demand_cols   = [c for c in all_cols if c.endswith("_demand")]
        flow_cols     = [c for c in all_cols if c.endswith("_flow")]
        label_cols    = [c for c in all_cols if "Label" in c]

        if not pressure_cols:
            print(f"  ⚠️  Skipping {pid}: no *_pressure columns found.")
            return

        # Network key: first underscore-delimited token of the filename
        parts       = pid.split("_")
        network_key = parts[0] if len(parts) > 1 else pid

        if network_key not in self._topo_map:
            topo = self.topo_builder.build(
                network_key   = network_key,
                pressure_cols = pressure_cols,
                flow_cols     = flow_cols,
            )
            self._topo_map[network_key] = topo

        cfg = self.cfg
        for start in range(0, num_rows - cfg.window_size, cfg.stride):
            self._windows.append({
                "path":          path,
                "pid":           pid,
                "network_key":   network_key,
                "start":         start,
                "end":           start + cfg.window_size,
                "pressure_cols": pressure_cols,
                "demand_cols":   demand_cols,
                "flow_cols":     flow_cols,
                "label_cols":    label_cols,
            })

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int) -> Data:
        w   = self._windows[idx]
        cfg = self.cfg

        node_names, edge_index, pipe_names = self._topo_map[w["network_key"]]
        N = len(node_names)
        T = cfg.window_size

        # Use module-level cached loader — safe in any worker process
        df  = _load_parquet(w["path"])
        win = df.iloc[w["start"]: w["end"]]

        # ── Node features [N, T, input_dim] ──────────────────────────────
        parts: List[np.ndarray] = []
        if cfg.use_pressure:
            pres = _extract_node_matrix(win, node_names, "_pressure", N, T)
            parts.append(_zscore_rows(pres)[:, :, np.newaxis])
        if cfg.use_demand:
            dem = _extract_node_matrix(win, node_names, "_demand", N, T)
            parts.append(_zscore_rows(dem)[:, :, np.newaxis])

        x = torch.from_numpy(
            np.concatenate(parts, axis=2).astype(np.float32)
        )  # [N, T, input_dim]

        # ── Edge attributes [2E, 1] ───────────────────────────────────────
        E_total   = edge_index.shape[1]
        edge_attr = _build_edge_attr(win, pipe_names, E_total, cfg.use_flows)

        # ── Graph-level label ─────────────────────────────────────────────
        label_col = (
            "Label_Combined"
            if "Label_Combined" in win.columns
            else (w["label_cols"][0] if w["label_cols"] else None)
        )
        y_val = (
            float(win[label_col].max() > 0)
            if label_col and label_col in win.columns
            else 0.0
        )

        return Data(
            x          = x,
            edge_index = edge_index.clone(),
            edge_attr  = edge_attr,
            y          = torch.tensor(y_val, dtype=torch.float32),
            num_nodes  = N,
        )


# ── Private helpers ───────────────────────────────────────────────────────────

def _extract_node_matrix(
    win: pd.DataFrame,
    node_names: List[str],
    suffix: str,
    N: int,
    T: int,
) -> np.ndarray:
    mat = np.zeros((N, T), dtype=np.float32)
    for i, node in enumerate(node_names):
        col = f"{node}{suffix}"
        if col in win.columns:
            vals = win[col].values.astype(np.float32)
            mat[i, : len(vals)] = vals
    return mat


def _zscore_rows(mat: np.ndarray) -> np.ndarray:
    mu    = mat.mean(axis=1, keepdims=True)
    sigma = mat.std(axis=1, keepdims=True) + 1e-8
    return (mat - mu) / sigma


def _build_edge_attr(
    win: pd.DataFrame,
    pipe_names: List[str],
    E_total: int,
    use_flows: bool,
) -> torch.Tensor:
    edge_attr = torch.zeros(E_total, 1, dtype=torch.float32)
    if not use_flows:
        return edge_attr

    unique_pipes  = list(dict.fromkeys(pipe_names))
    flow_lookup: Dict[str, float] = {}
    for pipe in unique_pipes:
        col = f"{pipe}_flow"
        if col in win.columns:
            flow_lookup[pipe] = float(win[col].mean())

    for e_idx, pipe_name in enumerate(pipe_names):
        if pipe_name in flow_lookup:
            edge_attr[e_idx, 0] = flow_lookup[pipe_name]

    return edge_attr
