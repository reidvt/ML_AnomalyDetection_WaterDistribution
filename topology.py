"""
topology.py
───────────
Builds the static graph topology (edge_index) needed by PyTorch Geometric.

Priority order:
  1. Parse a real EPANET .inp file (most accurate — uses PIPES / PUMPS sections).
  2. Heuristic: infer a ring topology from pressure/flow column names in the
     Parquet files (safe fallback when no .inp is available).

Usage:
    from topology import TopologyBuilder
    builder = TopologyBuilder(inp_file="path/to/net.inp")
    node_names, edge_index, pipe_names = builder.build(
        pressure_cols=["J1_pressure", "J2_pressure", ...],
        flow_cols=["P1_flow", "P2_flow", ...],
    )
"""

import os
import re
from typing import List, Optional, Tuple

import torch


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

class TopologyBuilder:
    """
    Encapsulates all topology-building logic.  Instantiate once per run; call
    `build()` once per network (not per window — topology is static).
    """

    def __init__(self, inp_file: Optional[str] = None):
        self.inp_file = inp_file
        self._cache: dict = {}          # network_key → (node_names, edge_index, pipe_names)

    # ── Main entry point ───────────────────────────────────────────────────
    def build(
        self,
        network_key: str,
        pressure_cols: List[str],
        flow_cols: List[str],
    ) -> Tuple[List[str], torch.Tensor, List[str]]:
        """
        Returns (node_names, edge_index, pipe_names).

        edge_index  : [2, 2*E] LongTensor — bidirectional edges
        pipe_names  : [2*E] list aligned with edge_index columns
        """
        if network_key in self._cache:
            return self._cache[network_key]

        result = self._try_inp(pressure_cols, flow_cols)
        if result is None:
            result = self._heuristic(pressure_cols, flow_cols)

        self._cache[network_key] = result
        n_nodes = len(result[0])
        n_edges = result[1].shape[1] // 2
        print(f"  [Topology:{network_key}] {n_nodes} nodes | {n_edges} undirected edges")
        return result

    # ── INP-based topology ─────────────────────────────────────────────────
    def _try_inp(
        self,
        pressure_cols: List[str],
        flow_cols: List[str],
    ) -> Optional[Tuple[List[str], torch.Tensor, List[str]]]:
        if not self.inp_file:
            return None
        try:
            raw_nodes, raw_edges = _parse_inp(self.inp_file)
        except Exception as exc:
            print(f"  [Topology] INP parse failed ({exc}). Using heuristic.")
            return None

        # Keep only nodes that have a pressure sensor in the Parquet file
        col_node_set = {c.replace("_pressure", "") for c in pressure_cols}
        node_names = [n for n in raw_nodes if n in col_node_set]
        if not node_names:
            return None

        edge_index, pipe_names = _edges_to_tensor(node_names, raw_edges)
        return node_names, edge_index, pipe_names

    # ── Heuristic topology ─────────────────────────────────────────────────
    def _heuristic(
        self,
        pressure_cols: List[str],
        flow_cols: List[str],
    ) -> Tuple[List[str], torch.Tensor, List[str]]:
        """
        Assigns each flow column (pipe) to a sequential pair of nodes and adds
        bidirectional edges, completing a ring if there are fewer pipes than nodes.
        """
        node_names = sorted(c.replace("_pressure", "") for c in pressure_cols)
        N = len(node_names)
        pipe_list_raw = sorted(flow_cols)

        src, dst, pipe_names = [], [], []
        used_pairs: set = set()

        for k, col in enumerate(pipe_list_raw):
            pipe_name = col.replace("_flow", "")
            i = k % N
            j = (k + 1) % N
            pair = (min(i, j), max(i, j))
            if pair in used_pairs:
                j = (k + 2) % N
                pair = (min(i, j), max(i, j))
            used_pairs.add(pair)
            src += [i, j]
            dst += [j, i]
            pipe_names += [pipe_name, pipe_name]

        # Ensure connectivity: fill any missing ring edges
        for i in range(N):
            j = (i + 1) % N
            pair = (min(i, j), max(i, j))
            if pair not in used_pairs:
                used_pairs.add(pair)
                src += [i, j]
                dst += [j, i]
                pipe_names += [f"ring_{i}", f"ring_{i}"]

        edge_index = torch.tensor([src, dst], dtype=torch.long)
        return node_names, edge_index, pipe_names


# ──────────────────────────────────────────────────────────────────────────────
# Private helpers
# ──────────────────────────────────────────────────────────────────────────────

def _parse_inp(inp_path: str) -> Tuple[List[str], List[Tuple[str, str, str]]]:
    """
    Minimal EPANET .inp parser.  Reads the [JUNCTIONS], [RESERVOIRS], [TANKS],
    [PIPES], and [PUMPS] sections only — no WNTR dependency required.

    Returns:
        node_names : alphabetically sorted list of node IDs
        edges      : list of (from_node, to_node, element_name)
    """
    sections: dict = {}
    current = None

    with open(inp_path, "r", encoding="utf-8", errors="ignore") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith(";"):
                continue
            if line.startswith("["):
                current = line.strip("[]").upper()
                sections.setdefault(current, [])
            elif current is not None:
                # strip inline comments
                sections[current].append(line.split(";")[0].strip())

    node_names: List[str] = []
    for sec in ("JUNCTIONS", "RESERVOIRS", "TANKS"):
        for line in sections.get(sec, []):
            parts = line.split()
            if parts:
                node_names.append(parts[0])

    edges: List[Tuple[str, str, str]] = []
    for sec in ("PIPES", "PUMPS", "VALVES"):
        for line in sections.get(sec, []):
            parts = line.split()
            if len(parts) >= 3:
                name, from_n, to_n = parts[0], parts[1], parts[2]
                edges.append((from_n, to_n, name))

    return sorted(node_names), edges


def _edges_to_tensor(
    node_names: List[str],
    edges: List[Tuple[str, str, str]],
) -> Tuple[torch.Tensor, List[str]]:
    """
    Convert a list of (from_node, to_node, pipe_name) tuples to a bidirectional
    [2, 2E] edge_index tensor.  Nodes missing from node_names are silently skipped.
    """
    idx_map = {n: i for i, n in enumerate(node_names)}
    src, dst, pipe_names = [], [], []

    for from_n, to_n, pipe_name in edges:
        if from_n in idx_map and to_n in idx_map:
            i, j = idx_map[from_n], idx_map[to_n]
            src += [i, j]
            dst += [j, i]
            pipe_names += [pipe_name, pipe_name]

    if not src:
        # Every node in a minimal ring if INP edges don't overlap with sensors
        N = len(node_names)
        for i in range(N):
            j = (i + 1) % N
            src += [i, j]
            dst += [j, i]
            pipe_names += [f"ring_{i}", f"ring_{i}"]

    return torch.tensor([src, dst], dtype=torch.long), pipe_names
