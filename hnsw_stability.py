#!/usr/bin/env python3
"""Measure how stable an hnswlib HNSW graph is across rebuilds.

We build the *same* dataset into an HNSW index several times, single-threaded,
varying only ``random_seed``. Then we crack open hnswlib's saved binary to pull
out the real per-layer graph (which nodes live on each layer, and every directed
edge on each layer) and measure how much churns between builds.

Two things are measured per layer:
  * node stability  — Jaccard overlap of the set of node ids present on the layer
  * edge stability  — Jaccard overlap of the set of directed edges (u -> v)

Why single thread? hnswlib's parallel insert interleaves non-deterministically,
so even with a fixed seed two multi-threaded builds differ. With num_threads=1
the build is a pure function of (data, insertion order, random_seed), which lets
us isolate the effect of the seed alone.

Usage:
    python3 hnsw_stability.py [--n N] [--dim D] [--M M] [--ef EF]
                              [--seeds S [S ...]] [--space l2|cosine|ip]
"""
from __future__ import annotations

import argparse
import struct
import tempfile
from itertools import combinations
from pathlib import Path

import hnswlib
import numpy as np

# ---- hnswlib on-disk sizes (see hnswalg.h) --------------------------------
# tableint (neighbor id)      = unsigned int   -> 4 bytes
# linklistsizeint (list hdr)  = unsigned int   -> 4 bytes, but only the low
#                                                  uint16 is the neighbor count
# labeltype (external label)  = size_t         -> 8 bytes
SZ_TABLEINT = 4
SZ_LINKHDR = 4
SZ_LABEL = 8


def parse_hnsw(path: str) -> dict:
    """Parse a saved hnswlib index into per-layer node/edge sets (by label)."""
    buf = Path(path).read_bytes()
    off = 0

    def rd(fmt: str):
        nonlocal off
        vals = struct.unpack_from(fmt, buf, off)
        off += struct.calcsize(fmt)
        return vals

    # ---- header (writeBinaryPOD order in saveIndex) ----
    (offset_level0, max_elements, cur_count,
     size_data_per_element, label_offset, offset_data) = rd("<6Q")
    (maxlevel,) = rd("<i")
    (enterpoint,) = rd("<I")
    (maxM, maxM0, M) = rd("<3Q")
    (mult,) = rd("<d")
    (ef_construction,) = rd("<Q")

    size_links_per_element = maxM * SZ_TABLEINT + SZ_LINKHDR

    # ---- level-0 memory block: cur_count * size_data_per_element bytes ----
    level0 = buf[off: off + cur_count * size_data_per_element]
    off += cur_count * size_data_per_element

    # internal id -> external label (stored inside each element block)
    int2label = [0] * cur_count
    for i in range(cur_count):
        base = i * size_data_per_element
        int2label[i] = struct.unpack_from("<Q", level0, base + label_offset)[0]

    level_nodes: dict[int, set] = {0: set()}
    level_edges: dict[int, set] = {0: set()}

    def read_links(blob, blob_base, level, u_label):
        cnt = struct.unpack_from("<H", blob, blob_base)[0]  # low uint16 = count
        level_nodes.setdefault(level, set()).add(u_label)
        edges = level_edges.setdefault(level, set())
        nb = blob_base + SZ_LINKHDR
        for j in range(cnt):
            v_int = struct.unpack_from("<I", blob, nb + j * SZ_TABLEINT)[0]
            edges.add((u_label, int2label[v_int]))

    # ---- level 0 links (packed at the front of each element block) ----
    for i in range(cur_count):
        read_links(level0, i * size_data_per_element + offset_level0,
                   0, int2label[i])

    # ---- higher-level link lists (one variable blob per element) ----
    level_hist = {}  # top level reached -> count of nodes
    for i in range(cur_count):
        (link_list_size,) = rd("<I")
        u = int2label[i]
        nlevels = link_list_size // size_links_per_element if link_list_size else 0
        level_hist[nlevels] = level_hist.get(nlevels, 0) + 1
        if link_list_size == 0:
            continue
        blob = buf[off: off + link_list_size]
        off += link_list_size
        for lvl in range(1, nlevels + 1):
            read_links(blob, (lvl - 1) * size_links_per_element, lvl, u)

    return {
        "cur_count": cur_count,
        "maxlevel": maxlevel,
        "M": M, "maxM": maxM, "maxM0": maxM0, "ef_construction": ef_construction,
        "level_nodes": level_nodes,
        "level_edges": level_edges,
        "level_hist": level_hist,
    }


def build(data: np.ndarray, seed: int, M: int, ef: int, space: str) -> dict:
    """Build one single-threaded HNSW index and parse its graph."""
    n, d = data.shape
    idx = hnswlib.Index(space=space, dim=d)
    idx.init_index(max_elements=n, ef_construction=ef, M=M, random_seed=seed)
    idx.set_num_threads(1)
    idx.add_items(data, np.arange(n), num_threads=1)
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as fh:
        path = fh.name
    idx.save_index(path)
    try:
        return parse_hnsw(path)
    finally:
        Path(path).unlink(missing_ok=True)


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def compare(g1: dict, g2: dict) -> dict:
    """Per-layer node & edge stability between two graphs."""
    layers = sorted(set(g1["level_nodes"]) | set(g2["level_nodes"]))
    rows = {}
    for L in layers:
        n1 = g1["level_nodes"].get(L, set())
        n2 = g2["level_nodes"].get(L, set())
        e1 = g1["level_edges"].get(L, set())
        e2 = g2["level_edges"].get(L, set())
        rows[L] = {
            "n1": len(n1), "n2": len(n2),
            "node_jac": jaccard(n1, n2),
            "node_shared": len(n1 & n2),
            "e1": len(e1), "e2": len(e2),
            "edge_jac": jaccard(e1, e2),
            "edge_shared": len(e1 & e2),
        }
    return rows


def fmt_pct(x: float) -> str:
    return f"{100 * x:6.2f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--M", type=int, default=16)
    ap.add_argument("--ef", type=int, default=200)
    ap.add_argument("--space", default="l2", choices=["l2", "cosine", "ip"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--data-seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.data_seed)
    data = rng.standard_normal((args.n, args.dim)).astype(np.float32)

    print(f"dataset: n={args.n} dim={args.dim} space={args.space} | "
          f"M={args.M} ef_construction={args.ef} | single-threaded")
    print(f"seeds: {args.seeds}\n")

    graphs = {}
    for s in args.seeds:
        graphs[s] = build(data, s, args.M, args.ef, args.space)

    # ---- layer size profile (from the first seed) ----
    g0 = graphs[args.seeds[0]]
    print(f"top level = {g0['maxlevel']}   maxM(L>0)={g0['maxM']}  maxM0={g0['maxM0']}")
    print("\nlayer size profile (seed %d):" % args.seeds[0])
    print(f"  {'layer':>5} {'#nodes':>9} {'#edges':>10} {'avg deg':>8}")
    for L in sorted(g0["level_nodes"]):
        nn = len(g0["level_nodes"][L])
        ne = len(g0["level_edges"][L])
        print(f"  {L:>5} {nn:>9} {ne:>10} {ne / nn:>8.2f}")

    # ---- determinism sanity check: same seed twice must be identical ----
    print("\n[sanity] rebuilding seed %d again (must be identical)..." % args.seeds[0])
    g_repeat = build(data, args.seeds[0], args.M, args.ef, args.space)
    det = compare(g0, g_repeat)
    identical = all(r["node_jac"] == 1.0 and r["edge_jac"] == 1.0 for r in det.values())
    print(f"  identical across every layer: {identical}")

    # ---- cross-seed stability ----
    pairs = list(combinations(args.seeds, 2))
    print(f"\ncross-seed stability, averaged over {len(pairs)} seed pairs:")
    print(f"  {'layer':>5} {'#nodes':>9} {'node stab':>10} {'#edges':>10} {'edge stab':>10}")

    all_layers = sorted(set().union(*[set(g["level_nodes"]) for g in graphs.values()]))
    agg = {L: {"node_jac": [], "edge_jac": [], "n": [], "e": []} for L in all_layers}
    for a, b in pairs:
        rows = compare(graphs[a], graphs[b])
        for L, r in rows.items():
            agg[L]["node_jac"].append(r["node_jac"])
            agg[L]["edge_jac"].append(r["edge_jac"])
            agg[L]["n"].append((r["n1"] + r["n2"]) / 2)
            agg[L]["e"].append((r["e1"] + r["e2"]) / 2)

    for L in all_layers:
        nn = np.mean(agg[L]["n"])
        ne = np.mean(agg[L]["e"])
        nj = np.mean(agg[L]["node_jac"])
        ej = np.mean(agg[L]["edge_jac"])
        print(f"  {L:>5} {nn:>9.0f} {fmt_pct(nj):>10} {ne:>10.0f} {fmt_pct(ej):>10}")

    # ---- whole-graph edge stability (union of all layers) ----
    def all_edges(g):
        out = set()
        for L, es in g["level_edges"].items():
            out |= {(L, u, v) for (u, v) in es}
        return out

    ejs = [jaccard(all_edges(graphs[a]), all_edges(graphs[b])) for a, b in pairs]
    print(f"\nwhole-graph edge stability (all layers): {fmt_pct(np.mean(ejs))}")
    print("(node id = insertion order; edges compared by external label)")


if __name__ == "__main__":
    main()
