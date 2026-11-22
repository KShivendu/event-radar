#!/usr/bin/env python3
"""Measure HNSW graph drift as an embedding model trains.

Companion to hnsw_stability.py. There we held the data fixed and varied the
HNSW random_seed. Here we do the opposite: hold the HNSW build *fixed* (same
seed, single thread) and feed it the SAME corpus embedded by successive
checkpoints of an embedding model. Any change in the graph is then attributable
to the embedding geometry evolving during training — not to HNSW's own RNG.

Pass an ordered list of "label=path.npz" checkpoint embeddings. We build one
HNSW per checkpoint and report, per layer, how much the node set and directed
edge set change (1 - Jaccard = drift) between consecutive checkpoints and
against the final checkpoint.

Usage:
    python3 hnsw_drift.py 0k=a.npz 20k=b.npz 100k=c.npz 200k=d.npz 2000k=e.npz \
        [--n N] [--M 16] [--ef 200] [--space cosine] [--hnsw-seed 42]
"""
from __future__ import annotations

import argparse

import numpy as np

from hnsw_stability import build, jaccard


def load(path: str, n: int) -> np.ndarray:
    loaded = np.load(path)
    arr = loaded["embeddings"] if hasattr(loaded, "files") else loaded
    return np.ascontiguousarray(arr[:n]).astype(np.float32)


def layer_drift(g1: dict, g2: dict):
    layers = sorted(set(g1["level_nodes"]) | set(g2["level_nodes"]))
    rows = []
    for L in layers:
        n1, n2 = g1["level_nodes"].get(L, set()), g2["level_nodes"].get(L, set())
        e1, e2 = g1["level_edges"].get(L, set()), g2["level_edges"].get(L, set())
        rows.append((L, len(n1), len(n2), jaccard(n1, n2),
                     len(e1), len(e2), jaccard(e1, e2)))
    return rows


def print_pair(title, g1, g2):
    print(f"\n{title}")
    print(f"  {'layer':>5} {'#nodes':>9} {'node stab':>10} {'#edges':>10} {'edge stab':>10}")
    for L, n1, n2, nj, e1, e2, ej in layer_drift(g1, g2):
        nn = (n1 + n2) / 2
        ne = (e1 + e2) / 2
        print(f"  {L:>5} {nn:>9.0f} {100*nj:>9.2f}% {ne:>10.0f} {100*ej:>9.2f}%")


def whole_graph_stab(g1, g2):
    def edges(g):
        return {(L, u, v) for L, es in g["level_edges"].items() for (u, v) in es}
    return jaccard(edges(g1), edges(g2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="+", help="label=path.npz, in training order")
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--M", type=int, default=16)
    ap.add_argument("--ef", type=int, default=200)
    ap.add_argument("--space", default="cosine", choices=["l2", "cosine", "ip"])
    ap.add_argument("--hnsw-seed", type=int, default=42)
    args = ap.parse_args()

    ckpts = [c.split("=", 1) for c in args.checkpoints]
    labels = [c[0] for c in ckpts]

    print(f"HNSW drift across checkpoints: {labels}")
    print(f"fixed hnsw_seed={args.hnsw_seed}, single-threaded | "
          f"M={args.M} ef={args.ef} space={args.space} n={args.n}\n")

    graphs = {}
    for label, path in ckpts:
        data = load(path, args.n)
        g = build(data, args.hnsw_seed, args.M, args.ef, args.space)
        graphs[label] = g
        top = g["maxlevel"]
        prof = " ".join(f"L{L}:{len(g['level_nodes'][L])}" for L in sorted(g["level_nodes"]))
        print(f"[{label:>6}] n={data.shape[0]} dim={data.shape[1]} top={top}  {prof}")

    # consecutive-checkpoint drift
    print("\n" + "=" * 62)
    print("CONSECUTIVE DRIFT (how much the graph moves each training interval)")
    print("=" * 62)
    for (l1, _), (l2, _) in zip(ckpts, ckpts[1:]):
        print_pair(f"{l1} -> {l2}   (whole-graph edge stab "
                   f"{100*whole_graph_stab(graphs[l1], graphs[l2]):.2f}%)",
                   graphs[l1], graphs[l2])

    # everything vs the final checkpoint
    final_label = labels[-1]
    print("\n" + "=" * 62)
    print(f"VS FINAL CHECKPOINT ({final_label})")
    print("=" * 62)
    for l, _ in ckpts[:-1]:
        print_pair(f"{l} -> {final_label}   (whole-graph edge stab "
                   f"{100*whole_graph_stab(graphs[l], graphs[final_label]):.2f}%)",
                   graphs[l], graphs[final_label])


if __name__ == "__main__":
    main()
