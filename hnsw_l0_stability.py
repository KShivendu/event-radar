#!/usr/bin/env python3
"""Can we stabilize HNSW layer-0 edges across rebuilds? Two levers:

  (1) canonical insertion order  — sort by a data-derived key instead of arrival
      order, so every rebuild inserts in the same sequence -> identical graph.
  (2) larger ef_construction     — wider build search converges toward the
      order-invariant "ideal" heuristic graph.

We measure L0 directed-edge Jaccard (by document label) across rebuilds that use
*different arrival orders*, comparing: random order (baseline), canonical order,
and an ef_construction sweep.
"""
from __future__ import annotations

import argparse
from itertools import combinations

import numpy as np

from hnsw_stability import build, jaccard


def l0_edges(g):
    return g["level_edges"].get(0, set())


def avg_l0_stability(graphs):
    pairs = list(combinations(range(len(graphs)), 2))
    return float(np.mean([jaccard(l0_edges(graphs[a]), l0_edges(graphs[b]))
                          for a, b in pairs]))


def pca1_order(data):
    """Deterministic total order: project onto the top principal component."""
    xc = data - data.mean(0)
    # top right-singular vector; randomized-free, fine at 10k x 768
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    key = xc @ vt[0]
    return np.argsort(key, kind="stable")


def load(path, n):
    x = np.load(path)
    arr = x["embeddings"] if hasattr(x, "files") else x
    return np.ascontiguousarray(arr[:n]).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--M", type=int, default=16)
    ap.add_argument("--space", default="cosine")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--arrival-seeds", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--ef-sweep", type=int, nargs="+", default=[40, 100, 200, 400, 800])
    args = ap.parse_args()

    data = load(args.data, args.n)
    print(f"data={args.data} n={data.shape[0]} dim={data.shape[1]} "
          f"space={args.space} M={args.M} seed={args.seed} single-thread")
    arrivals = [np.random.default_rng(s).permutation(data.shape[0])
                for s in args.arrival_seeds]

    # ---- baseline: random arrival order, ef_construction=200 ----
    print("\n(1) BASELINE — random arrival order (ef_construction=200):")
    base = [build(data, args.seed, args.M, 200, args.space, order=o) for o in arrivals]
    print(f"    L0 edge stability across {len(arrivals)} orders: "
          f"{100*avg_l0_stability(base):.2f}%")

    # ---- fix: canonical (data-derived) order — arrival order no longer matters ----
    print("\n(2) FIX — canonical PCA-1 order (same for every rebuild):")
    canon = pca1_order(data)
    # build twice to confirm determinism; arrival shuffles are irrelevant now
    c1 = build(data, args.seed, args.M, 200, args.space, order=canon)
    c2 = build(data, args.seed, args.M, 200, args.space, order=canon)
    print(f"    L0 edge stability (two rebuilds): "
          f"{100*jaccard(l0_edges(c1), l0_edges(c2)):.2f}%")

    # ---- lever: does bigger ef_construction reduce order sensitivity? ----
    print("\n(3) ef_construction sweep — L0 stability across random arrival orders:")
    print(f"    {'ef_c':>6} {'L0 edge stab':>14} {'#L0 edges':>12}")
    for ef in args.ef_sweep:
        gs = [build(data, args.seed, args.M, ef, args.space, order=o) for o in arrivals]
        ne = int(np.mean([len(l0_edges(g)) for g in gs]))
        print(f"    {ef:>6} {100*avg_l0_stability(gs):>13.2f}% {ne:>12}")


if __name__ == "__main__":
    main()
