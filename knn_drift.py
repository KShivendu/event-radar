#!/usr/bin/env python3
"""Ground-truth embedding drift: exact top-k neighbor overlap between checkpoints.

This is the clean, index-free drift signal. For each document we compute its
EXACT top-k nearest neighbors (brute force) under each checkpoint's embeddings,
then measure how much that neighbor set changes across checkpoints — by document
id, so it is directly comparable. No HNSW, no IVF, no build randomness: whatever
changes here is purely the embedding geometry moving.

Contrast with hnsw_drift.py / ivf_stability.py, whose numbers are this signal
seen through a noisy index build.

Usage:
    python3 knn_drift.py 0k=a.npz 20k=b.npz ... --k 10
"""
from __future__ import annotations

import argparse

import faiss
import numpy as np

faiss.omp_set_num_threads(1)


def load(path, n):
    x = np.load(path)
    arr = x["embeddings"] if hasattr(x, "files") else x
    return np.ascontiguousarray(arr[:n]).astype(np.float32)


def exact_knn(data, k):
    """Exact top-k neighbor ids per row (self excluded), via flat inner product.
    Embeddings are assumed L2-normalized, so IP == cosine."""
    n, d = data.shape
    index = faiss.IndexFlatIP(d)
    index.add(data)
    _, idx = index.search(data, k + 1)  # +1 because row is its own NN
    out = np.empty((n, k), dtype=np.int64)
    for i in range(n):
        row = idx[i][idx[i] != i][:k]
        out[i] = row
    return out


def overlap(nnA, nnB):
    """Mean over points of |A∩B|/k (neighbor-set retention)."""
    k = nnA.shape[1]
    vals = [len(set(a) & set(b)) / k for a, b in zip(nnA, nnB)]
    return float(np.mean(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="+", help="label=path.npz, in training order")
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    ckpts = [c.split("=", 1) for c in args.checkpoints]
    print(f"exact top-{args.k} neighbor overlap (index-free drift), n={args.n}\n")

    nns = {}
    for label, path in ckpts:
        nns[label] = exact_knn(load(path, args.n), args.k)
        print(f"  computed kNN for {label}")

    labels = [c[0] for c in ckpts]
    print(f"\nconsecutive checkpoints — top-{args.k} neighbor retention:")
    for l1, l2 in zip(labels, labels[1:]):
        print(f"  {l1:>6} -> {l2:<6} {100*overlap(nns[l1], nns[l2]):6.2f}%")

    fl = labels[-1]
    print(f"\nvs final ({fl}):")
    for l in labels[:-1]:
        print(f"  {l:>6} -> {fl:<6} {100*overlap(nns[l], nns[fl]):6.2f}%")


if __name__ == "__main__":
    main()
