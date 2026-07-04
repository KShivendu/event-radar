#!/usr/bin/env python3
"""Do two *different* embedding models share exploitable structure?

If yes, a model swap can warm-start / partially reuse the old index. If no, the
only path is full reindex + serving-layer cutover. We test three things between
model X and model Y embeddings of the SAME documents (by id):

  1. raw cross-model neighbor overlap — do X and Y agree on top-k neighbors with
     NO alignment? (expected: low, they're different spaces)
  2. linear CKA — representational-similarity up to a linear transform (0..1).
  3. learned linear map X->Y (ridge, fit on a train split, evaluated on held-out
     points): can we recover a point's true Y-neighbors from its X vector? This
     is the direct "is warm-start / alignment viable" test. Compared against a
     random-map control so we know the number isn't trivially inflated.

Usage: python3 align_test.py X=multibert.npz Y=minilm.npz --k 10
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


def normalize(x):
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-9, None)


def knn(x, k, xq=None):
    ix = faiss.IndexFlatIP(x.shape[1])
    ix.add(x)
    q = x if xq is None else xq
    extra = 1 if xq is None else 0
    _, I = ix.search(q, k + extra)
    if xq is None:
        return np.array([r[r != i][:k] for i, r in enumerate(I)])
    return I[:, :k]


def mean_overlap(nnA, nnB):
    k = nnA.shape[1]
    return float(np.mean([len(set(a) & set(b)) / k for a, b in zip(nnA, nnB)]))


def linear_cka(X, Y):
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    hsic = np.linalg.norm(X.T @ Y, "fro") ** 2
    return float(hsic / (np.linalg.norm(X.T @ X, "fro") * np.linalg.norm(Y.T @ Y, "fro")))


def ridge_map(Xtr, Ytr, alpha):
    # W minimizes ||Xtr W - Ytr||^2 + alpha||W||^2
    d = Xtr.shape[1]
    A = Xtr.T @ Xtr + alpha * np.eye(d, dtype=np.float64)
    return np.linalg.solve(A, Xtr.T @ Ytr.astype(np.float64))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs=2, help="X=path.npz Y=path.npz")
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--test-frac", type=float, default=0.2)
    args = ap.parse_args()

    (lx, px), (ly, py) = (m.split("=", 1) for m in args.models)
    X = normalize(load(px, args.n))
    Y = normalize(load(py, args.n))
    n = X.shape[0]
    print(f"X={lx} {X.shape}  Y={ly} {Y.shape}  n={n} k={args.k}\n")

    # 1. raw cross-model neighbor agreement (no alignment)
    nnX = knn(X, args.k)
    nnY = knn(Y, args.k)
    print(f"1) raw cross-model top-{args.k} neighbor overlap: "
          f"{100*mean_overlap(nnX, nnY):.2f}%   (how much X and Y agree unaligned)")

    # 2. linear CKA
    print(f"2) linear CKA(X, Y): {linear_cka(X, Y):.3f}   (0=unrelated, 1=identical up to linear map)")

    # 3. learned linear map X->Y, evaluated on held-out points
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    ntest = int(n * args.test_frac)
    te, tr = perm[:ntest], perm[ntest:]
    W = ridge_map(X[tr].astype(np.float64), Y[tr], args.alpha)
    Xte_mapped = normalize((X[te].astype(np.float64) @ W).astype(np.float32))

    # true Y-neighbors of the test points (ids into the full Y set)
    true_nnY = knn(Y, args.k)[te]
    # neighbors retrieved by querying the full Y index with the mapped X vectors
    mapped_nnY = knn(Y, args.k, xq=Xte_mapped)
    aligned = float(np.mean([len(set(a) & set(b)) / args.k
                             for a, b in zip(mapped_nnY, true_nnY)]))

    # random-map control: same shapes, random W
    Wr = rng.standard_normal(W.shape) / np.sqrt(W.shape[0])
    Xte_rand = normalize((X[te].astype(np.float64) @ Wr).astype(np.float32))
    rand_nnY = knn(Y, args.k, xq=Xte_rand)
    ctrl = float(np.mean([len(set(a) & set(b)) / args.k
                          for a, b in zip(rand_nnY, true_nnY)]))

    print(f"3) learned linear map X->Y (ridge, held-out {ntest} pts):")
    print(f"     recover true Y top-{args.k} neighbors from mapped X: {100*aligned:.2f}%")
    print(f"     random-map control:                                 {100*ctrl:.2f}%")
    print(f"     (raw same-space Y self-consistency ceiling is 100%)")

    print("\nverdict:")
    if aligned > 3 * max(ctrl, 0.01) and aligned > 0.20:
        print(f"  spaces ARE substantially alignable ({100*aligned:.0f}% recovery) "
              f"-> warm-start / partial transfer is viable.")
    else:
        print(f"  little exploitable shared structure ({100*aligned:.0f}% recovery vs "
              f"{100*ctrl:.0f}% random) -> full reindex + serving cutover only.")


if __name__ == "__main__":
    main()
