#!/usr/bin/env python3
"""Measure IVF index stability across rebuilds.

HNSW stability is about graph edges. An IVF index has no edges — its structure
is a k-means partition: `nlist` centroids, and every vector assigned to its
nearest centroid (its inverted list). So "how much did the index change on
rebuild" becomes "how much did the partition change". We measure it with
label-invariant metrics (cluster ids are arbitrary across runs):

  * ARI / NMI            — partition agreement, invariant to label permutation.
  * Rand (co-assignment) — fraction of point PAIRS whose same-list / different-
                           list relationship is preserved. Needs no matching.
  * reassignment %       — after optimally matching run A's lists to run B's
                           (Hungarian on the overlap matrix), what fraction of
                           points landed in a different list. The direct analog
                           of "how many points changed inverted list".
  * centroid drift       — centroids aren't real points and are unordered, so we
                           Hungarian-match them between runs and report how far
                           matched centroids moved, in units of the inter-centroid
                           spacing (drift << 1 => partition barely moved).

Two modes, mirroring the HNSW scripts:
  seeds : vary the k-means seed on ONE dataset (rebuild stability)
  drift : fix the seed, vary the embeddings (e.g. model checkpoints)

Usage:
    python3 ivf_stability.py --data emb.npz --nlist 256 --seeds 1 2 3 4 5
    python3 ivf_stability.py --nlist 256 --hnsw-seed 42 \
        --checkpoints 0k=a.npz 20k=b.npz 200k=c.npz
"""
from __future__ import annotations

import argparse
from itertools import combinations

import faiss
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             rand_score)

faiss.omp_set_num_threads(1)  # single-thread => deterministic given a seed


def build_ivf(data: np.ndarray, nlist: int, seed: int, niter: int = 25,
              spherical: bool = True):
    """Train the IVF quantizer (k-means) and return (centroids, assignment)."""
    d = data.shape[1]
    km = faiss.Kmeans(d, nlist, niter=niter, seed=seed, spherical=spherical,
                      verbose=False)
    km.train(data)
    centroids = np.asarray(km.centroids).reshape(nlist, d)
    _, assign = km.index.search(data, 1)
    return centroids, assign.ravel().astype(np.int64)


def centroid_drift(cA: np.ndarray, cB: np.ndarray):
    """Hungarian-match centroids, report matched displacement vs spacing."""
    # cost = pairwise euclidean distance between the two centroid sets
    d2 = ((cA[:, None, :] - cB[None, :, :]) ** 2).sum(-1)
    ri, ci = linear_sum_assignment(d2)
    disp = np.sqrt(d2[ri, ci])
    # inter-centroid spacing within run A: mean nearest-neighbour distance
    aa = ((cA[:, None, :] - cA[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(aa, np.inf)
    spacing = np.sqrt(aa.min(1)).mean()
    return disp.mean(), np.median(disp), disp.mean() / spacing


def reassignment_frac(aA: np.ndarray, aB: np.ndarray, nlist: int) -> float:
    """After optimally matching lists A->B, fraction of points that moved."""
    C = np.zeros((nlist, nlist), dtype=np.int64)
    np.add.at(C, (aA, aB), 1)
    ri, ci = linear_sum_assignment(-C)  # maximise overlap
    matched = C[ri, ci].sum()
    return 1.0 - matched / len(aA)


def compare(aA, aB, cA, cB, nlist):
    md, mdn, rel = centroid_drift(cA, cB)
    return {
        "ari": adjusted_rand_score(aA, aB),
        "nmi": normalized_mutual_info_score(aA, aB),
        "rand": rand_score(aA, aB),
        "reassign": reassignment_frac(aA, aB, nlist),
        "cdrift_mean": md, "cdrift_med": mdn, "cdrift_rel": rel,
    }


def list_size_stats(assign, nlist):
    sizes = np.bincount(assign, minlength=nlist)
    return sizes.min(), sizes.max(), sizes.mean(), sizes.std()


def load(path, n):
    x = np.load(path)
    arr = x["embeddings"] if hasattr(x, "files") else x
    return np.ascontiguousarray(arr[:n]).astype(np.float32)


def print_row(tag, m):
    print(f"  {tag:<16} ARI={m['ari']:.3f}  NMI={m['nmi']:.3f}  "
          f"Rand={m['rand']:.3f}  reassign={100*m['reassign']:5.1f}%  "
          f"centroid-drift={m['cdrift_rel']:.3f}x spacing "
          f"(abs {m['cdrift_mean']:.3f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--nlist", type=int, default=256)
    ap.add_argument("--niter", type=int, default=25)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--spherical", type=int, default=1,
                    help="1 for normalized/cosine embeddings, 0 for L2")
    ap.add_argument("--checkpoints", nargs="+", default=None,
                    help="drift mode: label=path.npz ... (fixed seed)")
    ap.add_argument("--kmeans-seed", type=int, default=42)
    args = ap.parse_args()
    sph = bool(args.spherical)

    if args.checkpoints:
        # ---- DRIFT MODE: fixed seed, evolving embeddings ----
        ckpts = [c.split("=", 1) for c in args.checkpoints]
        print(f"IVF DRIFT | nlist={args.nlist} niter={args.niter} "
              f"kmeans_seed={args.kmeans_seed} spherical={sph} single-thread")
        built = []
        for label, path in ckpts:
            data = load(path, args.n)
            c, a = build_ivf(data, args.nlist, args.kmeans_seed, args.niter, sph)
            built.append((label, c, a))
            mn, mx, mean, sd = list_size_stats(a, args.nlist)
            print(f"[{label:>6}] n={len(a)} dim={data.shape[1]} "
                  f"list size min/mean/max={mn}/{mean:.0f}/{mx} (sd {sd:.0f})")
        print("\nconsecutive checkpoints:")
        for (l1, c1, a1), (l2, c2, a2) in zip(built, built[1:]):
            print_row(f"{l1}->{l2}", compare(a1, a2, c1, c2, args.nlist))
        fl, fc, fa = built[-1]
        print(f"\nvs final ({fl}):")
        for l, c, a in built[:-1]:
            print_row(f"{l}->{fl}", compare(a, fa, c, fc, args.nlist))
        return

    # ---- SEED MODE: fixed data, varying k-means seed (rebuild stability) ----
    if args.data:
        data = load(args.data, args.n)
        src = args.data
    else:
        rng = np.random.default_rng(0)
        data = rng.standard_normal((args.n, args.dim)).astype(np.float32)
        src = "synthetic gaussian"
    print(f"IVF rebuild stability | data={src} n={data.shape[0]} dim={data.shape[1]}")
    print(f"nlist={args.nlist} niter={args.niter} spherical={sph} "
          f"seeds={args.seeds} single-thread\n")

    builds = {s: build_ivf(data, args.nlist, s, args.niter, sph) for s in args.seeds}
    mn, mx, mean, sd = list_size_stats(builds[args.seeds[0]][1], args.nlist)
    print(f"list size (seed {args.seeds[0]}): min={mn} mean={mean:.0f} "
          f"max={mx} sd={sd:.0f}")

    print("\n[sanity] same seed twice must be identical:")
    c2, a2 = build_ivf(data, args.nlist, args.seeds[0], args.niter, sph)
    same = np.array_equal(a2, builds[args.seeds[0]][1])
    print(f"  identical assignment: {same}")

    print("\ncross-seed stability (each seed pair):")
    accum = {k: [] for k in ("ari", "nmi", "rand", "reassign", "cdrift_rel")}
    for a, b in combinations(args.seeds, 2):
        cA, aA = builds[a]
        cB, aB = builds[b]
        m = compare(aA, aB, cA, cB, args.nlist)
        print_row(f"seed {a} vs {b}", m)
        for k in accum:
            accum[k].append(m[k])
    print("\naveraged over all seed pairs:")
    print(f"  ARI={np.mean(accum['ari']):.3f}  NMI={np.mean(accum['nmi']):.3f}  "
          f"Rand={np.mean(accum['rand']):.3f}  "
          f"reassign={100*np.mean(accum['reassign']):.1f}%  "
          f"centroid-drift={np.mean(accum['cdrift_rel']):.3f}x spacing")


if __name__ == "__main__":
    main()
