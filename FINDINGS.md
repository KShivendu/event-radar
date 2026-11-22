# HNSW / IVF rebuild stability & embedding drift — findings

An empirical study of how stable approximate-nearest-neighbor indexes are across
rebuilds, and how that relates to **embedding drift** (the vectors changing when
the embedding model changes). All numbers below are measured, not asserted.

## TL;DR

- With **fixed insertion order + fixed seed + single thread**, HNSW rebuilds are
  bit-identical. That is the *only* condition under which they are.
- The earlier-looking "HNSW base layer is 99.6% stable" was an **artifact of
  fixed insertion order**. Under realistic (order-varying) rebuilds, base-layer
  edge overlap collapses to **~31%** — a real rebuild changes **~70% of edges**.
- **Node-to-layer assignment is decided by the RNG in insertion order, not by the
  data.** So insertion order — not the random seed — is what actually drives
  instability. The seed barely touches layer 0.
- You can make L0 fully reproducible by **canonicalizing insertion order**
  (sort by a data-/id-derived key). Raising `ef_construction` does **not** help.
- **Indexes exaggerate embedding drift.** Where the true nearest neighbors were
  ~58% retained between checkpoints, the HNSW graph shared only ~19% of edges.
  Measure drift on the **raw vectors** (kNN overlap), not by diffing indexes.
- **IVF is less seed-stable than HNSW**: reseeding k-means on identical data
  reassigns **~54% of points**. Membership metrics are noisy; **centroid
  displacement** is the clean signal.
- You **cannot merge HNSW graphs from two different models** (different vector
  spaces). But two *different* models still share exploitable structure
  (CKA 0.61; a learned linear map recovers 36% of neighbors vs 0% random), so a
  **warm-start is viable** — though a full reindex is still needed for full recall.

---

## Setup

- **Corpus:** 10,000 AG News snippets (same texts, same order, across all runs).
- **Embedding models:**
  - `google/multiberts-seed_0-step_{0k,20k,100k,200k,2000k}` — from-scratch BERT
    pretraining checkpoints (mean-pooled, L2-normalized, 768-d). Chosen because
    the tokenizer/pooling are identical across checkpoints, so any change is pure
    weight evolution. `step_0k` = random init; `step_2000k` = fully trained.
  - `sentence-transformers/all-MiniLM-L6-v2` — a genuinely different model
    (contrastive sentence encoder, 384-d) for the cross-model test.
- **Indexes:** `hnswlib` (HNSW), `faiss` (IVF k-means + exact kNN).
- **HNSW params:** `M=16`, `ef_construction=200`, cosine space, single-threaded.
- **Metric of comparison:** everything compared **by document id**, so results are
  invariant to internal reordering. HNSW graphs parsed directly from hnswlib's
  saved binary to recover exact per-layer node and edge sets.

### Environment notes
- **Modal (GPU) can't run from the sandbox** — its gRPC is blocked by the proxy
  policy. `modal_embed.py` is provided to run GPU embedding from your own machine.
- **mxbai-embed-large was too slow** on the throttled 4-vCPU box (~8h for 10k), so
  we used the smaller MultiBERTs / MiniLM models — real embeddings, fast enough.

---

## 1. HNSW rebuild stability

HNSW layers: layer 0 holds every point and does the fine search; thin upper
layers are express lanes. We rebuilt the same data repeatedly and compared each
layer's node set and directed-edge set (Jaccard overlap = stability).

### 1a. Vary the random seed, fixed insertion order

| layer | node stability | edge stability |
|------:|---------------:|---------------:|
| 0     | 100%           | **99.98%**     |
| 1     | 3.5%           | 0.08%          |
| whole graph | —        | 92.0%          |

Layer 0 looks rock-solid. **But this is misleading** — see 1b.

### 1b. Vary the insertion order, fixed seed (the realistic case)

| layer | node stability | edge stability |
|------:|---------------:|---------------:|
| 0     | 100%\*         | **31.5%**      |
| 1     | 3.3%           | 0.09%          |
| whole graph | —        | **29.8%**      |

\*L0 node stability is 100% trivially (every point is on layer 0). The meaningful
L0 metric is edges.

**Interpretation.** Real rebuilds rarely preserve insertion order (re-ingestion,
segment merges, parallel writes). Once order varies, base-layer edge overlap
falls from ~100% to **~31.5%**, and whole-graph overlap from 92% to **~30%**.
So a realistic HNSW rebuild changes **~70% of its edges**.

Why order dominates and the seed doesn't:
- hnswlib draws each node's **level from the RNG in insertion order**, so changing
  the order reshuffles which nodes reach the upper layers even at a fixed seed.
- Layer 0 is nearly a deterministic function of *(data, insertion order)* and is
  almost independent of the seed — the seed only affects upper-layer promotion.

**Determinism check:** same seed + same order + single thread ⇒ bit-identical
(verified on every layer). Multi-threaded inserts break this (nondeterministic
interleaving), which is why single-thread was used throughout.

---

## 2. What decides an L0 edge, and how to stabilize it

When node X is inserted, its layer-0 neighbors are chosen by:
1. a **beam search** (width `ef_construction`) collecting nearby already-inserted
   candidates;
2. the **neighbor-selection heuristic** (`getNeighborsByHeuristic2`) — keeps a
   diverse subset, not simply the M nearest;
3. **bidirectional connect + prune** — links are added both ways, and an
   over-full neighbor's list is re-pruned.

All three depend on the accumulated graph state at insertion time ⇒ **order
dependence**.

### Stabilization experiment

| approach | L0 edge stability across rebuilds |
|---|---|
| Random insertion order (baseline) | 31.4% |
| **Canonical PCA-1 order** | **100.00%** |
| `ef_construction` 40 → 800 | 32.7% → 29.3% (flat/worse) |

- **Canonical insertion order works**: sort the vectors by a data-derived key
  before insertion → every rebuild uses the same sequence → identical graph.
- **`ef_construction` is not a stability lever** — bigger build search improves
  recall but *slightly worsens* structural stability (more borderline edges to
  flip).

**Practical caveat.** PCA-1 order is a function of the whole set, so adds/deletes
reshuffle it, and it moves under embedding drift (it's derived from the vectors).
For incremental/production use, order by **`hash(point_id)`** instead — per-point,
set-independent, and stable under drift. Optionally also derive each node's
**level from `hash(id)`** to pin the layer skeleton too.

---

## 3. Embedding drift — the ground truth

We embedded the same documents with 5 training checkpoints and measured how each
document's exact nearest neighbors changed — **index-free**, in the raw vector
space (this is the true drift the index only approximates).

### Exact top-10 neighbor retention

| transition | true kNN@10 retention | HNSW L0 edge overlap |
|---|---|---|
| 0k → 20k | 20.6% | 4.7% |
| 20k → 100k | 45.5% | 12.7% |
| 100k → 200k | 58.2% | 18.8% |
| 200k → 2000k | 50.5% | 14.8% |

Two findings:

**Indexes amplify drift.** At `100k→200k` the true neighbors were ~58% retained,
but the HNSW graph shared only ~19% of edges — the heuristic turns small geometry
shifts into large edge changes. **Measure drift on vectors, not indexes.**

**Drift is gentler and more local than the top-10 number suggests** (200k→2000k):

| view | value |
|---|---|
| A's top-10 still in B's top-10 | 50.4% |
| A's top-10 still in B's top-50 | 82.5% |
| A's top-10 still in B's **top-100** | **90.3%** |
| per-point retention p10 / p50 / p90 | 20% / 50% / 80% |
| points keeping 0/10 neighbors | 0.9% |
| points keeping ≥8/10 neighbors | 12.7% |

So 90% of old neighbors stay *within the local neighborhood* (they get
re-ranked, not ejected), and drift is **heterogeneous** — some regions barely
move, others churn. Note the MultiBERTs checkpoints are *pretraining* steps
(violent early reorganization); a gentle finetune of a converged model would
drift far less.

---

## 4. IVF stability & drift

IVF has no edges — its structure is a k-means partition (centroids + which
inverted list each point lands in). So we use label-invariant partition metrics.

### 4a. Rebuild stability (vary k-means seed, same data), nlist=256

| metric | value | note |
|---|---|---|
| **reassignment %** | **54.2%** | over half of points change list on reseed |
| ARI | 0.36 | moderate (1=identical, 0=random) |
| NMI | 0.74 | partitions share much information |
| centroid drift | 0.83× spacing | centroids move ~as far as the gap to their nearest neighbor |
| Rand | 0.993 | **degenerate** at large nlist — ignore |

### 4b. Training drift (fixed seed, vary checkpoints)

| transition | ARI | reassign % | centroid drift (abs) |
|---|---|---|---|
| 0k → 20k | 0.06 | 84% | 1.35 |
| 20k → 100k | 0.31 | 57% | 0.89 |
| 100k → 200k | 0.40 | 47% | 0.45 |
| 200k → 2000k | 0.35 | 51% | 0.39 |

**Interpretation.** IVF is **much less seed-stable than HNSW** — reseeding on
identical data reassigns ~54% of points (k-means has many near-equivalent optima
and cluster boundaries flip easily). Membership metrics (ARI/reassign) are noisy
and, late in training, indistinguishable from seed noise; **absolute centroid
displacement** is the cleaner drift signal (converges 1.35 → 0.39). Rand/co-
assignment is useless at realistic nlist (always ~0.99).

---

## 5. Merging indexes from two different embedding models

**You cannot merge the graphs.** Two models = two incompatible vector spaces;
cross-model distances are meaningless, so a graph union has lying edges. The
mergeable unit is the **collection of points (by id)**, not the graphs.

Options for a smooth transition (prod model A → new model B):
1. **Blue/green reindex + cutover** — build B offline, validate, then
   shadow → canary → switch. (Default; do this unless you have a reason not to.)
2. **Dual-serve with rank fusion (RRF)** during the window — RRF is rank-based, so
   it sidesteps incomparable cross-model scores. Shift weight A → B gradually.
3. **Warm-start B from A's topology** — seed B's construction with A's edges, then
   repair under B's distances. Viable only if A and B are correlated (see §7).
4. **Qdrant named vectors** — store both embeddings per point; two indexes under
   one collection; flip which vector the query targets.

Rules: a query must be embedded by the **same model** as the index it hits; never
use cross-model score thresholds (use RRF); validate on **recall**, not structure;
different dimensionalities ⇒ separate indexes are mandatory.

---

## 6. Would a slime-mold / ant-colony index be more drift-robust?

No, not for a model swap, and worse for reproducibility:
- These are **stochastic optimizers** (random walks / pheromone) — two builds
  differ *more* than HNSW, not less.
- They could be ergonomically nicer for **slow, continuous drift** (self-healing
  reinforcement/decay, no cutovers), but a big model swap is a discontinuous jump
  best handled by rebuild + cutover; incremental adaptation would lag with
  degraded recall, and reinforcement is query-distribution-dependent (cold regions
  never earn good edges).
- The transferable idea: a **usage-driven reinforce-used / decay-unused +
  local-repair maintenance layer** on top of HNSW — not a new index family.

---

## 7. Do two *different* models share exploitable structure?

MultiBERTs (768-d, MLM) vs all-MiniLM (384-d, contrastive) on the same 10k docs:

| test | result | meaning |
|---|---|---|
| raw cross-model top-10 overlap | **31.8%** | agreement with no alignment |
| linear CKA | **0.612** | moderately related up to a linear map |
| learned linear map (held-out) recovers true neighbors | **36.3%** | structure transfers |
| random-map control | 0.03% | confirms 36% is real |

**Verdict:** even "totally different" text models share substantial structure
(consistent with the Platonic-representation / relative-representations line of
work). So a **warm-start / partial transfer across a model swap is viable** — but
36% is far from perfect, so a full reindex is still required for full recall.
Alignment is a bootstrap, not a replacement. For any given pair, **measure**
alignability (CKA + map recovery) before relying on it.

---

## Practical recommendations

1. Index structure is **never** bit-stable across realistic rebuilds or model
   swaps — don't chase that. It's expected, not a bug.
2. Want reproducible builds? Canonicalize insertion order by **`hash(id)`**,
   derive **levels from `hash(id)`**, and build single-threaded (or with a
   deterministic parallel merge).
3. Monitor **embedding drift on the raw vectors** (kNN neighbor retention on a
   fixed anchor set) to decide *when* to rebuild — not by diffing indexes.
4. For a model swap: **full reindex + gradual serving cutover** (blue/green + RRF).
   Warm-start only if a CKA / map-recovery check says the models are alignable.
5. Judge any transition on **recall stability**, the invariant that actually
   matters — two structurally different indexes can retrieve nearly the same
   neighbors.

---

## Scripts (all on branch `claude/hnsw-rebuild-stability-l38zet`)

| file | what it does |
|---|---|
| `hnsw_stability.py` | Rebuild stability; `--vary seed` or `--vary order`; per-layer node/edge Jaccard. Parses hnswlib binary for exact graph. |
| `hnsw_drift.py` | HNSW graph drift across checkpoint embeddings (fixed seed). |
| `hnsw_l0_stability.py` | L0 stabilization: canonical PCA-1 order vs `ef_construction` sweep. |
| `ivf_stability.py` | IVF partition stability (seed) & drift (checkpoints): ARI/NMI/Rand/reassign/centroid drift. |
| `knn_drift.py` | Index-free ground-truth drift: exact top-k neighbor overlap. |
| `align_test.py` | Cross-model alignability: raw overlap, CKA, learned linear map vs control. |
| `modal_embed.py` | GPU embedding app to run from your own machine (Modal). |

### Reproduce
```bash
pip install hnswlib faiss-cpu scikit-learn scipy numpy sentence-transformers datasets

# rebuild stability: seed vs order
python3 hnsw_stability.py --data emb.npz --space cosine --vary seed  --seeds 104729 224737 350377 499979 611953
python3 hnsw_stability.py --data emb.npz --space cosine --vary order --seeds 104729 224737 350377 499979 611953

# L0 stabilization
python3 hnsw_l0_stability.py --data emb.npz --space cosine

# ground-truth embedding drift
python3 knn_drift.py 0k=a.npz 20k=b.npz 100k=c.npz 200k=d.npz 2000k=e.npz --k 10

# IVF stability & drift
python3 ivf_stability.py --data emb.npz --nlist 256 --seeds 104729 224737 350377 499979 611953
python3 ivf_stability.py --nlist 256 --checkpoints 0k=a.npz 20k=b.npz 200k=d.npz 2000k=e.npz

# cross-model alignability
python3 align_test.py X=multibert.npz Y=minilm.npz --k 10
```
