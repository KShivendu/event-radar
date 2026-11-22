#!/usr/bin/env python3
"""Embed a corpus on a Modal GPU, save .npz for the HNSW stability/drift scripts.

Modal uses gRPC, which the Claude-Code sandbox proxy blocks, so run this from
YOUR OWN machine:

    pip install modal
    modal token new                      # opens browser to authenticate
    # embed the same AG News 10k with every MultiBERTs seed_0 checkpoint:
    modal run modal_embed.py --preset multibert
    # or a sentence-transformers model (e.g. mixedbread large) at full 10k:
    modal run modal_embed.py --preset mxbai
    # or anything: --models "intfloat/e5-base-unsupervised,intfloat/e5-base" --pooling mean

Results land in ./embeddings/<label>.npz (key "embeddings"). Then locally:

    python3 hnsw_drift.py 0k=embeddings/multibert_step_0k.npz ... --space cosine
    python3 hnsw_stability.py --data embeddings/mxbai.npz --space cosine
"""
import modal

app = modal.App("hnsw-embed")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch", "transformers", "sentence-transformers",
        "datasets", "numpy",
    )
)

MULTIBERT_STEPS = ["0k", "20k", "40k", "100k", "200k", "1000k", "2000k"]


@app.function(image=image, gpu="A10G", timeout=60 * 30)
def embed_bert_meanpool(repo: str, n: int = 10000, max_len: int = 128) -> bytes:
    """Mean-pooled, L2-normalized embeddings from a BERT-style checkpoint."""
    import io
    import numpy as np
    import torch
    from datasets import load_dataset
    from transformers import AutoModel, AutoTokenizer

    texts = load_dataset("fancyzhx/ag_news", split="train")["text"][:n]
    tok = AutoTokenizer.from_pretrained(repo)
    model = AutoModel.from_pretrained(repo).eval().cuda()

    out = []
    with torch.no_grad():
        for i in range(0, len(texts), 256):
            enc = tok(texts[i:i + 256], padding=True, truncation=True,
                      max_length=max_len, return_tensors="pt").to("cuda")
            hs = model(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).float()
            e = (hs * m).sum(1) / m.sum(1).clamp(min=1e-9)
            e = torch.nn.functional.normalize(e, dim=1)
            out.append(e.cpu().numpy())
    emb = np.concatenate(out).astype("float32")
    buf = io.BytesIO()
    np.save(buf, emb)
    return buf.getvalue()


@app.function(image=image, gpu="A10G", timeout=60 * 30)
def embed_sentence_transformer(model_id: str, n: int = 10000) -> bytes:
    """Embeddings from a sentence-transformers model (e.g. mxbai-embed-large)."""
    import io
    import numpy as np
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer

    texts = load_dataset("fancyzhx/ag_news", split="train")["text"][:n]
    model = SentenceTransformer(model_id, device="cuda")
    emb = model.encode(texts, batch_size=256, normalize_embeddings=True,
                       convert_to_numpy=True, show_progress_bar=True).astype("float32")
    buf = io.BytesIO()
    np.save(buf, emb)
    return buf.getvalue()


@app.local_entrypoint()
def main(preset: str = "", models: str = "", pooling: str = "st",
         n: int = 10000, seed_labels: str = ""):
    import io
    import pathlib
    import numpy as np

    out_dir = pathlib.Path("embeddings")
    out_dir.mkdir(exist_ok=True)

    jobs = []  # (label, kind, repo)
    if preset == "multibert":
        for s in MULTIBERT_STEPS:
            jobs.append((f"multibert_step_{s}", "bert",
                         f"google/multiberts-seed_0-step_{s}"))
    elif preset == "mxbai":
        jobs.append(("mxbai", "st", "mixedbread-ai/mxbai-embed-large-v1"))
    elif models:
        for m in models.split(","):
            m = m.strip()
            label = m.split("/")[-1]
            kind = "bert" if pooling == "mean" else "st"
            jobs.append((label, kind, m))
    else:
        raise SystemExit("pass --preset multibert|mxbai or --models '<repo>,...'")

    # fan out across GPUs in parallel
    handles = []
    for label, kind, repo in jobs:
        fn = embed_bert_meanpool if kind == "bert" else embed_sentence_transformer
        handles.append((label, fn.spawn(repo, n)))

    for label, h in handles:
        emb = np.load(io.BytesIO(h.get()))
        path = out_dir / f"{label}.npz"
        np.savez(path, embeddings=emb)
        print(f"saved {emb.shape} -> {path}")
