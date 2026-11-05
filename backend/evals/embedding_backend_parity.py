"""
Measures how much switching from the PyTorch models (sentence-transformers)
to their ONNX equivalents (fastembed) would change retrieval -- before
committing to the switch. Tiny vector differences only matter if they change
what comes back, so this checks three levels:

1. Vector parity   -- cosine(torch_vec, onnx_vec) for the same text, plus
                      cosine(stored_db_vec, onnx_vec): can ONNX query vectors
                      search the existing index, or must it be re-embedded?
2. Dense retrieval -- per query, top-k over every chunk in the DB. Baseline is
                      prod today (torch query vs stored vectors); compared
                      against (a) ONNX query vs stored vectors, (b) ONNX query
                      vs ONNX re-embedded chunks. Reports top-1 agreement and
                      overlap@5 / overlap@15 (15 = DENSE_K).
3. Reranking       -- same candidates scored by both cross-encoders: top-5
                      agreement (5 = RERANK_TOP_K) and Spearman rank correlation.

Plus load time and per-call latency for each backend.

Needs fastembed, which isn't in requirements.txt. Install it WITHOUT deps so
it can't upgrade huggingface-hub/tokenizers out from under transformers:
  pip install --no-deps fastembed py-rust-stemmers loguru

Usage: python -m evals.embedding_backend_parity [--queries path.jsonl]
  queries file: JSONL with a "standalone_question" or "question" field per line
  (defaults to evals/dataset.jsonl).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sqlalchemy import select

from app.db.models import Chunk
from app.db.session import SyncSessionLocal
from app.services.retrieval import DENSE_K, HF_CACHE_DIR, RERANK_TOP_K, RERANKER_MODEL

EMBED_MODEL = "BAAI/bge-small-en"
ONNX_RERANKER_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"  # ONNX export of RERANKER_MODEL
DEFAULT_QUERIES = Path(__file__).parent / "dataset.jsonl"


def _unit(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    return m / np.linalg.norm(m, axis=-1, keepdims=True)


def _rowwise_cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(_unit(a) * _unit(b), axis=1)


def _top_k(query_vecs: np.ndarray, doc_vecs: np.ndarray, k: int) -> np.ndarray:
    return np.argsort(-(_unit(query_vecs) @ _unit(doc_vecs).T), axis=1)[:, :k]


def _timed(fn):
    t = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t


def load_queries(path: Path) -> list[str]:
    queries = []
    for line in open(path):
        if line.strip():
            row = json.loads(line)
            q = row.get("standalone_question") or row.get("question")
            if q and row.get("intent", "general") != "chitchat":
                queries.append(q)
    return queries


def overlap(a: np.ndarray, b: np.ndarray, k: int) -> float:
    return float(np.mean([len(set(x[:k]) & set(y[:k])) / k for x, y in zip(a, b)]))


def main(queries_path: Path) -> dict:
    from fastembed import TextEmbedding
    from fastembed.rerank.cross_encoder import TextCrossEncoder
    from langchain_huggingface import HuggingFaceEmbeddings
    from sentence_transformers import CrossEncoder

    queries = load_queries(queries_path)
    with SyncSessionLocal() as session:
        chunks = session.execute(select(Chunk)).scalars().all()
    texts = [c.content for c in chunks]
    stored = np.array([np.asarray(c.embedding, dtype=np.float32) for c in chunks])
    print(f"{len(queries)} queries, {len(chunks)} chunks")

    report: dict = {"queries": len(queries), "chunks": len(chunks)}

    # --- load both backends exactly as prod (torch) / a fastembed swap would ---
    torch_emb, report["load_s_torch_embedder"] = _timed(lambda: HuggingFaceEmbeddings(
        model_name=EMBED_MODEL, model_kwargs={"device": "cpu"}, cache_folder=HF_CACHE_DIR))
    onnx_emb, report["load_s_onnx_embedder"] = _timed(lambda: TextEmbedding(EMBED_MODEL))
    torch_rr, report["load_s_torch_reranker"] = _timed(lambda: CrossEncoder(RERANKER_MODEL, cache_folder=HF_CACHE_DIR))
    onnx_rr, report["load_s_onnx_reranker"] = _timed(lambda: TextCrossEncoder(ONNX_RERANKER_MODEL))

    # --- 1. vector parity ---
    tq, t_tq = _timed(lambda: np.array([torch_emb.embed_query(q) for q in queries]))
    oq, t_oq = _timed(lambda: np.array([next(iter(onnx_emb.embed([q]))) for q in queries]))
    report["query_embed_ms_torch"] = round(1000 * t_tq / len(queries), 1)
    report["query_embed_ms_onnx"] = round(1000 * t_oq / len(queries), 1)

    td, _ = _timed(lambda: np.array(torch_emb.embed_documents(texts)))
    od, _ = _timed(lambda: np.array(list(onnx_emb.embed(texts))))

    for name, a, b in [
        ("cos_query_torch_vs_onnx", tq, oq),
        ("cos_chunk_torch_vs_onnx", td, od),
        ("cos_chunk_stored_vs_torch", stored, td),  # sanity: does today's torch reproduce the index?
        ("cos_chunk_stored_vs_onnx", stored, od),
    ]:
        c = _rowwise_cos(a, b)
        report[name] = {"mean": round(float(c.mean()), 6), "min": round(float(c.min()), 6)}

    # --- 2. dense retrieval agreement vs prod today ---
    base = _top_k(tq, stored, DENSE_K)
    for name, q_vecs, d_vecs in [
        ("onnx_query_vs_stored_index", oq, stored),
        ("onnx_query_vs_onnx_reindex", oq, od),
    ]:
        cand = _top_k(q_vecs, d_vecs, DENSE_K)
        report[name] = {
            "top1_agreement": round(float(np.mean(base[:, 0] == cand[:, 0])), 3),
            "overlap@5": round(overlap(base, cand, 5), 3),
            f"overlap@{DENSE_K}": round(overlap(base, cand, DENSE_K), 3),
        }

    # --- 3. reranker agreement on identical candidates (prod's dense top-k) ---
    top_agree, rhos, t_torch, t_onnx = [], [], 0.0, 0.0
    for qi, q in enumerate(queries):
        docs = [texts[i] for i in base[qi]]
        ts, dt = _timed(lambda: torch_rr.predict([(q, d) for d in docs]))
        os_, do = _timed(lambda: np.array(list(onnx_rr.rerank(q, docs))))
        t_torch, t_onnx = t_torch + dt, t_onnx + do
        top_t = set(np.argsort(-ts)[:RERANK_TOP_K])
        top_o = set(np.argsort(-os_)[:RERANK_TOP_K])
        top_agree.append(len(top_t & top_o) / RERANK_TOP_K)
        rhos.append(spearmanr(ts, os_).statistic)
    report["rerank"] = {
        f"top{RERANK_TOP_K}_overlap": round(float(np.mean(top_agree)), 3),
        "spearman_mean": round(float(np.nanmean(rhos)), 4),
        "spearman_min": round(float(np.nanmin(rhos)), 4),
        f"ms_per_query_{DENSE_K}_docs_torch": round(1000 * t_torch / len(queries), 1),
        f"ms_per_query_{DENSE_K}_docs_onnx": round(1000 * t_onnx / len(queries), 1),
    }

    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    main(parser.parse_args().queries)
