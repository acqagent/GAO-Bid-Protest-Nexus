#!/usr/bin/env python3
"""Shared search core for the GAO decision corpus.

Hybrid retrieval: dense (BAAI/bge-base-en-v1.5) + BM25, fused with Reciprocal
Rank Fusion, then re-ranked with BAAI/bge-reranker-base. Results are capped at
two chunks per decision file so one decision cannot flood the top-k.

Used by scripts/serve.py (dashboard /api/search) and scripts/query.py (CLI).
"""

import json
import re
import time
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parent.parent
VEC_DIR = ROOT / "vector"
CHUNKS_JSONL = VEC_DIR / "chunks.jsonl"
EMBEDDINGS_NPY = VEC_DIR / "embeddings.npy"
META_JSON = VEC_DIR / "meta.json"

DENSE_MODEL = "BAAI/bge-base-en-v1.5"
QUERY_PREFIX = "Represent this query for retrieving relevant passages: "
RERANK_MODEL = "BAAI/bge-reranker-base"

RRF_K = 60            # RRF damping constant
POOL = 96             # candidates kept per channel
RERANK_CANDIDATES = 48
MAX_PER_FILE = 2      # max chunks from one decision file in the top-k
MAX_RERANK_CHARS = 2200  # ~<512 tokens for the cross-encoder


def tokenize(text):
    t = text.lower()
    # dotted/hyphenated tokens stay whole (b-416240.2, 4.c.f.r.); add bare subwords
    return (re.findall(r"[a-z0-9]+(?:[.-][a-z0-9]+)*", t)
            + re.findall(r"[a-z0-9]+", t))


class Store:
    def __init__(self, use_reranker=True):
        t0 = time.time()
        self.chunks = [json.loads(l) for l in CHUNKS_JSONL.open()]
        self.vecs = np.load(EMBEDDINGS_NPY)
        if len(self.vecs) != len(self.chunks):
            raise RuntimeError(
                f"embeddings ({len(self.vecs)}) != chunks ({len(self.chunks)})")
        meta = json.loads(META_JSON.read_text()) if META_JSON.exists() else {}
        if meta.get("model") and meta["model"] != DENSE_MODEL:
            raise RuntimeError(
                f"embeddings were built with {meta['model']} but this searchlib "
                f"expects {DENSE_MODEL} — run `python3 scripts/vectorize.py --rebuild`")
        from sentence_transformers import SentenceTransformer
        self.embedder = SentenceTransformer(DENSE_MODEL)
        self.reranker = None
        if use_reranker:
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder(RERANK_MODEL)
        self._toks = [tokenize(c["text"]) for c in self.chunks]
        self.bm25 = BM25Okapi(self._toks)
        self.load_seconds = round(time.time() - t0, 1)

    # ---------- channels ----------
    def _dense_scores(self, query):
        qv = self.embedder.encode([QUERY_PREFIX + query], normalize_embeddings=True,
                                  convert_to_numpy=True).astype("float32")[0]
        return self.vecs @ qv

    def bm25_rank(self, query, pool=POOL):
        scores = self.bm25.get_scores(tokenize(query))
        return [(int(i), float(scores[i])) for i in np.argsort(-scores)[:pool]]

    # ---------- fusion + re-rank ----------
    @staticmethod
    def _ok(file_no, ff, allowed):
        if ff and ff not in file_no.upper():
            return False
        if allowed and not (set(map_ids(file_no)) & allowed):
            return False
        return True

    def search(self, query, k=6, file_filter=None, mode="hybrid", pool=POOL,
               allowed_files=None):
        """mode='hybrid': dense+BM25+RRF+rerank. mode='dense': legacy dense only.
        allowed_files: optional iterable of base B-numbers (topic facet)."""
        ff = file_filter.upper() if file_filter else None
        allowed = ({x.upper() for x in allowed_files}
                   if allowed_files is not None else None)
        p = max(pool * 3, 200) if (ff or allowed) else pool
        d_scores = self._dense_scores(query)
        dense_rank = [(int(i), float(d_scores[i])) for i in np.argsort(-d_scores)[:p]]
        if mode == "hybrid":
            fused = {}
            for ranks in (dense_rank, self.bm25_rank(query, p)):
                for pos, (i, _) in enumerate(ranks):
                    fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + pos + 1)
            cands = [i for i, _ in sorted(fused.items(),
                                          key=lambda x: -x[1])[:RERANK_CANDIDATES]]
        else:
            cands = [i for i, _ in dense_rank[:RERANK_CANDIDATES]]
        cands = [i for i in cands
                 if self._ok(self.chunks[i]["file"], ff, allowed)]
        if len(cands) < k:  # top up from the dense scores within the filter
            s = d_scores.copy()
            for i in range(len(self.chunks)):
                if not self._ok(self.chunks[i]["file"], ff, allowed):
                    s[i] = -1.0
            cands += [int(i) for i in np.argsort(-s)[:k] if i not in set(cands)]
        dense_score = dict(dense_rank)

        reranked = False
        if mode == "hybrid" and cands:
            # Blend normalized RRF with the re-ranker sigmoid. The cross-encoder
            # alone over-ranks generic text on domain queries; RRF alone
            # under-ranks it — the blend outperforms both (eval-set measured).
            rrf = np.array([fused.get(i, 0.0) for i in cands])
            rrf_n = rrf / max(rrf.max(), 1e-9)
            if self.reranker is not None:
                pairs = [(query, self.chunks[i]["text"][:MAX_RERANK_CHARS])
                         for i in cands]
                rs = self.reranker.predict(pairs, show_progress_bar=False)
                sig = 1.0 / (1.0 + np.exp(-rs))
                final = (rrf_n + sig) / 2.0
                order = np.argsort(-final)
                ranked = [(int(cands[int(o)]),
                           float(dense_score.get(int(cands[int(o)]), 0.0)),
                           float(final[int(o)])) for o in order]
                reranked = True
            else:
                order = np.argsort(-rrf_n)
                ranked = [(int(cands[int(o)]),
                           float(dense_score.get(int(cands[int(o)]), 0.0)),
                           float(rrf_n[int(o)])) for o in order]
        else:
            ranked = [(i, dense_score.get(i, 0.0), dense_score.get(i, 0.0))
                      for i in cands]

        out, seen = [], {}
        for i, _, disp in ranked:
            f = self.chunks[i]["file"]
            if seen.get(f, 0) >= MAX_PER_FILE:
                continue
            seen[f] = seen.get(f, 0) + 1
            out.append({
                "chunk": self.chunks[i],
                "score": disp,
                "dense_score": dense_score.get(i),
                "reranked": reranked,
            })
            if len(out) == k:
                break
        return out


def map_ids(file_no):
    """Base B-numbers (upper) out of a chunk file field, deduped."""
    out = []
    for part in file_no.split(";"):
        base = re.sub(r"\.\d+$", "", part.strip().upper())
        if base and base not in out:
            out.append(base)
    return out
