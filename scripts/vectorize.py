#!/usr/bin/env python3
"""Build the Dynamic Search index from a list of decision PDF links.

Four stages, each resumable and each safe to interrupt. Run them together or
one at a time with --stage:

    fetch    the PDFs named in the links CSV      -> corpus/pdf/<B-NUMBER>.pdf
    extract  their text, page by page             -> corpus/text/<B-NUMBER>.txt
    chunk    overlapping windows + a digest chunk -> vector/chunks.jsonl
    embed    one vector per chunk                 -> vector/embeddings.npy
                                                     vector/meta.json

    python3 scripts/vectorize.py --links pdf-links-classified.csv

EMBEDDINGS
    Two backends, and the index records which one built it so searchlib.py can
    follow rather than guess:

      --embedder api    any OpenAI-compatible POST {base}/embeddings. This is
                        the one to use with a local server (llama.cpp, vLLM,
                        LM Studio, Ollama, TEI, Infinity):
                          --embed-base-url http://localhost:8080/v1 \\
                          --embed-model your-embedding-model
      --embedder local  sentence-transformers in this process
                          --embed-model BAAI/bge-base-en-v1.5

    A retrieval-trained embedding model is what this wants. A general chat
    model will produce vectors, but they are not trained to put a question near
    the passage that answers it, and a large one costs hours of forward passes
    for the ~78k chunks this corpus produces. Qwen3-Embedding-0.6B / 4B / 8B,
    BAAI/bge-*, and intfloat/e5-* are all built for the job.

RESUMING
    Every stage skips work it has already done. The embed stage writes into a
    memory-mapped .partial file and records how far it got, so a run
    interrupted at chunk 60,000 of 78,000 continues from there rather than
    starting over.

Needs numpy for the embed stage and pypdf for extract; both are in
requirements.txt. Nothing else, and no network beyond gao.gov and whatever
embedding endpoint you point it at.
"""

import argparse
import csv
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
USER_AGENT = ("GAO-Bid-Protest-Nexus/1.0 (index builder; "
              "+https://github.com/acqagent/GAO-Bid-Protest-Nexus)")
ALLOWED_HOSTS = {h.strip().lower() for h in os.environ.get(
    "VECTORIZE_HOSTS", "gao.gov,www.gao.gov").split(",") if h.strip()}
_lock = threading.Lock()

# Retrieval models want the query marked differently from the passage. Getting
# this wrong quietly costs accuracy, so it is derived from the model name and
# then stored in meta.json for searchlib.py to reuse.
PREFIXES = [
    (r"bge-.*-v1\.5|bge-(base|small|large)",
     "Represent this query for retrieving relevant passages: ", ""),
    (r"bge-m3|gte-|mxbai", "", ""),
    (r"e5-|multilingual-e5", "query: ", "passage: "),
    (r"qwen.*embedding",
     "Instruct: Given a search query, retrieve relevant passages that answer "
     "the query\nQuery: ", ""),
    (r"nomic-embed", "search_query: ", "search_document: "),
]


def prefixes_for(model):
    m = (model or "").lower()
    for pat, q, d in PREFIXES:
        if re.search(pat, m):
            return q, d
    return "", ""


def write_meta(path, updates):
    """Merge into vector/meta.json. Each stage records what it is responsible
    for, so running stages separately cannot leave a field describing a default
    that was never used."""
    meta = {}
    if os.path.exists(path):
        try:
            meta = json.load(open(path))
        except Exception:
            meta = {}
    meta.update(updates)
    meta.setdefault("created", now())
    meta["updated"] = now()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    json.dump(meta, open(path, "w"), indent=1)
    return meta


def now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def log(stage, msg):
    print(f"[{stage}] {msg}", flush=True)


# --------------------------------------------------------------- the links

def read_links(path):
    """[{ids: [...], file: 'B-1;B-2', url: ...}] out of the classified CSV."""
    rows, skipped = [], 0
    with open(path, encoding="utf-8-sig", newline="") as f:
        rdr = csv.DictReader(f)
        cols = {c.lower().strip(): c for c in (rdr.fieldnames or [])}
        idc = next((cols[c] for c in ("b_number", "bnumber", "id") if c in cols), None)
        urlc = next((cols[c] for c in ("url", "pdf", "link") if c in cols), None)
        if not idc or not urlc:
            sys.exit(f"{path}: need a b_number column and a url column, "
                     f"found {rdr.fieldnames}")
        for row in rdr:
            url = (row.get(urlc) or "").strip()
            ids = [p.strip().upper() for p in (row.get(idc) or "").split("|") if p.strip()]
            if not ids or not url.lower().endswith(".pdf"):
                skipped += 1
                continue
            # searchlib.map_ids splits a chunk's file field on ';'
            rows.append({"ids": ids, "file": ";".join(ids), "url": url,
                         "key": ids[0].replace("/", "_")})
    log("links", f"{len(rows)} decision PDFs, {skipped} row(s) with no PDF to fetch")
    return rows


# -------------------------------------------------------------- 1. fetch

def _get(url, timeout=90):
    host = (urllib.parse.urlparse(url).hostname or "").lower()
    if urllib.parse.urlparse(url).scheme not in ("http", "https"):
        raise ValueError("only http(s) URLs are fetched")
    if ALLOWED_HOSTS and host not in ALLOWED_HOSTS:
        raise ValueError(f"host not allowed: {host} (set VECTORIZE_HOSTS to widen)")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT,
                                               "Accept": "application/pdf,*/*"})
    return urllib.request.urlopen(req, timeout=timeout)


def stage_fetch(rows, args):
    os.makedirs(args.pdf_dir, exist_ok=True)
    todo = [r for r in rows
            if args.force or not os.path.exists(os.path.join(args.pdf_dir, r["key"] + ".pdf"))]
    if not todo:
        log("fetch", "every PDF is already on disk")
        return
    log("fetch", f"{len(todo)} to download, {args.workers} at a time, "
                 f"{args.sleep}s apart — Ctrl-C is safe")
    done = {"n": 0, "ok": 0, "fail": 0, "bytes": 0}
    t0 = time.time()

    def one(r):
        path = os.path.join(args.pdf_dir, r["key"] + ".pdf")
        err = ""
        for attempt in range(args.retries + 1):
            try:
                with _get(r["url"]) as resp:
                    data = resp.read()
                if not data.startswith(b"%PDF"):
                    raise ValueError("not a PDF (got HTML?)")
                tmp = path + ".part"
                with open(tmp, "wb") as f:
                    f.write(data)
                os.replace(tmp, path)
                err = ""
                break
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                if attempt < args.retries:
                    time.sleep(args.sleep * (2 ** attempt) + 0.5)
        time.sleep(args.sleep)
        with _lock:
            done["n"] += 1
            if err:
                done["fail"] += 1
                if done["fail"] <= 12:
                    log("fetch", f"  {r['ids'][0]} failed — {err}")
            else:
                done["ok"] += 1
                done["bytes"] += os.path.getsize(path)
            if done["n"] % 100 == 0 or done["n"] == len(todo):
                rate = done["n"] / max(time.time() - t0, 1e-9)
                left = int((len(todo) - done["n"]) / rate) if rate else 0
                log("fetch", f"  {done['n']}/{len(todo)}  ok {done['ok']}  "
                             f"failed {done['fail']}  {done['bytes'] // 1_000_000} MB  "
                             f"~{left // 60}m{left % 60:02d}s left")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(one, todo))
    log("fetch", f"done: {done['ok']} downloaded, {done['fail']} failed, "
                 f"{done['bytes'] // 1_000_000} MB")


# ------------------------------------------------------------ 2. extract

DATE_RE = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|October|"
    r"November|December)\s+(\d{1,2}),\s+(\d{4})\b")
MONTHS = {m: i for i, m in enumerate(
    "January February March April May June July August September October "
    "November December".split(), 1)}


def pdf_to_pages(path):
    from pypdf import PdfReader
    reader = PdfReader(path)
    out = []
    for i, page in enumerate(reader.pages, 1):
        t = page.extract_text() or ""
        t = re.sub(r"[ \t]+", " ", t)
        t = re.sub(r"[ \t]+\n", "\n", t).strip()
        if t:
            out.append((i, t))
    return out


def stage_extract(rows, args):
    os.makedirs(args.text_dir, exist_ok=True)
    todo = [r for r in rows
            if os.path.exists(os.path.join(args.pdf_dir, r["key"] + ".pdf"))
            and (args.force or not os.path.exists(
                os.path.join(args.text_dir, r["key"] + ".txt")))]
    if not todo:
        log("extract", "every PDF already has extracted text")
        return
    try:
        import pypdf  # noqa: F401
    except ImportError:
        sys.exit("extract needs pypdf — pip install pypdf")
    log("extract", f"{len(todo)} PDF(s) to read")
    ok = scanned = broken = 0
    for n, r in enumerate(todo, 1):
        src = os.path.join(args.pdf_dir, r["key"] + ".pdf")
        dst = os.path.join(args.text_dir, r["key"] + ".txt")
        try:
            pages = pdf_to_pages(src)
        except Exception as e:
            broken += 1
            if broken <= 10:
                log("extract", f"  {r['ids'][0]} unreadable — {type(e).__name__}: {e}")
            continue
        body = "\n\n".join(f"[p. {i}]\n{t}" for i, t in pages)
        if len(re.sub(r"\s", "", body)) < 200:
            scanned += 1
            if scanned <= 10:
                log("extract", f"  {r['ids'][0]} has almost no text — a scan? "
                               f"run OCR and drop the result in {args.text_dir}")
            continue
        with open(dst, "w", encoding="utf-8") as f:
            f.write(body)
        ok += 1
        if n % 200 == 0 or n == len(todo):
            log("extract", f"  {n}/{len(todo)}  text {ok}  scans {scanned}  broken {broken}")
    log("extract", f"done: {ok} extracted, {scanned} image-only, {broken} unreadable")


# -------------------------------------------------------------- 3. chunk

def find_date(text):
    m = DATE_RE.search(text[:4000])
    if not m:
        return None
    return f"{m.group(3)}-{MONTHS[m.group(1)]:02d}-{int(m.group(2)):02d}"


def find_digest(text):
    """GAO leads with a DIGEST section; it is the best one-shot summary there
    is, so it becomes its own chunk and gets its own vector."""
    m = re.search(r"\bDIGEST\b(.{80,6000}?)(?:\n\s*DECISION\b|\Z)", text,
                  re.S | re.I)
    if not m:
        return None
    d = re.sub(r"\[p\. \d+\]\n?", "", m.group(1)).strip()
    return d if len(d) > 120 else None


def words_with_pages(text):
    """[(word, page)] so a chunk can report the page span it came from."""
    out, page = [], 1
    for part in re.split(r"\[p\. (\d+)\]", text):
        if part.isdigit():
            page = int(part)
            continue
        for w in part.split():
            out.append((w, page))
    return out


def chunk_text(text, target, overlap):
    wp = words_with_pages(text)
    if not wp:
        return []
    step = max(target - overlap, 1)
    out = []
    for start in range(0, len(wp), step):
        window = wp[start:start + target]
        if not window:
            break
        # a trailing sliver adds noise and no recall
        if len(window) < overlap and out:
            break
        out.append((" ".join(w for w, _ in window),
                    [window[0][1], window[-1][1]]))
        if start + target >= len(wp):
            break
    return out


def stage_chunk(rows, args):
    have = [r for r in rows
            if os.path.exists(os.path.join(args.text_dir, r["key"] + ".txt"))]
    if not have:
        sys.exit("chunk: no extracted text — run the fetch and extract stages first")
    log("chunk", f"{len(have)} decision(s) with text · "
                 f"{args.chunk_words}-word windows, {args.overlap} overlap")
    os.makedirs(os.path.dirname(args.chunks) or ".", exist_ok=True)
    n_chunks = n_digest = 0
    with open(args.chunks, "w", encoding="utf-8") as out:
        for i, r in enumerate(have, 1):
            text = open(os.path.join(args.text_dir, r["key"] + ".txt"),
                        encoding="utf-8").read()
            date = find_date(text)
            digest = find_digest(text)
            if digest:
                out.write(json.dumps({
                    "id": f"{r['key']}__digest", "file": r["file"],
                    "pages": [1, 1], "date": date, "text": digest}) + "\n")
                n_chunks += 1
                n_digest += 1
            for j, (body, pages) in enumerate(
                    chunk_text(text, args.chunk_words, args.overlap)):
                out.write(json.dumps({
                    "id": f"{r['key']}__{j}", "file": r["file"],
                    "pages": pages, "date": date, "text": body}) + "\n")
                n_chunks += 1
            if i % 500 == 0 or i == len(have):
                log("chunk", f"  {i}/{len(have)} decisions · {n_chunks} chunks")
    write_meta(args.meta, {"chunk_target_words": args.chunk_words,
                           "chunk_overlap_words": args.overlap,
                           "n_chunks": n_chunks, "digests": n_digest})
    log("chunk", f"done: {n_chunks} chunks ({n_digest} digests) -> {args.chunks}")


# -------------------------------------------------------------- 4. embed

class ApiEmbedder:
    """Any endpoint that speaks OpenAI's POST {base}/embeddings."""

    def __init__(self, base_url, model, api_key, timeout=300):
        self.url = base_url.rstrip("/") + "/embeddings"
        self.model = model
        self.key = api_key
        self.timeout = timeout
        self.name = model

    def __call__(self, texts):
        payload = {"model": self.model, "input": texts}
        body = json.dumps(payload).encode()
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = "Bearer " + self.key
        req = urllib.request.Request(self.url, data=body, headers=headers,
                                     method="POST")
        host = (urllib.parse.urlparse(self.url).hostname or "").lower()
        opener = (urllib.request.build_opener(urllib.request.ProxyHandler({}))
                  if host in ("localhost", "127.0.0.1", "::1")
                  else urllib.request.build_opener())
        try:
            with opener.open(req, timeout=self.timeout) as r:
                data = json.loads(r.read().decode("utf-8", "replace"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:500]
            raise RuntimeError(f"{e.code} from {self.url} — {detail}") from None
        rows = sorted(data["data"], key=lambda d: d.get("index", 0))
        return [d["embedding"] for d in rows]


class LocalEmbedder:
    def __init__(self, model):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            sys.exit("the local embedder needs sentence-transformers — "
                     "pip install sentence-transformers, or use --embedder api "
                     "with --embed-base-url pointing at a local server")
        self.m = SentenceTransformer(model)
        self.name = model

    def __call__(self, texts):
        return self.m.encode(texts, normalize_embeddings=False,
                             convert_to_numpy=True).tolist()


def stage_embed(args):
    import numpy as np
    if not os.path.exists(args.chunks):
        sys.exit(f"embed: {args.chunks} not found — run the chunk stage first")
    chunks = [json.loads(l) for l in open(args.chunks, encoding="utf-8") if l.strip()]
    n = len(chunks)
    if not n:
        sys.exit("embed: chunks file is empty")

    if args.embedder == "local":
        emb = LocalEmbedder(args.embed_model)
    else:
        if not args.embed_base_url:
            sys.exit("embed: --embed-base-url is required for the api embedder "
                     "(e.g. http://localhost:8080/v1)")
        emb = ApiEmbedder(args.embed_base_url, args.embed_model, args.embed_api_key)
    qpre, dpre = prefixes_for(args.embed_model)
    if args.query_prefix is not None:
        qpre = args.query_prefix
    if args.doc_prefix is not None:
        dpre = args.doc_prefix

    partial = args.embeddings + ".partial"
    progress = args.embeddings + ".progress.json"
    start, dim = 0, args.embed_dim
    if os.path.exists(progress) and not args.force:
        try:
            st = json.load(open(progress))
            if st.get("n") == n and st.get("model") == emb.name:
                start, dim = st["done"], st["dim"]
                log("embed", f"resuming at chunk {start}/{n}")
        except Exception:
            pass
    if not dim:                      # one probe call to learn the width
        v = emb([dpre + chunks[0]["text"]])
        dim = len(v[0])
        log("embed", f"vector width {dim}, learned from a probe call")

    mm = np.lib.format.open_memmap(
        partial, mode=("r+" if start and os.path.exists(partial) else "w+"),
        dtype=args.dtype, shape=(n, dim))
    log("embed", f"{n} chunks · {emb.name} · dim {dim} · {args.dtype} · "
                 f"batch {args.embed_batch} "
                 f"(~{n * dim * np.dtype(args.dtype).itemsize // 1_000_000} MB)")
    if qpre:
        log("embed", f"query prefix: {qpre!r}")

    t0 = time.time()
    i = start
    try:
        while i < n:
            batch = chunks[i:i + args.embed_batch]
            texts = [dpre + c["text"] for c in batch]
            for attempt in range(args.retries + 1):
                try:
                    vecs = emb(texts)
                    break
                except Exception as e:
                    if attempt >= args.retries:
                        raise
                    log("embed", f"  batch at {i} failed ({e}); retrying")
                    time.sleep(2 ** attempt)
            arr = np.asarray(vecs, dtype=args.dtype)
            if arr.shape != (len(batch), dim):
                sys.exit(f"embed: endpoint returned {arr.shape}, expected "
                         f"{(len(batch), dim)} — is the model consistent?")
            if args.normalize:
                norms = np.linalg.norm(arr.astype("float32"), axis=1, keepdims=True)
                arr = (arr.astype("float32") / np.maximum(norms, 1e-12)).astype(args.dtype)
            mm[i:i + len(batch)] = arr
            i += len(batch)
            if (i // args.embed_batch) % 20 == 0 or i >= n:
                mm.flush()
                json.dump({"n": n, "done": i, "dim": dim, "model": emb.name},
                          open(progress, "w"))
                rate = (i - start) / max(time.time() - t0, 1e-9)
                left = int((n - i) / rate) if rate else 0
                log("embed", f"  {i}/{n}  {rate:.1f} chunks/s  "
                             f"~{left // 3600}h{(left % 3600) // 60:02d}m left")
    except KeyboardInterrupt:
        mm.flush()
        json.dump({"n": n, "done": i, "dim": dim, "model": emb.name},
                  open(progress, "w"))
        sys.exit(f"\n[embed] interrupted at {i}/{n} — rerun to continue")

    mm.flush()
    del mm
    os.replace(partial, args.embeddings)
    if os.path.exists(progress):
        os.remove(progress)

    fields = {
        "model": emb.name,
        "backend": "sentence-transformers" if args.embedder == "local" else "openai-api",
        "dim": dim,
        "dtype": args.dtype,
        "normalized": bool(args.normalize),
        "query_prefix": qpre,
        "doc_prefix": dpre,
        "n_chunks": n,
        "digests": sum(1 for c in chunks if str(c["id"]).endswith("__digest")),
    }
    if args.embedder == "api":
        # the endpoint, never the key — searchlib needs it to embed queries the
        # same way, and EMBED_BASE_URL overrides it at query time
        fields["base_url"] = args.embed_base_url
    write_meta(args.meta, fields)
    log("embed", f"done in {int(time.time() - t0)}s -> {args.embeddings}")
    log("embed", f"meta -> {args.meta}")


# ------------------------------------------------------------------ cli

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--links", default="", help="classified CSV of decision PDF links")
    ap.add_argument("--stage", default="all",
                    choices=("all", "fetch", "extract", "chunk", "embed"))
    ap.add_argument("--pdf-dir", default=os.path.join(ROOT, "corpus", "pdf"))
    ap.add_argument("--text-dir", default=os.path.join(ROOT, "corpus", "text"))
    ap.add_argument("--chunks", default=os.path.join(ROOT, "vector", "chunks.jsonl"))
    ap.add_argument("--embeddings", default=os.path.join(ROOT, "vector", "embeddings.npy"))
    ap.add_argument("--meta", default=os.path.join(ROOT, "vector", "meta.json"))
    ap.add_argument("--limit", type=int, default=0, help="stop after N decisions")
    ap.add_argument("--force", action="store_true", help="redo work already done")
    # fetch
    ap.add_argument("--workers", type=int, default=4, help="parallel downloads (4)")
    ap.add_argument("--sleep", type=float, default=0.3, help="pause per request (0.3s)")
    ap.add_argument("--retries", type=int, default=2)
    # chunk
    ap.add_argument("--chunk-words", type=int, default=350)
    ap.add_argument("--overlap", type=int, default=40)
    # embed
    ap.add_argument("--embedder", default="api", choices=("api", "local"))
    ap.add_argument("--embed-model", default=os.environ.get(
        "EMBED_MODEL", "BAAI/bge-base-en-v1.5"))
    ap.add_argument("--embed-base-url", default=os.environ.get("EMBED_BASE_URL", ""))
    ap.add_argument("--embed-api-key", default=os.environ.get("EMBED_API_KEY", ""))
    ap.add_argument("--embed-batch", type=int, default=32)
    ap.add_argument("--embed-dim", type=int, default=0,
                    help="vector width; probed from the endpoint when omitted")
    ap.add_argument("--dtype", default="float32", choices=("float32", "float16"))
    ap.add_argument("--no-normalize", dest="normalize", action="store_false",
                    help="keep raw vectors; by default they are L2-normalized so "
                         "searchlib's dot product is cosine similarity")
    ap.add_argument("--query-prefix", default=None,
                    help="override the retrieval prefix for queries")
    ap.add_argument("--doc-prefix", default=None,
                    help="override the retrieval prefix for passages")
    args = ap.parse_args()

    stages = (["fetch", "extract", "chunk", "embed"] if args.stage == "all"
              else [args.stage])
    rows = []
    if set(stages) & {"fetch", "extract", "chunk"}:
        if not args.links:
            sys.exit("--links is required for the fetch, extract and chunk stages")
        rows = read_links(args.links)
        if args.limit:
            rows = rows[:args.limit]

    if "fetch" in stages:
        stage_fetch(rows, args)
    if "extract" in stages:
        stage_extract(rows, args)
    if "chunk" in stages:
        stage_chunk(rows, args)
    if "embed" in stages:
        stage_embed(args)


if __name__ == "__main__":
    main()
