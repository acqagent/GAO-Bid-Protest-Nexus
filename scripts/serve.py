#!/usr/bin/env python3
"""Local server for the GAO Bid Protest Nexus dashboard.

Serves the static site (visualization/, data/, ...) plus a small API backed by
the local vector store in vector/ and an OpenAI-compatible model endpoint:

    GET  /api/search?q=<question>&k=<n>[&file=<B-number>][&mode=hybrid|dense]
         -> {"query": ..., "results": [{file, pages, date, score, text,
                                         isDigest, mapIds, links}]}
    GET  /api/config
         -> what this server can do: search, corpus text, PDF extraction, and
            whether a model endpoint and key are already configured here
    GET  /api/pdf?id=<B-number>[&url=<gao.gov page>]
         -> {"pdf": "https://www.gao.gov/assets/....pdf"} — resolves a
            /products/ landing page to the actual PDF and caches the answer
    POST /api/analyze[?stream=1]
         -> key-element summary of one decision (Server-Sent Events when
            stream=1, otherwise one JSON object)
    POST /api/compare[?stream=1]
         -> one comparative analysis across several selected decisions, built
            from their individual analyses where the dashboard already has them

Retrieval is hybrid (dense + BM25, RRF fusion) with cross-encoder re-ranking;
mode=dense disables both (legacy behaviour).

The vector index is optional: without vector/chunks.jsonl + embeddings.npy the
server still starts and still serves the dashboard and the analysis API — only
Dynamic Search is switched off.

Run with the project venv:

    .venv/bin/python scripts/serve.py
    # then open http://127.0.0.1:8765/visualization/map.html
"""

import functools
import http.server
import json
import os
import posixpath
import sys
import time
import urllib.parse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analysis  # noqa: E402  (sit next to this script)
import llm       # noqa: E402

HOST = "127.0.0.1"   # loopback only; reach it remotely via `tailscale serve`
PORT = 8765
MAX_K = 50           # cap on ?k= — every candidate costs one cross-encoder pass
MAX_POST_BYTES = 8 * 1024 * 1024   # the browser may post extracted PDF text
MAX_COMPARE = int(os.environ.get("ANALYZE_MAX_COMPARE", "12"))
                     # decisions in one comparison — a context-window guard
# Only the dashboard's own assets are served; the rest of the project tree
# (.venv/, logs/, llm-work/, vector/, *.csv) stays off the wire.
ALLOWED_PREFIXES = ("/visualization/", "/data/")


def load_store():
    """The vector store, or None with the reason — analysis does not need it."""
    t0 = time.time()
    print("[serve] loading store (embedder + reranker + BM25) ...", flush=True)
    try:
        import searchlib
        store = searchlib.Store(use_reranker=True)
    except Exception as e:
        print(f"[serve] no vector index ({type(e).__name__}: {e})", flush=True)
        print("[serve] Dynamic Search is off; Map, Table and Analysis still work.",
              flush=True)
        return None, f"{type(e).__name__}: {e}"
    body = sum(1 for c in store.chunks if not c["id"].endswith("__digest"))
    print(f"[serve] ready in {store.load_seconds}s "
          f"({len(store.chunks)} chunks, {body} body chunks)", flush=True)
    return store, None


STORE, STORE_ERROR = load_store()
CHUNK_INDEX = analysis.ChunkIndex(STORE.chunks) if STORE else None

def _linkmeta(d):
    """url + pdf, and pdfok when the pdf link was never confirmed, so a search
    result labels a link exactly the way the Table and the Map popups do."""
    meta = {"url": d.get("url"), "pdf": d.get("pdf")}
    if d.get("pdfok") == 0:
        meta["pdfok"] = 0
    return meta


MAPMETA = {d["id"].lower(): _linkmeta(d)
           for d in json.load(open(os.path.join(ROOT, "data", "map.json")))["decisions"]}
FACETED = {}
for d in json.load(open(os.path.join(ROOT, "data", "mapped-decisions.json"))):
    key = d["id"].lower()
    FACETED[key] = d
    MAPMETA.setdefault(key, _linkmeta(d))


def map_ids(file_no):
    """Base B-numbers out of a chunk file field — mirrors searchlib.map_ids so
    the analysis path does not drag in numpy."""
    if STORE is not None:
        import searchlib
        return searchlib.map_ids(file_no)
    return analysis.base_ids(file_no)


class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def _gate(self):
        """True if this static request may proceed; else respond and return False."""
        path = posixpath.normpath(
            urllib.parse.unquote(urllib.parse.urlparse(self.path).path))
        if path in ("/", "/index.html"):
            # Relative Location, so this works both at :8765/ and behind a
            # path-prefix proxy (`tailscale serve` mounts us at /csv/).
            self.send_response(302)
            self.send_header("Location", "visualization/map.html")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return False
        if not path.startswith(ALLOWED_PREFIXES):
            self.send_error(404, "Not Found")
            return False
        return True

    def do_GET(self):
        if self.path.startswith("/api/search"):
            self.handle_search()
            return
        if self.path.startswith("/api/config"):
            self.handle_config()
            return
        if self.path.startswith("/api/pdf"):
            self.handle_pdf()
            return
        if self.path.startswith("/api/decision"):
            self.handle_decision()
            return
        if self._gate():
            super().do_GET()

    def do_HEAD(self):
        if self._gate():
            super().do_HEAD()

    def do_POST(self):
        if self.path.startswith("/api/analyze"):
            self.handle_analyze()
            return
        if self.path.startswith("/api/compare"):
            self.handle_compare()
            return
        self.send_error(404, "Not Found")

    # ------------------------------------------------------------- search

    def handle_search(self):
        p = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        q = p.get("q", [""])[0].strip()
        kraw = p.get("k", [""])[0]
        k = int(kraw) if kraw.isdigit() and len(kraw) <= 6 else 6
        k = max(1, min(k, MAX_K))
        filef = p.get("file", [""])[0].strip()
        mode = p.get("mode", ["hybrid"])[0] if p.get("mode", ["hybrid"])[0] in ("hybrid", "dense") else "hybrid"
        if STORE is None:
            self.send_json({"query": q, "results": [], "error": "no vector index",
                            "detail": STORE_ERROR}, status=503)
            return
        if not q:
            self.send_json({"query": "", "results": []})
            return

        t0 = time.time()
        raw = STORE.search(q, k=k, file_filter=filef or None, mode=mode)
        results = []
        for r in raw:
            c = r["chunk"]
            bases = map_ids(c["file"])
            links = [{"id": b.lower(), **MAPMETA[b.lower()]}
                     for b in bases if b.lower() in MAPMETA]
            results.append({
                "file": c["file"],
                "pages": c["pages"],
                "date": c.get("date"),
                "score": round(r["score"], 4),
                "dense_score": round(r["dense_score"], 4) if r.get("dense_score") is not None else None,
                "text": c["text"],
                "isDigest": c["id"].endswith("__digest"),
                "mapIds": [b.lower() for b in bases],
                "links": links,
            })
        self.send_json({"query": q, "mode": mode,
                        "took_ms": int((time.time() - t0) * 1000),
                        "results": results})

    # ------------------------------------------------------------ analysis

    def handle_config(self):
        """What the dashboard may offer. Never leaks the key itself."""
        self.send_json({
            "search": STORE is not None,
            "search_error": STORE_ERROR,
            "corpus": CHUNK_INDEX is not None,
            "pdf_extract": analysis.pdf_extractor(),
            "pdf_fetch": True,
            "llm": llm.defaults(),
            "pdf_links_cached": len(analysis.link_cache()),
            "max_compare": MAX_COMPARE,
        })

    def handle_pdf(self):
        p = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        did = p.get("id", [""])[0].strip()
        page = p.get("url", [""])[0].strip() or None
        known = MAPMETA.get(did.lower(), {})
        try:
            url, how = analysis.resolve_pdf_url(
                did, known_pdf=known.get("pdf"),
                page_url=page or known.get("url") or known.get("pdf"))
        except Exception as e:
            self.send_json({"id": did, "pdf": None,
                            "error": f"{type(e).__name__}: {e}"}, status=502)
            return
        self.send_json({"id": did, "pdf": url, "how": how})

    def handle_decision(self):
        """Metadata + whether the local corpus holds this decision's text."""
        p = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        did = p.get("id", [""])[0].strip()
        rec = FACETED.get(did.lower()) or MAPMETA.get(did.lower()) or {}
        out = {"id": did, "meta": rec, "corpus": False}
        if CHUNK_INDEX is not None:
            files = CHUNK_INDEX.files_for(did)
            out["corpus"] = bool(files)
            out["files"] = files
        self.send_json(out)

    def _read_json_body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_POST_BYTES:
            raise ValueError(f"request body over the {MAX_POST_BYTES} byte cap")
        return json.loads(self.rfile.read(n).decode("utf-8", "replace") or "{}")

    def handle_analyze(self):
        q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        want_stream = q.get("stream", ["0"])[0] in ("1", "true", "yes")
        try:
            body = self._read_json_body()
        except Exception as e:
            self.send_json({"error": f"bad request body: {e}"}, status=400)
            return

        did = (body.get("id") or "").strip()
        rec = FACETED.get(did.lower(), {})
        known = MAPMETA.get(did.lower(), {})
        meta = dict(rec)
        meta.update(body.get("meta") or {})
        meta["id"] = did or meta.get("id") or "(not given)"

        try:
            text, info = analysis.resolve_text(
                decision_id=did or None,
                supplied_text=body.get("text"),
                pdf_url=(body.get("pdf_url") or "").strip() or None,
                chunk_index=CHUNK_INDEX,
                known_pdf=known.get("pdf"),
                page_url=known.get("url") or known.get("pdf"),
                prefer=body.get("prefer") or "corpus")
        except Exception as e:
            self.send_json({"error": str(e)}, status=422)
            return
        meta["source_label"] = info.get("label")

        kw = dict(base_url=(body.get("base_url") or "").strip() or None,
                  api_key=body.get("api_key") if body.get("api_key") else None,
                  model=(body.get("model") or "").strip() or None)
        if not want_stream:
            try:
                out = analysis.analyze(text, meta, **kw)
            except Exception as e:
                self.send_json({"error": str(e), "source": info}, status=502)
                return
            self.send_json({"id": did, "source": info, "markdown": out["text"],
                            "model": out.get("model"), "usage": out.get("usage")})
            return

        self._stream(lambda: analysis.analyze(text, meta, stream=True, **kw),
                     opening=info)

    def handle_compare(self):
        """Comparative read across several decisions.

        Each item may arrive as a finished individual analysis (``markdown``) or
        as bare text; anything missing both is filled from the local corpus or
        the PDF, condensed to a share of one context budget so the set fits.
        """
        q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        want_stream = q.get("stream", ["0"])[0] in ("1", "true", "yes")
        try:
            body = self._read_json_body()
        except Exception as e:
            self.send_json({"error": f"bad request body: {e}"}, status=400)
            return

        incoming = body.get("items") or [{"id": i} for i in (body.get("ids") or [])]
        if len(incoming) < 2:
            self.send_json({"error": "pick at least two decisions to compare"},
                           status=400)
            return
        if len(incoming) > MAX_COMPARE:
            self.send_json({"error": f"comparing more than {MAX_COMPARE} decisions "
                                     f"at once will not fit a useful context window"},
                           status=400)
            return

        budget = max(6000, analysis.MAX_TEXT_CHARS // max(len(incoming), 1))
        items, notes = [], []
        for raw in incoming:
            did = (raw.get("id") or "").strip()
            meta = dict(FACETED.get(did.lower(), {}))
            meta.update(raw.get("meta") or {})
            item = {"id": did or meta.get("id") or "(unidentified)", "meta": meta}
            if raw.get("markdown"):
                item["markdown"] = raw["markdown"]
            else:
                try:
                    known = MAPMETA.get(did.lower(), {})
                    text, info = analysis.resolve_text(
                        decision_id=did or None, supplied_text=raw.get("text"),
                        chunk_index=CHUNK_INDEX, known_pdf=known.get("pdf"),
                        page_url=known.get("url") or known.get("pdf"),
                        prefer=body.get("prefer") or "corpus")
                    text, cut = analysis.condense(text, budget)
                    item["text"] = text
                    notes.append({"id": item["id"], "source": info.get("label"),
                                  "condensed": cut})
                except Exception as e:
                    self.send_json({"error": f"{item['id']}: {e}"}, status=422)
                    return
            items.append(item)

        kw = dict(base_url=(body.get("base_url") or "").strip() or None,
                  api_key=body.get("api_key") if body.get("api_key") else None,
                  model=(body.get("model") or "").strip() or None,
                  question=(body.get("question") or "").strip() or None)
        if not want_stream:
            try:
                out = analysis.compare(items, **kw)
            except Exception as e:
                self.send_json({"error": str(e)}, status=502)
                return
            self.send_json({"ids": [i["id"] for i in items], "sources": notes,
                            "markdown": out["text"], "model": out.get("model"),
                            "usage": out.get("usage")})
            return
        self._stream(lambda: analysis.compare(items, stream=True, **kw),
                     opening={"sources": notes,
                              "ids": [i["id"] for i in items]})

    def _stream(self, make_generator, opening=None):
        """Server-Sent Events: one opening frame, then content deltas."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        if opening:
            self._sse("source", opening)
        try:
            for piece in make_generator():
                self._sse("delta", {"t": piece})
            self._sse("done", {"ok": True})
        except Exception as e:
            self._sse("error", {"error": str(e)})

    def _sse(self, event, payload):
        try:
            self.wfile.write(
                f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
                .encode("utf-8"))
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass   # the tab was closed mid-answer

    def send_json(self, obj, status=200):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


if __name__ == "__main__":
    cfg = llm.defaults()
    server = http.server.ThreadingHTTPServer(
        (HOST, PORT),
        functools.partial(Handler, directory=ROOT),
    )
    print(f"[serve] analysis endpoint {cfg['base_url']} · model {cfg['model']} · "
          f"key {'set' if cfg['has_key'] else 'NOT set (the browser must supply one)'}",
          flush=True)
    print(f"[serve] http://{HOST}:{PORT}  ->  /visualization/map.html", flush=True)
    server.serve_forever()
