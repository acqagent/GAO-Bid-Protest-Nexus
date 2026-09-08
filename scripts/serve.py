#!/usr/bin/env python3
"""Local server for the GAO Bid Protest Nexus dashboard.

Serves the static site (visualization/, data/, ...) and a small search API
backed by the local vector store in vector/ :

    GET /api/search?q=<question>&k=<n>[&file=<B-number>][&mode=hybrid|dense]
    -> {"query": ..., "results": [{file, pages, date, score, text,
                                    isDigest, mapIds, links}]}

Retrieval is hybrid (dense + BM25, RRF fusion) with cross-encoder re-ranking;
mode=dense disables both (legacy behaviour).

Run with the project venv:

    .venv/bin/python scripts/serve.py
    # then open http://127.0.0.1:8765/visualization/map.html
"""

import functools
import http.server
import json
import os
import posixpath
import time
import urllib.parse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import searchlib  # noqa: E402  (sits next to this script)

HOST = "127.0.0.1"   # loopback only; reach it remotely via `tailscale serve`
PORT = 8765
MAX_K = 50           # cap on ?k= — every candidate costs one cross-encoder pass
# Only the dashboard's own assets are served; the rest of the project tree
# (.venv/, logs/, llm-work/, vector/, *.csv) stays off the wire.
ALLOWED_PREFIXES = ("/visualization/", "/data/")


def load_store():
    t0 = time.time()
    print("[serve] loading store (embedder + reranker + BM25) ...", flush=True)
    store = searchlib.Store(use_reranker=True)
    print(f"[serve] ready in {store.load_seconds}s "
          f"({len(store.chunks)} chunks, {len(store.chunks) - sum(1 for c in store.chunks if c['id'].endswith('__digest'))} body chunks)",
          flush=True)
    return store


STORE = load_store()
MAPMETA = {d["id"]: {"url": d.get("url"), "pdf": d.get("pdf")}
            for d in json.load(open(os.path.join(ROOT, "data", "map.json")))["decisions"]}
for d in json.load(open(os.path.join(ROOT, "data", "mapped-decisions.json"))):
    MAPMETA.setdefault(d["id"].lower(), {"url": d.get("url"), "pdf": d.get("pdf")})


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
        if self._gate():
            super().do_GET()

    def do_HEAD(self):
        if self._gate():
            super().do_HEAD()

    def handle_search(self):
        p = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        q = p.get("q", [""])[0].strip()
        kraw = p.get("k", [""])[0]
        k = int(kraw) if kraw.isdigit() and len(kraw) <= 6 else 6
        k = max(1, min(k, MAX_K))
        filef = p.get("file", [""])[0].strip()
        mode = p.get("mode", ["hybrid"])[0] if p.get("mode", ["hybrid"])[0] in ("hybrid", "dense") else "hybrid"
        if not q:
            self.send_json({"query": "", "results": []})
            return

        t0 = time.time()
        raw = STORE.search(q, k=k, file_filter=filef or None, mode=mode)
        results = []
        for r in raw:
            c = r["chunk"]
            bases = searchlib.map_ids(c["file"])
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

    def send_json(self, obj):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


if __name__ == "__main__":
    server = http.server.ThreadingHTTPServer(
        (HOST, PORT),
        functools.partial(Handler, directory=ROOT),
    )
    print(f"[serve] http://{HOST}:{PORT}  ->  /visualization/map.html", flush=True)
    server.serve_forever()
