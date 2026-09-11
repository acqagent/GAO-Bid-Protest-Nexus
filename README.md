# GAO Bid Protest Nexus

A self-hosted, interactive dashboard over **5,986 GAO bid protest decisions**
(~5,978 decision PDFs). One HTML file carries the Map, Table and Analysis
views; a small Python server adds the Dynamic Search view (grounded search over
the full decision text) and takes over the work the Analysis tab would
otherwise ask of your browser.

## The five tabs

- **Map** — an interactive **3D** constellation: the 33 protest grounds sit on
  a sphere, each surrounded by its own cluster of decisions, with the four
  sub-filter facets (Disposition, Overall Outcome, Procurement Authority,
  Protest Posture) forming a core at the centre that every connection runs
  through. **All 5,986 decisions are rendered** as dots, and a ground's cluster
  grows with the number of decisions on it, so volume is readable before you
  read a label. Dots that don't match the active filters are drawn dim. Node sizes and edge weights scale with the number of
  matching decisions. Drag to rotate, scroll to zoom, right/ctrl-drag to pan,
  double-click to reset. Hover a node to trace its connections; click any
  node for a popup with sub-filter breakdowns and the matching decisions
  (B-number, short description, gao.gov and PDF links). The left-panel
  filters drive every view. The **Select all / Select none** buttons (top of
  the side panel) set every value at once, and each filter facet also has
  **all / none** quick toggles in its header.
- **Table** — every matching decision: B-number, short description, all five
  filter values, and links to the gao.gov decision page and PDF.
- **Dynamic Search** — hybrid retrieval over the full decision text (dense
  vectors + BM25, RRF fusion, cross-encoder re-ranking). Returns the matching
  passages, each linked back to its source decision and PDF.
- **Analysis** — key-element summaries of individual decisions, and
  comparisons across a selection of them, written by any **OpenAI-compatible**
  model endpoint. Select decisions anywhere in the dashboard — the checkbox in
  the Table's first column, a Map popup, a Dynamic Search result — or hit the
  **✦ Analyze** button on any single decision. Output lands on this tab, never
  in the view you picked from. See [Decision analysis](#decision-analysis).
- **Ground detail** — how often one ground is sustained, against the 12.2%
  corpus base rate, with the 95% confidence interval its sample size supports,
  a disposition breakdown, splits by authority and posture, and links to every
  sustained decision on that ground.

Ground detail reads the whole corpus and ignores the side-panel filters: it
answers what the odds are on a ground, which is a property of the data rather
than of the current selection. It carries its intervals everywhere, because
most of the apparent spread between grounds is sampling noise —
of the 33 grounds only three (OCI, Corrective Action Challenge and Technical
Evaluation) are distinguishable from the base rate once you correct for
testing 33 of them.

**All decisions are in the 3D map.** Unlike a flat list, the map renders every
one of the 5,986 decision dots at once, so nothing is hidden behind a cap —
the filters only change which dots are bright (matching) versus dim. Each
ground ball also shows its live count (e.g. "Past Performance 808"). For
scrollable, link-forward views of a single ground's full decision set, use a
ground's popup (first 60 listed, "…and N more") or the Table tab (up to 1,000
rows).

All decision text and links come from the U.S. Government Accountability
Office (public domain) at gao.gov. The dashboard itself is a single
self-contained HTML file — Map, Table and Analysis all work with no server at
all (Analysis needs a model endpoint, and asks you for the decision).

## Downloads

- **Basic** (`gao-bid-protest-nexus-basic.zip`, ~0.3 MB) — just
  `visualization/map.html`. Unzip and open the file in any browser: you get
  the Map and Table tabs (all filters, the 3D map with all 5,986 decisions,
  the full decision table). No Python, no server, no models — fully offline.
  The **Analysis** tab works here too, once you point it at a model endpoint
  and hand it the decision — see
  [Decision analysis](#decision-analysis).
  **Download from the [Releases page](../../releases/latest).**
- **Full** (`gao-bid-protest-nexus-full.zip`, ~250 MB) — everything in the
  basic zip plus the Dynamic Search backend (server scripts, the
  77,979-chunk vector index, and the decision metadata). Unzip and follow
  the Quick start to also get Dynamic Search over the full decision text
  (local models; the first run downloads ~1.4 GB of models,
  then it is fully offline).
  **Download:
  [acqagent.ai/downloads/gao-bid-protest-nexus-full.zip](https://acqagent.ai/downloads/gao-bid-protest-nexus-full.zip)**
  (resumable, so an interrupted download can pick up where it stopped).

Use the basic zip to browse and filter; use the full zip if you also want
to search the full text of the decisions.

### What is in this repository

This repo holds the **source tree** (~4.3 MB): the dashboard, the server
scripts, the decision metadata, and the index metadata. It does **not**
contain the vector index — `vector/chunks.jsonl` (146 MB) and
`vector/embeddings.npy` (229 MB) are each over GitHub's 100 MB per-file
limit. Get them from the
[full zip](https://acqagent.ai/downloads/gao-bid-protest-nexus-full.zip).

Cloning the repo is enough to open the Map, Table and Analysis tabs. **Dynamic
Search additionally requires the two `vector/` files** — without them
`scripts/serve.py` still runs, it just has no index to search (and the Analysis
tab then sources decisions from their PDFs instead of from the index).

## Quick start

Map and Table only — no download needed beyond the repo:

```bash
git clone https://github.com/acqagent/GAO-Bid-Protest-Nexus.git
# then open visualization/map.html in a browser
```

With Dynamic Search — copy `vector/chunks.jsonl` and
`vector/embeddings.npy` from the full zip into `vector/` first, then:

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/serve.py
# open http://127.0.0.1:8765/
```

To also use the **Analysis** tab without pasting a key into the browser, set
the endpoint in the server's environment first:

```bash
export OPENAI_BASE_URL=https://api.openai.com/v1   # or any compatible endpoint
export OPENAI_MODEL=gpt-4o-mini
export OPENAI_API_KEY=sk-...
.venv/bin/python scripts/serve.py
```

The first run downloads the two local models (~1.4 GB) from
huggingface.co; they are cached in `~/.cache/huggingface` and afterwards the
server runs fully offline.

**Don't need Dynamic Search?** Just open `visualization/map.html` in a browser (or
serve `visualization/` with any static file server).

Configuration (top of `scripts/serve.py`): `HOST` (loopback by default),
`PORT` (8765), `MAX_K` (search result cap, 50), `MAX_COMPARE` (decisions in one
comparison, 12). The server exposes:

| Endpoint | Used by |
|---|---|
| `GET /api/search?q=<question>&k=<n>` | Dynamic Search |
| `GET /api/config` | what this server can do (index loaded? PDF extraction? key set?) |
| `GET /api/pdf?id=<B-number>` | resolving a gao.gov landing page to the actual PDF |
| `POST /api/analyze[?stream=1]` | one decision's key-element summary |
| `POST /api/compare[?stream=1]` | one comparison across several decisions |

**The vector index is now optional.** Without `vector/chunks.jsonl` and
`vector/embeddings.npy` the server still starts and still serves the dashboard,
the PDF resolver and the analysis API — only Dynamic Search switches off, and
`/api/config` says why.

## Decision analysis

The Analysis tab summarizes a decision into a fixed set of key elements —
snapshot, procurement authority and posture, grounds raised, **GAO's ruling
ground by ground**, the reasoning, competitive prejudice, timeliness and
jurisdiction, disposition and recommendation, authorities cited, the practical
takeaway, and an explicit "gaps and low confidence" section. Comparing a
selection produces a different shape: a side-by-side table, what the decisions
share, where they diverge and why, and the line GAO is drawing across them.

**Selecting.** Every decision in the dashboard carries a **Select** control
(the Table's first column, Map popups, Dynamic Search results) and a
**✦ Analyze** button that jumps straight to the Analysis tab and runs that one
on its own. A tray along the bottom shows what is selected; the cap is 12,
because a comparison has to fit one context window. The selection survives a
reload, and `#tab=an&pick=b-407234,b-417327` deep-links into it.

**Where the text comes from**, in order, reported on every result:

1. the **local corpus** — the same chunks Dynamic Search searches, so no
   network and no PDF is needed (full build, server running);
2. the **decision PDF** — fetched from gao.gov and extracted server-side
   (needs `pypdf`; `pip install pypdf`);
3. **what you give it** — drop a PDF on the card or paste the text. This is the
   basic build's normal path, since a browser cannot read gao.gov's files from
   another page and the single HTML file carries no decision text.

**Configuring the endpoint.** The *Model endpoint* panel on the Analysis tab
takes a base URL, a model name and a key. Anything that speaks the OpenAI
`/v1/chat/completions` shape works: OpenAI, OpenRouter, Together, Groq,
Fireworks, LiteLLM, vLLM, llama.cpp, LM Studio, Ollama's `/v1` shim. With the
server running, set `OPENAI_API_KEY` in its environment instead and the key
never reaches the browser; typed keys are kept in that browser's `localStorage`
only. Without the server the page calls the endpoint directly, so the endpoint
must send CORS headers.

**What leaves your machine.** The decision text and your question go to
whichever endpoint you configure. The decisions are public record, but review
that against your own rules — and note that a local endpoint (LM Studio, Ollama,
vLLM) keeps everything in-house.

**Read the decision.** These summaries are a reading aid over a public record,
not legal advice, and a model can misread a holding. Every result links back to
the decision it was written from; the prompts are tuned to say "not stated in
the decision" rather than guess, and to flag their own gaps.

## Decision PDF links

The corpus carries a direct `gao.gov/assets/....pdf` link for some decisions and
only a `gao.gov/products/` landing page for the rest. The dashboard labels the
two apart rather than calling both "PDF", so a link never lands somewhere you
did not expect.

To turn the landing pages into direct PDF links for good — which the basic
build ships with, since the data is embedded in `map.html`:

```bash
python3 scripts/resolve_pdfs.py --dry-run   # count the work, fetch nothing
python3 scripts/resolve_pdfs.py             # resolve, caching as it goes
python3 scripts/resolve_pdfs.py --apply     # write them into the data and the dashboard
```

**How it resolves.** gao.gov files a decision's PDF under one of two spellings,
`/assets/b-417327.pdf` or `/assets/417327.pdf`, and the record does not say
which. Of the links the corpus already has, 82% are one of the two, and the
B-number predicts the order to try (the bare form dominates from B-420000 up).
So each decision costs one or two `HEAD` requests, and only the ~18% that are
consolidated dockets fall back to fetching and parsing the landing page. A link
is written only when gao.gov confirms the file is there — nothing is guessed.

It is resumable (Ctrl-C is safe, progress is saved as it goes), rate-limited,
and needs only the standard library plus network access to gao.gov. While the
server is running, the **find PDF** link in the dashboard does the same thing
for one decision at a time.

Current state in this repo: **1,915 of 5,986** decisions carry a direct PDF
link. The remaining 4,071 show as landing pages until the script is run.

## Minimum hardware requirements

The Map and Table tabs are plain HTML with embedded data — no server or model
needed. The Map is an interactive 3D view that draws all 5,986 decision dots,
so use a recent version of a hardware-accelerated browser (Chrome, Edge,
Firefox, or Safari) on a machine with a discrete or modern integrated GPU for
smooth rotation; an older machine will still render it, just less fluently.
The Dynamic Search tab runs local transformer models:

| Resource  | Minimum            | Comfortable                     |
|-----------|--------------------|---------------------------------|
| RAM       | 4 GB               | 8 GB                            |
| Disk      | ~1.8 GB (~0.4 GB index + ~1.4 GB model download) | 3 GB |
| CPU       | any modern x86-64 / ARM64, 2+ cores | 4+ cores — hybrid queries ≈ 2–5 s |
| GPU       | not required       | optional (CUDA/MPS) — makes re-ranking much faster |
| Python    | 3.10+              | 3.12                            |

What drives the footprint: the in-memory vector store (~240 MB of embeddings
for 77,979 text chunks), an in-RAM BM25 index, and two transformer models.
Expect a one-time ~30–60 s startup while the store loads.

## Models and cloud API options (Dynamic Search tab)

**Default: fully local, offline after first download.**
- Dense embeddings: `BAAI/bge-base-en-v1.5` (768-d) via sentence-transformers
- Re-ranking: `BAAI/bge-reranker-base` (cross-encoder)
- Lexical: BM25 (`rank-bm25`), fused with dense via Reciprocal Rank Fusion

**Other local models.** Change `DENSE_MODEL` / `RERANK_MODEL` in
`scripts/searchlib.py`. For example `BAAI/bge-large-en-v1.5` (stronger, ~330
MB) or `BAAI/bge-m3` (multilingual) for dense; any `CrossEncoder` model for
re-ranking. Note the shipped `vector/embeddings.npy` was built with
bge-base-en-v1.5 (the model is checked against `vector/meta.json` at
startup) — if you switch the dense model you must regenerate
`vector/embeddings.npy` with the new model or search quality will be wrong.

**Cloud API alternatives.** Instead of (or in addition to) the local models,
the embedder and/or reranker in `scripts/searchlib.py` can call a hosted
service — e.g. OpenAI `text-embedding-3-small`/`3-large`, Cohere
`embed-v3` + `rerank`, Jina embeddings + reranker, Voyage, etc. Two patterns:

1. *Query-time API, local index* — embed queries with the API at search time.
   The stored vectors must have been produced by the same model, so
   `vector/embeddings.npy` has to be regenerated offline with that model's
   API. Re-ranking can stay local or move to the API.
2. *Fully API-driven* — drop the local index entirely; fetch embeddings and
   call a rerank API per query. Requires network + an API key (set via
   environment variable, never hardcoded), adds per-query latency and cost,
   and sends question text to the provider. The underlying decisions are
   public record, but review this against your own privacy/compliance rules.

Only the embedder/reranker plumbing in `searchlib.py` needs touching; the
fusion, re-ranking blend and result shaping in `Store.search()` work
unchanged with any model or API.

## Folder contents

In the repo *and* the full zip:

```
visualization/map.html   the dashboard (all Map/Table/Analysis data embedded)
scripts/serve.py         HTTP server: static files + search + analysis API
scripts/searchlib.py     hybrid search core (dense + BM25 + RRF + re-rank)
scripts/analysis.py      decision text sourcing, PDF handling, analysis prompts
scripts/llm.py           OpenAI-compatible chat client (standard library only)
scripts/resolve_pdfs.py  one-off: gao.gov landing pages -> direct PDF links
vector/meta.json         index metadata (model, dimensionality, chunk count)
data/map.json            decision metadata (B-numbers, gao.gov/PDF links)
data/mapped-decisions.json  per-decision ground + four filter values
requirements.txt         Python dependencies for the Dynamic Search backend
```

Full zip only (too large for GitHub — from
[acqagent.ai](https://acqagent.ai/downloads/gao-bid-protest-nexus-full.zip)):

```
vector/chunks.jsonl      146 MB — 77,979 text chunks from the decision corpus
vector/embeddings.npy    229 MB — precomputed dense embeddings (bge-base-en-v1.5, 768-d)
```

## License

- **Dashboard** (scripts, visualization, search index):
  [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/)
  (CC BY 4.0). Share and adapt freely, including commercially, with
  attribution.
- **Corpus credit**: Kevin Misener
  ([@kmisener90](https://github.com/kmisener90),
  [GAO-Bid-Protest-Dataset](https://github.com/kmisener90/GAO-Bid-Protest-Dataset))
  provided the corpus of ~5,700 GAO bid protest decisions, in which AI was
  used to derive the URLs to the decision PDFs. Please credit this source
  when reusing the corpus.
- **Underlying government data**: GAO bid protest decisions are works of the
  U.S. federal government, not protected by U.S. copyright (17 U.S.C. § 105)
  — public domain. Cite GAO as the source.
