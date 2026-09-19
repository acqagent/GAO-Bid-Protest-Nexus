# GAO Bid Protest Nexus

A self-hosted, interactive dashboard over **33,136 GAO bid protest decisions**,
1925 through 2026. One HTML file carries the Map, Table, Ground detail,
Analysis and License tabs; a small Python server adds the Dynamic Search tab
(grounded search over the full decision text) and takes over the work the
Analysis tab would otherwise ask of your browser.

**v2.1.** Read the numbers as a map of the corpus, not as findings: 27,232 of
the 33,136 grounds were assigned by a model rather than a person, and 16,451
decisions carry no recorded outcome. The License tab spells this out in full,
and every decision links back to gao.gov, which is the authority. See
[Accuracy](#accuracy).

## The six tabs

- **Map** — an interactive **3D** constellation: the 33 protest grounds sit on
  a sphere, each surrounded by its own cluster of decisions, with the
  sub-filter facets (Overall Outcome, Procurement Authority, Protest Posture,
  Decade) forming a core at the centre that every connection runs
  through. **All 33,136 decisions are rendered** as dots, and a ground's cluster
  grows with the number of decisions on it, so volume is readable before you
  read a label. Dots that don't match the active filters are drawn dim. Node
  sizes and edge weights scale with the number of matching decisions. Drag to
  rotate, scroll to zoom, right/ctrl-drag to pan, double-click to reset. Hover
  a node to trace its connections; click a **labelled** node — a ground or a
  facet value — for a popup with sub-filter breakdowns and the matching
  decisions (B-number, short description, gao.gov and PDF links). Individual
  decision dots are scenery, not targets: at 33,136 of them, clicking one by
  accident was easier than clicking the ground you meant. The left-panel
  filters drive every view. The **Select all / Select none** buttons (top of
  the side panel) set every value at once, and each filter facet also has
  **all / none** quick toggles in its header.
- **Table** — every matching decision: B-number, short description, year,
  case name, ground, outcome, authority, posture, and links to the gao.gov
  decision page and PDF. It pages through the full matching set 200 rows at a
  time, and the first column carries the checkbox that feeds the Analysis tab.
- **Dynamic Search** — hybrid retrieval over the full decision text (dense
  vectors + BM25, RRF fusion, cross-encoder re-ranking). Returns the matching
  passages, each linked back to its source decision and PDF.
- **Analysis** — key-element summaries of individual decisions, and
  comparisons across a selection of them, written by any **OpenAI-compatible**
  model endpoint. Select decisions anywhere in the dashboard — the checkbox in
  the Table's first column, a Map popup, a Dynamic Search result — or hit the
  **✦ Analyze** button on any single decision. Output lands on this tab, never
  in the view you picked from. See [Decision analysis](#decision-analysis).
- **Ground detail** — how often one ground is sustained, against the 6.0%
  corpus base rate, with the 95% confidence interval its sample size supports,
  a disposition breakdown, splits by authority and posture, and links to every
  sustained decision on that ground.
- **License** — the licensing table, the citation to copy, the credits, and a
  plain-language account of what this data can and cannot tell you.

Ground detail reads the whole corpus and ignores the side-panel filters: it
answers what the odds are on a ground, which is a property of the data rather
than of the current selection. It carries its 95% intervals everywhere, because
at this corpus size the spread between grounds is mostly real but the tails are
not: 25 of the 33 grounds separate from the 6.0% base rate (19 above, 6 below),
and the remaining 8 do not — their intervals still straddle it.

**All decisions are in the 3D map.** Unlike a flat list, the map renders every
one of the 33,136 decision dots at once, so nothing is hidden behind a cap —
the filters only change which dots are bright (matching) versus dim. Each
ground ball also shows its live count (e.g. "Past Performance 934"). For
scrollable, link-forward views of a single ground's full decision set, use a
ground's popup (first 60 listed, "…and N more") or the Table tab, which pages
through the whole matching set 200 rows at a time.

All decision text and links come from the U.S. Government Accountability
Office (public domain) at gao.gov. The dashboard itself is a single
self-contained HTML file — Map, Table and Analysis all work with no server at
all (Analysis needs a model endpoint, and asks you for the decision).

## Downloads

- **Basic** (`gao-bid-protest-nexus-basic.zip`, ~1.8 MB) —
  `visualization/map.html` plus the license files (`LICENSE`, `LICENSE-DATA`,
  `LICENSE-CORPUS-MIT`, `NOTICE`). Unzip and open the HTML file in any
  browser: you get the Map, Table, Ground detail and License tabs (all
  filters, the 3D map with all 33,136 decisions, the full decision table). No
  Python, no server, no models — fully offline. The **Analysis** tab works
  here too, once you point it at a model endpoint and hand it the decision —
  see [Decision analysis](#decision-analysis).
  **Download from the [Releases page](../../releases/latest).**
- **Full** (`gao-bid-protest-nexus-full.zip`, ~250 MB) — everything in the
  basic zip plus the Dynamic Search backend (server scripts, the
  77,979-chunk vector index, and the decision metadata). Unzip and follow
  the Quick start to also get Dynamic Search over the full decision text
  (local models; the first run downloads ~1.4 GB of models,
  then it is fully offline).
  **A v2.1 full bundle is coming soon.** The download available today still
  carries the index built over the earlier 5,986-decision corpus, so Dynamic
  Search covers that subset while Map, Table and Ground detail read the
  embedded data and are current across all 33,136. The rebuilt index covering
  the whole corpus is finished and in review; this section will point at it
  once it ships. If you would rather not wait, `scripts/vectorize.py` builds
  the index yourself.
  **Download:
  [acqagent.ai/downloads/gao-bid-protest-nexus-full.zip](https://acqagent.ai/downloads/gao-bid-protest-nexus-full.zip)**
  (resumable, so an interrupted download can pick up where it stopped).

Use the basic zip to browse and filter; use the full zip if you also want
to search the full text of the decisions.

### What is in this repository

This repo holds the **source tree** (~25 MB): the dashboard, the hosting build
of it, the server scripts, the decision metadata, and the index metadata. It
does **not** contain the vector index — `vector/chunks.jsonl` (146 MB) and
`vector/embeddings.npy` (229 MB) are each over GitHub's 100 MB per-file
limit. Get them from the
[full zip](https://acqagent.ai/downloads/gao-bid-protest-nexus-full.zip).

Cloning the repo is enough to open the Map, Table, Ground detail, Analysis and
License tabs. **Dynamic Search additionally requires the two `vector/` files** —
without them `scripts/serve.py` still runs, it just has no index to search (and
the Analysis tab then sources decisions from their PDFs instead of from the
index).

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

**Using a reasoning model.** o-series, DeepSeek-R1, Qwen3 and friends bill their
thinking against the same completion budget as the answer, and a GAO decision is
a long prompt. Left alone, such a model can spend the entire budget reasoning and
return **an empty analysis** — the tab renders nothing and the failure is silent.
Raising the ceiling does not fix it; the model simply thinks longer and then
outruns the timeout. Cap the effort instead:

```bash
export OPENAI_REASONING_EFFORT=low   # sent as reasoning_effort; the actual fix
export ANALYZE_MAX_TOKENS=8000       # per decision   (default 2600)
export COMPARE_MAX_TOKENS=10000      # per comparison (default 3200)
export OPENAI_TIMEOUT=1800           # seconds; a local 27B streams for minutes
```

All four are read by the server and by `scripts/analyze.py` (whose
`--max-tokens` overrides the budget for one run). When an endpoint rejects
`reasoning_effort` with a 400 that names the parameter, the client drops it and
retries, so leaving the variable set is usually safe across providers. If an
analysis comes back blank, check `finish_reason` — `length`, with the whole
budget spent on reasoning tokens and the prose sitting in `reasoning_content`,
is this problem.

**What leaves your machine.** The decision text and your question go to
whichever endpoint you configure. The decisions are public record, but review
that against your own rules — and note that a local endpoint (LM Studio, Ollama,
vLLM) keeps everything in-house.

### Headless

Everything the Analysis tab does is available from a terminal, with no browser
and no display — over SSH, in cron, in CI, in a container:

```bash
export OPENAI_BASE_URL=http://localhost:1234/v1   # LM Studio, Ollama, vLLM...
export OPENAI_MODEL=your-local-model
export OPENAI_REASONING_EFFORT=low                # if it is a reasoning model

python3 scripts/analyze.py B-417327                       # one, to stdout
python3 scripts/analyze.py --ids-file ids.txt --out reports/
python3 scripts/analyze.py --filter ground=oci,disposition=sustained \
        --limit 8 --compare --out reports/ --json reports/all.json
```

Select with B-numbers, `--ids-file`, stdin, or `--filter` over the corpus
facets. `--list` prints the selection without calling anything; `--dry-run`
reports where each decision's text would come from; `--compare` adds one
comparison across the set, reusing the individual analyses it just wrote.
Output is one Markdown file per decision plus `comparison.md`, and `--json`
gives the same machine-readable. Exit status is 0 only if every decision
produced an analysis.

Sourcing text from the local corpus this way **needs no models at all** —
`vector/chunks.jsonl` is read directly, so numpy and sentence-transformers are
never imported. Those are only for Dynamic Search. `scripts/resolve_pdfs.py` is
headless on the same terms, and imports nothing outside the standard library.

**Read the decision.** These summaries are a reading aid over a public record,
not legal advice, and a model can misread a holding. Every result links back to
the decision it was written from; the prompts are tuned to say "not stated in
the decision" rather than guess, and to flag their own gaps.

## Decision PDF links

**5,893 of 33,136 decisions link straight to their PDF.** The rest link to
their page on gao.gov, where the document is one click further on. The
dashboard shows one of three labels, so a link never promises more than it is:

| Label | Count | Meaning |
|---|---|---|
| `PDF ↗` | 5,893 | a resolver pass fetched this URL and got a PDF back |
| `PDF ↗?` (amber, dotted) | 4 | a `.pdf` URL the corpus carries that no run has confirmed |
| `find PDF` / gao.gov only | 27,239 | no direct PDF URL yet — the gao.gov page is the link |

The `find PDF` link appears when `scripts/serve.py` is running: it asks the
local server to resolve that one decision's PDF on demand. To resolve the
whole set ahead of time, run `scripts/resolve_pdfs.py` (below).

To resolve links yourself, or to import a set someone else resolved — either
way the basic build ships them, since the data is embedded in `map.html`:

```bash
# import an already-resolved set: b_number,url,classification
# (b_number may hold several pipe-joined B-numbers for one consolidated docket)
python3 scripts/resolve_pdfs.py --import links.csv --apply
```

Under `--import` the file is treated as the whole truth: any decision it has no
PDF for keeps whatever link it already had and is flagged unconfirmed rather
than quietly presented as a PDF.

To resolve from scratch instead:

```bash
python3 scripts/resolve_pdfs.py --dry-run   # count the work, fetch nothing
python3 scripts/resolve_pdfs.py --csv audit.csv   # resolve, saving as it goes
python3 scripts/resolve_pdfs.py --apply     # write them into the data and the dashboard
```

The script is **standalone**: one file, standard library only, no pip install,
no API key and no model — it only speaks HTTP to gao.gov. Copy it anywhere and
point `--input` at `mapped-decisions.json`, `map.json`, or even the single-file
`map.html` from the basic zip. `--csv` writes a per-decision audit trail (id,
url, which method found it, status) and `--recheck` re-verifies links already
held rather than trusting them. Ctrl-C is safe; a rerun continues where it
stopped.

**How it resolves.** gao.gov files a decision's PDF under one of three
spellings and the record does not say which: `/assets/b-417327.pdf`,
`/assets/417327.pdf`, or the older nested `/assets/330/325381.pdf`. The
B-number predicts the order to try — the bare form dominates from B-420000 up,
the `b-` form below B-300000, the nested form in the oldest decisions — so most
cost a single `HEAD`. Only consolidated dockets, whose filenames carry several
B-numbers, fall back to fetching and parsing the landing page. A link is written
only when gao.gov confirms the file is there — nothing is guessed.

Four workers 0.4s apart by default — be kind, gao.gov is a public service.
While the dashboard server is running, its **find PDF** link does the same
thing for one decision at a time.

Before it fetches anything, the script also takes the links that need no
network at all: GAO publishes **one document per consolidated docket** and the
asset filename names every B-number it covers
(`b-414706,b-414380.2.pdf`), so each name in a filename already known points at
that same PDF. It runs again after fetching, because every consolidated filename
a run discovers unlocks its siblings too.

Reruns are cheap and idempotent: `--recheck` re-verifies links already held
rather than trusting them.

## Minimum hardware requirements

The Map and Table tabs are plain HTML with embedded data — no server or model
needed. The Map is an interactive 3D view that draws all 33,136 decision dots,
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

**Rebuilding the index.** `scripts/vectorize.py` builds `vector/chunks.jsonl`
and `vector/embeddings.npy` from a list of decision PDF links, in four resumable
stages — fetch the PDFs, extract their text, chunk it, embed it:

```bash
python3 scripts/vectorize.py --links pdf-links-classified.csv \
        --embedder api --embed-base-url http://localhost:8080/v1 \
        --embed-model Qwen3-Embedding-8B
```

`--embedder api` targets any OpenAI-compatible `/v1/embeddings` endpoint, so a
local server (llama.cpp, vLLM, LM Studio, Ollama, TEI, Infinity) can do the
embedding; `--embedder local` runs sentence-transformers in-process instead.
Each stage skips work it has already done, and the embed stage writes into a
memory-mapped file and records its position, so a run interrupted at chunk
60,000 of 78,000 resumes there rather than starting over.

The index records what built it — model, backend, vector width, and the
retrieval prefix that model expects — in `vector/meta.json`, and
`scripts/searchlib.py` reads that rather than assuming. So a rebuild with a
different embedding model needs no code change: build it, and search follows.
For an API-built index, set `EMBED_BASE_URL` (and `EMBED_API_KEY` if the
endpoint wants one) so queries are embedded the same way the passages were.

Use a model trained for retrieval. A general chat model will emit vectors, but
they are not trained to put a question near the passage that answers it, and a
large one costs many hours of forward passes over the ~78k chunks this corpus
produces. `Qwen3-Embedding-0.6B/4B/8B`, `BAAI/bge-*` and `intfloat/e5-*` are all
built for it.

**Other local models.** Change `--embed-model`, or `RERANK_MODEL` in
`scripts/searchlib.py`. For example `BAAI/bge-large-en-v1.5` (stronger, ~330
MB) or `BAAI/bge-m3` (multilingual) for dense; any `CrossEncoder` model for
re-ranking. The shipped `vector/embeddings.npy` was built with bge-base-en-v1.5. Switching
the dense model means regenerating it — `scripts/vectorize.py --stage embed` —
or search quality will be wrong; the vector width is checked against
`vector/meta.json` at startup to catch the obvious version of that mistake.

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
visualization/map-connect.html  hosting build: same six tabs plus a Setup tab
                         explaining how to connect your own model endpoint
scripts/build_static.py  produces the static / web / connect hosting builds
scripts/serve.py         HTTP server: static files + search + analysis API
scripts/searchlib.py     hybrid search core (dense + BM25 + RRF + re-rank)
scripts/analysis.py      decision text sourcing, PDF handling, analysis prompts
scripts/llm.py           OpenAI-compatible chat client (standard library only)
scripts/analyze.py       headless CLI: analyze and compare without a browser
scripts/vectorize.py     builds the search index: fetch, extract, chunk, embed
scripts/resolve_pdfs.py  one-off: gao.gov landing pages -> direct PDF links
vector/meta.json         index metadata (model, dimensionality, chunk count)
data/map.json            decision metadata (B-numbers, gao.gov/PDF links)
data/mapped-decisions.json  per-decision ground + filter values (earlier corpus)
requirements.txt         Python dependencies for the Dynamic Search backend
```

Full zip only (too large for GitHub — from
[acqagent.ai](https://acqagent.ai/downloads/gao-bid-protest-nexus-full.zip)):

```
vector/chunks.jsonl      146 MB — 77,979 text chunks from the decision corpus
vector/embeddings.npy    229 MB — precomputed dense embeddings (bge-base-en-v1.5, 768-d)
```

## Hosting it

Three builds come out of the same source file, each one switch away:

| Build | Command | Tabs | Needs |
|---|---|---|---|
| Local (default) | `visualization/map.html` | all six | nothing for Map/Table/Ground/License; `scripts/serve.py` for Dynamic Search; a model endpoint for Analysis |
| Static | `python3 scripts/build_static.py` | four | a file server, nothing else — Dynamic Search and Analysis are removed outright |
| Web | `python3 scripts/build_static.py --web` | all six | a file server; the two backend tabs explain themselves to a visitor instead of naming a server they cannot start |
| Connect | `python3 scripts/build_static.py --connect` | all six + Setup | same as Web, plus a Setup tab walking a visitor through pointing it at their own OpenAI-compatible endpoint |

Upload the output as `index.html`. It is ~10.9 MB raw and ~1.8 MB gzipped —
mostly embedded JSON, so turn gzip or brotli on at the server and it compresses
about 6:1. No build step, no dependencies, no backend.

The switches (`STATIC_BUILD`, `WEB_BUILD`, `SETUP_GUIDE`) are three `const`
lines in `visualization/map.html`; the script flips one and lets the page strip
its own markup after it has initialized. Rebuild whenever `map.html` changes —
it is a one-line transform, not a fork, so the builds cannot drift.

**Do not put `scripts/serve.py` on a public address.** It binds loopback for a
reason: `/api/analyze` and `/api/compare` would let anyone spend the key in
`OPENAI_API_KEY`, and `/api/pdf` becomes an open fetch proxy. If you want
Dynamic Search hosted, put a reverse proxy in front that forwards only
`/api/search` and the static files, and leave `OPENAI_API_KEY` unset.

## License

This project uses two licenses, one for code and one for data.

| What | License |
| --- | --- |
| Dashboard code, search backend, scripts | [Apache 2.0](LICENSE) |
| Protest taxonomy, per-decision coding, decision metadata, vector index | [CC BY 4.0](LICENSE-DATA) |
| The upstream decision corpus | [MIT](LICENSE-CORPUS-MIT) / CC BY, per source |
| The GAO decisions themselves | Public domain (17 U.S.C. 105) |

In short: use it, change it, build on it, at work or commercially. Just keep the
credit. See [NOTICE](NOTICE) for the attribution you need to carry forward.

`visualization/map.html` embeds the decision data inside the application code.
In that file the code is Apache 2.0 and the embedded data is CC BY 4.0.

### Credits

The Nexus is built on someone else's work, and the corpus came first.

**The decision corpus** came from [Kevin Misener](https://github.com/kmisener90),
in two collections:
[GAO-Bid-Protest](https://huggingface.co/datasets/Kmisener/GAO-Bid-Protest) on
Hugging Face, the larger corpus, under the **MIT License**; and the earlier
[GAO-Bid-Protest-Dataset](https://github.com/kmisener90/GAO-Bid-Protest-Dataset),
roughly 5,700 decisions, under a Creative Commons Attribution license. Both are
used the same way here: the PDF URLs, the 33-ground taxonomy, and the
per-decision coding were derived from them using AI. If you reuse the corpus,
credit that source too — see [LICENSE-CORPUS-MIT](LICENSE-CORPUS-MIT).

**The decisions themselves** are works of the U.S. federal government and are
not protected by U.S. copyright (17 U.S.C. 105), so they are in the public
domain. Cite GAO as the source.

### Name and marks

The code and data are yours to reuse. The AcqAgent name and logo are not. Say
your work is based on the GAO Bid Protest Nexus, but do not present a fork as an
official AcqAgent release.

### Earlier releases

Releases before v1.0 were published under CC BY 4.0 in full, including the code.
Those releases stay available under that license. From v1.0 forward, code is
Apache 2.0 and data is CC BY 4.0.

### Accuracy

The coding was derived using AI and has not been verified decision by decision.
This is not legal advice. Verify anything you rely on against the original GAO
decision. Found a miscoded decision, or want a ground split differently? Open an
issue.
