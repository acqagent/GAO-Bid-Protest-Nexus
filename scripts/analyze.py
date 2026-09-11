#!/usr/bin/env python3
"""Headless decision analysis — the Analysis tab without a browser.

Same prompts, same text sourcing and same output as the dashboard, driven from
a terminal so it can run over SSH, in cron, in CI, or in a container with no
display. Points at any OpenAI-compatible endpoint, including a local one.

    export OPENAI_BASE_URL=http://localhost:1234/v1   # LM Studio, Ollama, vLLM…
    export OPENAI_MODEL=your-local-model
    export OPENAI_API_KEY=sk-...                      # omit it for a local server

    # one decision, straight to stdout
    python3 scripts/analyze.py B-417327

    # a batch, one markdown file each
    python3 scripts/analyze.py --ids-file ids.txt --out reports/

    # everything sustained on OCI, analysed and then compared as a set
    python3 scripts/analyze.py --filter ground=oci,disposition=sustained \\
            --limit 8 --compare --out reports/

    # a decision that is not in the corpus at all
    python3 scripts/analyze.py B-999999 --text-file some-decision.txt

WHERE THE TEXT COMES FROM  (same order the dashboard uses)
    corpus   vector/chunks.jsonl, if you have it — no network, no PDF, and it
             needs NO models: the chunk file is read directly, so numpy and
             sentence-transformers are not imported. Those are only for search.
    pdf      fetched from gao.gov and extracted (needs `pip install pypdf`)
    file     --text-file / --text-dir, for anything the other two cannot reach

Exit status is 0 only if every requested decision produced an analysis.
"""

import argparse
import concurrent.futures
import json
import os
import re
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import analysis  # noqa: E402
import llm       # noqa: E402

MAPPED = os.path.join(ROOT, "data", "mapped-decisions.json")
CHUNKS = os.path.join(ROOT, "vector", "chunks.jsonl")
FACETS = ("ground", "disposition", "overall_outcome",
          "procurement_authority", "protest_posture")
_out_lock = threading.Lock()


def load_records():
    try:
        return {r["id"].lower(): r for r in json.load(open(MAPPED))}
    except OSError:
        return {}


def load_corpus(path, quiet=False):
    """ChunkIndex straight off chunks.jsonl — no numpy, no models."""
    if not os.path.exists(path):
        return None
    t0 = time.time()
    if not quiet:
        print(f"[corpus] reading {os.path.relpath(path, ROOT)} …",
              file=sys.stderr, flush=True)
    chunks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    ix = analysis.ChunkIndex(chunks)
    if not quiet:
        print(f"[corpus] {len(chunks)} chunks in {time.time() - t0:.1f}s",
              file=sys.stderr, flush=True)
    return ix


def pick_ids(args, records):
    ids = [i.strip().lower() for i in args.ids if i.strip()]
    if args.ids_file:
        with open(args.ids_file) as f:
            ids += [l.strip().lower() for l in f
                    if l.strip() and not l.startswith("#")]
    if not sys.stdin.isatty() and args.stdin:
        ids += [l.strip().lower() for l in sys.stdin if l.strip()]
    if args.filter:
        want = {}
        for part in args.filter.split(","):
            if "=" not in part:
                sys.exit(f"--filter wants key=value, got {part!r}")
            k, v = part.split("=", 1)
            k = k.strip()
            if k not in FACETS:
                sys.exit(f"--filter key must be one of {', '.join(FACETS)}")
            want[k] = v.strip()
        ids += [i for i, r in sorted(records.items())
                if all(r.get(k) == v for k, v in want.items())]
    seen, out = set(), []
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out[:args.limit] if args.limit else out


def meta_for(did, records):
    r = records.get(did, {})
    meta = {"id": did.upper()}
    for k in FACETS:
        if r.get(k):
            meta[k] = r[k]
    return meta


def text_for(did, args, corpus, records):
    """(text, info). Raises with a readable reason if nothing works."""
    supplied = None
    if args.text_file:
        supplied = open(args.text_file, encoding="utf-8").read()
    elif args.text_dir:
        for ext in (".txt", ".md"):
            p = os.path.join(args.text_dir, did.upper() + ext)
            if os.path.exists(p):
                supplied = open(p, encoding="utf-8").read()
                break
    rec = records.get(did, {})
    return analysis.resolve_text(
        decision_id=did, supplied_text=supplied, chunk_index=corpus,
        known_pdf=rec.get("pdf"), page_url=rec.get("url") or rec.get("pdf"),
        prefer=args.source)


def write_out(args, name, markdown):
    if not args.out:
        return None
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(markdown.rstrip() + "\n")
    return path


def analyze_one(did, args, corpus, records, kw):
    started = time.time()
    try:
        text, info = text_for(did, args, corpus, records)
    except Exception as e:
        return {"id": did.upper(), "ok": False, "error": str(e), "stage": "text"}
    meta = meta_for(did, records)
    meta["source_label"] = info.get("label")
    try:
        res = analysis.analyze(text, meta, **kw)
    except Exception as e:
        return {"id": did.upper(), "ok": False, "error": str(e), "stage": "model",
                "source": info}
    md = res["text"]
    header = (f"<!-- {did.upper()} · source: {info.get('label')}"
              f" ({info.get('chars')} chars"
              f"{', truncated' if info.get('truncated') else ''})"
              f" · model: {res.get('model')} -->\n\n")
    path = write_out(args, f"{did.upper()}.md", header + md)
    with _out_lock:
        print(f"  {did.upper():16} {info.get('label','?'):26} "
              f"{len(md):6} chars  {time.time() - started:5.1f}s"
              + (f"  -> {os.path.relpath(path, os.getcwd())}" if path else ""),
              file=sys.stderr, flush=True)
    return {"id": did.upper(), "ok": True, "markdown": md, "source": info,
            "model": res.get("model"), "usage": res.get("usage"), "meta": meta}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ids", nargs="*", help="B-numbers, e.g. B-417327")
    ap.add_argument("--ids-file", help="file of B-numbers, one per line (# comments ok)")
    ap.add_argument("--stdin", action="store_true", help="also read B-numbers from stdin")
    ap.add_argument("--filter", help="ground=oci,disposition=sustained — corpus facets")
    ap.add_argument("--limit", type=int, default=0, help="cap how many are taken")
    ap.add_argument("--list", action="store_true", help="print the selection and stop")
    ap.add_argument("--compare", action="store_true",
                    help="also write one comparison across the whole selection")
    ap.add_argument("--compare-only", action="store_true",
                    help="skip the individual analyses, just compare")
    ap.add_argument("--question", default="", help="a question to answer across the set")
    ap.add_argument("--out", help="directory for one .md per decision")
    ap.add_argument("--json", dest="json_out", help="write every result as one JSON file")
    ap.add_argument("--source", choices=("corpus", "pdf", "supplied"), default="corpus",
                    help="which text source to try first (default corpus)")
    ap.add_argument("--chunks", default=CHUNKS, help="path to vector/chunks.jsonl")
    ap.add_argument("--no-corpus", action="store_true", help="ignore the local corpus")
    ap.add_argument("--text-file", help="use this file's text (one decision)")
    ap.add_argument("--text-dir", help="directory of <B-NUMBER>.txt files")
    ap.add_argument("--base-url", default="", help="overrides OPENAI_BASE_URL")
    ap.add_argument("--model", default="", help="overrides OPENAI_MODEL")
    ap.add_argument("--api-key", default="", help="overrides OPENAI_API_KEY")
    ap.add_argument("--workers", type=int, default=2, help="decisions at a time (2)")
    ap.add_argument("--max-tokens", type=int, default=2600)
    ap.add_argument("--dry-run", action="store_true",
                    help="report the plan and the text source, call no model")
    args = ap.parse_args()

    records = load_records()
    ids = pick_ids(args, records)
    if not ids:
        ap.error("no decisions selected — give B-numbers, --ids-file, or --filter")
    if args.list:
        for i in ids:
            r = records.get(i, {})
            print(f"{i.upper():16} {r.get('ground','?'):32} {r.get('disposition','?')}")
        print(f"\n{len(ids)} decision(s)", file=sys.stderr)
        return 0

    corpus = None if args.no_corpus else load_corpus(args.chunks)
    if corpus is None and args.source == "corpus" and not args.no_corpus:
        print(f"[corpus] {os.path.relpath(args.chunks, ROOT)} not present — "
              f"falling back to the PDF or --text-dir", file=sys.stderr)

    cfg = llm.defaults()
    base = args.base_url or cfg["base_url"]
    model = args.model or cfg["model"]
    print(f"[llm] {model} at {base} · key "
          f"{'set' if (args.api_key or cfg['has_key']) else 'NOT set'}",
          file=sys.stderr)
    print(f"[plan] {len(ids)} decision(s)"
          + (", then one comparison" if args.compare or args.compare_only else ""),
          file=sys.stderr)

    if args.dry_run:
        for i in ids:
            try:
                _, info = text_for(i, args, corpus, records)
                print(f"  {i.upper():16} would use {info.get('label')} "
                      f"({info.get('chars')} chars)", file=sys.stderr)
            except Exception as e:
                print(f"  {i.upper():16} NO TEXT — {e}", file=sys.stderr)
        return 0

    kw = dict(base_url=args.base_url or None, model=args.model or None,
              api_key=args.api_key or None, max_tokens=args.max_tokens)
    results = []
    if not args.compare_only:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(
                lambda i: analyze_one(i, args, corpus, records, kw), ids))
        for r in results:
            if not r["ok"]:
                print(f"  {r['id']:16} FAILED at {r['stage']}: {r['error']}",
                      file=sys.stderr)

    comparison = None
    if (args.compare or args.compare_only) and len(ids) >= 2:
        items = []
        done = {r["id"].lower(): r for r in results if r.get("ok")}
        for i in ids:
            it = {"id": i.upper(), "meta": meta_for(i, records)}
            if i in done:
                it["markdown"] = done[i]["markdown"]     # cheaper and sharper
            else:
                try:
                    text, _ = text_for(i, args, corpus, records)
                    budget = max(6000, analysis.MAX_TEXT_CHARS // len(ids))
                    it["text"], _ = analysis.condense(text, budget)
                except Exception as e:
                    print(f"  {i.upper():16} left out of the comparison: {e}",
                          file=sys.stderr)
                    continue
            items.append(it)
        if len(items) < 2:
            print("[compare] fewer than two decisions had material — skipped",
                  file=sys.stderr)
        else:
            t0 = time.time()
            try:
                res = analysis.compare(items, question=args.question or None,
                                       base_url=args.base_url or None,
                                       model=args.model or None,
                                       api_key=args.api_key or None,
                                       max_tokens=max(args.max_tokens, 3200))
                comparison = res["text"]
                p = write_out(args, "comparison.md", comparison)
                print(f"  {'comparison':16} {len(items)} decisions"
                      f"{'':17}{len(comparison):6} chars  {time.time() - t0:5.1f}s"
                      + (f"  -> {os.path.relpath(p, os.getcwd())}" if p else ""),
                      file=sys.stderr)
            except Exception as e:
                print(f"  comparison FAILED: {e}", file=sys.stderr)
    elif args.compare or args.compare_only:
        print("[compare] needs at least two decisions", file=sys.stderr)

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump({"decisions": results, "comparison": comparison},
                      f, ensure_ascii=False, indent=1)
        print(f"[json] {args.json_out}", file=sys.stderr)

    if not args.out and not args.json_out:
        for r in results:
            if r.get("ok"):
                print(f"\n{'=' * 70}\n{r['id']}\n{'=' * 70}\n")
                print(r["markdown"])
        if comparison:
            print(f"\n{'=' * 70}\nCOMPARISON\n{'=' * 70}\n")
            print(comparison)

    failed = [r for r in results if not r.get("ok")]
    print(f"[summary] {len(results) - len(failed)}/{len(results)} analysed"
          + (f" · {len(failed)} failed" if failed else "")
          + (" · comparison written" if comparison else ""), file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
