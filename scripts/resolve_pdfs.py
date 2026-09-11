#!/usr/bin/env python3
"""Turn gao.gov landing-page links into direct PDF links, once, offline.

Most decisions in the corpus carry a ``pdf`` field that is really a gao.gov
``/products/`` page — HTML, not the PDF. That is why a "PDF" link in the
dashboard can land you on a web page instead of the document. The page itself
links the actual ``/assets/....pdf``; this script follows each one, records what
it finds, and (with --apply) writes the direct links back into the data and into
the single-file dashboard, so the basic build ships with them too.

    # see the scale of it, fetch nothing
    python3 scripts/resolve_pdfs.py --dry-run

    # resolve everything into data/pdf-links.json (resumable — rerun any time)
    python3 scripts/resolve_pdfs.py

    # write what has been resolved into the data and the dashboard
    python3 scripts/resolve_pdfs.py --apply

Only needs the standard library, and needs network access to gao.gov. The
dashboard's own "find PDF" link does the same thing for one decision at a time
while scripts/serve.py is running; this is the bulk version.
"""

import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analysis  # noqa: E402

ROOT = analysis.ROOT
MAPPED = ROOT / "data" / "mapped-decisions.json"
MAPJSON = ROOT / "data" / "map.json"
HTML = ROOT / "visualization" / "map.html"
LINKS = analysis.PDF_LINKS

_print_lock = threading.Lock()


def needs_pdf(rec):
    return not str(rec.get("pdf") or "").lower().endswith(".pdf")


def page_for(rec):
    for key in ("url", "pdf"):
        v = rec.get(key)
        if v and "/products/" in v:
            return v
    return rec.get("url") or rec.get("pdf")


def resolve_all(todo, cache, sleep, workers, retries):
    done = {"n": 0, "hit": 0, "miss": 0, "asset": 0, "scraped": 0}
    total = len(todo)

    def one(rec):
        did, page = rec["id"].lower(), page_for(rec)
        found, how, err = None, "", ""
        for attempt in range(retries + 1):
            try:
                # Tries the two /assets/ spellings first (one HEAD each, and the
                # answer for ~82% of decisions), then reads the landing page.
                found, how = analysis.resolve_pdf_url(
                    did, page_url=page, use_cache=False)
                break
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                if attempt < retries:
                    time.sleep(sleep * (2 ** attempt) + 0.5)
        time.sleep(sleep)
        with _print_lock:
            done["n"] += 1
            if found:
                cache[did] = found
                done["hit"] += 1
                done["asset" if how == "asset url" else "scraped"] += 1
            else:
                done["miss"] += 1
            if done["n"] % 25 == 0 or done["n"] == total:
                print(f"  {done['n']}/{total}  found {done['hit']} "
                      f"({done['asset']} by asset url, {done['scraped']} by page)  "
                      f"missed {done['miss']}", flush=True)
                save(cache)
            if not found and done["miss"] <= 10:
                print(f"    no pdf for {rec['id']} ({page}) {err}", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(one, todo))
    save(cache)
    return done


def save(cache):
    LINKS.parent.mkdir(parents=True, exist_ok=True)
    LINKS.write_text(json.dumps(cache, indent=1, sort_keys=True))


def apply_links(cache):
    """Write resolved links into the data files and the single-file dashboard."""
    touched = 0
    records = json.loads(MAPPED.read_text())
    for rec in records:
        got = cache.get(rec["id"].lower())
        if got and needs_pdf(rec):
            rec["pdf"] = got
            touched += 1
    MAPPED.write_text(json.dumps(records, indent=1) + "\n")
    print(f"[apply] data/mapped-decisions.json — {touched} link(s) replaced")

    html = HTML.read_text(encoding="utf-8")
    m = re.search(r"^const DATA = (\{.*\});$", html, re.M)
    if not m:
        print("[apply] could not find the embedded DATA in visualization/map.html",
              file=sys.stderr)
        return touched
    data = json.loads(m.group(1))
    inline = 0
    for rec in data.get("mapped", []):
        got = cache.get(rec["id"].lower())
        if got and needs_pdf(rec):
            rec["pdf"] = got
            inline += 1
    blob = json.dumps(data, separators=(",", ":"))
    HTML.write_text(html[:m.start(1)] + blob + html[m.end(1):], encoding="utf-8")
    print(f"[apply] visualization/map.html — {inline} link(s) replaced "
          f"(the basic build now ships them)")
    return touched


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="count the work, fetch nothing")
    ap.add_argument("--apply", action="store_true",
                    help="write already-resolved links into the data and the dashboard")
    ap.add_argument("--limit", type=int, default=0, help="stop after N decisions")
    ap.add_argument("--only", default="", help="one B-number, for a spot check")
    ap.add_argument("--sleep", type=float, default=0.4,
                    help="pause after each request, per worker (default 0.4s)")
    ap.add_argument("--workers", type=int, default=4, help="parallel requests (default 4)")
    ap.add_argument("--retries", type=int, default=2, help="retries per page (default 2)")
    ap.add_argument("--refresh", action="store_true",
                    help="re-resolve decisions already in data/pdf-links.json")
    args = ap.parse_args()

    cache = analysis.link_cache()
    records = json.loads(MAPPED.read_text())
    # map.json's own entries are already direct PDFs — free answers for any
    # decision the two files share.
    seeded = 0
    for d in json.loads(MAPJSON.read_text())["decisions"]:
        u = d.get("pdf") or d.get("url") or ""
        if u.lower().endswith(".pdf") and d["id"].lower() not in cache:
            cache[d["id"].lower()] = u
            seeded += 1
    if seeded:
        save(cache)
        print(f"[seed] {seeded} direct link(s) taken from data/map.json")

    if args.only:
        records = [r for r in records if r["id"].lower() == args.only.lower()]
        if not records:
            sys.exit(f"no decision with id {args.only}")

    todo = [r for r in records if needs_pdf(r) and (page_for(r) or r.get("id"))
            and (args.refresh or r["id"].lower() not in cache)]
    have = sum(1 for r in records if not needs_pdf(r))
    print(f"[scan] {len(records)} decisions · {have} already direct · "
          f"{len(cache)} in the cache · {len(todo)} to fetch")

    if args.apply:
        apply_links(cache)
        return
    if args.dry_run or not todo:
        if not todo:
            print("[scan] nothing to fetch — run with --apply to write them in")
        return
    if args.limit:
        todo = todo[:args.limit]
    print(f"[fetch] {len(todo)} gao.gov page(s), {args.workers} at a time, "
          f"{args.sleep}s apart — Ctrl-C is safe, progress is saved as it goes")
    t0 = time.time()
    try:
        done = resolve_all(todo, cache, args.sleep, args.workers, args.retries)
    except KeyboardInterrupt:
        save(cache)
        sys.exit(f"\n[stop] interrupted — {len(cache)} link(s) saved to {LINKS}")
    print(f"[done] {done['hit']} found ({done['asset']} by asset url, "
          f"{done['scraped']} by reading the page), {done['miss']} not found, "
          f"{int(time.time() - t0)}s → {LINKS}")
    print("[next] python3 scripts/resolve_pdfs.py --apply")


if __name__ == "__main__":
    main()
