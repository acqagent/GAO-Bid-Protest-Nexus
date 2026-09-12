#!/usr/bin/env python3
"""Find the real PDF link for every GAO decision in the corpus.

Standalone: one file, standard library only, no API key, no model, no pip
install. It just talks HTTP to gao.gov. Copy it anywhere and run it.

WHY IT EXISTS
    Most decisions in the corpus carry a gao.gov ``/products/`` landing page in
    their ``pdf`` field. That is an HTML page, not the document, which is why a
    "PDF" link can drop you on a web page. This walks the whole corpus and
    records the actual ``/assets/....pdf`` for each decision.

HOW IT RESOLVES  (cheapest first; nothing is ever guessed)
    1. cache          a previous run already found it
    2. docket         GAO publishes ONE document per consolidated docket and the
                      filename names every B-number it covers
                      (b-414706,b-414380.2.pdf), so siblings come free
    3. asset url      gao.gov files a decision as /assets/b-417327.pdf or
                      /assets/417327.pdf and the record does not say which; both
                      are tested with a HEAD. 82% of known links are one of the
                      two, and the B-number predicts the order. One request.
    4. landing page   fetch the /products/ page and read the link out of it
    A link is recorded only when gao.gov confirms a PDF is really there.

USAGE
    python3 resolve_pdfs.py --dry-run        # count the work, fetch nothing
    python3 resolve_pdfs.py                  # resolve, saving as it goes
    python3 resolve_pdfs.py --apply          # write them into the data + dashboard
    python3 resolve_pdfs.py --recheck        # also re-verify links already held

    --input   mapped-decisions.json, map.json, or map.html (found automatically
              next to the script if the repo layout is intact)
    --out     link map, {"b-417327": "https://..."}   default data/pdf-links.json
    --csv     per-decision audit trail: id, pdf, method, status, page
    --workers 4 by default. Be kind; gao.gov is a public service.

Ctrl-C is safe: progress is written as it goes and a rerun picks up where it
stopped. Nothing here needs the rest of the project.
"""

import argparse
import csv
from collections import Counter
import json
import math
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
# Overridable so the resolver can be exercised against a stand-in server;
# leave them alone for real runs.
ASSET_BASE = os.environ.get("GAO_ASSET_BASE", "https://www.gao.gov/assets/")
PRODUCT_BASE = os.environ.get("GAO_PRODUCT_BASE", "https://www.gao.gov/products/")
USER_AGENT = ("GAO-Bid-Protest-Nexus/1.0 (link resolver; "
              "+https://github.com/acqagent/GAO-Bid-Protest-Nexus)")
BNUM = re.compile(r"b-\d+(?:\.\d+)?", re.I)
ASSET_HREF = re.compile(r'href="([^"]*?/assets/[^"]*?\.pdf)"', re.I)

# The URL-shape logic below is mirrored in scripts/analysis.py, which the server
# uses for one-at-a-time lookups. Change one, change the other.

_lock = threading.Lock()


# ----------------------------------------------------------------- input

def load_decisions(path=None):
    """[{id, url, pdf}] out of mapped-decisions.json, map.json, or map.html."""
    if path is None:
        for guess in ("data/mapped-decisions.json", "data/map.json",
                      "visualization/map.html"):
            p = os.path.join(ROOT, guess)
            if os.path.exists(p):
                path = p
                break
    if not path or not os.path.exists(path):
        sys.exit("no input found — pass --input path/to/mapped-decisions.json "
                 "(or map.json, or map.html)")
    raw = open(path, encoding="utf-8").read()
    if path.endswith(".html"):
        m = re.search(r"^const DATA = (\{.*\});$", raw, re.M)
        if not m:
            sys.exit(f"no embedded decision data in {path}")
        data = json.loads(m.group(1))
        records = data.get("mapped") or data.get("decisions") or []
    else:
        data = json.loads(raw)
        records = data if isinstance(data, list) else (
            data.get("mapped") or data.get("decisions") or [])
    out, seen = [], set()
    for r in records:
        i = str(r.get("id", "")).strip().lower()
        if not i or i in seen:
            continue
        seen.add(i)
        out.append({"id": i, "url": r.get("url"), "pdf": r.get("pdf")})
    print(f"[input] {len(out)} decisions from {os.path.relpath(path, ROOT)}")
    return out


def import_csv(path):
    """Links from a classified CSV: b_number,url,classification.

    A row's b_number may carry several pipe-joined B-numbers — one consolidated
    document covering all of them — so each member gets the link. Only rows that
    actually name a .pdf are taken as resolved; a row with no URL, or one
    pointing at a /products/ page, leaves that decision unresolved so --apply
    falls back to whatever link it already has and flags it as unconfirmed.
    """
    links, skipped = {}, {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        rdr = csv.DictReader(f)
        cols = {c.lower().strip(): c for c in (rdr.fieldnames or [])}
        idc = next((cols[c] for c in ("b_number", "bnumber", "id") if c in cols), None)
        urlc = next((cols[c] for c in ("url", "pdf", "link") if c in cols), None)
        clsc = cols.get("classification")
        if not idc or not urlc:
            sys.exit(f"{path}: need a b_number column and a url column, "
                     f"found {rdr.fieldnames}")
        for row in rdr:
            url = (row.get(urlc) or "").strip()
            cls = (row.get(clsc) or "").strip().lower() if clsc else ""
            for part in (row.get(idc) or "").split("|"):
                did = part.strip().lower()
                if not did:
                    continue
                if is_pdf_url(url) and cls in ("", "pdf"):
                    links[did] = url
                else:
                    skipped[did] = cls or "no pdf in the row"
    print(f"[import] {os.path.basename(path)}: {len(links)} link(s), "
          f"{len(skipped)} decision(s) the file has no PDF for")
    if skipped:
        for why, n in sorted(Counter(skipped.values()).items(), key=lambda kv: -kv[1]):
            print(f"           {n:5} classified {why}")
    return links, skipped


def is_pdf_url(u):
    return bool(u) and u.lower().split("?")[0].endswith(".pdf")


def page_for(rec):
    for key in ("url", "pdf"):
        v = rec.get(key)
        if v and "/products/" in v:
            return v
    return rec.get("url") or rec.get("pdf") or (PRODUCT_BASE + rec["id"])


# ------------------------------------------------------------- http bits

def _open(url, method="GET", timeout=40, extra=None):
    if urllib.parse.urlparse(url).scheme not in ("http", "https"):
        raise ValueError("only http(s) URLs are fetched")
    headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "identity"}
    if extra:
        headers.update(extra)
    req = urllib.request.Request(url, headers=headers, method=method)
    return urllib.request.urlopen(req, timeout=timeout)


def is_pdf_at(url, timeout=25):
    """True only if that URL really serves a PDF."""
    try:
        with _open(url, "HEAD", timeout) as r:
            ctype = (r.headers.get("Content-Type") or "").lower()
            if r.status != 200:
                return False
            if "pdf" in ctype:
                return True
            if ctype:
                return False        # 200 with an HTML type is gao.gov's soft 404
    except urllib.error.HTTPError as e:
        if e.code not in (405, 501):
            return False
    except Exception:
        return False
    try:                            # servers that will not answer HEAD
        with _open(url, "GET", timeout, {"Range": "bytes=0-3"}) as r:
            return r.read(4).startswith(b"%PDF")
    except Exception:
        return False


def _subdir(did):
    """GAO's older layout nests by B-number band: /assets/330/325381.pdf. The
    folder is ceil(number/10000)*10 — verified against all 47 such links in the
    corpus. Missing this form is why a resolver pass can report no-pdf for a
    decision whose PDF is really there."""
    m = re.match(r"b-(\d+)$", did)
    if not m:
        return None
    n = int(m.group(1))
    return f"{ASSET_BASE}{math.ceil(n / 10000) * 10}/{n}.pdf"


def candidate_urls(decision_id):
    """The spellings worth testing, likeliest first."""
    did = decision_id.strip().lower()
    if not did.startswith("b-"):
        return []
    bare, prefixed = ASSET_BASE + did[2:] + ".pdf", ASSET_BASE + did + ".pdf"
    m = re.match(r"b-(\d+)", did)
    out = ([bare, prefixed] if m and int(m.group(1)) >= 410000
           else [prefixed, bare])
    nested = _subdir(did)
    if nested:
        out.append(nested)
    return out


def scrape_page(page_url, timeout=45):
    """The asset PDF link out of a gao.gov /products/ page."""
    with _open(page_url, "GET", timeout, {"Accept": "text/html"}) as r:
        html = r.read(4_000_000).decode("utf-8", "replace")
    seen, hits = set(), []
    for m in ASSET_HREF.finditer(html):
        u = urllib.parse.urljoin(page_url, m.group(1).replace("&amp;", "&"))
        if u not in seen:
            seen.add(u)
            hits.append(u)
    if not hits:
        return None
    slug = urllib.parse.unquote(page_url.rstrip("/").rsplit("/", 1)[-1]).lower()
    keys = {k for k in re.split(r"[^a-z0-9.]+", slug) if k}
    hits.sort(key=lambda u: -sum(
        1 for k in keys if k in urllib.parse.unquote(u).lower()))
    return hits[0]


# ------------------------------------------------------- free inferences

def infer_dockets(records, links, seed_records=True):
    """One document per consolidated docket: every B-number in a filename we
    already know points at that same PDF. Costs nothing, so run it before the
    fetch and again after — each consolidated name a run finds unlocks more.

    seed_records=False keeps it to links that actually resolved, so an imported
    set stays the only source of confirmed links and a legacy URL already in the
    data cannot launder itself into that set through a sibling."""
    known = dict(links)
    if seed_records:
        for rec in records:
            if is_pdf_url(rec.get("pdf")):
                known.setdefault(rec["id"], rec["pdf"])
    added = 0
    for url in list(known.values()):
        tail = urllib.parse.unquote(url.rsplit("/", 1)[-1]).lower()[:-4]
        members = BNUM.findall(tail)
        if len(members) < 2:
            continue
        for m in members:
            if m not in known:
                links[m] = known[m] = url
                added += 1
    return added


# ------------------------------------------------------------- the work

def resolve_one(rec, recheck):
    """(url, method, status) for one decision. Never guesses."""
    did = rec["id"]
    if is_pdf_url(rec.get("pdf")) and not recheck:
        return rec["pdf"], "already direct", "ok"
    if is_pdf_url(rec.get("pdf")) and is_pdf_at(rec["pdf"]):
        return rec["pdf"], "already direct", "verified"
    for cand in candidate_urls(did):
        if is_pdf_at(cand):
            return cand, "asset url", "verified"
    page = page_for(rec)
    try:
        found = scrape_page(page)
    except Exception as e:
        return None, "landing page", f"{type(e).__name__}: {e}"
    if found:
        return found, "landing page", "scraped"
    return None, "landing page", "no pdf linked on that page"


def run(records, links, rows, args):
    todo = [r for r in records
            if args.recheck or (r["id"] not in links and not is_pdf_url(r.get("pdf")))]
    if args.limit:
        todo = todo[:args.limit]
    total = len(todo)
    if not total:
        print("[fetch] nothing left to fetch")
        return
    print(f"[fetch] {total} decision(s), {args.workers} at a time, "
          f"{args.sleep}s apart. Ctrl-C is safe — progress is saved as it goes.")
    tally = {"n": 0, "hit": 0, "miss": 0}
    by_method = {}
    t0 = time.time()

    def one(rec):
        url = method = status = None
        for attempt in range(args.retries + 1):
            try:
                url, method, status = resolve_one(rec, args.recheck)
                break
            except Exception as e:
                status = f"{type(e).__name__}: {e}"
                if attempt < args.retries:
                    time.sleep(args.sleep * (2 ** attempt) + 0.5)
        time.sleep(args.sleep)
        with _lock:
            tally["n"] += 1
            if url:
                links[rec["id"]] = url
                tally["hit"] += 1
                by_method[method] = by_method.get(method, 0) + 1
            else:
                tally["miss"] += 1
                if tally["miss"] <= 12:
                    print(f"    no pdf for {rec['id'].upper()}: {status}", flush=True)
            rows.append({"id": rec["id"].upper(), "pdf": url or "",
                         "method": method or "", "status": status or "",
                         "page": page_for(rec)})
            if tally["n"] % 50 == 0 or tally["n"] == total:
                done = tally["n"]
                rate = done / max(time.time() - t0, 1e-9)
                left = int((total - done) / rate) if rate else 0
                print(f"  {done}/{total}  found {tally['hit']}  missed {tally['miss']}"
                      f"  ~{left // 60}m{left % 60:02d}s left", flush=True)
                save(links, rows, args)

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(one, todo))
    except KeyboardInterrupt:
        save(links, rows, args)
        sys.exit(f"\n[stop] interrupted — {len(links)} link(s) saved to {args.out}. "
                 f"Rerun to continue.")
    print(f"[done] {tally['hit']} found, {tally['miss']} not found, "
          f"{int(time.time() - t0)}s")
    for m, c in sorted(by_method.items(), key=lambda kv: -kv[1]):
        print(f"         {c:5} by {m}")


# ------------------------------------------------------------- output

def save(links, rows, args):
    with open(args.out, "w") as f:
        json.dump(links, f, indent=1, sort_keys=True)
    if args.csv and rows:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["id", "pdf", "method", "status", "page"])
            w.writeheader()
            w.writerows(sorted(rows, key=lambda r: r["id"]))


def _apply_to(records, links, strict):
    """pdf <- the resolved link. Outside the resolved set, keep whatever link the
    record already has but mark it pdfok:0, so the dashboard can label it as a
    link that was never confirmed to be a PDF rather than promising one."""
    changed = flagged = 0
    for rec in records:
        did = rec["id"].lower()
        got = links.get(did)
        if got:
            if rec.get("pdf") != got:
                rec["pdf"] = got
                changed += 1
            rec.pop("pdfok", None)
        elif strict:
            if rec.get("pdfok") != 0:
                rec["pdfok"] = 0
                flagged += 1
        elif not is_pdf_url(rec.get("pdf")):
            rec["pdfok"] = 0
    return changed, flagged


def apply_links(links, strict=False):
    """Write the links into the corpus data and into the single-file dashboard.

    strict: treat `links` as the whole truth — every decision outside it is
    flagged unconfirmed. Use it after importing a complete resolved set.
    """
    touched = 0
    mapped = os.path.join(ROOT, "data", "mapped-decisions.json")
    if os.path.exists(mapped):
        records = json.loads(open(mapped).read())
        touched, flagged = _apply_to(records, links, strict)
        open(mapped, "w").write(json.dumps(records, indent=1) + "\n")
        print(f"[apply] data/mapped-decisions.json — {touched} link(s) written, "
              f"{flagged} flagged unconfirmed")

    html_path = os.path.join(ROOT, "visualization", "map.html")
    if not os.path.exists(html_path):
        return touched
    html = open(html_path, encoding="utf-8").read()
    m = re.search(r"^const DATA = (\{.*\});$", html, re.M)
    if not m:
        print("[apply] could not find the embedded DATA in visualization/map.html",
              file=sys.stderr)
        return touched
    data = json.loads(m.group(1))
    inline, iflag = _apply_to(data.get("mapped", []), links, strict)
    open(html_path, "w", encoding="utf-8").write(
        html[:m.start(1)] + json.dumps(data, separators=(",", ":")) + html[m.end(1):])
    print(f"[apply] visualization/map.html — {inline} link(s) written, "
          f"{iflag} flagged (the basic build ships them now)")
    return touched


def report(records, links):
    have = sum(1 for r in records if is_pdf_url(r.get("pdf")) or r["id"] in links)
    n = len(records)
    print(f"\n[report] {have}/{n} decisions have a direct PDF "
          f"({100 * have // max(n, 1)}%) · {n - have} still landing-page only")


# ------------------------------------------------------------------ cli

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", help="mapped-decisions.json, map.json or map.html")
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "pdf-links.json"),
                    help="link map to write (default data/pdf-links.json)")
    ap.add_argument("--csv", default="", help="also write a per-decision audit CSV")
    ap.add_argument("--dry-run", action="store_true",
                    help="do the free passes and report; fetch nothing")
    ap.add_argument("--apply", action="store_true",
                    help="write resolved links into the data and the dashboard")
    ap.add_argument("--import", dest="import_csv", default="",
                    help="load links from a classified CSV (b_number,url,"
                         "classification) instead of fetching; combine with --apply")
    ap.add_argument("--recheck", action="store_true",
                    help="re-verify links already held instead of trusting them")
    ap.add_argument("--only", default="", help="one B-number, for a spot check")
    ap.add_argument("--limit", type=int, default=0, help="stop after N decisions")
    ap.add_argument("--workers", type=int, default=4, help="parallel requests (4)")
    ap.add_argument("--sleep", type=float, default=0.4,
                    help="pause after each request, per worker (0.4s)")
    ap.add_argument("--retries", type=int, default=2, help="retries per decision (2)")
    args = ap.parse_args()

    records = load_decisions(args.input)
    if args.only:
        records = [r for r in records if r["id"] == args.only.strip().lower()]
        if not records:
            sys.exit(f"no decision with id {args.only}")

    links = {}
    if os.path.exists(args.out):
        try:
            links = {k.lower(): v for k, v in json.load(open(args.out)).items()}
            print(f"[cache] {len(links)} link(s) from a previous run in {args.out}")
        except Exception:
            pass

    strict = False
    if args.import_csv:
        imported, _ = import_csv(args.import_csv)
        links.update(imported)
        strict = True          # the file is the whole truth about what resolved

    seeded = infer_dockets(records, links, seed_records=not strict)
    if seeded:
        print(f"[docket] {seeded} link(s) inferred from consolidated filenames")

    direct = sum(1 for r in records if is_pdf_url(r.get("pdf")))
    pending = sum(1 for r in records
                  if not is_pdf_url(r.get("pdf")) and r["id"] not in links)
    print(f"[scan] {len(records)} decisions · {direct} already direct in the data "
          f"· {len(links)} known links · {pending} to fetch")

    rows = []
    if not args.dry_run and not args.import_csv:
        run(records, links, rows, args)
        more = infer_dockets(records, links, seed_records=not strict)
        if more:
            print(f"[docket] {more} more inferred from filenames this run turned up")
        save(links, rows, args)
        print(f"[save] {len(links)} link(s) → {args.out}"
              + (f" · audit → {args.csv}" if args.csv else ""))

    if args.apply:
        apply_links(links, strict=strict)
    report(records, links)
    if not args.apply and not args.dry_run:
        print("[next] python3 scripts/resolve_pdfs.py --apply")


if __name__ == "__main__":
    main()
