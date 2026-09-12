#!/usr/bin/env python3
"""Decision analysis: get the text of one GAO bid protest decision, then have an
OpenAI-compatible model summarize it into a fixed set of key elements.

Three text sources, tried in this order and reported back to the caller so the
dashboard can say where the words came from:

  1. ``corpus``  — the local vector store's own chunks (full build only; no
                   network, no PDF, and it is the same text Dynamic Search
                   searches, so a citation can be checked against a passage)
  2. ``pdf``     — the decision PDF fetched from gao.gov and extracted here
  3. ``supplied``— text the browser sent up (a PDF the user picked, or pasted)

Only the transport lives in llm.py; everything domain-shaped is here.
"""

import json
import math
import os
import re
import threading
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import llm

ROOT = Path(__file__).resolve().parent.parent
PDF_LINKS = ROOT / "data" / "pdf-links.json"

# A GAO decision runs ~5-40 pages. The cap is a backstop against a pathological
# consolidated record, not a normal-path truncation.
MAX_TEXT_CHARS = int(os.environ.get("ANALYZE_MAX_CHARS", "180000"))
MAX_PDF_BYTES = int(os.environ.get("ANALYZE_MAX_PDF_BYTES", str(40 * 1024 * 1024)))
# Fetching an arbitrary URL on the user's behalf is a wider door than this tool
# needs; the corpus only ever points at gao.gov.
ALLOWED_PDF_HOSTS = {h.strip().lower() for h in os.environ.get(
    "ANALYZE_PDF_HOSTS", "gao.gov,www.gao.gov").split(",") if h.strip()}
USER_AGENT = ("GAO-Bid-Protest-Nexus/1.0 (local research dashboard; "
              "+https://github.com/acqagent/GAO-Bid-Protest-Nexus)")

SECTIONS = [
    "Snapshot",
    "Procurement and posture",
    "Grounds raised",
    "GAO's ruling, ground by ground",
    "Why it came out that way",
    "Competitive prejudice",
    "Timeliness and jurisdiction",
    "Disposition and recommendation",
    "Authorities cited",
    "Practical takeaway",
    "Gaps and low confidence",
]

SYSTEM_PROMPT = """\
You are a federal procurement attorney summarizing a U.S. Government \
Accountability Office (GAO) bid protest decision for a contracting officer and \
a proposal team. You are precise, plain-spoken, and you never invent facts.

Absolute rules:
- Use ONLY the decision text supplied. It is the whole record you get.
- If the text does not state something, write "not stated in the decision". \
Never guess a party, a dollar figure, a date, a solicitation number or a citation.
- Quote sparingly and exactly, in quotation marks, and only where the wording \
carries the holding. Cite the page as (p. N) when page markers are present.
- Distinguish what GAO held from what a party argued. Attribute arguments.
- If the supplied text is truncated, partial, or looks like the wrong document, \
say so plainly under "Gaps and low confidence" instead of filling gaps.
"""

_FORMAT = """\
Write GitHub-flavored Markdown. Use exactly these level-2 headings, in this \
order, and nothing above the first one:

## Snapshot
A compact bullet list: B-number(s); decision date; protester; agency/activity; \
awardee or intervenor; solicitation number; what is being procured; dollar \
value; decision length. One bullet each, "not stated in the decision" where absent.

## Procurement and posture
Procurement authority (FAR part 15 negotiated, sealed bidding, FSS/GSA order, \
task or delivery order under FAR 16.505, 8(a), SBIR, commercial items, OTA, \
etc.), and the posture (pre-award, post-award, challenge to corrective action, \
reconsideration, protest of a cancellation, size/status referral). Say which \
words in the decision told you.

## Grounds raised
Number each ground the protester actually pressed, in the decision's own terms. \
Note any ground GAO treated as abandoned, withdrawn or untimely.

## GAO's ruling, ground by ground
One entry per ground from the section above, each in this shape:
**N. <ground>** — *sustained / denied / dismissed / academic / not reached* — \
one or two sentences on the reason GAO gave.

## Why it came out that way
Three to six bullets on the pivotal facts and the standard of review GAO \
applied (reasonableness, consistency with the solicitation's stated criteria, \
adequacy of documentation, the record before the agency). This is the part a \
reader would otherwise have to read the whole decision for.

## Competitive prejudice
Whether GAO reached prejudice, and what it concluded — a substantial chance of \
receiving the award but for the agency's action. "Not reached" is a valid answer.

## Timeliness and jurisdiction
Any 4 C.F.R. § 21.2 timeliness ruling, the pre-bid-opening rule for \
solicitation improprieties, jurisdictional limits (task order threshold, matters \
of contract administration, affirmative responsibility determinations, small \
business size/status), and any dismissal that rested on one.

## Disposition and recommendation
The bottom line, then GAO's recommendation verbatim in substance: reevaluate, \
amend the solicitation and re-solicit, terminate and re-award, document the \
record — and whether GAO recommended the protester recover its costs of filing \
and pursuing the protest, including attorneys' fees.

## Authorities cited
The decisions, statutes and regulations the holding leans on. B-number and \
short name only. Omit the heading's content and write "none identified" if the \
text cites nothing.

## Practical takeaway
Two to four bullets: what an agency should do differently, and what an offeror \
should take from this. Concrete, not generic.

## Gaps and low confidence
Anything you could not determine from the supplied text, any sign the text is \
truncated or is not the decision it claims to be, and any element above you are \
unsure of. Write "none" if the text was complete and unambiguous.
"""


def build_messages(text, meta=None):
    """The two-message chat payload for one decision."""
    meta = meta or {}
    lines = []
    for label, key in (("B-number", "id"), ("Ground (dashboard label)", "ground"),
                       ("Disposition (dashboard label)", "disposition"),
                       ("Overall outcome (dashboard label)", "overall_outcome"),
                       ("Procurement authority (dashboard label)", "procurement_authority"),
                       ("Protest posture (dashboard label)", "protest_posture"),
                       ("Source", "source_label")):
        if meta.get(key):
            lines.append(f"- {label}: {meta[key]}")
    header = ""
    if lines:
        header = ("Dashboard metadata for this decision. It is a coarse "
                  "classification made elsewhere, NOT part of the decision — "
                  "correct it in your answer if the text disagrees.\n"
                  + "\n".join(lines) + "\n\n")
    user = (header + _FORMAT
            + "\n\n---\nDECISION TEXT BEGINS\n---\n\n" + text
            + "\n\n---\nDECISION TEXT ENDS\n---\n")
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user}]


# ---------------------------------------------------------------- corpus text

def _base(part):
    return re.sub(r"\.\d+$", "", part.strip().upper())


def base_ids(file_no):
    """Base B-numbers out of a chunk ``file`` field, deduped and upper-cased.
    Mirrors searchlib.map_ids so this module never needs numpy."""
    out = []
    for part in (file_no or "").split(";"):
        b = _base(part)
        if b and b not in out:
            out.append(b)
    return out


class ChunkIndex:
    """Maps a dashboard B-number onto the vector store's chunk ``file`` keys.

    The two identifier spaces do not line up cleanly: a chunk's ``file`` can
    carry several consolidated B-numbers separated by ';', and the dashboard's
    id is sometimes the base number and sometimes a docket suffix. Match exact
    first, then on the base number, then on a substring.
    """

    def __init__(self, chunks):
        self.chunks = chunks
        self.exact, self.base = {}, {}
        for i, c in enumerate(chunks):
            f = c.get("file", "")
            for part in f.split(";"):
                p = part.strip().upper()
                if not p:
                    continue
                self.exact.setdefault(p, set()).add(f)
                self.base.setdefault(_base(p), set()).add(f)

    def files_for(self, decision_id):
        q = (decision_id or "").strip().upper()
        if not q:
            return []
        for table, key in ((self.exact, q), (self.base, q), (self.base, _base(q))):
            if key in table:
                return sorted(table[key])
        hits = {c["file"] for c in self.chunks if q in c.get("file", "").upper()}
        return sorted(hits)

    def text_for(self, decision_id):
        """Returns (text, info) — info carries files, pages, date, chunk count."""
        files = self.files_for(decision_id)
        if not files:
            return "", {"files": [], "chunks": 0}
        picked = [c for c in self.chunks if c.get("file") in set(files)]
        if not picked:
            return "", {"files": files, "chunks": 0}
        digests = [c for c in picked if str(c.get("id", "")).endswith("__digest")]
        body = [c for c in picked if not str(c.get("id", "")).endswith("__digest")]
        body.sort(key=lambda c: (c.get("file", ""),
                                 (c.get("pages") or [0, 0])[0],
                                 (c.get("pages") or [0, 0])[1]))
        parts = []
        if digests:
            parts.append("[GAO DIGEST]\n" + "\n".join(d.get("text", "") for d in digests))
        out, last_file, prev = [], None, ""
        for c in body:
            f = c.get("file")
            if f != last_file:
                last_file, prev = f, ""
                if len(files) > 1:
                    out.append(f"[DECISION {f}]")
            piece = _join_overlap(prev, c.get("text", ""))
            prev = c.get("text", "")
            if not piece.strip():
                continue
            pg = (c.get("pages") or [None])[0]
            out.append((f"[p. {pg}]\n" if pg else "") + piece)
        text = "\n".join(parts + out).strip()
        pages = [p for c in body for p in (c.get("pages") or [])]
        info = {"files": files, "chunks": len(picked),
                "pages": [min(pages), max(pages)] if pages else None,
                "date": next((c.get("date") for c in picked if c.get("date")), None)}
        return text, info


def _join_overlap(prev, nxt, max_overlap_words=90):
    """Drop the window overlap the chunker left between consecutive chunks."""
    if not prev or not nxt:
        return nxt
    pw, nw = prev.split(), nxt.split()
    for n in range(min(max_overlap_words, len(pw), len(nw)), 8, -1):
        if [w.lower() for w in pw[-n:]] == [w.lower() for w in nw[:n]]:
            return " ".join(nw[n:])
    return nxt


def clamp(text, limit=None):
    """Keep the head and the tail — a decision's holding lives at both ends."""
    limit = limit or MAX_TEXT_CHARS
    if len(text) <= limit:
        return text, False
    head = int(limit * 0.62)
    tail = limit - head
    return (text[:head] + "\n\n[... " + str(len(text) - limit)
            + " characters omitted from the middle of this decision ...]\n\n"
            + text[-tail:]), True


# ------------------------------------------------------------------- pdf side

def _host_ok(url):
    host = (urllib.parse.urlparse(url).hostname or "").lower()
    return host in ALLOWED_PDF_HOSTS or not ALLOWED_PDF_HOSTS


def _get(url, timeout=45, accept="*/*"):
    if urllib.parse.urlparse(url).scheme not in ("http", "https"):
        raise ValueError("only http(s) URLs are fetched")
    if not _host_ok(url):
        raise ValueError(
            f"host not allowed: {urllib.parse.urlparse(url).hostname} "
            f"(set ANALYZE_PDF_HOSTS to widen)")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT,
                                               "Accept": accept})
    return urllib.request.urlopen(req, timeout=timeout)


_link_lock = threading.Lock()
_link_cache = None


def link_cache():
    global _link_cache
    if _link_cache is None:
        try:
            _link_cache = json.loads(PDF_LINKS.read_text())
        except Exception:
            _link_cache = {}
    return _link_cache


def _remember(decision_id, pdf_url):
    with _link_lock:
        cache = link_cache()
        cache[decision_id.lower()] = pdf_url
        try:
            PDF_LINKS.parent.mkdir(parents=True, exist_ok=True)
            PDF_LINKS.write_text(json.dumps(cache, indent=1, sort_keys=True))
        except OSError:
            pass



# gao.gov files a decision's PDF under one of two spellings — /assets/b-417327.pdf
# or /assets/417327.pdf — and which one is not derivable from the record. Of the
# links the corpus already has, 85% are one of these two, and the B-number says
# which to try first: the bare form dominates from B-420000 up, the b- form below
# B-300000. Trying them is one HEAD request, against a full HTML page fetch for
# the landing page, so it is both faster and lighter on gao.gov — and nothing is
# recorded unless the server confirms the file is really there.
ASSET_BASE = "https://www.gao.gov/assets/"


def candidate_pdf_urls(decision_id):
    """The URLs worth testing for a decision, likeliest first. The third is
    GAO's older nested layout, /assets/330/325381.pdf, whose folder is
    ceil(number/10000)*10 — verified against all 47 such links in the corpus."""
    did = (decision_id or "").strip().lower()
    if not did.startswith("b-"):
        return []
    bare, prefixed = ASSET_BASE + did[2:] + ".pdf", ASSET_BASE + did + ".pdf"
    m = re.match(r"b-(\d+)", did)
    n = int(m.group(1)) if m else 0
    out = [bare, prefixed] if n >= 410000 else [prefixed, bare]
    if re.match(r"b-\d+$", did):
        out.append(f"{ASSET_BASE}{math.ceil(n / 10000) * 10}/{n}.pdf")
    return out


def is_pdf_at(url, timeout=20):
    """True only if that URL really serves a PDF. HEAD, or a 4-byte GET for
    servers that will not answer HEAD."""
    if urllib.parse.urlparse(url).scheme not in ("http", "https") or not _host_ok(url):
        return False
    try:
        req = urllib.request.Request(url, method="HEAD",
                                     headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            if r.status == 200 and "pdf" in r.headers.get("Content-Type", "").lower():
                return True
            if r.status == 200 and not r.headers.get("Content-Type"):
                pass          # no type header — fall through to the byte check
            elif r.status == 200:
                return False  # 200 with a non-PDF type is gao.gov's soft 404
            else:
                return False
    except urllib.error.HTTPError as e:
        if e.code not in (405, 501):
            return False
    except Exception:
        return False
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": USER_AGENT, "Range": "bytes=0-3"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read(4).startswith(b"%PDF")
    except Exception:
        return False


_ASSET_RE = re.compile(r'href="([^"]*?/assets/[^"]*?\.pdf)"', re.I)


def scrape_pdf_url(page_url):
    """Pull the asset PDF link out of a gao.gov /products/ page."""
    with _get(page_url, accept="text/html") as resp:
        html = resp.read(4_000_000).decode("utf-8", "replace")
    seen, out = set(), []
    for m in _ASSET_RE.finditer(html):
        u = urllib.parse.urljoin(page_url, m.group(1).replace("&amp;", "&"))
        if u not in seen:
            seen.add(u)
            out.append(u)
    if not out:
        return None
    # Prefer the asset whose filename shares the most with the page's own slug.
    slug = urllib.parse.unquote(page_url.rstrip("/").rsplit("/", 1)[-1]).lower()
    keys = {k for k in re.split(r"[^a-z0-9.]+", slug) if k}
    out.sort(key=lambda u: -sum(
        1 for k in keys if k in urllib.parse.unquote(u).lower()))
    return out[0]


def resolve_pdf_url(decision_id, known_pdf=None, page_url=None, use_cache=True,
                    try_candidates=True):
    """Best real .pdf URL for a decision. Returns (url, how)."""
    did = (decision_id or "").lower()
    if known_pdf and known_pdf.lower().endswith(".pdf"):
        return known_pdf, "already a pdf"
    if use_cache and did and did in link_cache():
        return link_cache()[did], "cache"
    if try_candidates:
        for cand in candidate_pdf_urls(did):
            if is_pdf_at(cand):
                if did:
                    _remember(did, cand)
                return cand, "asset url"
    page = page_url or known_pdf
    if not page:
        return None, "no gao.gov page known, and neither asset url exists"
    found = scrape_pdf_url(page)
    if found and did:
        _remember(did, found)
    return found, ("scraped " + page if found else "no .pdf on " + page)


def fetch_pdf(url, timeout=90):
    with _get(url, timeout=timeout, accept="application/pdf,*/*") as resp:
        data = resp.read(MAX_PDF_BYTES + 1)
    if len(data) > MAX_PDF_BYTES:
        raise ValueError(f"PDF larger than the {MAX_PDF_BYTES} byte cap")
    if not data.startswith(b"%PDF"):
        raise ValueError("that URL did not return a PDF "
                         "(a gao.gov /products/ page is HTML, not the PDF itself)")
    return data


def pdf_extractor():
    """Name of the available PDF text extractor, or None."""
    for name in ("pypdf", "PyPDF2", "pdfminer"):
        try:
            __import__(name)
            return name
        except ImportError:
            continue
    return None


def pdf_to_text(data):
    """Extract text with whichever extractor is installed. Page-marked."""
    name = pdf_extractor()
    if name is None:
        raise RuntimeError(
            "no PDF text extractor installed — run `pip install pypdf` "
            "(or analyze from the local corpus instead)")
    import io
    buf = io.BytesIO(data)
    if name in ("pypdf", "PyPDF2"):
        mod = __import__(name)
        reader = mod.PdfReader(buf)
        pages = [(i + 1, (p.extract_text() or "")) for i, p in enumerate(reader.pages)]
    else:
        from pdfminer.high_level import extract_text
        whole = extract_text(buf) or ""
        pages = [(i + 1, t) for i, t in enumerate(whole.split("\f"))]
    out = []
    for n, t in pages:
        t = re.sub(r"[ \t]+\n", "\n", t).strip()
        if t:
            out.append(f"[p. {n}]\n{t}")
    text = "\n\n".join(out)
    if len(text.strip()) < 200:
        raise RuntimeError(
            "almost no text came out of that PDF — it is probably a scan. "
            "Run it through OCR first, or paste the text in.")
    return text


# ----------------------------------------------------------------- the runner

def resolve_text(*, decision_id=None, supplied_text=None, pdf_url=None,
                 chunk_index=None, known_pdf=None, page_url=None,
                 prefer="corpus"):
    """Find the decision text. Returns (text, source_info)."""
    tried = []

    def from_corpus():
        if chunk_index is None or not decision_id:
            tried.append("corpus: no local index" if chunk_index is None
                         else "corpus: no decision id")
            return None
        text, info = chunk_index.text_for(decision_id)
        if not text:
            tried.append(f"corpus: {decision_id} not in the index")
            return None
        return text, {"source": "corpus", "label": "the local decision corpus",
                      "detail": f"{info['chunks']} chunk(s) from "
                                f"{', '.join(info['files'])}", **info}

    def from_pdf():
        url = pdf_url
        how = "supplied url"
        if not url:
            url, how = resolve_pdf_url(decision_id, known_pdf=known_pdf,
                                       page_url=page_url)
        if not url:
            tried.append(f"pdf: {how}")
            return None
        data = fetch_pdf(url)
        return pdf_to_text(data), {"source": "pdf", "label": "the decision PDF",
                                   "detail": f"{url} ({how}, {len(data) // 1024} KB)",
                                   "pdf": url}

    def from_supplied():
        if not supplied_text or len(supplied_text.strip()) < 200:
            tried.append("supplied: nothing usable sent")
            return None
        return supplied_text, {"source": "supplied", "label": "text you supplied",
                               "detail": f"{len(supplied_text)} characters"}

    order = {"corpus": (from_corpus, from_supplied, from_pdf),
             "pdf": (from_pdf, from_corpus, from_supplied),
             "supplied": (from_supplied, from_corpus, from_pdf)}[prefer]
    errors = []
    for step in order:
        try:
            got = step()
        except Exception as e:                       # one source failing is normal
            errors.append(f"{step.__name__[5:]}: {e}")
            continue
        if got:
            text, info = got
            text, truncated = clamp(text)
            info["truncated"] = truncated
            info["chars"] = len(text)
            info["tried"] = tried + errors
            return text, info
    raise RuntimeError("could not get the decision text — " +
                       "; ".join(tried + errors or ["no source available"]))


def analyze(text, meta=None, *, base_url=None, api_key=None, model=None,
            stream=False, max_tokens=2600):
    messages = build_messages(text, meta)
    if stream:
        return llm.stream(messages, base_url=base_url, api_key=api_key,
                          model=model, max_tokens=max_tokens)
    return llm.complete(messages, base_url=base_url, api_key=api_key,
                        model=model, max_tokens=max_tokens)


# ------------------------------------------------------------------ compare

COMPARE_SECTIONS = [
    "The set",
    "Side by side",
    "What these decisions share",
    "Where they diverge, and why",
    "The line GAO is drawing",
    "What this set does not tell you",
    "Practical takeaway",
]

COMPARE_SYSTEM = """\
You are a federal procurement attorney comparing several U.S. Government \
Accountability Office (GAO) bid protest decisions for a contracting officer and \
a proposal team. You reason across decisions: what they share, where they split, \
and what rule explains the split.

Absolute rules:
- Use ONLY the material supplied for each decision. Never import outside \
knowledge of a B-number, a company, an agency or a holding.
- Every comparative claim must be traceable to at least one decision in the set. \
Name the B-number inline, like (B-417327).
- A difference in outcome is not automatically a difference in rule. Say when \
two decisions differ on their facts rather than on the law.
- This is a small, self-selected set, not a sample. Never state a rate, a trend \
or a likelihood from it, and never generalize to the corpus.
- If the material for a decision is thin or truncated, say so rather than \
inferring around it.
"""

_COMPARE_FORMAT = """\
Write GitHub-flavored Markdown. Use exactly these level-2 headings, in this \
order, and nothing above the first one:

## The set
One line per decision: **B-number** — protester v. agency, date, one clause on \
what it decided. Nothing else.

## Side by side
A Markdown table, one row per decision, with these columns exactly:
B-number | Date | Authority | Posture | Grounds pressed | GAO's disposition | Remedy
Keep cells short. Use "not stated" where the material does not say.

## What these decisions share
Three to six bullets: the facts, procurement structures, evaluation practices or \
legal standards that genuinely recur across the set. Cite B-numbers.

## Where they diverge, and why
The heart of it. For each real split, state the split, then the reason GAO gave \
that explains it — a difference in the solicitation's terms, in what the record \
documented, in timeliness, in prejudice, or in the standard applied. Cite \
B-numbers on both sides of every split.

## The line GAO is drawing
Two to five bullets stating, as a rule an agency could act on, what puts a \
protest on the sustained side of this set versus the denied side. If the set is \
too mixed to support a rule, say that instead of inventing one.

## What this set does not tell you
The limits: decisions that are not comparable, material that was thin or \
truncated, questions the set leaves open, and an explicit reminder that these \
decisions were hand-picked and carry no statistical weight.

## Practical takeaway
Two to four bullets, concrete: what an agency should do differently and what an \
offeror should take from the set as a whole.
"""


def condense(text, budget):
    """Squeeze one decision to fit a shared context budget, head and tail."""
    text = (text or "").strip()
    if len(text) <= budget:
        return text, False
    head = int(budget * 0.7)
    return (text[:head] + "\n\n[... middle omitted for the comparison ...]\n\n"
            + text[-(budget - head):]), True


def build_compare_messages(items, question=None):
    """items: [{id, meta, markdown?, text?}] — a per-decision brief beats raw text.

    Two-stage by design: comparing eight full decisions does not fit a useful
    context window, and a comparison built from per-decision analyses is both
    cheaper and sharper than one built from eight truncated transcripts.
    """
    blocks = []
    for it in items:
        meta = it.get("meta") or {}
        label = it.get("id") or meta.get("id") or "(unidentified)"
        tags = [f"{k.replace('_', ' ')}: {meta[k]}" for k in
                ("ground", "disposition", "overall_outcome",
                 "procurement_authority", "protest_posture") if meta.get(k)]
        kind = "PRIOR ANALYSIS" if it.get("markdown") else "DECISION TEXT"
        body = it.get("markdown") or it.get("text") or "(no material supplied)"
        blocks.append(f"===== DECISION {label} =====\n"
                      + (f"Dashboard labels: {'; '.join(tags)}\n" if tags else "")
                      + f"--- {kind} ---\n{body}\n")
    ask = (f"\nThe reader also asked: {question.strip()}\nAnswer it inside the "
           "sections below, never as an extra heading.\n" if question else "")
    user = (f"Compare these {len(items)} GAO bid protest decisions.\n"
            "Dashboard labels are a coarse classification made elsewhere, not part "
            "of any decision — correct them if the material disagrees.\n"
            + ask + "\n" + _COMPARE_FORMAT
            + "\n\n---\nMATERIAL BEGINS\n---\n\n" + "\n".join(blocks)
            + "\n---\nMATERIAL ENDS\n---\n")
    return [{"role": "system", "content": COMPARE_SYSTEM},
            {"role": "user", "content": user}]


def compare(items, *, question=None, base_url=None, api_key=None, model=None,
            stream=False, max_tokens=3200):
    messages = build_compare_messages(items, question)
    if stream:
        return llm.stream(messages, base_url=base_url, api_key=api_key,
                          model=model, max_tokens=max_tokens)
    return llm.complete(messages, base_url=base_url, api_key=api_key,
                        model=model, max_tokens=max_tokens)
