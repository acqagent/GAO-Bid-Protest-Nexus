#!/usr/bin/env python3
# Copyright 2026 Tommy Kim
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Produce the static build of the dashboard, for hosting on any web server.

The dashboard carries one switch, STATIC_BUILD. Flipping it drops the two tabs
that need a backend — Dynamic Search, which needs the vector index behind
scripts/serve.py, and Analysis, which needs a model endpoint — and removes their
markup from the page, so a stale deep link cannot reach them either. Map, Table,
Ground detail and License remain, and they need nothing but a file server.

    python3 scripts/build_static.py
    # -> visualization/map-static.html, ready to upload

Rebuild it whenever visualization/map.html changes; this script is a one-line
transform, not a fork, so the two never drift.
"""

import argparse
import gzip
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "visualization", "map.html")
OUT = os.path.join(ROOT, "visualization", "map-static.html")
FLAG = re.compile(r"^const STATIC_BUILD = (true|false);$", re.M)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=SRC, help="the full dashboard to build from")
    ap.add_argument("--out", default=OUT, help="where to write the static build")
    args = ap.parse_args()

    html = open(args.src, encoding="utf-8").read()
    hits = FLAG.findall(html)
    if len(hits) != 1:
        sys.exit(f"{args.src}: expected exactly one STATIC_BUILD line, found {len(hits)}. "
                 f"Has the switch been renamed?")
    out = FLAG.sub("const STATIC_BUILD = true;", html, count=1)
    if out == html:
        sys.exit(f"{args.src} is already a static build — build from the full dashboard")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(out)

    raw = len(out.encode())
    gz = len(gzip.compress(out.encode(), 9))
    print(f"[build] {os.path.relpath(args.out, ROOT)}")
    print(f"        {raw / 1048576:.1f} MB raw · {gz / 1048576:.1f} MB gzipped")
    print("        tabs: Map, Table, Ground detail, License")
    print("        dropped: Dynamic Search (needs the vector index), "
          "Analysis (needs a model endpoint)")
    print("[serve] upload it as index.html; it needs nothing but a file server.")
    print("        Turn on gzip or brotli — it is mostly embedded JSON and "
          "compresses about 10:1.")


if __name__ == "__main__":
    main()
