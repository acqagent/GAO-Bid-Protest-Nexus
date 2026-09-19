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

"""Produce a hosting build of the dashboard. Two shapes, one switch each.

    python3 scripts/build_static.py          # -> visualization/map-static.html
    python3 scripts/build_static.py --web    # -> visualization/map-web.html

--static (the default) flips STATIC_BUILD. That drops the two tabs that need a
backend — Dynamic Search, which needs the vector index behind scripts/serve.py,
and Analysis, which needs a model endpoint — and removes their markup, so a
stale deep link cannot reach them either. Four tabs remain and want nothing but
a file server.

--web keeps all six tabs and flips WEB_BUILD instead. Everything still renders;
the two backend tabs explain themselves to a visitor rather than telling them to
start a server they have no access to, and the page stops probing /api/config,
so a public site is not logging a 404 on every pageview. Analysis still works
for a visitor who supplies their own endpoint and their own decision text.

Rebuild whenever visualization/map.html changes; this is a one-line transform,
not a fork, so they cannot drift.
"""

import argparse
import gzip
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "visualization", "map.html")
SWITCHES = {
    "static": ("STATIC_BUILD",
               "drops Dynamic Search and Analysis, and their markup with them"),
    "web":    ("WEB_BUILD",
               "no /api/config probe, and the backend tabs explain themselves "
               "to a visitor instead of naming a server they cannot start"),
    "connect": ("SETUP_GUIDE",
                "adds a Setup tab: how to point the dashboard at an "
                "OpenAI-compatible endpoint and turn on every feature"),
}
DEFAULT_OUT = {"static": "map-static.html", "web": "map-web.html",
               "connect": "map-connect.html"}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=SRC, help="the full dashboard to build from")
    ap.add_argument("--out", default="", help="where to write it (defaults per mode)")
    for name, (_, help_text) in SWITCHES.items():
        ap.add_argument(f"--{name}", action="store_true", help=help_text)
    args = ap.parse_args()

    on = [k for k in SWITCHES if getattr(args, k)] or ["static"]
    if "static" in on and len(on) > 1:
        sys.exit("--static drops the tabs the other modes are about; pick one or the other")
    # name the file after the most specific switch asked for, so
    # "--connect --web" does not quietly overwrite the plain web build
    named = next(k for k in ("connect", "web", "static") if k in on)
    out_path = args.out or os.path.join(ROOT, "visualization", DEFAULT_OUT[named])

    out = open(args.src, encoding="utf-8").read()
    for name in on:
        flag_name = SWITCHES[name][0]
        flag = re.compile(rf"^const {flag_name} = (true|false);$", re.M)
        hits = flag.findall(out)
        if len(hits) != 1:
            sys.exit(f"{args.src}: expected exactly one {flag_name} line, found "
                     f"{len(hits)}. Has the switch been renamed?")
        if hits[0] == "true":
            sys.exit(f"{args.src} already has {flag_name} on — build from the full dashboard")
        out = flag.sub(f"const {flag_name} = true;", out, count=1)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(out)

    raw = len(out.encode())
    gz = len(gzip.compress(out.encode(), 9))
    print(f"[build] {os.path.relpath(out_path, ROOT)}  "
          f"({', '.join(SWITCHES[k][0] for k in on)} on)")
    print(f"        {raw / 1048576:.1f} MB raw · {gz / 1048576:.1f} MB gzipped")
    for k in on:
        print(f"        --{k}: {SWITCHES[k][1]}")
    print("[serve] upload it as index.html; it needs nothing but a file server.")
    print("        Turn on gzip or brotli — it is mostly embedded JSON and "
          "compresses about 10:1.")


if __name__ == "__main__":
    main()
