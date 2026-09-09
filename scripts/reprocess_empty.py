"""Re-run the articles whose run logged a given error, and splice their fresh rows back into the
existing production batch files in place (same filenames, same article->batch assignment).

Default target: the empty-LLM-content articles (the describe step that returned an empty final
channel). Runs AFTER the full corpus run, once the fix is in place.

Usage:
    python scripts/reprocess_empty.py <curation_dir> --plan          # dry run
    python scripts/reprocess_empty.py <curation_dir> --run           # reprocess + splice (backs up first)
    python scripts/reprocess_empty.py <curation_dir> --run --select "Empty LLM content"
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ABSTRACTS = REPO / "data" / "raw" / "abstracts"
CONFIG = REPO / "config.yaml"
GRAPH_PY = REPO / "src" / "agent" / "graph.py"
PYTHON = sys.executable  # same interpreter that launches this script

FULL_INPUT = ABSTRACTS / "abstracts_full.jsonl"
REPROCESS_INPUT_NAME = "abstracts_reprocess.jsonl"
TMP_DIR_NAME = "reprocess-tmp"
STREAMS = ["curations", "runlog", "violations", "errors", "reasonings", "metadata"]


def _rows(d: Path, stream: str):
    for f in sorted(d.glob(f"{stream}_batch_*.jsonl")):
        for line in f.read_text().splitlines():
            if line.strip():
                yield json.loads(line)


def collect_pmids(d: Path, substring: str) -> tuple[set, set]:
    """(processing_ids, pmids) for every article whose errors row contains `substring`."""
    pids, pmids = set(), set()
    for o in _rows(d, "errors"):
        msgs = o.get("error_messages", [])
        if isinstance(msgs, dict):
            msgs = [msgs]
        text = " ".join((m or {}).get("content", "") if isinstance(m, dict) else str(m) for m in (msgs or []))
        if substring in text:
            if o.get("processing_id"):
                pids.add(o["processing_id"])
            if o.get("pmid"):
                pmids.add(str(o["pmid"]))
    return pids, pmids


def provenance_map(d: Path) -> dict:
    m = {}
    for o in _rows(d, "curations"):
        pv = o.get("provenance", {}) or {}
        m[str(o.get("pmid"))] = {
            "article_index": pv.get("article_index"),
            "batch_number": pv.get("batch_number"),
            "batch_offset": pv.get("batch_offset"),
            "pid": o.get("pid"),
        }
    return m


def build_filtered_input(pids: set) -> Path:
    out = ABSTRACTS / REPROCESS_INPUT_NAME
    kept = [ln for ln in FULL_INPUT.read_text().splitlines()
            if ln.strip() and json.loads(ln).get("processing_id") in pids]
    out.write_text("\n".join(kept) + "\n")
    return out


def run_reprocess(curation_dir: Path) -> Path:
    tmp = curation_dir.parent / TMP_DIR_NAME
    if tmp.exists():
        shutil.rmtree(tmp)
    backup = CONFIG.read_text()
    new = []
    for ln in backup.splitlines():
        s = ln.lstrip()
        if s.startswith("input_abstracts:"):
            new.append(ln[: len(ln) - len(s)] + f'input_abstracts: "{REPROCESS_INPUT_NAME}"')
        elif s.startswith("curation_output_dir:"):
            new.append(ln[: len(ln) - len(s)] + f'curation_output_dir: "{TMP_DIR_NAME}"')
        else:
            new.append(ln)
    try:
        CONFIG.write_text("\n".join(new) + "\n")
        print("  running pipeline on the affected subset ...")
        subprocess.run([PYTHON, "-u", str(GRAPH_PY)], cwd=str(GRAPH_PY.parent), check=True)
    finally:
        CONFIG.write_text(backup)  # always restore config
    return tmp


def load_new_rows(tmp: Path) -> dict:
    return {s: {str(o.get("pmid")): o for o in _rows(tmp, s)} for s in STREAMS}


def rebuild(d: Path, stream: str, pmids: set, new_rows: dict, prov: dict) -> tuple[int, int]:
    kept = {str(o.get("pmid")): o for o in _rows(d, stream) if str(o.get("pmid")) not in pmids}
    replaced = dropped = 0
    for pm in pmids:
        nr = new_rows.get(stream, {}).get(pm)
        if nr is not None:
            if stream in ("curations", "runlog") and pm in prov:  # keep original provenance/order
                pv = nr.setdefault("provenance", {})
                for k in ("article_index", "batch_number", "batch_offset"):
                    if prov[pm].get(k) is not None:
                        pv[k] = prov[pm][k]
            kept[pm] = nr
            replaced += 1
        else:
            dropped += 1  # e.g. an errors row that the fix cleared
    buckets: dict[int, list] = {}
    for pm, row in kept.items():
        ai = (prov.get(pm) or {}).get("article_index")
        b = (ai // 100 + 1) if isinstance(ai, int) else 1
        buckets.setdefault(b, []).append((ai if isinstance(ai, int) else 1 << 30, row))
    for f in d.glob(f"{stream}_batch_*.jsonl"):
        f.unlink()
    for b, items in buckets.items():
        items.sort(key=lambda x: x[0])
        (d / f"{stream}_batch_{b:03d}.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for _, r in items) + "\n")
    return replaced, dropped


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("curation_dir")
    ap.add_argument("--select", default="Empty LLM content", help="error substring selecting articles")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--run", action="store_true")
    args = ap.parse_args()
    if not (args.plan or args.run):
        ap.error("pass --plan or --run")

    d = Path(args.curation_dir)
    pids, pmids = collect_pmids(d, args.select)
    prov = provenance_map(d)
    print(f"articles matching {args.select!r}: {len(pmids)} (pids {len(pids)})")

    if args.plan:
        by_batch: dict[int, int] = {}
        for pm in pmids:
            b = ((prov.get(pm) or {}).get("article_index") or 0) // 100 + 1
            by_batch[b] = by_batch.get(b, 0) + 1
        print("  per batch:", dict(sorted(by_batch.items())))
        print("  --plan only; no files changed.")
        return 0

    backup = d.parent / f"{d.name}.backup-{int(time.time())}"
    print(f"backing up {d.name} -> {backup.name}")
    shutil.copytree(d, backup)
    build_filtered_input(pids)
    tmp = run_reprocess(d)
    new_rows = load_new_rows(tmp)
    print("  reprocessed rows: " + ", ".join(f"{s}={len(new_rows[s])}" for s in STREAMS))
    for s in STREAMS:
        rep, dr = rebuild(d, s, pmids, new_rows, prov)
        print(f"  {s:11s}: replaced {rep}, dropped {dr}")
    shutil.rmtree(tmp)
    (ABSTRACTS / REPROCESS_INPUT_NAME).unlink(missing_ok=True)
    print(f"done. backup retained at {backup.name}; verify then delete it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
