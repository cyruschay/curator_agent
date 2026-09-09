"""Flatten the curation JSONL output into a glycan biomarker TSV (one row per relation).

Usage:
    python scripts/build_biomarker_tsv.py <curation_output_dir> [-o out.tsv]

Reads curations_batch_*.jsonl from the output dir and writes one row per extracted
glycan-disease biomarker relation, with the mapped ontology fields, the biomarker role(s),
the role-specific annotations (as JSON), and the evidence.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

COLUMNS = [
    "pid", "pmid", "pmcid", "study_type",
    "glycan_name", "glycan_mapped_name", "glycan_id", "glycan_alignment", "glycan_aglycon", "glycan_metadata",
    "disease_name", "disease_mapped_name", "disease_id", "disease_annotation",
    "direction", "biomarker_role", "is_multicomponent",
    "specimen_original", "specimen_mapped_name", "specimen_mapped_id", "specimen_category", "specimen_ontology",
    "species_name", "species_mapped_name", "species_id",
    "protein_name", "protein_mapped_name", "protein_id",
    "method_names", "metrics", "negated_or_hedged",
    "role_annotations",
    "evidence_count", "evidence_sections", "evidence_sentences",
    "title",
]

_WS = re.compile(r"[\t\r\n]+")


def _cell(v) -> str:
    """One TSV cell: JSON-encode containers, flatten whitespace, never emit a raw tab/newline."""
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (dict, list)):
        v = json.dumps(v, ensure_ascii=False)
    return _WS.sub(" ", str(v)).strip()


def rows_from_record(c: dict):
    scr = c.get("screening") or {}
    for r in c.get("relations") or []:
        sp = r.get("specimen") or {}
        gem = r.get("glycan_entity_metadata") or {}
        ev = r.get("evidence_sentences") or []
        metrics = "; ".join(
            f"{m.get('name')}={m.get('value') if m.get('value') is not None else m.get('raw')}"
            for m in (r.get("metrics") or [])
        )
        yield {
            "pid": c.get("pid"), "pmid": c.get("pmid"), "pmcid": c.get("pmcid"),
            "study_type": scr.get("study_type"),
            "glycan_name": r.get("glycan_name"), "glycan_mapped_name": r.get("glycan_mapped_name"),
            "glycan_id": r.get("glycan_id"),
            "glycan_alignment": gem.get("alignment"), "glycan_aglycon": gem.get("aglycon"),
            "glycan_metadata": r.get("glycan_metadata"),
            "disease_name": r.get("disease_name"), "disease_mapped_name": r.get("disease_mapped_name"),
            "disease_id": r.get("disease_id"), "disease_annotation": r.get("disease_annotation"),
            "direction": r.get("direction"),
            "biomarker_role": ",".join(r.get("biomarker_role") or []),
            "is_multicomponent": r.get("is_multicomponent"),
            "specimen_original": sp.get("original"), "specimen_mapped_name": sp.get("mapped_name"),
            "specimen_mapped_id": sp.get("mapped_id"), "specimen_category": sp.get("category"),
            "specimen_ontology": sp.get("ontology"),
            "species_name": r.get("species_name"), "species_mapped_name": r.get("species_mapped_name"),
            "species_id": r.get("species_id"),
            "protein_name": r.get("protein_name"), "protein_mapped_name": r.get("protein_mapped_name"),
            "protein_id": r.get("protein_id"),
            "method_names": "; ".join(r.get("method_names") or []),
            "metrics": metrics,
            "negated_or_hedged": r.get("negated_or_hedged"),
            "role_annotations": r.get("role_annotations") or {},
            "evidence_count": len(ev),
            "evidence_sections": "; ".join(sorted({e.get("section") for e in ev if e.get("section")})),
            "evidence_sentences": " | ".join(e.get("sentence", "") for e in ev),
            "title": c.get("title"),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir", help="curation output dir containing curations_batch_*.jsonl")
    ap.add_argument("-o", "--out", default=None, help="output TSV path (default: <outdir>/glycan_biomarker_table.tsv)")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    out_tsv = Path(args.out) if args.out else outdir / "glycan_biomarker_table.tsv"

    files = sorted(glob.glob(str(outdir / "curations_batch_*.jsonl")))
    if not files:
        raise SystemExit(f"No curations_batch_*.jsonl in {outdir}")

    n_articles = n_rel_articles = n_rows = 0
    role_counts: dict[str, int] = {}
    id_cov = {"glycan": 0, "disease": 0, "specimen": 0}
    with open(out_tsv, "w", encoding="utf-8") as out:
        out.write("\t".join(COLUMNS) + "\n")
        for f in files:
            for line in open(f, encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                c = json.loads(line)
                n_articles += 1
                rels = c.get("relations") or []
                if rels:
                    n_rel_articles += 1
                for row in rows_from_record(c):
                    n_rows += 1
                    for role in (row["biomarker_role"].split(",") if row["biomarker_role"] else []):
                        role_counts[role] = role_counts.get(role, 0) + 1
                    if row["glycan_id"]:
                        id_cov["glycan"] += 1
                    if row["disease_id"]:
                        id_cov["disease"] += 1
                    if row["specimen_mapped_id"]:
                        id_cov["specimen"] += 1
                    out.write("\t".join(_cell(row[c2]) for c2 in COLUMNS) + "\n")

    print(f"Wrote {out_tsv}")
    print(f"  articles scanned:        {n_articles}")
    print(f"  articles with relations: {n_rel_articles}")
    print(f"  biomarker rows:          {n_rows}")
    print(f"  role distribution:       {dict(sorted(role_counts.items(), key=lambda kv: -kv[1]))}")
    if n_rows:
        print(f"  id coverage: glycan {id_cov['glycan']}/{n_rows} ({100*id_cov['glycan']//n_rows}%) "
              f"| disease {id_cov['disease']}/{n_rows} ({100*id_cov['disease']//n_rows}%) "
              f"| specimen {id_cov['specimen']}/{n_rows} ({100*id_cov['specimen']//n_rows}%)")


if __name__ == "__main__":
    main()
