#!/usr/bin/env python3
"""Step 1 — Fetch Nov 1–8, 2026 arXiv submissions, compute category overlap, sample, and split."""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import defusedxml.ElementTree as ET

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from common import ALL_QUERY_CATEGORIES, ALT_CATEGORIES_SET_A, CATEGORIES, collapse_whitespace  # noqa: E402

ATOM_NS = "{http://www.w3.org/2005/Atom}"
ARXIV_NS = "{http://arxiv.org/schemas/atom}"
OPENSEARCH_NS = "{http://a9.com/-/spec/opensearch/1.1/}"

DATE_WINDOW = "202611010000+TO+202611082359"
MAX_RESULTS = 2000
SLEEP_SECONDS = 3.1
SEED = 0
SAMPLE_PER_CAT = 70
FIT_PER_CAT = 12
TEST_PER_CAT = 58

# Standard arXiv archive/category identifier pattern (excludes ACM like 'I.2.7' or MSC like '68T07')
ARXIV_CAT_RE = re.compile(r"^[a-z-]+(\.[A-Za-z-]+)?$")
VERSION_RE = re.compile(r"v\d+$")


def strip_arxiv_id(raw_id: str) -> str:
    """Convert 'http://arxiv.org/abs/2611.01234v1' -> '2611.01234'."""
    cleaned = raw_id.strip()
    for prefix in ("http://arxiv.org/abs/", "https://arxiv.org/abs/"):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):]
            break
    return VERSION_RE.sub("", cleaned)


def fetch_url(url: str, retries: int = 4) -> bytes:
    """Fetch URL with polite User-Agent and exponential backoff on transient errors."""
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "decision-confidence-eval/1.0 (https://github.com/DavidFlorez/decision-confidence)"},
    )
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read()
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            if attempt == retries - 1:
                raise
            wait = SLEEP_SECONDS * (2 ** (attempt + 1))
            print(f"  [retry {attempt + 1}/{retries}] {exc} — sleeping {wait:.1f}s...", flush=True)
            time.sleep(wait)
    raise RuntimeError("Unreachable")


def download_raw_xml(raw_dir: Path, refresh: bool = False) -> list[Path]:
    """Download Atom XML for each queried category with >=3s spacing between API calls."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    saved_files: list[Path] = []
    made_network_call = False

    for cat in ALL_QUERY_CATEGORIES:
        out_path = raw_dir / f"{cat}.xml"
        if out_path.exists() and not refresh:
            print(f"[cached] {out_path.relative_to(ROOT)}", flush=True)
            saved_files.append(out_path)
            root = ET.fromstring(out_path.read_bytes())
        else:
            if made_network_call:
                time.sleep(SLEEP_SECONDS)
            url = (
                f"https://export.arxiv.org/api/query?"
                f"search_query=cat:{cat}+AND+submittedDate:[{DATE_WINDOW}]&max_results={MAX_RESULTS}"
            )
            print(f"[fetch]  {cat} -> {url}", flush=True)
            xml_bytes = fetch_url(url)
            made_network_call = True
            out_path.write_bytes(xml_bytes)
            saved_files.append(out_path)
            root = ET.fromstring(xml_bytes)

        total_el = root.find(f"{OPENSEARCH_NS}totalResults")
        total_results = int(total_el.text.strip()) if (total_el is not None and total_el.text) else 0
        entries = root.findall(f"{ATOM_NS}entry")
        print(f"         {cat}: totalResults={total_results}, entries_in_page={len(entries)}", flush=True)

        # Paginate if totalResults exceeds MAX_RESULTS so zero papers are truncated
        start = MAX_RESULTS
        while start < total_results:
            page_path = raw_dir / f"{cat}_start{start}.xml"
            if page_path.exists() and not refresh:
                print(f"[cached] {page_path.relative_to(ROOT)}", flush=True)
                saved_files.append(page_path)
            else:
                if made_network_call:
                    time.sleep(SLEEP_SECONDS)
                page_url = (
                    f"https://export.arxiv.org/api/query?"
                    f"search_query=cat:{cat}+AND+submittedDate:[{DATE_WINDOW}]"
                    f"&start={start}&max_results={MAX_RESULTS}"
                )
                print(f"[fetch]  {cat} (start={start}) -> {page_url}", flush=True)
                page_bytes = fetch_url(page_url)
                made_network_call = True
                page_path.write_bytes(page_bytes)
                saved_files.append(page_path)
            start += MAX_RESULTS

    return saved_files


def parse_xml_files(xml_paths: list[Path]) -> tuple[dict[str, dict], set[str]]:
    """Parse all Atom XML files and deduplicate papers by version-stripped arXiv ID."""
    papers_by_id: dict[str, dict] = {}
    non_arxiv_terms: set[str] = set()

    for path in xml_paths:
        root = ET.fromstring(path.read_bytes())
        for entry in root.findall(f"{ATOM_NS}entry"):
            id_el = entry.find(f"{ATOM_NS}id")
            title_el = entry.find(f"{ATOM_NS}title")
            summary_el = entry.find(f"{ATOM_NS}summary")
            pub_el = entry.find(f"{ATOM_NS}published")
            prim_el = entry.find(f"{ARXIV_NS}primary_category")

            if id_el is None or not id_el.text or prim_el is None:
                continue

            paper_id = strip_arxiv_id(id_el.text)
            primary = prim_el.attrib.get("term", "").strip()
            title = collapse_whitespace(title_el.text or "") if title_el is not None else ""
            abstract = collapse_whitespace(summary_el.text or "") if summary_el is not None else ""
            published = (pub_el.text or "").strip() if pub_el is not None else ""

            raw_terms: list[str] = []
            arxiv_terms: list[str] = []
            for cat_el in entry.findall(f"{ATOM_NS}category"):
                term = cat_el.attrib.get("term", "").strip()
                if not term:
                    continue
                if term not in raw_terms:
                    raw_terms.append(term)
                if ARXIV_CAT_RE.match(term):
                    if term not in arxiv_terms:
                        arxiv_terms.append(term)
                else:
                    non_arxiv_terms.add(term)

            if paper_id not in papers_by_id:
                papers_by_id[paper_id] = {
                    "id": paper_id,
                    "title": title,
                    "abstract": abstract,
                    "primary": primary,
                    "all_terms_raw": raw_terms,
                    "all_terms_arxiv": arxiv_terms,
                    "published": published,
                }

    return papers_by_id, non_arxiv_terms


def analyze_set(
    papers_by_id: dict[str, dict],
    cat_set: list[str],
    use_arxiv_only_terms: bool = True,
) -> dict:
    """Compute per-category eligible counts and cross-list overlap within cat_set."""
    cat_set_lookup = set(cat_set)
    per_cat: dict[str, list[dict]] = {c: [] for c in cat_set}
    total_eligible = 0
    in_set_crosslist_count = 0

    for pid in sorted(papers_by_id.keys()):
        p = papers_by_id[pid]
        prim = p["primary"]
        if prim not in cat_set_lookup:
            continue
        terms = p["all_terms_arxiv"] if use_arxiv_only_terms else p["all_terms_raw"]
        cross_lists = [t for t in terms if t != prim]
        if len(cross_lists) <= 1:
            total_eligible += 1
            if any(t in cat_set_lookup for t in cross_lists):
                in_set_crosslist_count += 1
            per_cat[prim].append(
                {
                    "id": p["id"],
                    "title": p["title"],
                    "abstract": p["abstract"],
                    "primary": prim,
                    "cross_lists": cross_lists,
                    "published": p["published"],
                }
            )

    overlap_rate = (in_set_crosslist_count / total_eligible) if total_eligible > 0 else 0.0
    return {
        "categories": cat_set,
        "counts_per_category": {c: len(per_cat[c]) for c in cat_set},
        "total_eligible": total_eligible,
        "in_set_crosslist_count": in_set_crosslist_count,
        "overlap_rate": overlap_rate,
        "eligible_by_category": per_cat,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Nov 1-8, 2026 arXiv 8-category dataset.")
    parser.add_argument("--refresh", action="store_true", help="Re-download Atom XML even if cached")
    args = parser.parse_args()

    raw_dir = ROOT / "data" / "raw"
    xml_files = download_raw_xml(raw_dir, refresh=args.refresh)
    papers_by_id, non_arxiv_terms = parse_xml_files(xml_files)
    print(f"\nTotal unique papers parsed across {len(xml_files)} XML files: {len(papers_by_id)}", flush=True)
    print(f"Non-arXiv subject classification terms seen (ACM/MSC): {len(non_arxiv_terms)}", flush=True)

    # Compare raw <category> terms vs archive-filtered terms to verify cross-list counting
    set_b_arxiv = analyze_set(papers_by_id, CATEGORIES, use_arxiv_only_terms=True)
    set_a_arxiv = analyze_set(papers_by_id, ALT_CATEGORIES_SET_A, use_arxiv_only_terms=True)
    set_b_raw = analyze_set(papers_by_id, CATEGORIES, use_arxiv_only_terms=False)
    set_a_raw = analyze_set(papers_by_id, ALT_CATEGORIES_SET_A, use_arxiv_only_terms=False)

    print("\n=== Overlap Check (arXiv archive terms only) ===")
    print(
        f"Set B ({', '.join(CATEGORIES)}): "
        f"{set_b_arxiv['in_set_crosslist_count']}/{set_b_arxiv['total_eligible']} "
        f"({set_b_arxiv['overlap_rate'] * 100:.2f}%)"
    )
    print(f"  Per-category: {set_b_arxiv['counts_per_category']}")
    print(
        f"Set A ({', '.join(ALT_CATEGORIES_SET_A)}): "
        f"{set_a_arxiv['in_set_crosslist_count']}/{set_a_arxiv['total_eligible']} "
        f"({set_a_arxiv['overlap_rate'] * 100:.2f}%)"
    )
    print(f"  Per-category: {set_a_arxiv['counts_per_category']}")

    print("\n=== Overlap Check (all raw <category> terms including ACM/MSC) ===")
    print(
        f"Set B raw: {set_b_raw['in_set_crosslist_count']}/{set_b_raw['total_eligible']} "
        f"({set_b_raw['overlap_rate'] * 100:.2f}%) | {set_b_raw['counts_per_category']}"
    )
    print(
        f"Set A raw: {set_a_raw['in_set_crosslist_count']}/{set_a_raw['total_eligible']} "
        f"({set_a_raw['overlap_rate'] * 100:.2f}%) | {set_a_raw['counts_per_category']}"
    )

    # Use raw <category> terms if ACM/MSC were counted by reference, or archive-only if that matches:
    # Note: ACM/MSC terms are classification codes, not archives ("counting any archive, including non-cs ones").
    # We'll inspect both above; use archive terms as specified ("counting any archive, including non-cs ones").
    chosen_b = set_b_arxiv
    chosen_a = set_a_arxiv

    for cat, cnt in chosen_b["counts_per_category"].items():
        if cnt < SAMPLE_PER_CAT:
            raise RuntimeError(f"Category {cat} only has {cnt} eligible papers (< {SAMPLE_PER_CAT})!")

    # Sample 70 per category (seed 0) and stratified split: 12 fit + 58 test per category (seed 0)
    sample_rng = random.Random(SEED)
    split_rng = random.Random(SEED)

    sampled_records: list[dict] = []
    sampled_ids_by_cat: dict[str, dict[str, list[str]]] = {}

    for cat in CATEGORIES:
        pool = list(chosen_b["eligible_by_category"][cat])  # already sorted by id
        picked = sample_rng.sample(pool, SAMPLE_PER_CAT)
        # Sort picked by id before shuffling for stratified split so split is clean and reproducible
        picked_sorted = sorted(picked, key=lambda r: r["id"])
        split_rng.shuffle(picked_sorted)

        fit_slice = sorted(picked_sorted[:FIT_PER_CAT], key=lambda r: r["id"])
        test_slice = sorted(picked_sorted[FIT_PER_CAT:FIT_PER_CAT + TEST_PER_CAT], key=lambda r: r["id"])
        assert len(fit_slice) == FIT_PER_CAT and len(test_slice) == TEST_PER_CAT

        sampled_ids_by_cat[cat] = {
            "fit": [r["id"] for r in fit_slice],
            "test": [r["id"] for r in test_slice],
        }

        for r in fit_slice:
            sampled_records.append(
                {
                    "id": r["id"],
                    "title": r["title"],
                    "abstract": r["abstract"],
                    "primary": r["primary"],
                    "cross_lists": r["cross_lists"],
                    "split": "fit",
                }
            )
        for r in test_slice:
            sampled_records.append(
                {
                    "id": r["id"],
                    "title": r["title"],
                    "abstract": r["abstract"],
                    "primary": r["primary"],
                    "cross_lists": r["cross_lists"],
                    "split": "test",
                }
            )

    papers_path = ROOT / "data" / "papers.jsonl"
    with papers_path.open("w", encoding="utf-8") as fh:
        for rec in sampled_records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    counts_payload = {
        "date_window": DATE_WINDOW,
        "pulled_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": SEED,
        "sample_per_category": SAMPLE_PER_CAT,
        "fit_per_category": FIT_PER_CAT,
        "test_per_category": TEST_PER_CAT,
        "total_sampled": len(sampled_records),
        "total_fit": sum(1 for r in sampled_records if r["split"] == "fit"),
        "total_test": sum(1 for r in sampled_records if r["split"] == "test"),
        "set_b": {
            "categories": CATEGORIES,
            "available_counts": chosen_b["counts_per_category"],
            "total_eligible": chosen_b["total_eligible"],
            "in_set_crosslist_count": chosen_b["in_set_crosslist_count"],
            "overlap_rate": chosen_b["overlap_rate"],
        },
        "set_a": {
            "categories": ALT_CATEGORIES_SET_A,
            "available_counts": chosen_a["counts_per_category"],
            "total_eligible": chosen_a["total_eligible"],
            "in_set_crosslist_count": chosen_a["in_set_crosslist_count"],
            "overlap_rate": chosen_a["overlap_rate"],
        },
        "raw_category_tag_comparison": {
            "non_arxiv_classification_terms_count": len(non_arxiv_terms),
            "set_b_raw_total": set_b_raw["total_eligible"],
            "set_b_raw_in_set_crosslist": set_b_raw["in_set_crosslist_count"],
            "set_b_raw_counts": set_b_raw["counts_per_category"],
            "set_a_raw_total": set_a_raw["total_eligible"],
            "set_a_raw_in_set_crosslist": set_a_raw["in_set_crosslist_count"],
            "set_a_raw_counts": set_a_raw["counts_per_category"],
        },
        "sampled_ids": sampled_ids_by_cat,
    }

    counts_path = ROOT / "data" / "counts.json"
    counts_path.write_text(json.dumps(counts_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nWrote {len(sampled_records)} papers to {papers_path.relative_to(ROOT)}", flush=True)
    print(f"Wrote counts & overlap summary to {counts_path.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
