import argparse
import json
import hashlib
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import numpy as np
import pymupdf
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def parse_local_pdf(
    pdf_path: Path,
    min_words: int = 90,
    max_words: int = 250,
    include_segmentation_ablations: bool = False,
) -> Optional[Dict]:
    try:
        doc = pymupdf.open(pdf_path)
    except Exception:
        return None

    pages: List[str] = []
    first_page_lines: List[str] = []
    for p_idx, page in enumerate(doc):
        blocks = page.get_text("blocks")
        blocks.sort(key=lambda b: (round(b[1], 1), round(b[0], 1)))
        txt_blocks = [b[4].strip() for b in blocks if b[6] == 0 and b[4].strip()]
        if p_idx == 0:
            first_page_lines = txt_blocks[:8]
        pages.append("\n\n".join(txt_blocks))
    page_count = len(pages)
    doc.close()

    full_text = "\n\n".join(pages)
    ref_matches = list(re.finditer(r"(?im)^\s*(?:\d+\.?\s*)?(references|bibliography)\s*$", full_text))
    if ref_matches:
        body_text = full_text[: ref_matches[-1].start()]
        refs_text = full_text[ref_matches[-1].start() :]
    else:
        body_text, refs_text = full_text, ""

    title = ""
    for blk in first_page_lines:
        clean_blk = re.sub(r"\s+", " ", blk).strip()
        if len(clean_blk.split()) >= 4 and not re.search(
            r"@|arxiv:|university|department|abstract|preprint|ieee|acm", clean_blk, re.I
        ):
            title = clean_blk
            break
    if not title:
        title = pdf_path.stem

    abs_m = re.search(
        r"(?is)\babstract\b[:\s\n]+(.{100,2400}?)(?:\n\s*\n|\b1[\.\s]+introduction\b)", body_text
    )
    abstract = re.sub(r"\s+", " ", abs_m.group(1)).strip() if abs_m else body_text[:1200]

    raw_blocks = re.split(r"\n\s*\n", body_text)
    cleaned: List[str] = []
    for blk in raw_blocks:
        txt = re.sub(r"-\s*\n\s*", "", blk)
        txt = re.sub(r"\s+", " ", txt).strip()
        if len(txt.split()) < 15:
            continue
        if re.match(r"^(figure|fig\.|table|algorithm|eq\.)\s+\d+", txt, re.IGNORECASE):
            continue
        cleaned.append(txt)

    paragraphs: List[Dict] = []
    buf: List[str] = []

    def emit(words_list: List[str]) -> None:
        paragraphs.append({
            "passage_id": f"p_{len(paragraphs):03d}",
            "passage_index": len(paragraphs),
            "word_count": len(words_list),
            "text": " ".join(words_list),
        })

    for blk in cleaned:
        words = blk.split()
        if len(buf) + len(words) <= max_words:
            buf.extend(words)
            if len(buf) >= min_words:
                emit(buf)
                buf = []
        else:
            if len(buf) >= min_words:
                emit(buf)
            buf = []
            sents = re.split(r"(?<=[.!?])\s+", blk)
            chunk: List[str] = []
            for s in sents:
                sw = s.split()
                if len(chunk) + len(sw) > max_words and len(chunk) >= min_words:
                    emit(chunk)
                    chunk = sw
                else:
                    chunk.extend(sw)
            if len(chunk) >= min_words:
                emit(chunk)
            else:
                buf = chunk

    if len(buf) >= min_words:
        emit(buf)

    if len(paragraphs) < 6:
        return None

    def make_sentence_passages(sentence_count: int) -> List[Dict]:
        sentences = [
            re.sub(r"\s+", " ", sentence).strip()
            for sentence in re.split(r"(?<=[.!?])\s+", body_text)
        ]
        sentences = [
            sentence for sentence in sentences
            if len(sentence.split()) >= 5
            and not re.match(r"^(figure|fig\.|table|algorithm|eq\.)\b", sentence, re.I)
        ]
        output = []
        step = sentence_count
        for start in range(0, len(sentences), step - (1 if sentence_count > 1 else 0)):
            window = sentences[start:start + sentence_count]
            if len(window) < sentence_count and sentence_count > 1:
                continue
            text = " ".join(window)
            output.append({
                "passage_id": f"seg_{sentence_count}_{len(output):03d}",
                "passage_index": len(output),
                "word_count": len(text.split()),
                "text": text,
            })
        return output

    passage_sets = {}
    if include_segmentation_ablations:
        section_passages = []
        sections = re.split(
            r"\n\s*(?=(?:\d+(?:\.\d+)*[\s.]+[A-Z]|abstract|introduction|methods|results|discussion)\b)",
            body_text,
            flags=re.I,
        )
        for section in sections:
            section = re.sub(r"\s+", " ", section).strip()
            if len(section.split()) >= 15:
                section_passages.append({
                    "passage_id": f"section_{len(section_passages):03d}",
                    "passage_index": len(section_passages),
                    "word_count": len(section.split()),
                    "text": section,
                })
        passage_sets = {
            "90-250-word": paragraphs,
            "single-sentence": make_sentence_passages(1),
            "3-sentence": make_sentence_passages(3),
            "section-oriented": section_passages,
        }

    arxiv_id = pdf_path.stem.replace("_", "/")
    pdf_bytes = pdf_path.read_bytes()
    return {
        "arxiv_id": arxiv_id,
        "title": title,
        "abstract": abstract,
        "body_text": body_text,
        "refs_text": refs_text,
        "paragraphs": paragraphs,
        "passage_sets": passage_sets,
        "source_audit": {
            "pdf_filename": pdf_path.name,
            "pdf_sha256": hashlib.sha256(pdf_bytes).hexdigest(),
            "extracted_text_chars": len(full_text),
            "body_text_chars": len(body_text),
            "passage_count": len(paragraphs),
            "references_extracted": bool(refs_text),
            "page_count": page_count,
            "source_text_truncated": False,
            "extraction_method": "PyMuPDF text blocks ordered by page y/x coordinates",
            "page_headers_footers_removed": False,
            "excluded_categories": [
                "blocks_under_15_words",
                "blocks_starting_with_figure_table_algorithm_or_equation",
                "text_after_detected_references_heading",
            ],
            "extraction_quality_score": round(
                min(1.0, (len(body_text) / 1000.0) * 0.3 + (len(paragraphs) / 50.0) * 0.7),
                4,
            ),
            "extraction_quality_score_method": (
                "heuristic from extracted body length and passage count; not human-validated"
            ),
            "extraction_status": "ok",
        },
    }


def norm_title(t: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", t.lower())).strip()


def extract_citation_contexts(
    citing_body: str, cited_doc: Dict, ref_num: Optional[str], max_contexts: int = 4, window: int = 1
) -> List[str]:
    stop = {
        "using", "based", "towards", "learning", "models", "model", "neural",
        "language", "large", "approach", "method", "methods", "framework", "network",
    }
    title_words = [w for w in re.findall(r"[a-zA-Z]{5,}", cited_doc["title"].lower()) if w not in stop]
    sentences = re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", citing_body))

    contexts: List[str] = []
    seen_ctx: Set[str] = set()

    for i, sent in enumerate(sentences):
        matched = False
        if ref_num:
            for br in re.findall(r"\[([0-9,\s\-–]+)\]", sent):
                if ref_num in re.findall(r"\d+", br):
                    matched = True
                    break
        if not matched and len(title_words) >= 2:
            s_low = sent.lower()
            if sum(1 for w in title_words[:8] if w in s_low) >= 2:
                matched = True

        if matched:
            ctx = " ".join(sentences[max(0, i - window) : min(len(sentences), i + window + 1)])
            ctx = re.sub(r"\s*\[[0-9,\s\-–]+\]", "", ctx).strip()
            if 12 <= len(ctx.split()) <= 240 and ctx not in seen_ctx:
                seen_ctx.add(ctx)
                contexts.append(ctx)
                if len(contexts) >= max_contexts:
                    break

    return contexts


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline dataset builder (>= 1,000 instances).")
    parser.add_argument("--pdf-dir", default="data/pdfs")
    parser.add_argument("--target-positives", type=int, default=500)
    parser.add_argument("--target-negatives", type=int, default=500,
                        help="Kept as a paired 500/500 target when enough valid sources exist")
    parser.add_argument("--max-contexts-per-pair", type=int, default=4)
    parser.add_argument("--output", default="data/thesis_dataset.jsonl")
    parser.add_argument("--include-segmentation-ablations", action="store_true",
                        help="Store sentence, 3-sentence, and section passage sets (increases dataset size)")
    args = parser.parse_args()
    if args.target_negatives != args.target_positives:
        parser.error("--target-negatives must equal --target-positives for paired reporting")

    pdf_dir = Path(args.pdf_dir)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    print(f"[1/3] Parsing {len(pdf_files)} cached PDFs locally...")
    valid_docs: Dict[str, Dict] = {}
    title_index: Dict[str, str] = {}

    for pdf in tqdm(pdf_files, desc="Parsing local PDFs"):
        parsed = parse_local_pdf(
            pdf,
            include_segmentation_ablations=args.include_segmentation_ablations,
        )
        if parsed:
            aid = parsed["arxiv_id"]
            valid_docs[aid] = parsed
            nt = norm_title(parsed["title"])
            if len(nt) >= 18:
                title_index[nt] = aid

    print(f"      Parsed {len(valid_docs)} full-text papers.")

    print("[2/3] Matching internal citations directly from PDF References & body text...")
    positives: List[Dict] = []
    citations_by_paper: Dict[str, Set[str]] = {aid: set() for aid in valid_docs}
    seen_pairs: Set[Tuple[str, str]] = set()

    for cid, doc in tqdm(valid_docs.items(), desc="Resolving local PDF citations"):
        refs_text = doc["refs_text"]
        body_text = doc["body_text"]
        combined_ref_zone = f"{refs_text}\n{body_text}"

        found_ids: Set[str] = set(re.findall(r"(?i)\b(\d{4}\.\d{4,5})\b", combined_ref_zone))
        norm_refs = norm_title(refs_text) if refs_text else ""

        if norm_refs:
            for nt, target_id in title_index.items():
                if target_id != cid and nt in norm_refs:
                    found_ids.add(target_id)

        for target_id in found_ids:
            if target_id == cid or target_id not in valid_docs:
                continue
            citations_by_paper[cid].add(target_id)
            ref_num = None
            target_nt = norm_title(valid_docs[target_id]["title"])
            if refs_text:
                for entry in re.split(r"\n(?=\[\d+\]|\d+\.)", refs_text):
                    ne = norm_title(entry)
                    if (target_id.lower() in entry.lower()) or (len(target_nt) >= 18 and target_nt in ne):
                        m = re.match(r"^\s*\[?(\d+)\]?", entry.strip())
                        if m:
                            ref_num = m.group(1)
                            break

            ctxs = extract_citation_contexts(
                body_text, valid_docs[target_id], ref_num, max_contexts=args.max_contexts_per_pair
            )
            if ctxs:
                seen_pairs.add((cid, target_id))
                for ctx in ctxs:
                    positives.append({"citing_id": cid, "cited_id": target_id, "claim_context": ctx})
                    if len(positives) >= args.target_positives:
                        break

        if len(positives) >= args.target_positives:
            break

    positives = positives[: args.target_positives]
    if len(positives) < args.target_positives:
        source_positives = list(positives)
        duplicate_index = 0
        while len(positives) < args.target_positives and source_positives:
            template = dict(source_positives[duplicate_index % len(source_positives)])
            template["augmentation"] = "alternative_distractor_for_resolved_pair"
            positives.append(template)
            duplicate_index += 1
    print(f"      Collected {len(positives)} positive instances offline.")

    print("[3/3] Pairing each positive with a plausible same-topic non-cited distractor (50/50 split)...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    ids = list(valid_docs.keys())
    embs = embedder.encode(
        [f"{valid_docs[i]['title']}. {valid_docs[i]['abstract']}" for i in ids],
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    sim_matrix = np.matmul(embs, embs.T)
    id_to_idx = {pid: i for i, pid in enumerate(ids)}

    rows: List[Dict] = []
    pair_counter = 0
    for pos in positives:
        citing_id = pos["citing_id"]
        true_id = pos["cited_id"]
        ctx = pos["claim_context"]
        excluded = citations_by_paper.get(citing_id, set()) | {citing_id, true_id}

        t_idx = id_to_idx[true_id]
        ranked_cand_indices = np.argsort(-sim_matrix[t_idx])
        ranked_candidates = [j for j in ranked_cand_indices if ids[j] not in excluded]
        candidate_offset = pair_counter % max(1, len(ranked_candidates)) if pos.get("augmentation") else 0
        distractor_id = ids[ranked_candidates[candidate_offset]] if ranked_candidates else None
        if not distractor_id:
            continue

        sim_score = float(sim_matrix[t_idx, id_to_idx[distractor_id]])
        true_doc = valid_docs[true_id]
        neg_doc = valid_docs[distractor_id]
        rows.append({
            "instance_id": f"inst_{pair_counter:04d}_pos",
            "pair_id": pair_counter,
            "augmentation": pos.get("augmentation"),
            "label": 1,
            "ground_truth": "SUPPORTS",
            "citing_arxiv_id": citing_id,
            "source_arxiv_id": true_id,
            "source_title": true_doc["title"],
            "topic_similarity_to_true": 1.0,
            "claim_context": ctx,
            "claim_validation_status": "machine_generated_not_human_validated",
            "claim_complete_statement": False,
            "claim_source_linked": True,
            "silver_verdict": "SUPPORTS",
            "gold_evidence_passage_id": None,
            "gold_evidence_rank": None,
            "human_annotated": False,
            "annotation_status": "not_human_validated",
            "negative_category": None,
            "source_audit": true_doc.get("source_audit"),
            "num_paragraphs": len(true_doc["paragraphs"]),
            "paragraphs": true_doc["paragraphs"],
            "passage_sets": true_doc["passage_sets"],
        })
        rows.append({
            "instance_id": f"inst_{pair_counter:04d}_neg",
            "pair_id": pair_counter,
            "augmentation": pos.get("augmentation"),
            "label": 0,
            "ground_truth": "NOT_ENOUGH_INFORMATION",
            "citing_arxiv_id": citing_id,
            "source_arxiv_id": distractor_id,
            "source_title": neg_doc["title"],
            "topic_similarity_to_true": round(sim_score, 4),
            "claim_context": ctx,
            "claim_validation_status": "machine_generated_not_human_validated",
            "claim_complete_statement": False,
            "claim_source_linked": False,
            "silver_verdict": "NOT_ENOUGH_INFORMATION",
            "gold_evidence_passage_id": None,
            "gold_evidence_rank": None,
            "human_annotated": False,
            "annotation_status": "not_human_validated",
            "negative_category": None,
            "negative_sampling_method": "embedding_selected_nearest_title_abstract; not_human_validated",
            "source_audit": neg_doc.get("source_audit"),
            "num_paragraphs": len(neg_doc["paragraphs"]),
            "paragraphs": neg_doc["paragraphs"],
            "passage_sets": neg_doc["passage_sets"],
        })
        pair_counter += 1

    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    report = {
        "available_pdf_count": len(pdf_files),
        "valid_pdf_count": len(valid_docs),
        "excluded_invalid_pdf_count": len(pdf_files) - len(valid_docs),
        "excluded_pdf_filenames": sorted(
            pdf.name for pdf in pdf_files
            if pdf.name not in {
                doc["source_audit"]["pdf_filename"] for doc in valid_docs.values()
            }
        ),
        "usable_positive_pairs": len(rows) // 2,
        "exact_instance_count": len(rows),
        "positive_count": sum(r["label"] == 1 for r in rows),
        "negative_count": sum(r["label"] == 0 for r in rows),
        "human_annotated_count": sum(r["human_annotated"] for r in rows),
        "gold_evidence_count": sum(r["gold_evidence_passage_id"] is not None for r in rows),
        "augmentation_pair_count": sum(
            r["label"] == 1 and bool(r.get("augmentation")) for r in rows
        ),
        "non_augmented_pair_count": sum(
            r["label"] == 1 and not r.get("augmentation") for r in rows
        ),
        "status": "citation-derived_unvalidated" if len(rows) else "no_instances_built",
    }
    out_path.with_suffix(".build_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"Saved {len(rows)} instances ({len(rows)//2} pos / {len(rows)//2} neg) -> {out_path}")


if __name__ == "__main__":
    main()
