"""
RAG Text Cleaner — v2
Reads .txt files from `data/rag_documents/cleaned/`
Writes cleaned versions to `data/rag_documents/cleaned_v2/`
Preserves the full subfolder structure.

Handles: academic papers, books, business reports, recipes,
         nutrition references, Arabic-English mixed content.
"""

import re
import os
import unicodedata
from pathlib import Path


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
INPUT_ROOT  = Path("data/rag_documents/cleaned")
OUTPUT_ROOT = Path("data/rag_documents/cleaned_v2")


# ─────────────────────────────────────────────
# CLEANING STEPS
# ─────────────────────────────────────────────

def normalize_unicode(text: str) -> str:
    """Normalize unicode characters (ligatures, smart quotes, dashes, etc.)."""
    text = unicodedata.normalize("NFKC", text)
    # Smart quotes → straight
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    # Em/en dash → hyphen
    text = text.replace("\u2014", " - ").replace("\u2013", " - ")
    # Bullet variants → dash
    text = re.sub(r"[•·▪▸►✓✔◦‣]", "-", text)
    return text


def remove_page_numbers(text: str) -> str:
    """
    Remove standalone page numbers in many forms:
      - lone digits on a line:  "42"
      - Page N / Page N of M
      - "- 42 -"  or  "— 42 —"
      - Roman numerals on their own line: "iv", "XIV"
    """
    # "Page 42" / "page 42 of 100" / "PAGE 42"
    text = re.sub(r"(?im)^\s*page\s+\d+(\s+of\s+\d+)?\s*$", "", text)
    # Centered dashes around number:  "- 42 -"
    text = re.sub(r"(?m)^\s*[-–—]\s*\d+\s*[-–—]\s*$", "", text)
    # Lone digit(s) on a line (1–4 digits, not inside a sentence)
    text = re.sub(r"(?m)^\s*\d{1,4}\s*$", "", text)
    # Roman numerals on their own line (i, ii, iii … xxiv, etc.)
    text = re.sub(
        r"(?im)^\s*M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})\s*$",
        "",
        text,
    )
    return text


def remove_table_of_contents(text: str) -> str:
    """
    Detect and strip a TOC block.
    Patterns: lines ending with  "........ 12"  or  "  12"  repeatedly.
    Also strips "Table of Contents" / "Contents" header lines.
    """
    # Header variants
    text = re.sub(
        r"(?im)^\s*(table\s+of\s+)?contents\s*$", "", text
    )
    # TOC entry lines:  "Some Title ......... 23"
    text = re.sub(
        r"(?m)^.{3,80}[.\s]{4,}\s*\d{1,4}\s*$", "", text
    )
    # TOC entry lines with tab:  "Some Title\t23"
    text = re.sub(r"(?m)^.{3,80}\t\s*\d{1,4}\s*$", "", text)
    return text


def remove_headers_footers(text: str) -> str:
    """
    Remove repetitive running headers/footers.
    Heuristic: short lines (<= 80 chars) that appear 3+ times verbatim.
    """
    lines = text.splitlines()
    from collections import Counter
    stripped = [l.strip() for l in lines]
    freq = Counter(s for s in stripped if 5 < len(s) <= 80)
    repeated = {s for s, c in freq.items() if c >= 3}
    cleaned = [l for l in lines if l.strip() not in repeated]
    return "\n".join(cleaned)


def remove_artifacts(text: str) -> str:
    """Remove common PDF-extraction noise."""
    # Hyphenation across lines: "nutri-\ntion" → "nutrition"
    text = re.sub(r"-\n(\w)", lambda m: m.group(1), text)
    # URLs / DOIs that are unlikely RAG-useful
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"doi:\s*\S+", "", text, flags=re.IGNORECASE)
    # Copyright / license lines
    text = re.sub(
        r"(?im)^.*?(©|copyright|\(c\)|all rights reserved|licensed under).*$",
        "",
        text,
    )
    # Watermark-style repeated short caps lines  "CONFIDENTIAL" / "DRAFT"
    text = re.sub(r"(?m)^\s*(CONFIDENTIAL|DRAFT|INTERNAL USE ONLY)\s*$", "", text)
    # Form-feed characters
    text = text.replace("\f", "\n")
    return text


def clean_whitespace(text: str) -> str:
    """
    - Collapse 3+ blank lines → 2 (paragraph separator)
    - Strip trailing spaces on each line
    - Ensure single space between words (but preserve indentation intent)
    """
    # Trailing spaces
    text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)
    # Multiple spaces inside a line (not leading) → single space
    text = re.sub(r"(?<=\S) {2,}", " ", text)
    # 3+ consecutive blank lines → 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Leading/trailing whitespace for the whole document
    text = text.strip()
    return text


def fix_sentence_breaks(text: str) -> str:
    """
    Re-join lines that were split mid-sentence (common in PDF extractions).
    A line break is a mid-sentence break when:
      - current line does NOT end with  . ! ? : ; or a digit (list item)
      - next line starts with a lowercase letter
    """
    text = re.sub(
        r"(?<![.!?:;\d\n])\n(?=[a-z])",
        " ",
        text,
    )
    return text


def clean_references_section(text: str) -> str:
    """
    Optionally strip the References / Bibliography section at the end
    (academic papers). Keeps it if <= 5 lines (might be inline citations).
    """
    # Find where references section starts
    match = re.search(
        r"(?im)^\s*(references|bibliography|works cited)\s*$", text
    )
    if match:
        ref_text = text[match.start():]
        ref_lines = [l for l in ref_text.splitlines() if l.strip()]
        if len(ref_lines) > 5:
            text = text[: match.start()].rstrip() + "\n"
    return text


def normalize_section_headings(text: str) -> str:
    """
    Ensure section headings are clearly separated with blank lines.
    Heuristic: a short ALL-CAPS line or a line ending with ':' that is
    not inside a sentence gets blank lines around it.
    """
    def add_spacing(m):
        heading = m.group(0).strip()
        return f"\n\n{heading}\n\n"

    # ALL-CAPS short headings (3–60 chars, no trailing period)
    text = re.sub(
        r"(?m)^[ \t]*([A-Z][A-Z0-9 /&,\-]{2,58})[ \t]*$",
        add_spacing,
        text,
    )
    # Re-clean excess blank lines created above
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def remove_mcq_answer_noise(text: str) -> str:
    """
    For MCQ / FAQ files: strip answer-key lines like "Answer: A" or "(A)" alone.
    Keep the question and choices but remove stray answer markers if isolated.
    """
    text = re.sub(r"(?im)^\s*answer\s*[:\-]?\s*[a-d]\s*$", "", text)
    return text


# ─────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────

PIPELINE = [
    ("Normalize unicode",          normalize_unicode),
    ("Remove page numbers",        remove_page_numbers),
    ("Remove table of contents",   remove_table_of_contents),
    ("Remove headers/footers",     remove_headers_footers),
    ("Remove PDF artifacts",       remove_artifacts),
    ("Fix mid-sentence linebreaks",fix_sentence_breaks),
    ("Clean references section",   clean_references_section),
    ("Normalize section headings", normalize_section_headings),
    ("Remove MCQ answer noise",    remove_mcq_answer_noise),
    ("Clean whitespace",           clean_whitespace),   # always last
]


def clean_text(text: str, filename: str = "") -> str:
    for name, fn in PIPELINE:
        try:
            text = fn(text)
        except Exception as e:
            print(f"    ⚠  Step '{name}' failed on {filename}: {e}")
    return text


# ─────────────────────────────────────────────
# FILE I/O
# ─────────────────────────────────────────────

def process_file(src: Path, dst: Path) -> dict:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        raw = src.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return {"file": str(src), "status": "error", "detail": str(e)}

    original_len = len(raw)
    cleaned = clean_text(raw, src.name)
    cleaned_len = len(cleaned)

    dst.write_text(cleaned, encoding="utf-8")
    reduction = round((1 - cleaned_len / max(original_len, 1)) * 100, 1)
    return {
        "file": src.name,
        "original_chars": original_len,
        "cleaned_chars": cleaned_len,
        "reduction_%": reduction,
        "status": "ok",
    }


def run(input_root: Path = INPUT_ROOT, output_root: Path = OUTPUT_ROOT):
    txt_files = sorted(input_root.rglob("*.txt"))
    if not txt_files:
        print(f"No .txt files found under {input_root}")
        return

    print(f"Found {len(txt_files)} files  →  writing to {output_root}\n")
    results = []
    for src in txt_files:
        rel = src.relative_to(input_root)
        dst = output_root / rel
        print(f"  Processing: {rel}")
        result = process_file(src, dst)
        results.append(result)
        if result["status"] == "ok":
            print(f"    ✓  {result['original_chars']:,} → {result['cleaned_chars']:,} chars  ({result['reduction_%']}% reduction)")
        else:
            print(f"    ✗  ERROR: {result['detail']}")

    ok    = sum(1 for r in results if r["status"] == "ok")
    error = len(results) - ok
    avg_red = round(
        sum(r.get("reduction_%", 0) for r in results if r["status"] == "ok") / max(ok, 1),
        1,
    )
    print(f"\n{'─'*50}")
    print(f"Done.  {ok} cleaned  |  {error} errors  |  avg reduction {avg_red}%")
    print(f"Output: {output_root.resolve()}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Clean PDF-extracted .txt files for RAG.")
    parser.add_argument(
        "--input",  default=str(INPUT_ROOT),
        help=f"Root folder of cleaned txts (default: {INPUT_ROOT})",
    )
    parser.add_argument(
        "--output", default=str(OUTPUT_ROOT),
        help=f"Root folder for cleaned_v2 output (default: {OUTPUT_ROOT})",
    )
    args = parser.parse_args()
    run(Path(args.input), Path(args.output))
