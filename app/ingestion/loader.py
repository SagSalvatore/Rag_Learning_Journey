# app/ingestion/loader.py
# -----------------------------------------------------------
# Production-grade PDF loader for SEC 10-K reports
# TOC extraction: 4 cascading strategies
# Stack: PyMuPDF (fitz) + LangChain splitter
# -----------------------------------------------------------

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import fitz  # PyMuPDF
import os
import re


# -----------------------------------------------------------
# PATH UTILS
# -----------------------------------------------------------
def get_abs_path(rel_path: str) -> str:
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(base, rel_path))


# -----------------------------------------------------------
# STRATEGY 1: Embedded PDF bookmarks
# -----------------------------------------------------------
def _strategy_bookmarks(pdf: fitz.Document) -> list[dict]:
    raw_toc = pdf.get_toc()
    if not raw_toc:
        return []
    return [
        {"title": item[1].strip(), "page": item[2], "level": item[0] - 1}
        for item in raw_toc
        if len(item[1].strip()) > 3
    ]


# -----------------------------------------------------------
# STRATEGY 2: Block-based dict scan (x0 indent detection)
# Collects ALL lines across ALL blocks per page before scanning
# so toc_started flag persists across block boundaries
# -----------------------------------------------------------
def _strategy_block_scan(pdf: fitz.Document, max_pages: int = 6) -> list[dict]:
    toc = []
    toc_started = False
    toc_page_found = -1
    page_number_pattern = re.compile(r"(\d{1,3})\s*$")

    for page_idx in range(min(max_pages, len(pdf))):
        page = pdf[page_idx]
        blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]

        # Flatten ALL lines from ALL blocks on this page
        all_lines: list[tuple[str, float]] = []
        for block in blocks:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                if not spans:
                    continue
                # Join all spans in a line
                line_text = " ".join(s["text"].strip() for s in spans).strip()
                x0 = spans[0]["origin"][0]
                if line_text:
                    all_lines.append((line_text, x0))

        # Scan lines with persistent state
        for line_text, x0 in all_lines:

            # --- Detect TOC header (flexible matching) ---
            normalized = re.sub(r"\s+", " ", line_text).upper()
            if "TABLE OF CONTENTS" in normalized or "TABLE OF\nCONTENTS" in normalized:
                toc_started = True
                toc_page_found = page_idx
                continue

            # Also activate if we're past the page where TOC was found
            # and we haven't found any entries yet (handles split-page case)
            if not toc_started:
                continue

            # Stop if we've moved 2+ pages past TOC page without entries
            if toc_page_found >= 0 and page_idx > toc_page_found + 2 and len(toc) == 0:
                return []

            # Check for page number
            page_match = page_number_pattern.search(line_text)

            # Skip all-caps section banners without a page number
            if line_text.upper() == line_text and not page_match:
                continue

            # Must have a page number at the end
            if not page_match:
                continue

            page_num = int(page_match.group(1))
            title = line_text[:page_match.start()].strip()

            # Clean dotted leaders and trailing dots/spaces
            title = re.sub(r"[.\s]{2,}$", "", title).strip()
            title = re.sub(r"\s+", " ", title)  # normalize spaces

            if len(title) < 4:
                continue
            if re.match(r"^\d+$", title):
                continue

            # x0-based indent level
            level = 0 if x0 < 80 else 1

            toc.append({"title": title, "page": page_num, "level": level})

            if len(toc) >= 50:
                break

    return toc


# -----------------------------------------------------------
# STRATEGY 3: Plain text scan (fallback when dict mode fails)
# Uses simple get_text() string and scans for TOC pattern
# -----------------------------------------------------------
def _strategy_plain_text_scan(pdf: fitz.Document, max_pages: int = 6) -> list[dict]:
    toc = []
    toc_started = False
    page_number_pattern = re.compile(r"^(.+?)\s{2,}(\d{1,3})\s*$")

    for page_idx in range(min(max_pages, len(pdf))):
        page_text = pdf[page_idx].get_text("text")
        lines = page_text.split("\n")

        for line in lines:
            line = line.strip()

            normalized = re.sub(r"\s+", " ", line).upper()
            if "TABLE OF CONTENTS" in normalized:
                toc_started = True
                continue

            if not toc_started:
                continue

            # This pattern requires 2+ spaces between title and page number
            # which is how PDF text extractors render right-aligned page nums
            match = page_number_pattern.match(line)
            if match:
                title = re.sub(r"[.\s]{2,}$", "", match.group(1)).strip()
                page_num = int(match.group(2))

                if len(title) >= 4 and not re.match(r"^\d+$", title):
                    toc.append({"title": title, "page": page_num, "level": 0})

            if len(toc) >= 50:
                break

    return toc


# -----------------------------------------------------------
# STRATEGY 4: Page 0-5 brute force with loose pattern
# Last resort — catches even weirdly formatted TOCs
# -----------------------------------------------------------
def _strategy_brute_force(pdf: fitz.Document, max_pages: int = 6) -> list[dict]:
    toc = []
    toc_started = False

    # Loose: any line ending in 1-3 digits after at least 5 chars of title
    loose_pattern = re.compile(r"^([A-Za-z].{4,}?)\s+(\d{1,3})\s*$")

    for page_idx in range(min(max_pages, len(pdf))):
        page_text = pdf[page_idx].get_text("text")

        if "TABLE OF CONTENTS" in page_text.upper():
            toc_started = True

        if not toc_started:
            continue

        for line in page_text.split("\n"):
            line = line.strip()
            match = loose_pattern.match(line)
            if match:
                title = re.sub(r"[.\s]{2,}$", "", match.group(1)).strip()
                page_num = int(match.group(2))
                if len(title) >= 5:
                    toc.append({"title": title, "page": page_num, "level": 0})

            if len(toc) >= 50:
                break

    return toc


# -----------------------------------------------------------
# MASTER TOC EXTRACTOR — runs all 4 strategies in order
# -----------------------------------------------------------
def extract_toc(path: str) -> list[dict]:
    """
    Cascading TOC extraction: tries 4 strategies, returns first that works.
    """
    pdf = fitz.open(path)

    print("  [TOC] Trying Strategy 1: Embedded PDF bookmarks...")
    result = _strategy_bookmarks(pdf)
    if result:
        print(f"  [TOC] ✅ Strategy 1 succeeded → {len(result)} entries")
        return result

    print("  [TOC] Trying Strategy 2: Block-based dict scan...")
    result = _strategy_block_scan(pdf, max_pages=6)
    if result:
        print(f"  [TOC] ✅ Strategy 2 succeeded → {len(result)} entries")
        return result

    print("  [TOC] Trying Strategy 3: Plain text multi-space scan...")
    result = _strategy_plain_text_scan(pdf, max_pages=6)
    if result:
        print(f"  [TOC] ✅ Strategy 3 succeeded → {len(result)} entries")
        return result

    print("  [TOC] Trying Strategy 4: Brute-force loose pattern...")
    result = _strategy_brute_force(pdf, max_pages=6)
    if result:
        print(f"  [TOC] ✅ Strategy 4 succeeded → {len(result)} entries")
        return result

    print("  [TOC] ⚠️  All strategies failed — no TOC extracted")
    return []


# -----------------------------------------------------------
# FULL TEXT EXTRACTION
# -----------------------------------------------------------
def load_full_text(path: str, max_pages: int = 150) -> str:
    pdf = fitz.open(path)
    total_pages = min(max_pages, len(pdf))
    texts = []

    for i in range(total_pages):
        page_text = pdf[i].get_text("text")
        if page_text.strip():
            texts.append(page_text)

    return "\n\n".join(texts)


# -----------------------------------------------------------
# DEBUG HELPER
# Run: python -c "from loader import debug_toc_blocks; debug_toc_blocks('../../data/documents/Mcdonalds_report/mcd-20251231.pdf')"
# -----------------------------------------------------------
def debug_toc_blocks(path: str, max_pages: int = 4) -> None:
    """
    Prints every text line with its x0 coordinate for first N pages.
    Use output to determine correct x0 threshold for indent detection
    and to see exactly how 'TABLE OF CONTENTS' appears in raw extraction.
    """
    abs_path = get_abs_path(path)
    pdf = fitz.open(abs_path)

    for page_idx in range(min(max_pages, len(pdf))):
        print(f"\n{'='*25} PAGE {page_idx} {'='*25}")
        blocks = pdf[page_idx].get_text("dict")["blocks"]
        for block in blocks:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                if not spans:
                    continue
                x0 = spans[0]["origin"][0]
                text = " ".join(s["text"] for s in spans).strip()
                if text:
                    print(f"  x0={x0:6.1f} | '{text[:100]}'")


# -----------------------------------------------------------
# MAIN LOADER
# -----------------------------------------------------------
def load_documents(path: str) -> list[Document]:
    abs_path = get_abs_path(path)

    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"File not found: {abs_path}")

    print(f"\n✅ Using file: {abs_path}\n")

    # ---- TOC ----
    toc = extract_toc(abs_path)

    print("\n========== CLEAN TOC ==========")
    if toc:
        for entry in toc:
            indent = "    " if entry["level"] == 1 else ""
            marker = "├─" if entry["level"] == 1 else "◆"
            print(f"  {marker} {indent}{entry['title']} → Page {entry['page']}")
    else:
        print("  ⚠️  No TOC extracted")
    print("================================\n")

    # ---- Full text ----
    text = load_full_text(abs_path)

    if not text.strip():
        raise ValueError(
            f"No text extracted from {abs_path}.\n"
            "PDF may be scanned/image-only → use docling with do_ocr=True."
        )

    print(f"✅ Extracted {len(text):,} characters from PDF")

    # ---- Build Document with metadata ----
    raw_doc = Document(
        page_content=text,
        metadata={
            "source": abs_path,
            "filename": os.path.basename(abs_path),
            "toc_entries": len(toc),
            "toc": [e["title"] for e in toc],
            "toc_structured": toc,
        }
    )

    # ---- Chunk ----
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""],
        length_function=len,
        is_separator_regex=False,
    )

    chunks = splitter.split_documents([raw_doc])
    print(f"✅ Split into {len(chunks)} chunks\n")

    return chunks


# -----------------------------------------------------------
# ENTRY POINT
# -----------------------------------------------------------
if __name__ == "__main__":
    FILE_PATH = "../../data/documents/Mcdonalds_report/mcd-20251231.pdf"

    docs = load_documents(FILE_PATH)

    print(f"Total chunks: {len(docs)}\n")

    print("--- Sample Chunk 1 ---")
    print(docs[0].page_content[:300])
    print("\n--- Metadata (preview) ---")
    print({k: v for k, v in docs[0].metadata.items() if k != "toc_structured"})
    print("=" * 50)

    print("\n--- Sample Chunk 2 ---")
    print(docs[1].page_content[:300])
    print("=" * 50)