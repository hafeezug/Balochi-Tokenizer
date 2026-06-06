"""
Balochi Data Cleaning Pipeline
================================
Reads all .txt and .docx files from the current directory,
removes all unwanted text (English/Latin script, numbers, URLs, etc.),
and keeps ONLY text written in Arabic/Balochi script.

Output:
  - cleaned_data/           → Individual cleaned files
  - cleaned_data/combined/  → Single combined cleaned corpus
  - cleaned_data/cleaning_report.txt → Detailed statistics

Usage:
    python data_cleaning_pipeline.py
"""

import os
import re
import glob
import sys
import time
from datetime import datetime


# ─── Configuration ───────────────────────────────────────────────────────────
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(DATA_DIR, "cleaned_data")
COMBINED_DIR = os.path.join(OUTPUT_DIR, "combined")
REPORT_FILE = os.path.join(OUTPUT_DIR, "cleaning_report.txt")

# Minimum number of Arabic/Balochi characters a file must have after cleaning
# to be included in the output (filters out files that were entirely English)
MIN_CLEAN_CHARS = 20

# ─── Unicode Ranges for Arabic / Balochi Script ─────────────────────────────
# Arabic script block:         U+0600 – U+06FF
# Arabic Supplement:           U+0750 – U+077F
# Arabic Extended-A:           U+08A0 – U+08FF
# Arabic Extended-B:           U+0870 – U+089F
# Arabic Presentation Forms-A: U+FB50 – U+FDFF
# Arabic Presentation Forms-B: U+FE70 – U+FEFF
# These cover standard Arabic, Urdu, Balochi, and Persian characters including
# diacritics (zabar, zer, pesh, tashdid, etc.)

# Pattern matching Arabic/Balochi script characters (including diacritics)
ARABIC_SCRIPT_PATTERN = re.compile(
    r'[\u0600-\u06FF\u0750-\u077F\u0870-\u089F\u08A0-\u08FF'
    r'\uFB50-\uFDFF\uFE70-\uFEFF]'
)

# Pattern matching characters to REMOVE:
# - Basic Latin letters (A-Z, a-z)
# - Extended Latin characters (accented, etc.)
# - ASCII digits (0-9)
# - Arabic-Indic digits (٠-٩)
# - Extended Arabic-Indic digits (۰-۹) — keeping these since they are used in Balochi
# - Various Latin-based Unicode blocks
LATIN_AND_DIGITS_PATTERN = re.compile(
    r'[A-Za-z0-9'
    r'\u00C0-\u024F'   # Latin Extended-A and B (accented characters)
    r'\u1E00-\u1EFF'   # Latin Extended Additional
    r'\u2C60-\u2C7F'   # Latin Extended-C
    r'\uA720-\uA7FF'   # Latin Extended-D
    r'\u0080-\u00BF'   # Latin-1 Supplement (control and symbols)
    r']+'
)

# URL pattern (http, https, www, ftp)
URL_PATTERN = re.compile(
    r'https?://[^\s<>\"\']+|www\.[^\s<>\"\']+|ftp://[^\s<>\"\']+'
)

# Email pattern
EMAIL_PATTERN = re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}')

# Punctuation and symbols to REMOVE (non-Arabic punctuation)
# Keep: Arabic comma ، Arabic semicolon ؛ Arabic question mark ؟
# Keep: Standard punctuation used in Balochi: . : ! ۔ ؟ ، ؛ « » ( ) [ ]
# Remove: #, $, %, ^, &, *, +, =, <, >, |, \, ~, `, @, {, }
UNWANTED_SYMBOLS_PATTERN = re.compile(
    r'[#$%^&*+=<>|\\~`@{}_/;]'
)


# ─── File Reading ────────────────────────────────────────────────────────────
def read_txt_file(filepath):
    """Read a .txt file trying multiple encodings."""
    encodings = ["utf-8", "utf-8-sig", "utf-16", "utf-16-le", "utf-16-be",
                 "cp1256", "cp1252", "latin-1"]
    for enc in encodings:
        try:
            with open(filepath, "r", encoding=enc) as f:
                content = f.read()
                # Quick sanity check: if we got mostly null bytes, wrong encoding
                if '\x00' in content[:100] and enc not in ('utf-16', 'utf-16-le', 'utf-16-be'):
                    continue
                return content
        except (UnicodeDecodeError, UnicodeError):
            continue
    print(f"  ⚠ Could not decode: {os.path.basename(filepath)}")
    return ""


def read_docx_file(filepath):
    """Read a .docx file and return its text."""
    try:
        from docx import Document
        doc = Document(filepath)
        paragraphs = []
        for para in doc.paragraphs:
            if para.text.strip():
                paragraphs.append(para.text)
        return "\n".join(paragraphs)
    except ImportError:
        print("  ⚠ python-docx not installed. Install with: pip install python-docx")
        return ""
    except Exception as e:
        print(f"  ⚠ Error reading {os.path.basename(filepath)}: {e}")
        return ""


# ─── Text Cleaning Functions ────────────────────────────────────────────────

def contains_arabic_script(text):
    """Check if text contains any Arabic/Balochi script characters."""
    return bool(ARABIC_SCRIPT_PATTERN.search(text))


def count_arabic_chars(text):
    """Count the number of Arabic/Balochi script characters."""
    return len(ARABIC_SCRIPT_PATTERN.findall(text))


def clean_text(text):
    """
    Clean text by removing everything except Arabic/Balochi script.

    Steps:
    1. Remove URLs
    2. Remove email addresses
    3. Remove English/Latin characters and digits
    4. Remove unwanted symbols
    5. Clean up Arabic-Indic digits (٠-٩) — remove these too
    6. Keep Extended Arabic-Indic digits (۰-۹) used in Balochi/Urdu — configurable
    7. Remove lines that have no Arabic script content
    8. Normalize whitespace
    """

    # Step 1: Remove URLs
    text = URL_PATTERN.sub(' ', text)

    # Step 2: Remove email addresses
    text = EMAIL_PATTERN.sub(' ', text)

    # Step 3: Remove all Latin/English characters and ASCII digits
    text = LATIN_AND_DIGITS_PATTERN.sub(' ', text)

    # Step 4: Remove unwanted symbols
    text = UNWANTED_SYMBOLS_PATTERN.sub(' ', text)

    # Step 5: Remove Arabic-Indic digits (٠١٢٣٤٥٦٧٨٩)
    text = re.sub(r'[٠-٩]', ' ', text)

    # Step 6: Remove Extended Arabic-Indic digits (۰۱۲۳۴۵۶۷۸۹) used in Urdu/Balochi
    # NOTE: Comment out the next line if you want to KEEP these digits
    text = re.sub(r'[۰-۹]', ' ', text)

    # Step 7: Remove remaining standalone ASCII digits that might have survived
    text = re.sub(r'\d+', ' ', text)

    # Step 8: Remove common non-Arabic punctuation leftovers
    # Keep: ، ؛ ؟ ۔ period(.) exclamation(!) colon(:) quotes("") parens
    # Remove: dashes (—, –, -), underscores, pipes, etc. when standalone
    text = re.sub(r'[—–\-]+', ' ', text)

    # Step 9: Process line by line — remove lines with no Arabic content
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Check if line has meaningful Arabic/Balochi content
        if contains_arabic_script(line):
            # Remove any remaining isolated Latin punctuation artifacts
            # but keep Arabic punctuation
            cleaned_lines.append(line)

    text = '\n'.join(cleaned_lines)

    # Step 10: Normalize whitespace
    # Collapse multiple spaces into one
    text = re.sub(r'[ \t]+', ' ', text)
    # Collapse multiple newlines (3+) into double newline
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Clean up spaces around newlines
    text = re.sub(r' *\n *', '\n', text)

    return text.strip()


# ─── Statistics ──────────────────────────────────────────────────────────────
def compute_stats(original, cleaned):
    """Compute cleaning statistics."""
    orig_chars = len(original)
    clean_chars = len(cleaned)
    arabic_chars = count_arabic_chars(cleaned)
    removed_chars = orig_chars - clean_chars

    return {
        "original_chars": orig_chars,
        "cleaned_chars": clean_chars,
        "arabic_chars": arabic_chars,
        "removed_chars": removed_chars,
        "removal_pct": (removed_chars / orig_chars * 100) if orig_chars > 0 else 0,
        "original_lines": original.count('\n') + 1,
        "cleaned_lines": cleaned.count('\n') + 1 if cleaned else 0,
    }


# ─── Main Pipeline ──────────────────────────────────────────────────────────
def main():
    # Fix Windows console encoding for Arabic text
    if sys.platform == "win32":
        import io
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding='utf-8', errors='replace'
        )

    start_time = time.time()

    print("=" * 70)
    print("  Balochi Data Cleaning Pipeline")
    print("  (Balochi Data Safai Pipeline)")
    print("=" * 70)
    print(f"\n  Data directory : {DATA_DIR}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Min clean chars : {MIN_CLEAN_CHARS}\n")

    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(COMBINED_DIR, exist_ok=True)

    # Find all files
    txt_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.txt")))
    docx_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.docx")))
    doc_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.doc")))

    # Exclude this script and other Python files from processing
    txt_files = [f for f in txt_files if not f.endswith('.py')]

    print(f"  Found {len(txt_files)} .txt files")
    print(f"  Found {len(docx_files)} .docx files")
    if doc_files:
        print(f"  Found {len(doc_files)} .doc files (skipped — convert to .docx first)")
    print()

    # Statistics tracking
    all_stats = []
    skipped_files = []
    failed_files = []
    combined_clean_text = []
    total_files = len(txt_files) + len(docx_files)
    processed = 0

    # ─── Process .txt files ──────────────────────────────────────────────
    print("-" * 70)
    print("  Processing .txt files...")
    print("-" * 70)

    for filepath in txt_files:
        basename = os.path.basename(filepath)
        processed += 1
        progress = f"[{processed}/{total_files}]"

        # Read file
        raw_text = read_txt_file(filepath)
        if not raw_text.strip():
            failed_files.append((basename, "Empty or unreadable"))
            print(f"  {progress} SKIP (empty): {basename}")
            continue

        # Clean text
        cleaned = clean_text(raw_text)

        # Check if enough content remains
        arabic_count = count_arabic_chars(cleaned)
        if arabic_count < MIN_CLEAN_CHARS:
            skipped_files.append((basename, f"Only {arabic_count} Arabic chars after cleaning"))
            print(f"  {progress} SKIP (no Balochi): {basename}")
            continue

        # Compute stats
        stats = compute_stats(raw_text, cleaned)
        stats["filename"] = basename
        stats["type"] = "txt"
        all_stats.append(stats)

        # Save cleaned file
        out_path = os.path.join(OUTPUT_DIR, basename)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(cleaned)

        # Add to combined
        combined_clean_text.append(cleaned)

        pct = stats["removal_pct"]
        print(f"  {progress} ✓ {basename}")
        print(f"         {stats['original_chars']:,} → {stats['cleaned_chars']:,} chars "
              f"({pct:.1f}% removed)")

    # ─── Process .docx files ─────────────────────────────────────────────
    print()
    print("-" * 70)
    print("  Processing .docx files...")
    print("-" * 70)

    for filepath in docx_files:
        basename = os.path.basename(filepath)
        processed += 1
        progress = f"[{processed}/{total_files}]"

        # Read file
        raw_text = read_docx_file(filepath)
        if not raw_text.strip():
            failed_files.append((basename, "Empty or unreadable"))
            print(f"  {progress} SKIP (empty): {basename}")
            continue

        # Clean text
        cleaned = clean_text(raw_text)

        # Check if enough content remains
        arabic_count = count_arabic_chars(cleaned)
        if arabic_count < MIN_CLEAN_CHARS:
            skipped_files.append((basename, f"Only {arabic_count} Arabic chars after cleaning"))
            print(f"  {progress} SKIP (no Balochi): {basename}")
            continue

        # Compute stats
        stats = compute_stats(raw_text, cleaned)
        stats["filename"] = basename
        stats["type"] = "docx"
        all_stats.append(stats)

        # Save cleaned file (as .txt)
        out_name = os.path.splitext(basename)[0] + ".txt"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(cleaned)

        # Add to combined
        combined_clean_text.append(cleaned)

        pct = stats["removal_pct"]
        print(f"  {progress} ✓ {basename}")
        print(f"         {stats['original_chars']:,} → {stats['cleaned_chars']:,} chars "
              f"({pct:.1f}% removed)")

    # ─── Save combined corpus ────────────────────────────────────────────
    print()
    print("-" * 70)
    print("  Saving combined corpus...")
    print("-" * 70)

    combined = "\n\n".join(combined_clean_text)
    combined_path = os.path.join(COMBINED_DIR, "balochi_clean_corpus.txt")
    with open(combined_path, "w", encoding="utf-8") as f:
        f.write(combined)

    combined_chars = len(combined)
    combined_arabic = count_arabic_chars(combined)
    combined_words = len(combined.split())

    print(f"  Combined corpus saved: {combined_path}")
    print(f"  Total characters: {combined_chars:,}")
    print(f"  Arabic/Balochi characters: {combined_arabic:,}")
    print(f"  Approximate words: {combined_words:,}")

    # ─── Generate Report ─────────────────────────────────────────────────
    elapsed = time.time() - start_time

    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("  BALOCHI DATA CLEANING REPORT")
    report_lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("=" * 70)
    report_lines.append("")
    report_lines.append(f"  Source directory: {DATA_DIR}")
    report_lines.append(f"  Output directory: {OUTPUT_DIR}")
    report_lines.append(f"  Processing time: {elapsed:.1f} seconds")
    report_lines.append("")
    report_lines.append("-" * 70)
    report_lines.append("  SUMMARY")
    report_lines.append("-" * 70)
    report_lines.append(f"  Total files found:     {total_files}")
    report_lines.append(f"  Files cleaned:         {len(all_stats)}")
    report_lines.append(f"  Files skipped:         {len(skipped_files)}")
    report_lines.append(f"  Files failed:          {len(failed_files)}")
    report_lines.append("")

    total_orig = sum(s["original_chars"] for s in all_stats)
    total_clean = sum(s["cleaned_chars"] for s in all_stats)
    total_arabic = sum(s["arabic_chars"] for s in all_stats)
    total_removed = sum(s["removed_chars"] for s in all_stats)

    report_lines.append(f"  Total original characters:   {total_orig:>15,}")
    report_lines.append(f"  Total cleaned characters:    {total_clean:>15,}")
    report_lines.append(f"  Total Arabic/Balochi chars:  {total_arabic:>15,}")
    report_lines.append(f"  Total characters removed:    {total_removed:>15,}")
    if total_orig > 0:
        report_lines.append(f"  Overall removal percentage:  {total_removed/total_orig*100:>14.1f}%")
    report_lines.append("")
    report_lines.append(f"  Combined corpus:")
    report_lines.append(f"    Characters: {combined_chars:,}")
    report_lines.append(f"    Arabic/Balochi characters: {combined_arabic:,}")
    report_lines.append(f"    Approximate words: {combined_words:,}")
    report_lines.append("")

    if skipped_files:
        report_lines.append("-" * 70)
        report_lines.append("  SKIPPED FILES (no/insufficient Balochi content)")
        report_lines.append("-" * 70)
        for fname, reason in skipped_files:
            report_lines.append(f"  • {fname}")
            report_lines.append(f"    Reason: {reason}")
        report_lines.append("")

    if failed_files:
        report_lines.append("-" * 70)
        report_lines.append("  FAILED FILES")
        report_lines.append("-" * 70)
        for fname, reason in failed_files:
            report_lines.append(f"  • {fname}")
            report_lines.append(f"    Reason: {reason}")
        report_lines.append("")

    report_lines.append("-" * 70)
    report_lines.append("  PER-FILE STATISTICS")
    report_lines.append("-" * 70)
    report_lines.append(f"  {'File':<50} {'Original':>10} {'Cleaned':>10} {'Removed%':>8}")
    report_lines.append(f"  {'─'*50} {'─'*10} {'─'*10} {'─'*8}")

    # Sort by cleaned size (largest first)
    for s in sorted(all_stats, key=lambda x: x["cleaned_chars"], reverse=True):
        name = s["filename"][:48]
        report_lines.append(
            f"  {name:<50} {s['original_chars']:>10,} {s['cleaned_chars']:>10,} "
            f"{s['removal_pct']:>7.1f}%"
        )

    report_lines.append("")
    report_lines.append("=" * 70)
    report_lines.append("  END OF REPORT")
    report_lines.append("=" * 70)

    report_text = "\n".join(report_lines)

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(report_text)

    # ─── Print Summary ───────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  ✅ CLEANING COMPLETE")
    print("=" * 70)
    print(f"  Files cleaned:    {len(all_stats)}")
    print(f"  Files skipped:    {len(skipped_files)}")
    print(f"  Files failed:     {len(failed_files)}")
    print(f"  Time elapsed:     {elapsed:.1f} seconds")
    print()
    print(f"  Original chars:   {total_orig:,}")
    print(f"  Cleaned chars:    {total_clean:,}")
    print(f"  Removed:          {total_removed:,} ({total_removed/total_orig*100:.1f}%)" if total_orig > 0 else "")
    print()
    print(f"  Combined corpus:  {combined_path}")
    print(f"  Cleaning report:  {REPORT_FILE}")
    print()
    print(f"  Output files in:  {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
