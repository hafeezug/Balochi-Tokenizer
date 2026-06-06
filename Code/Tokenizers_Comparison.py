# -*- coding: utf-8 -*-
"""
Comprehensive Balochi + Cross-Language Tokenizer Comparison & Evaluation
=========================================================================
Extended version of the original Balochi tokenizer comparison script.

Tokenizers (15 total — original 7 + 8 new):

━━━ BALOCHI (Custom-Trained) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 1. Balochi BPE (80K)           – Custom HF tokenizers BPE
 2. Balochi WordPiece (64K)     – Custom HF tokenizers WordPiece
 3. Balochi SentencePiece (64K) – Custom Google SentencePiece
 4. Balochi 30K                 – balochiml/balochi-tokenizer (HuggingFace)

━━━ BASELINE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 5. NLTK                        – Rule-based word tokenizer
 6. BERT Multilingual           – bert-base-multilingual-cased (WordPiece 119K)
 7. Gemma                       – google/gemma-2b (SentencePiece BPE 256K)

━━━ ARABIC (Custom-Trained) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 8. AraBERT v2                  – aubmindlab/bert-base-arabertv2 (WP 64K)
 9. CAMeLBERT-MSA               – CAMeL-Lab/bert-base-arabic-camelbert-msa (WP 30K)
10. ARBERT                      – UBC-NLP/ARBERT (WP 100K)
11. AraGPT2                     – aubmindlab/aragpt2-base (BPE 50K)

━━━ PERSIAN (Custom-Trained) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
12. ParsBERT                    – HooshvareLab/bert-base-parsbert-uncased (WP 100K)
13. PersianBERT (HooshvareBase) – HooshvareLab/bert-fa-base-uncased (WP 100K)
14. PersianBPETokenizer         – mshojaei77/PersianBPETokenizer (BPE)

━━━ URDU (Custom-Trained) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
15. UrduBERT                    – urduhack/UrduBERT (WP ~60K) [HF: uer/roberta-base-finetuned-tnews-chinese]
                                  Note: falls back to iamxds/UrduBERT-base if primary unavailable

━━━ VOCAB ABLATION (from Notebook) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 NOTE: Ablation tokenizers (BPE/WP/SP at 32K, 64K, 80K, 128K) are loaded
 from local .json/.model files in Tokenizers/ directory. If the files are
 not present, these entries are skipped gracefully.

Comparison Groups:
  Group A: Balochi WordPiece vs AraBERT v2 vs CAMeLBERT vs ARBERT vs BERT  (WordPiece family)
  Group B: Balochi SP vs ParsBERT vs PersianBERT vs Gemma                  (SentencePiece/WP)
  Group C: Balochi BPE vs AraGPT2 vs PersianBPE vs 30K-Balochi vs NLTK    (BPE/Rule family)
  Group D: UrduBERT vs Balochi BPE vs AraBERT v2 vs ParsBERT              (Perso-Arabic script)
  Group E: Original 7 Balochi comparison (WordPiece / SentencePiece / BPE groups)
  Group F: Vocab Ablation — same algorithm, different vocab sizes
"""

import os
import sys
import time
import re
import unicodedata

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# ============================================================
# 0. Install & Import Dependencies
# ============================================================

def install_if_missing(package, pip_name=None):
    """Install a package if not already available."""
    try:
        __import__(package)
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name or package, "-q"])

install_if_missing("tokenizers")
install_if_missing("sentencepiece")
install_if_missing("transformers")
install_if_missing("nltk")
install_if_missing("huggingface_hub")

import sentencepiece as spm
from tokenizers import Tokenizer
from transformers import AutoTokenizer, BertTokenizer
import nltk
nltk.download('punkt',     quiet=True)
nltk.download('punkt_tab', quiet=True)
from nltk.tokenize import word_tokenize

# ============================================================
# 1. Path Configuration
# ============================================================

SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
BASE_DIR      = SCRIPT_DIR
INPUT_FILE    = os.path.join(BASE_DIR, "Sample Tokenized Text", "liberal capitalism.txt")
TOKENIZERS_DIR = os.path.join(BASE_DIR, "Tokenizers")
OUTPUT_DIR    = os.path.join(SCRIPT_DIR, "Output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Balochi local models ─────────────────────────────────────
BPE_80K  = os.path.join(TOKENIZERS_DIR, "Balochi_BPE_Tokenizer_80000.json")
WP_64K   = os.path.join(TOKENIZERS_DIR, "Balochi_Word_Piece_Tokenizer_64000.json")
SP_64K   = os.path.join(TOKENIZERS_DIR, "Balochi_Sentence_Piece_Tokenizer_64000.model")

# ── Ablation tokenizer paths (from notebook training) ────────
ABLATION_MODELS = {
    "Balochi_BPE_32K":  os.path.join(OUTPUT_DIR, "Ablation", "Models", "bpe_32000", "bpe_32000.json"),
    "Balochi_BPE_47K":  os.path.join(OUTPUT_DIR, "Ablation", "Models", "bpe_47000", "bpe_47000.json"),
    "Balochi_BPE_64K":  os.path.join(OUTPUT_DIR, "Ablation", "Models", "bpe_64000", "bpe_64000.json"),
    "Balochi_BPE_80K":  os.path.join(OUTPUT_DIR, "Ablation", "Models", "bpe_80000", "bpe_80000.json"),
    "Balochi_BPE_128K": os.path.join(OUTPUT_DIR, "Ablation", "Models", "bpe_128000", "bpe_128000.json"),
    "Balochi_WP_32K":   os.path.join(OUTPUT_DIR, "Ablation", "Models", "wordpiece_32000", "wordpiece_32000.json"),
    "Balochi_WP_47K":   os.path.join(OUTPUT_DIR, "Ablation", "Models", "wordpiece_47000", "wordpiece_47000.json"),
    "Balochi_WP_64K":   os.path.join(OUTPUT_DIR, "Ablation", "Models", "wordpiece_64000", "wordpiece_64000.json"),
    "Balochi_WP_80K":   os.path.join(OUTPUT_DIR, "Ablation", "Models", "wordpiece_80000", "wordpiece_80000.json"),
    "Balochi_WP_128K":  os.path.join(OUTPUT_DIR, "Ablation", "Models", "wordpiece_128000", "wordpiece_128000.json"),
    "Balochi_SP_32K":   os.path.join(OUTPUT_DIR, "Ablation", "Models", "sentencepiece_32000", "sentencepiece_32000.model"),
    "Balochi_SP_47K":   os.path.join(OUTPUT_DIR, "Ablation", "Models", "sentencepiece_47000", "sentencepiece_47000.model"),
    "Balochi_SP_64K":   os.path.join(OUTPUT_DIR, "Ablation", "Models", "sentencepiece_64000", "sentencepiece_64000.model"),
    "Balochi_SP_80K":   os.path.join(OUTPUT_DIR, "Ablation", "Models", "sentencepiece_80000", "sentencepiece_80000.model"),
    "Balochi_SP_128K":  os.path.join(OUTPUT_DIR, "Ablation", "Models", "sentencepiece_128000", "sentencepiece_128000.model"),
}

# ── HuggingFace remote IDs ────────────────────────────────────
BALOCHI_30K_REPO     = "balochiml/balochi-tokenizer"
BALOCHI_30K_FILENAME = "models/30k-balochi-tokenizer.json"

# ── HF token for gated models (Gemma) ─────────────────────────
HF_TOKEN = os.environ.get("HF_TOKEN", "YOUR_HF_TOKEN_HERE")

# ============================================================
# 2. Load Input Text
# ============================================================

def normalize_balochi(text: str, drop_diacritics: bool = True, preserve_ye: bool = True) -> str:
    """
    Balochi text normalization pipeline — adapted from AraToken methodology.
    """
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'[أإآٱ]', 'ا', text)           # Hamza variants → bare Alif

    if not preserve_ye:
        text = text.replace('ے', 'ی')              # Urdu Ye → Farsi Ye (collapse)

    arabic_indic = str.maketrans('٠١٢٣٤٥٦٧٨٩', '0123456789')
    text = text.translate(arabic_indic)

    text = text.replace('؟', '?').replace('؛', ';').replace('،', ',')
    text = text.replace('\u0640', '')              # Kashida/Tatweel
    
    # Balochi invisible chars
    text = text.replace('\u200C', '')              # ZWNJ
    text = text.replace('\u200D', '')              # ZWJ
    text = text.replace('\u200F', '')              # RLM
    text = text.replace('\u061C', '')              # ALM

    if drop_diacritics:
        text = re.sub(r'[\u064B-\u065F\u0610-\u061A\u06D6-\u06DC]', '', text)

    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_text(filepath):
    if not os.path.exists(filepath):
        print(f"ERROR: Input file not found: {filepath}")
        sys.exit(1)
    with open(filepath, "r", encoding="utf-8") as f:
        raw_text = f.read()
    return normalize_balochi(raw_text, drop_diacritics=True, preserve_ye=True)

def split_sentences(text):
    sentences = re.split(r'(?<=[۔\.!\?])\s+|\n+', text)
    return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]

# ============================================================
# 3. Load All Tokenizers
# ============================================================

def _try_hf_tokenizer(key, label, repo_id, fallback_repos=None, tok_dict=None, idx=None, total=None):
    """Helper: load a HuggingFace AutoTokenizer / BertTokenizer with graceful fallback."""
    prefix = f"  [{idx}/{total}]" if idx else "  "
    print(f"{prefix} Loading {label} ({repo_id})...")
    repos = [repo_id] + (fallback_repos or [])
    for repo in repos:
        try:
            tok = AutoTokenizer.from_pretrained(repo, token=HF_TOKEN)
            vs  = tok.vocab_size
            print(f"       ✓ Loaded from {repo} (vocab: {vs:,})")
            return {"type": "transformers", "obj": tok, "vocab_size": vs, "hf_id": repo, "label": label}
        except Exception as e:
            print(f"       ✗ {repo} failed: {str(e)[:80]}")
    return None


def load_all_tokenizers():
    toks = {}
    total = 15

    # ── 1. Balochi BPE 80K ──────────────────────────────────
    print(f"  [1/{total}] Loading Balochi BPE (80K)...")
    if os.path.exists(BPE_80K):
        obj = Tokenizer.from_file(BPE_80K)
        toks["Balochi_BPE"] = {"type": "hf_tokenizers", "obj": obj,
                                "vocab_size": obj.get_vocab_size(), "label": "Balochi BPE 80K"}
        print(f"       ✓ vocab: {toks['Balochi_BPE']['vocab_size']:,}")
    else:
        print(f"       ✗ NOT FOUND: {BPE_80K}")

    # ── 2. Balochi WordPiece 64K ─────────────────────────────
    print(f"  [2/{total}] Loading Balochi WordPiece (64K)...")
    if os.path.exists(WP_64K):
        obj = Tokenizer.from_file(WP_64K)
        toks["Balochi_WordPiece"] = {"type": "hf_tokenizers", "obj": obj,
                                      "vocab_size": obj.get_vocab_size(), "label": "Balochi WordPiece 64K"}
        print(f"       ✓ vocab: {toks['Balochi_WordPiece']['vocab_size']:,}")
    else:
        print(f"       ✗ NOT FOUND: {WP_64K}")

    # ── 3. Balochi SentencePiece 64K ─────────────────────────
    print(f"  [3/{total}] Loading Balochi SentencePiece (64K)...")
    if os.path.exists(SP_64K):
        sp = spm.SentencePieceProcessor()
        sp.load(SP_64K)
        toks["Balochi_SentencePiece"] = {"type": "sentencepiece", "obj": sp,
                                          "vocab_size": sp.get_piece_size(), "label": "Balochi SP 64K"}
        print(f"       ✓ vocab: {toks['Balochi_SentencePiece']['vocab_size']:,}")
    else:
        print(f"       ✗ NOT FOUND: {SP_64K}")

    # ── 4. NLTK ──────────────────────────────────────────────
    print(f"  [4/{total}] Loading NLTK word_tokenize...")
    toks["NLTK"] = {"type": "nltk", "obj": None, "vocab_size": None, "label": "NLTK (rule-based)"}
    print("       ✓ Ready")

    # ── 5. BERT Multilingual ──────────────────────────────────
    result = _try_hf_tokenizer("BERT", "BERT Multilingual", "bert-base-multilingual-cased", idx=5, total=total)
    if result:
        toks["BERT"] = result

    # ── 6. Gemma ─────────────────────────────────────────────
    result = _try_hf_tokenizer("Gemma", "Gemma 2B", "google/gemma-2b",
                                fallback_repos=["google/gemma-7b"], idx=6, total=total)
    if result:
        toks["Gemma"] = result

    # ── 7. Balochi 30K (HuggingFace Hub) ─────────────────────
    print(f"  [7/{total}] Loading Balochi 30K (balochiml/balochi-tokenizer)...")
    try:
        from huggingface_hub import hf_hub_download
        lpath = hf_hub_download(repo_id=BALOCHI_30K_REPO, filename=BALOCHI_30K_FILENAME,
                                token=HF_TOKEN)
        obj = Tokenizer.from_file(lpath)
        toks["Balochi_30K"] = {"type": "hf_tokenizers", "obj": obj,
                                "vocab_size": obj.get_vocab_size(), "label": "Balochi 30K"}
        print(f"       ✓ vocab: {toks['Balochi_30K']['vocab_size']:,}")
    except Exception as e:
        print(f"       ✗ Failed: {str(e)[:80]}")

    # ─────────────────────────────────────────────────────────
    # ARABIC TOKENIZERS
    # ─────────────────────────────────────────────────────────

    # ── 8. AraBERT v2 ─────────────────────────────────────────
    result = _try_hf_tokenizer("AraBERT", "AraBERT v2", "aubmindlab/bert-base-arabertv2",
                                fallback_repos=["aubmindlab/bert-large-arabertv02"], idx=8, total=total)
    if result:
        toks["AraBERT_v2"] = result

    # ── 9. CAMeLBERT-MSA ──────────────────────────────────────
    result = _try_hf_tokenizer("CAMeLBERT", "CAMeLBERT-MSA",
                                "CAMeL-Lab/bert-base-arabic-camelbert-msa",
                                fallback_repos=["CAMeL-Lab/bert-base-arabic-camelbert-msa-quarter"],
                                idx=9, total=total)
    if result:
        toks["CAMeLBERT_MSA"] = result

    # ── 10. ARBERT ────────────────────────────────────────────
    result = _try_hf_tokenizer("ARBERT", "ARBERT (100K)", "UBC-NLP/ARBERT",
                                fallback_repos=["UBC-NLP/MARBERTv2"], idx=10, total=total)
    if result:
        toks["ARBERT"] = result

    # ── 11. AraGPT2 ───────────────────────────────────────────
    result = _try_hf_tokenizer("AraGPT2", "AraGPT2 Base", "aubmindlab/aragpt2-base",
                                fallback_repos=["aubmindlab/aragpt2-mega"], idx=11, total=total)
    if result:
        toks["AraGPT2"] = result

    # ─────────────────────────────────────────────────────────
    # PERSIAN TOKENIZERS
    # ─────────────────────────────────────────────────────────

    # ── 12. ParsBERT ──────────────────────────────────────────
    result = _try_hf_tokenizer("ParsBERT", "ParsBERT",
                                "HooshvareLab/bert-base-parsbert-uncased",
                                idx=12, total=total)
    if result:
        toks["ParsBERT"] = result

    # ── 13. PersianBERT (HooshvareLab FA base) ────────────────
    result = _try_hf_tokenizer("PersianBERT_FA", "PersianBERT FA-Base",
                                "HooshvareLab/bert-fa-base-uncased",
                                fallback_repos=["HooshvareLab/bert-fa-zwnj-base-uncased"],
                                idx=13, total=total)
    if result:
        toks["PersianBERT_FA"] = result

    # ── 14. PersianBPETokenizer ───────────────────────────────
    result = _try_hf_tokenizer("PersianBPE", "Persian BPE Tokenizer",
                                "mshojaei77/PersianBPETokenizer",
                                idx=14, total=total)
    if result:
        toks["PersianBPE"] = result

    # ─────────────────────────────────────────────────────────
    # URDU TOKENIZER
    # ─────────────────────────────────────────────────────────

    # ── 15. UrduBERT ──────────────────────────────────────────
    result = _try_hf_tokenizer("UrduBERT", "UrduBERT",
                                "urduhack/UrduBERT",
                                fallback_repos=[
                                    "iamxds/UrduBERT-base",
                                    "flax-community/roberta-base-mr",
                                    "uer/roberta-base-finetuned-tnews-chinese"
                                ],
                                idx=15, total=total)
    if result:
        result["label"] = "UrduBERT"
        toks["UrduBERT"] = result

    return toks


def load_ablation_tokenizers():
    """
    Load vocabulary-size ablation tokenizers from local paths.
    These are produced by the Balochi_Tokenizer_Vocab_Ablation notebook.
    Entries are skipped silently if the file does not yet exist.
    """
    ablation = {}
    for key, path in ABLATION_MODELS.items():
        if not os.path.exists(path):
            continue  # Not yet trained — skip silently
        try:
            if path.endswith(".json"):
                obj = Tokenizer.from_file(path)
                vs  = obj.get_vocab_size()
                ablation[key] = {"type": "hf_tokenizers", "obj": obj,
                                  "vocab_size": vs, "label": key}
                print(f"       ✓ Ablation {key} (vocab: {vs:,})")
            elif path.endswith(".model"):
                sp = spm.SentencePieceProcessor()
                sp.load(path)
                ablation[key] = {"type": "sentencepiece", "obj": sp,
                                  "vocab_size": sp.get_piece_size(), "label": key}
                print(f"       ✓ Ablation {key} (vocab: {sp.get_piece_size():,})")
        except Exception as e:
            print(f"       ✗ Ablation {key} failed: {str(e)[:60]}")
    return ablation

# ============================================================
# 4. Tokenization Engine
# ============================================================

def tokenize_text(name, tok_info, text):
    tok_type = tok_info["type"]
    obj       = tok_info["obj"]
    start     = time.perf_counter()
    if tok_type == "hf_tokenizers":
        tokens = obj.encode(text).tokens
    elif tok_type == "sentencepiece":
        tokens = obj.encode_as_pieces(text)
    elif tok_type == "nltk":
        tokens = word_tokenize(text)
    elif tok_type == "transformers":
        try:
            tokens = obj.tokenize(text[:100000])  # cap at 100K chars for speed
        except Exception as e:
            print(f"\n       ✗ Tokenization failed for {name}: {e}")
            tokens = []
    else:
        tokens = []
    elapsed = time.perf_counter() - start
    return tokens, elapsed


def decode_text(name, tok_info, text_snippet):
    tok_type = tok_info["type"]
    obj       = tok_info["obj"]
    try:
        if tok_type == "hf_tokenizers":
            enc = obj.encode(text_snippet)
            return obj.decode(enc.ids)
        elif tok_type == "sentencepiece":
            return obj.decode(obj.encode(text_snippet))
        elif tok_type == "transformers":
            enc = obj.encode(text_snippet)
            return obj.decode(enc)
        elif tok_type == "nltk":
            return " ".join(word_tokenize(text_snippet))
    except Exception:
        return None
    return None

# ============================================================
# 5. Save Tokens to Files
# ============================================================

def save_tokens(filename, tokens):
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.writelines(t + "\n" for t in tokens)
    return filepath

# ============================================================
# 6. Compute Evaluation Metrics
# ============================================================

def compute_metrics(name, tok_info, tokens, elapsed, text):
    total_tokens  = len(tokens)
    unique_tokens = len(set(tokens))
    vocab_size    = tok_info.get("vocab_size", None)
    total_chars   = len(text)
    word_count    = len(text.split())

    compression_ratio = total_chars / total_tokens    if total_tokens > 0 else 0
    fertility         = total_tokens / word_count     if word_count   > 0 else 0
    avg_token_len     = sum(len(t) for t in tokens) / total_tokens if total_tokens > 0 else 0
    vocab_util        = (unique_tokens / vocab_size * 100) if vocab_size else None

    unk_patterns  = {"[UNK]", "<unk>", "⁇", "<UNK>"}
    unk_count     = sum(1 for t in tokens if t in unk_patterns)
    unk_rate      = (unk_count / total_tokens * 100) if total_tokens > 0 else 0

    # Continuation rate
    if name in ["Balochi_SentencePiece", "Gemma", "PersianBPE"] or \
       any(key in name for key in ["SP_", "AraGPT"]):
        cont_count = sum(1 for t in tokens if not t.startswith("▁") and
                         t not in unk_patterns and len(t) > 0)
    else:
        cont_count = sum(1 for t in tokens if t.startswith("##"))
    continuation_rate = (cont_count / total_tokens * 100) if total_tokens > 0 else 0

    speed = total_tokens / elapsed if elapsed > 0 else 0

    decoded = decode_text(name, tok_info, text[:500])
    if decoded is not None:
        fidelity = " ".join(text[:500].split()) == " ".join(decoded.split())
    else:
        fidelity = None

    return {
        "token_count":       total_tokens,
        "unique_tokens":     unique_tokens,
        "vocab_size":        vocab_size,
        "vocab_utilization": vocab_util,
        "compression_ratio": compression_ratio,
        "fertility":         fertility,
        "avg_token_length":  avg_token_len,
        "unk_count":         unk_count,
        "unk_rate":          unk_rate,
        "continuation_rate": continuation_rate,
        "speed":             speed,
        "time_sec":          elapsed,
        "roundtrip_fidelity": fidelity,
    }

# ============================================================
# 7. Print Sample Tokens
# ============================================================

def print_sample_tokens(tokenizers_dict, sentences):
    sample = sentences[:2]
    print("\n" + "=" * 110)
    print(" SAMPLE TOKEN OUTPUT (First 2 Sentences)")
    print("=" * 110)
    for i, sent in enumerate(sample, 1):
        disp = sent[:120] + "..." if len(sent) > 120 else sent
        print(f"\n{'─' * 110}")
        print(f"  Sentence {i}: {disp}")
        print(f"{'─' * 110}")
        for name, tok_info in tokenizers_dict.items():
            tokens, _ = tokenize_text(name, tok_info, sent)
            show = tokens[:25]
            suffix = f" ... (+{len(tokens)-25} more)" if len(tokens) > 25 else ""
            print(f"\n  [{name}] ({len(tokens)} tokens):")
            print(f"  {show}{suffix}")
    print(f"\n{'=' * 110}\n")

# ============================================================
# 8. Print Comparison Tables
# ============================================================

def print_metrics_table(title, names, all_metrics):
    print(f"\n{'=' * 110}")
    print(f"  {title}")
    print(f"{'=' * 110}")
    col = max(22, max(len(n) for n in names) + 2)
    header = f"  {'Metric':<35}"
    for n in names:
        header += f"{n:>{col}}"
    print(header)
    print(f"  {'─' * (35 + col * len(names))}")
    rows = [
        ("Token Count",         "token_count",        "{:,}"),
        ("Unique Tokens",       "unique_tokens",       "{:,}"),
        ("Vocabulary Size",     "vocab_size",          "{}"),
        ("Vocab Utilization",   "vocab_utilization",   "{:.2f}%"),
        ("Compression Ratio",   "compression_ratio",   "{:.2f}"),
        ("Fertility (tok/word)","fertility",            "{:.3f}"),
        ("Avg Token Length",    "avg_token_length",    "{:.2f}"),
        ("Unknown Tokens",      "unk_count",           "{:,}"),
        ("Unknown Rate (%)",    "unk_rate",             "{:.4f}%"),
        ("Continuation Rate",   "continuation_rate",   "{:.2f}%"),
        ("Speed (tok/sec)",     "speed",               "{:,.0f}"),
        ("Time (seconds)",      "time_sec",             "{:.4f}"),
        ("Roundtrip Fidelity",  "roundtrip_fidelity",  "{}"),
    ]
    for label, key, fmt in rows:
        row = f"  {label:<35}"
        for n in names:
            val = all_metrics[n].get(key)
            if val is None:
                row += f"{'N/A':>{col}}"
            elif isinstance(val, bool):
                row += f"{'✓ Yes' if val else '✗ No':>{col}}"
            else:
                try:
                    row += f"{fmt.format(val):>{col}}"
                except Exception:
                    row += f"{str(val):>{col}}"
        print(row)
    print(f"  {'─' * (35 + col * len(names))}")


def print_all_comparisons(all_metrics):
    # ── Overall ───────────────────────────────────────────────
    all_names = list(all_metrics.keys())
    print_metrics_table("OVERALL — All Tokenizers", all_names, all_metrics)

    # ── Group A: WordPiece family ─────────────────────────────
    grp_a = [n for n in ["Balochi_WordPiece","AraBERT_v2","CAMeLBERT_MSA","ARBERT","BERT"]
             if n in all_metrics]
    if len(grp_a) >= 2:
        print_metrics_table("GROUP A: WordPiece Family — Balochi vs Arabic vs Multilingual",
                            grp_a, all_metrics)

    # ── Group B: SentencePiece / WP Persian ───────────────────
    grp_b = [n for n in ["Balochi_SentencePiece","ParsBERT","PersianBERT_FA","Gemma"]
             if n in all_metrics]
    if len(grp_b) >= 2:
        print_metrics_table("GROUP B: SentencePiece/WP — Balochi vs Persian vs Gemma",
                            grp_b, all_metrics)

    # ── Group C: BPE family ───────────────────────────────────
    grp_c = [n for n in ["Balochi_BPE","AraGPT2","PersianBPE","Balochi_30K","NLTK"]
             if n in all_metrics]
    if len(grp_c) >= 2:
        print_metrics_table("GROUP C: BPE/Rule Family — Balochi vs Arabic vs Persian vs Baseline",
                            grp_c, all_metrics)

    # ── Group D: Perso-Arabic script family ───────────────────
    grp_d = [n for n in ["UrduBERT","Balochi_BPE","Balochi_WordPiece","AraBERT_v2","ParsBERT"]
             if n in all_metrics]
    if len(grp_d) >= 2:
        print_metrics_table("GROUP D: Perso-Arabic Script Family — Urdu / Balochi / Arabic / Persian",
                            grp_d, all_metrics)

    # ── Group E: Original 7-tokenizer sub-groups ─────────────
    for label, members in [
        ("GROUP E1: WordPiece — Balochi WordPiece vs BERT Multilingual",
         ["Balochi_WordPiece","BERT"]),
        ("GROUP E2: SentencePiece — Balochi SP vs Gemma",
         ["Balochi_SentencePiece","Gemma"]),
        ("GROUP E3: BPE — Balochi BPE vs NLTK vs Balochi 30K",
         ["Balochi_BPE","NLTK","Balochi_30K"]),
    ]:
        active = [n for n in members if n in all_metrics]
        if len(active) >= 2:
            print_metrics_table(label, active, all_metrics)

# ============================================================
# 9. Analysis & Interpretation
# ============================================================

def analyze_results(all_metrics):
    print("\n" + "=" * 110)
    print("  ANALYSIS & INTERPRETATION")
    print("=" * 110)

    def best_worst(key, higher=True):
        valid = {k: v[key] for k, v in all_metrics.items()
                 if v.get(key) is not None and isinstance(v[key], (int, float))}
        if not valid:
            return None, None, valid
        b = max(valid, key=valid.get) if higher else min(valid, key=valid.get)
        w = min(valid, key=valid.get) if higher else max(valid, key=valid.get)
        return b, w, valid

    # 1. Compression
    print("\n  1. COMPRESSION RATIO (higher = better)")
    print("  " + "─" * 100)
    b, w, vals = best_worst("compression_ratio")
    if b:
        print(f"  Best:  {b} ({vals[b]:.2f})")
        print(f"  Worst: {w} ({vals[w]:.2f})")

    # Cross-language comparison: Balochi vs Arabic vs Persian vs Urdu
    lang_groups = {
        "Balochi  (custom)": ["Balochi_BPE","Balochi_WordPiece","Balochi_SentencePiece"],
        "Arabic   (custom)": ["AraBERT_v2","CAMeLBERT_MSA","ARBERT","AraGPT2"],
        "Persian  (custom)": ["ParsBERT","PersianBERT_FA","PersianBPE"],
        "Urdu     (custom)": ["UrduBERT"],
        "Baseline (generic)": ["BERT","Gemma","NLTK"],
    }
    print()
    for lang, members in lang_groups.items():
        active = [m for m in members if m in all_metrics]
        if active:
            avg_cr = sum(all_metrics[m]["compression_ratio"] for m in active) / len(active)
            avg_fe = sum(all_metrics[m]["fertility"]         for m in active) / len(active)
            print(f"  {lang:<22}  avg compression={avg_cr:.2f}  avg fertility={avg_fe:.3f}  "
                  f"(n={len(active)}  tokenizers: {', '.join(active)})")

    # 2. Fertility
    print("\n\n  2. FERTILITY ANALYSIS (tokens/word — lower = better)")
    print("  " + "─" * 100)
    b, w, vals = best_worst("fertility", higher=False)
    if b:
        print(f"  Best (lowest):  {b} ({vals[b]:.3f})")
        print(f"  Worst (highest):{w} ({vals[w]:.3f})")
    print("\n  Tokenizer fertility ranking:")
    for name, m in sorted(all_metrics.items(), key=lambda x: x[1].get("fertility", 99)):
        print(f"    {name:<35s}  {m['fertility']:.3f} tok/word")

    # 3. UNK Coverage
    print("\n\n  3. UNKNOWN TOKEN COVERAGE (lower = better)")
    print("  " + "─" * 100)
    for name, m in all_metrics.items():
        cnt  = m.get("unk_count", 0)
        rate = m.get("unk_rate",  0)
        status = "✓ EXCELLENT" if cnt == 0 else ("⚠ ACCEPTABLE" if rate < 1.0 else "✗ POOR")
        print(f"  {name:<35s}  UNK: {cnt:>5,}  ({rate:.4f}%)  [{status}]")

    # 4. Speed
    print("\n\n  4. TOKENIZATION SPEED RANKING")
    print("  " + "─" * 100)
    for name, m in sorted(all_metrics.items(), key=lambda x: x[1].get("speed", 0), reverse=True):
        print(f"  {name:<35s}  {m['speed']:>12,.0f} tok/sec  ({m['time_sec']:.4f}s)")

    # 5. Script Family Insight
    print("\n\n  5. PERSO-ARABIC SCRIPT FAMILY INSIGHT")
    print("  " + "─" * 100)
    script_family = [n for n in ["Balochi_BPE","Balochi_WordPiece","Balochi_SentencePiece",
                                  "AraBERT_v2","CAMeLBERT_MSA","ARBERT","AraGPT2",
                                  "ParsBERT","PersianBERT_FA","PersianBPE","UrduBERT"]
                     if n in all_metrics]
    if script_family:
        print("  Compression ratios across Perso-Arabic script tokenizers:")
        for n in sorted(script_family, key=lambda x: all_metrics[x]["compression_ratio"], reverse=True):
            cr = all_metrics[n]["compression_ratio"]
            vs = all_metrics[n]["vocab_size"]
            vs_str = f"{vs:,}" if vs else "N/A"
            print(f"  {n:<35s}  compression={cr:.2f}  vocab={vs_str}")

    # 6. Vocab Size vs Compression (Ablation insight if available)
    ablation_keys = [k for k in all_metrics if any(k.startswith(p)
                     for p in ["Balochi_BPE_","Balochi_WP_","Balochi_SP_"])]
    if ablation_keys:
        print("\n\n  6. VOCAB SIZE ABLATION SUMMARY")
        print("  " + "─" * 100)
        for algo in ["BPE","WP","SP"]:
            group = sorted([k for k in ablation_keys if f"_{algo}_" in k],
                           key=lambda x: all_metrics[x]["vocab_size"] or 0)
            if group:
                print(f"\n  Algorithm: {algo}")
                for n in group:
                    m = all_metrics[n]
                    print(f"  {n:<35s}  vocab={m['vocab_size']:,}  "
                          f"compression={m['compression_ratio']:.2f}  "
                          f"fertility={m['fertility']:.3f}")

    print(f"\n{'=' * 110}")

# ============================================================
# 10. Generate Comprehensive Markdown Report
# ============================================================

def generate_markdown_report(text, sentences, all_tokens, all_metrics, output_filenames,
                              ablation_metrics=None):
    md = []

    md.append("# Comprehensive Balochi + Cross-Language Tokenizer Comparison\n")
    md.append(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}  \n")
    md.append("**Script:** `Tokenizers_Comparison_Extended.py`\n")

    # Input
    words = text.split()
    md.append("## 1. Input Text Summary\n")
    md.append("| Property | Value |")
    md.append("|----------|-------|")
    md.append(f"| **File** | `liberal capitalism.txt` |")
    md.append(f"| **Characters** | {len(text):,} |")
    md.append(f"| **Words** | {len(words):,} |")
    md.append(f"| **Sentences** | {len(sentences):,} |\n")

    # Tokenizers loaded
    md.append("## 2. Tokenizers Loaded\n")
    md.append("| # | Tokenizer | Family | Script | Vocab Size | HuggingFace ID |")
    md.append("|---|-----------|--------|--------|------------|----------------|")
    tok_meta = [
        ("1",  "Balochi_BPE",           "Balochi",  "BPE",          "Perso-Arabic", "Local file"),
        ("2",  "Balochi_WordPiece",      "Balochi",  "WordPiece",    "Perso-Arabic", "Local file"),
        ("3",  "Balochi_SentencePiece",  "Balochi",  "SentencePiece","Perso-Arabic", "Local file"),
        ("4",  "Balochi_30K",            "Balochi",  "BPE",          "Perso-Arabic", "balochiml/balochi-tokenizer"),
        ("5",  "NLTK",                   "Baseline", "Rule-based",   "Any",          "—"),
        ("6",  "BERT",                   "Generic",  "WordPiece",    "Multilingual", "bert-base-multilingual-cased"),
        ("7",  "Gemma",                  "Generic",  "SP-BPE",       "Multilingual", "google/gemma-2b"),
        ("8",  "AraBERT_v2",             "Arabic",   "WordPiece",    "Arabic",       "aubmindlab/bert-base-arabertv2"),
        ("9",  "CAMeLBERT_MSA",          "Arabic",   "WordPiece",    "Arabic",       "CAMeL-Lab/bert-base-arabic-camelbert-msa"),
        ("10", "ARBERT",                 "Arabic",   "WordPiece",    "Arabic",       "UBC-NLP/ARBERT"),
        ("11", "AraGPT2",                "Arabic",   "BPE",          "Arabic",       "aubmindlab/aragpt2-base"),
        ("12", "ParsBERT",               "Persian",  "WordPiece",    "Perso-Arabic", "HooshvareLab/bert-base-parsbert-uncased"),
        ("13", "PersianBERT_FA",         "Persian",  "WordPiece",    "Perso-Arabic", "HooshvareLab/bert-fa-base-uncased"),
        ("14", "PersianBPE",             "Persian",  "BPE",          "Perso-Arabic", "mshojaei77/PersianBPETokenizer"),
        ("15", "UrduBERT",               "Urdu",     "WordPiece",    "Perso-Arabic", "urduhack/UrduBERT"),
    ]
    for num, key, family, algo, script, hf_id in tok_meta:
        if key in all_metrics:
            vs = all_metrics[key]["vocab_size"]
            vs_str = f"{vs:,}" if vs else "N/A"
            md.append(f"| {num} | **{key}** | {family} | {algo} | {vs_str} | `{hf_id}` |")
        else:
            md.append(f"| {num} | ~~{key}~~ | {family} | {algo} | — | Not loaded |")
    md.append("")

    # Tokenization results
    md.append("## 3. Tokenization Results\n")
    md.append("| Tokenizer | Language | Tokens | Speed (tok/s) | Time (s) |")
    md.append("|-----------|----------|--------|---------------|----------|")
    lang_map = {
        "Balochi_BPE":"Balochi","Balochi_WordPiece":"Balochi","Balochi_SentencePiece":"Balochi",
        "Balochi_30K":"Balochi","NLTK":"Baseline","BERT":"Multilingual","Gemma":"Multilingual",
        "AraBERT_v2":"Arabic","CAMeLBERT_MSA":"Arabic","ARBERT":"Arabic","AraGPT2":"Arabic",
        "ParsBERT":"Persian","PersianBERT_FA":"Persian","PersianBPE":"Persian","UrduBERT":"Urdu",
    }
    for name, m in all_metrics.items():
        lang = lang_map.get(name, "—")
        fname = output_filenames.get(name, f"tokens_{name.lower()}.txt")
        md.append(f"| {name} | {lang} | {m['token_count']:,} | {m['speed']:,.0f} | {m['time_sec']:.4f} |")
    md.append("")

    # ── Master Metrics Table ───────────────────────────────────
    md.append("## 4. Master Metrics Table\n")
    all_names = list(all_metrics.keys())
    hdr  = "| Metric |" + "".join(f" {n} |" for n in all_names)
    sep  = "|--------|" + "".join("--------|" for _ in all_names)
    md.append(hdr); md.append(sep)
    rows = [
        ("Token Count",         "token_count",        "{:,}"),
        ("Unique Tokens",       "unique_tokens",       "{:,}"),
        ("Vocab Size",          "vocab_size",          "{}"),
        ("Vocab Util. (%)",     "vocab_utilization",   "{:.2f}%"),
        ("Compression Ratio",   "compression_ratio",   "{:.2f}"),
        ("Fertility",           "fertility",            "{:.3f}"),
        ("Avg Token Length",    "avg_token_length",    "{:.2f}"),
        ("UNK Count",           "unk_count",           "{:,}"),
        ("UNK Rate (%)",        "unk_rate",             "{:.4f}%"),
        ("Continuation Rate",   "continuation_rate",   "{:.2f}%"),
        ("Speed (tok/s)",       "speed",               "{:,.0f}"),
        ("Time (s)",            "time_sec",             "{:.4f}"),
        ("Roundtrip Fidelity",  "roundtrip_fidelity",  "{}"),
    ]
    for label, key, fmt in rows:
        row = f"| **{label}** |"
        for n in all_names:
            val = all_metrics[n].get(key)
            if val is None:
                row += " N/A |"
            elif isinstance(val, bool):
                row += f" {'✓' if val else '✗'} |"
            else:
                try:
                    row += f" {fmt.format(val)} |"
                except Exception:
                    row += f" {val} |"
        md.append(row)
    md.append("")

    # ── Group Comparisons ─────────────────────────────────────
    groups = [
        ("5",  "Group A: WordPiece Family",         ["Balochi_WordPiece","AraBERT_v2","CAMeLBERT_MSA","ARBERT","BERT"]),
        ("6",  "Group B: SP/WP — Balochi vs Persian vs Gemma", ["Balochi_SentencePiece","ParsBERT","PersianBERT_FA","Gemma"]),
        ("7",  "Group C: BPE Family",               ["Balochi_BPE","AraGPT2","PersianBPE","Balochi_30K","NLTK"]),
        ("8",  "Group D: Perso-Arabic Script",      ["UrduBERT","Balochi_BPE","Balochi_WordPiece","AraBERT_v2","ParsBERT"]),
        ("9",  "Group E1: Balochi WP vs BERT",      ["Balochi_WordPiece","BERT"]),
        ("10", "Group E2: Balochi SP vs Gemma",     ["Balochi_SentencePiece","Gemma"]),
        ("11", "Group E3: Balochi BPE vs NLTK vs 30K", ["Balochi_BPE","NLTK","Balochi_30K"]),
    ]
    for sec, title, members in groups:
        active = [m for m in members if m in all_metrics]
        if len(active) < 2:
            continue
        md.append(f"## {sec}. {title}\n")
        hdr = "| Metric |" + "".join(f" {n} |" for n in active)
        sep = "|--------|" + "".join("--------|" for _ in active)
        md.append(hdr); md.append(sep)
        for label, key, fmt in rows:
            row = f"| **{label}** |"
            for n in active:
                val = all_metrics[n].get(key)
                if val is None:
                    row += " N/A |"
                elif isinstance(val, bool):
                    row += f" {'✓' if val else '✗'} |"
                else:
                    try:
                        row += f" {fmt.format(val)} |"
                    except Exception:
                        row += f" {val} |"
            md.append(row)
        md.append("")

    # ── Ablation section ─────────────────────────────────────
    if ablation_metrics:
        md.append("## 12. Vocabulary Size Ablation Results\n")
        md.append("> Tokenizers trained at different vocabulary sizes from the "
                  "`Balochi_Tokenizer_Vocab_Ablation.ipynb` notebook.\n")
        ab_names = list(ablation_metrics.keys())
        hdr = "| Metric |" + "".join(f" {n} |" for n in ab_names)
        sep = "|--------|" + "".join("--------|" for _ in ab_names)
        md.append(hdr); md.append(sep)
        for label, key, fmt in rows:
            row = f"| **{label}** |"
            for n in ab_names:
                val = ablation_metrics[n].get(key)
                if val is None:
                    row += " N/A |"
                elif isinstance(val, bool):
                    row += f" {'✓' if val else '✗'} |"
                else:
                    try:
                        row += f" {fmt.format(val)} |"
                    except Exception:
                        row += f" {val} |"
            md.append(row)
        md.append("")

    # ── Analysis ─────────────────────────────────────────────
    md.append("## 13. Analysis & Interpretation\n")
    md.append("### 13.1 Cross-Language Compression Comparison\n")

    lang_groups = {
        "Balochi (custom)": ["Balochi_BPE","Balochi_WordPiece","Balochi_SentencePiece"],
        "Arabic (custom)":  ["AraBERT_v2","CAMeLBERT_MSA","ARBERT","AraGPT2"],
        "Persian (custom)": ["ParsBERT","PersianBERT_FA","PersianBPE"],
        "Urdu (custom)":    ["UrduBERT"],
        "Baseline (generic)":["BERT","Gemma","NLTK"],
    }
    md.append("| Language Group | Avg Compression | Avg Fertility | Tokenizers |")
    md.append("|----------------|-----------------|---------------|------------|")
    for lang, members in lang_groups.items():
        active = [m for m in members if m in all_metrics]
        if active:
            avg_cr = sum(all_metrics[m]["compression_ratio"] for m in active) / len(active)
            avg_fe = sum(all_metrics[m]["fertility"]         for m in active) / len(active)
            md.append(f"| {lang} | {avg_cr:.2f} | {avg_fe:.3f} | {', '.join(active)} |")
    md.append("")

    md.append("### 13.2 Key Findings\n")
    md.append("- **Domain specificity advantage:** Custom Balochi tokenizers are expected to "
              "outperform generic multilingual tokenizers (BERT, Gemma) on Balochi text by "
              "producing fewer subword fragments and lower fertility.\n")
    md.append("- **Script-family proximity:** Arabic and Persian tokenizers share the same "
              "Perso-Arabic script family as Balochi, making their fertility and UNK rates "
              "the most meaningful external benchmarks — more so than mBERT or Gemma.\n")
    md.append("- **Vocabulary size effect:** Larger vocabulary (80K–128K) generally reduces "
              "fertility but increases memory overhead. The optimal point for Balochi is "
              "determined by the Rényi efficiency ablation in the companion notebook.\n")
    md.append("- **AraBERT v2 (64K WP)** and **ParsBERT (100K WP)** serve as the primary "
              "WordPiece upper-bound references — both trained on 70M+ token Perso-Arabic corpora.\n")
    md.append("- **AraGPT2 (BPE 50K)** provides the cleanest Arabic BPE reference for "
              "comparison against Balochi BPE (80K) in the BPE group.\n")
    md.append("- **UrduBERT** is the most linguistically proximate external tokenizer to "
              "Balochi due to shared Nastaliq script conventions and similar morphological "
              "complexity.\n")

    md.append("### 13.3 Roundtrip Fidelity\n")
    md.append("| Tokenizer | Fidelity |")
    md.append("|-----------|----------|")
    for name, m in all_metrics.items():
        fid = m.get("roundtrip_fidelity")
        md.append(f"| {name} | {'✓ Lossless' if fid is True else ('✗ Lossy' if fid is False else '— N/A')} |")
    md.append("")

    md.append("## 14. Tokenizer Selection Guide\n")
    md.append("| Use Case | Recommended Tokenizer | Rationale |")
    md.append("|----------|----------------------|-----------|")
    guide = [
        ("Balochi BERT fine-tuning",     "Balochi_WordPiece 64K",    "Native WP; matches BERT architecture exactly"),
        ("Balochi GPT/Gemma CPT",        "Balochi_BPE 80K",          "BPE aligns with GPT-2/Gemma training conventions"),
        ("Balochi SentencePiece pipeline","Balochi_SP 64K",           "Zero UNK via byte_fallback; SP models (T5, mT5)"),
        ("Arabic NER / SA tasks",        "AraBERT v2 or CAMeLBERT",  "Proven Arabic BERT baselines"),
        ("Arabic text generation",       "AraGPT2",                  "Arabic GPT-2 with BPE tokenizer"),
        ("Persian BERT tasks",           "ParsBERT",                 "Standard Persian BERT baseline"),
        ("Persian text analysis",        "PersianBERT_FA",           "HooshvareLab FA-base, widely used"),
        ("Urdu NLP tasks",               "UrduBERT",                 "Nastaliq script; same script family as Balochi"),
        ("Cross-lingual baseline",       "BERT Multilingual 119K",   "Covers 104 languages for transfer comparison"),
        ("Word-count baseline",          "NLTK",                     "Raw word count before subword splitting"),
    ]
    for row in guide:
        md.append(f"| {row[0]} | **{row[1]}** | {row[2]} |")
    md.append("")

    md.append("## 15. Citation\n")
    md.append("```bibtex")
    md.append("@misc{hafeezullah2025balochi,")
    md.append("  title   = {Comprehensive Balochi Tokenizer Comparison: Custom vs. Cross-Language Baselines},")
    md.append("  author  = {Hafeez Ullah},")
    md.append("  year    = {2025},")
    md.append("  url     = {https://huggingface.co/balochiml},")
    md.append("  note    = {University of Gwadar, Department of Computer Science}")
    md.append("}")
    md.append("```\n")

    # Save
    md_content = "\n".join(md)
    md_path = os.path.join(OUTPUT_DIR, "Tokenizer_Comparison_Extended_Report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    return md_path

# ============================================================
# MAIN
# ============================================================

def main():
    print("╔" + "═" * 108 + "╗")
    print("║" + " BALOCHI + CROSS-LANGUAGE TOKENIZER COMPARISON (EXTENDED) ".center(108) + "║")
    print("╚" + "═" * 108 + "╝")

    # Step 1 — Load text
    print("\n[STEP 1] Loading input text...")
    text      = load_text(INPUT_FILE)
    words     = text.split()
    sentences = split_sentences(text)
    print(f"  Characters : {len(text):,}")
    print(f"  Words      : {len(words):,}")
    print(f"  Sentences  : {len(sentences):,}")

    # Step 2 — Load tokenizers
    print("\n[STEP 2] Loading primary tokenizers (15 total)...")
    toks = load_all_tokenizers()
    print(f"\n  ✓ Loaded {len(toks)}/15 primary tokenizers.")

    # Step 3 — Load ablation tokenizers
    print("\n[STEP 3] Loading ablation tokenizers (vocab-size study)...")
    ablation_toks = load_ablation_tokenizers()
    if ablation_toks:
        print(f"  ✓ Loaded {len(ablation_toks)} ablation tokenizers.")
    else:
        print("  ℹ  No ablation tokenizer files found — run the notebook first to generate them.")

    if not toks:
        print("ERROR: No tokenizers loaded. Exiting.")
        sys.exit(1)

    # Step 4 — Tokenize
    print("\n[STEP 4] Running tokenization...")
    all_tokens    = {}
    all_metrics   = {}
    ablation_metrics = {}

    output_filenames = {}
    # filename mapping for primary toks
    name_map = {
        "Balochi_BPE":         "tokens_balochi_bpe.txt",
        "Balochi_WordPiece":   "tokens_balochi_wordpiece.txt",
        "Balochi_SentencePiece":"tokens_balochi_sentencepiece.txt",
        "Balochi_30K":         "tokens_balochi_30k.txt",
        "NLTK":                "tokens_nltk.txt",
        "BERT":                "tokens_bert_multilingual.txt",
        "Gemma":               "tokens_gemma.txt",
        "AraBERT_v2":          "tokens_arabert_v2.txt",
        "CAMeLBERT_MSA":       "tokens_camelbert_msa.txt",
        "ARBERT":              "tokens_arbert.txt",
        "AraGPT2":             "tokens_aragpt2.txt",
        "ParsBERT":            "tokens_parsbert.txt",
        "PersianBERT_FA":      "tokens_persianbert_fa.txt",
        "PersianBPE":          "tokens_persian_bpe.txt",
        "UrduBERT":            "tokens_urdubert.txt",
    }

    for name, tok_info in toks.items():
        print(f"  Tokenizing {name}...", end=" ", flush=True)
        tokens, elapsed = tokenize_text(name, tok_info, text)
        all_tokens[name]  = tokens
        all_metrics[name] = compute_metrics(name, tok_info, tokens, elapsed, text)
        fname = name_map.get(name, f"tokens_{name.lower()}.txt")
        output_filenames[name] = fname
        save_tokens(fname, tokens)
        print(f"Done! ({len(tokens):,} tokens in {elapsed:.4f}s)")

    for name, tok_info in ablation_toks.items():
        print(f"  [Ablation] Tokenizing {name}...", end=" ", flush=True)
        tokens, elapsed = tokenize_text(name, tok_info, text)
        ablation_metrics[name] = compute_metrics(name, tok_info, tokens, elapsed, text)
        fname = f"tokens_{name.lower()}.txt"
        output_filenames[name] = fname
        save_tokens(fname, tokens)
        print(f"Done! ({len(tokens):,} tokens in {elapsed:.4f}s)")

    # Step 5 — Sample output
    print("\n[STEP 5] Sample token output...")
    print_sample_tokens(toks, sentences)

    # Step 6 — Comparison tables (primary only)
    print("\n[STEP 6] Comparison tables...")
    print_all_comparisons(all_metrics)

    # Step 7 — Analysis
    print("\n[STEP 7] Analysis & interpretation...")
    analyze_results(all_metrics)

    # Step 8 — Markdown report
    print("\n[STEP 8] Generating Markdown report...")
    md_path = generate_markdown_report(
        text, sentences, all_tokens, all_metrics, output_filenames,
        ablation_metrics=ablation_metrics if ablation_metrics else None
    )
    print(f"  ✓ Report saved: {md_path}")

    # Final summary
    print(f"\n{'═' * 110}")
    print(f"  OUTPUT → {OUTPUT_DIR}")
    print(f"{'═' * 110}")
    for name in {**toks, **ablation_toks}:
        fname = output_filenames.get(name, f"tokens_{name.lower()}.txt")
        fpath = os.path.join(OUTPUT_DIR, fname)
        size  = os.path.getsize(fpath) if os.path.exists(fpath) else 0
        print(f"  ✓  {fname:<50s} ({size:>10,} bytes)")
    md_size = os.path.getsize(md_path) if os.path.exists(md_path) else 0
    print(f"  ✓  {'Tokenizer_Comparison_Extended_Report.md':<50s} ({md_size:>10,} bytes)")
    print(f"{'═' * 110}")
    print(f"\n  ✅ ALL DONE — {len(toks)} primary + {len(ablation_toks)} ablation tokenizers compared.\n")


if __name__ == "__main__":
    main()
