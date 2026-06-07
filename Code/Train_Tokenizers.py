# -*- coding: utf-8 -*-
"""
Balochi Tokenizer Training — Ablation Models
=============================================
This script handles the training of BPE, WordPiece, and SentencePiece 
tokenizers across multiple vocabulary sizes (32K, 64K, 80K, 128K).

It concatenates the training corpora and runs the training loop.
"""

import os
import sys
import time
import glob
import string

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

def install_if_missing(package, pip_name=None):
    try:
        __import__(package)
    except ImportError:
        import subprocess
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", pip_name or package, "-q"
        ])

install_if_missing("tokenizers")
install_if_missing("sentencepiece")

import sentencepiece as spm
from tokenizers import Tokenizer
from tokenizers.models import BPE, WordPiece
from tokenizers.trainers import BpeTrainer, WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace, Punctuation, Digits, Sequence as PretokSequence
from tokenizers.normalizers import NFKC

# ============================================
# 1. Configuration
# ============================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = SCRIPT_DIR
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "Output", "Ablation")
MODELS_DIR = os.path.join(OUTPUT_DIR, "Models")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

CONFIG = {
    "CORPUS_FILES": [
        os.path.join(BASE_DIR, "..", "..", "Final Tokenizers", "balochi_clean_corpus_dictionary.txt"),
        os.path.join(BASE_DIR, "..", "..", "Final Tokenizers", "balochi_dedup_corpus.txt"),
        os.path.join(BASE_DIR, "..", "..", "Final Tokenizers", "english_corpus_2M.txt"),
    ],
    "VOCAB_SIZES": [32000, 47000, 64000, 80000, 128000],
    "ALGORITHMS": ["bpe", "wordpiece", "sentencepiece"],
    "SKIP_EXISTING": False,
    "SP_MODEL_TYPE": "unigram",
    "SP_CHARACTER_COVERAGE": 0.9995,
    "WP_MIN_FREQUENCY": 2,
    "BPE_MIN_FREQUENCY": 2,
    "SPECIAL_TOKENS": ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
}

MANDATORY_ALPHABET = (
    list(string.ascii_lowercase) +
    list(string.ascii_uppercase) +
    list("0123456789") +
    list("۰۱۲۳۴۵۶۷۸۹") +
    list("٠١٢٣٤٥٦٧٨٩")
)

# ============================================
# 2. Corpus Loading
# ============================================

def load_corpus_files(corpus_files):
    existing_files = []
    for f in corpus_files:
        if os.path.exists(f):
            existing_files.append(f)
        else:
            print(f"      ⚠️ Warning: Corpus file not found: {f}")
    return existing_files

def prepare_corpus_for_training(corpus_files, output_path):
    from Tokenizers_Comparison import normalize_balochi
    
    # We will force-recreate if it exists to ensure normalization is applied
    print(f"    Concatenating and Normalizing {len(corpus_files)} corpus files...")
    total_lines = 0
    with open(output_path, "w", encoding="utf-8") as out_f:
        for fpath in corpus_files:
            try:
                with open(fpath, "r", encoding="utf-8") as in_f:
                    for line in in_f:
                        line = line.strip()
                        if len(line) >= 2:
                            norm_line = normalize_balochi(line, drop_diacritics=True, preserve_ye=True)
                            if len(norm_line) >= 2:
                                out_f.write(norm_line + "\n")
                                total_lines += 1
            except Exception as e:
                print(f"      ⚠️ Skipped {fpath}: {e}")

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"    ✓ Wrote {total_lines:,} normalized lines ({size_mb:.1f} MB)")
    return output_path

# ============================================
# 3. Training Functions
# ============================================

def train_bpe_tokenizer(corpus_file, vocab_size, output_dir):
    model_path = os.path.join(output_dir, f"bpe_{vocab_size}.json")
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.normalizer = NFKC()
    tokenizer.pre_tokenizer = PretokSequence([
        Whitespace(),
        Punctuation(),
        Digits(individual_digits=True)
    ])
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=CONFIG["BPE_MIN_FREQUENCY"],
        special_tokens=CONFIG["SPECIAL_TOKENS"],
        initial_alphabet=MANDATORY_ALPHABET,
        show_progress=True,
    )
    print(f"      Training BPE (vocab={vocab_size:,})...")
    start = time.perf_counter()
    tokenizer.train([corpus_file], trainer)
    elapsed = time.perf_counter() - start
    print(f"      ✓ Trained in {elapsed:.1f}s (actual vocab: {tokenizer.get_vocab_size():,})")
    tokenizer.save(model_path)
    return model_path, elapsed

def train_wordpiece_tokenizer(corpus_file, vocab_size, output_dir):
    model_path = os.path.join(output_dir, f"wordpiece_{vocab_size}.json")
    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    tokenizer.normalizer = NFKC()
    tokenizer.pre_tokenizer = PretokSequence([
        Whitespace(),
        Punctuation(),
        Digits(individual_digits=True)
    ])
    trainer = WordPieceTrainer(
        vocab_size=vocab_size,
        min_frequency=CONFIG["WP_MIN_FREQUENCY"],
        special_tokens=CONFIG["SPECIAL_TOKENS"],
        initial_alphabet=MANDATORY_ALPHABET,
        show_progress=True,
    )
    print(f"      Training WordPiece (vocab={vocab_size:,})...")
    start = time.perf_counter()
    tokenizer.train([corpus_file], trainer)
    elapsed = time.perf_counter() - start
    print(f"      ✓ Trained in {elapsed:.1f}s (actual vocab: {tokenizer.get_vocab_size():,})")
    tokenizer.save(model_path)
    return model_path, elapsed

def train_sentencepiece_tokenizer(corpus_file, vocab_size, output_dir):
    model_prefix = os.path.join(output_dir, f"sentencepiece_{vocab_size}")
    model_path = model_prefix + ".model"
    print(f"      Training SentencePiece (vocab={vocab_size:,})...")
    start = time.perf_counter()
    spm.SentencePieceTrainer.train(
        input=corpus_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type=CONFIG["SP_MODEL_TYPE"],
        character_coverage=CONFIG["SP_CHARACTER_COVERAGE"],
        bos_id=2,
        eos_id=3,
        pad_id=0,
        unk_id=1,
        pad_piece='<pad>',
        unk_piece='<unk>',
        bos_piece='<s>',
        eos_piece='</s>',
        byte_fallback=True,
        user_defined_symbols=["[CLS]", "[SEP]", "[MASK]"],
        train_extremely_large_corpus=True,
    )
    elapsed = time.perf_counter() - start
    sp = spm.SentencePieceProcessor(model_file=model_path)
    print(f"      ✓ Trained in {elapsed:.1f}s (actual vocab: {sp.get_piece_size():,})")
    return model_path, elapsed

# ============================================
# 4. Main Execution
# ============================================

def main():
    print("=" * 80)
    print("  🚀 Vocabulary Size Ablation — Tokenizer Training")
    print("=" * 80)

    corpus_files = load_corpus_files(CONFIG["CORPUS_FILES"])
    if not corpus_files:
        print("\n  ❌ Corpus files not found. Cannot train models. Exiting.")
        return

    print(f"\n  ✓ Found {len(corpus_files)} corpus files")
    corpus_file = prepare_corpus_for_training(corpus_files, os.path.join(OUTPUT_DIR, "training_corpus_normalized.txt"))

    print(f"\n  Training {len(CONFIG['ALGORITHMS'])} algorithms × "
          f"{len(CONFIG['VOCAB_SIZES'])} vocab sizes = "
          f"{len(CONFIG['ALGORITHMS']) * len(CONFIG['VOCAB_SIZES'])} models\n")

    for algo in CONFIG["ALGORITHMS"]:
        for vs in CONFIG["VOCAB_SIZES"]:
            print(f"  [{algo.upper()} / {vs:,}]", end=" ")
            ablation_model_dir = os.path.join(MODELS_DIR, f"{algo}_{vs}")
            
            if algo in ("bpe", "wordpiece"):
                ablation_model_path = os.path.join(ablation_model_dir, f"{algo}_{vs}.json")
            else:
                ablation_model_path = os.path.join(ablation_model_dir, f"sentencepiece_{vs}.model")

            if os.path.exists(ablation_model_path) and CONFIG["SKIP_EXISTING"]:
                print(f"→ Skipping (already trained)")
                continue

            os.makedirs(ablation_model_dir, exist_ok=True)
            print(f"→ Training new model...")
            try:
                if algo == "bpe":
                    train_bpe_tokenizer(corpus_file, vs, ablation_model_dir)
                elif algo == "wordpiece":
                    train_wordpiece_tokenizer(corpus_file, vs, ablation_model_dir)
                elif algo == "sentencepiece":
                    train_sentencepiece_tokenizer(corpus_file, vs, ablation_model_dir)
            except Exception as e:
                print(f"      ❌ Training failed: {e}")

    print("\n  ✅ Training Complete!")

if __name__ == "__main__":
    main()
