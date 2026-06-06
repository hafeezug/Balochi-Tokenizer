# -*- coding: utf-8 -*-
"""
Vocabulary Pruning Analyzer
===========================
This script determines the optimal vocabulary size by analyzing token frequencies
across the entire training corpus. It calculates the cumulative frequency distribution
to find the exact vocabulary size needed to cover 99% or 99.5% of the text, allowing
us to safely prune the "long tail" of rare tokens.
"""

import os
import sys
import time
import json
from collections import Counter

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

try:
    from tokenizers import Tokenizer
    import sentencepiece as spm
except ImportError:
    print("Please install tokenizers and sentencepiece: pip install tokenizers sentencepiece")
    sys.exit(1)

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CORPUS_PATH = os.path.join(SCRIPT_DIR, "Output", "Ablation", "training_corpus_normalized.txt")
TOKENIZER_PATH = os.path.join(SCRIPT_DIR, "Output", "Ablation", "Models", "sentencepiece_64000", "sentencepiece_64000.model")

def main():
    print("=" * 80)
    print(" ✂️ Vocabulary Pruning Analyzer — Cumulative Frequency Thresholding")
    print("=" * 80)

    if not os.path.exists(CORPUS_PATH):
        print(f"❌ Corpus not found at {CORPUS_PATH}")
        return
    
    if not os.path.exists(TOKENIZER_PATH):
        print(f"❌ Tokenizer not found at {TOKENIZER_PATH}")
        return

    print(f"\n[1] Loading SentencePiece Tokenizer (64K)...")
    sp = spm.SentencePieceProcessor()
    sp.load(TOKENIZER_PATH)
    vocab_size = sp.get_piece_size()
    print(f"    ✓ Loaded vocabulary size: {vocab_size:,}")

    print(f"\n[2] Tokenizing Corpus & Counting Frequencies...")
    print(f"    (This may take a few minutes for a ~370MB corpus)")
    
    token_counts = Counter()
    total_tokens = 0
    lines_processed = 0
    start_time = time.time()

    # Read and tokenize line by line to save memory
    with open(CORPUS_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            # Encode as IDs
            ids = sp.encode_as_ids(line)
            token_counts.update(ids)
            total_tokens += len(ids)
            lines_processed += 1
            
            if lines_processed % 500000 == 0:
                print(f"    ... processed {lines_processed:,} lines")

    elapsed = time.time() - start_time
    print(f"    ✓ Tokenization complete in {elapsed:.1f}s")
    print(f"    ✓ Total tokens in corpus: {total_tokens:,}")
    print(f"    ✓ Unique tokens utilized: {len(token_counts):,}")

    if total_tokens == 0:
        print("❌ No tokens generated. Corpus might be empty.")
        return

    print(f"\n[3] Calculating Cumulative Frequency...")
    # Sort tokens by frequency (highest first)
    sorted_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)
    
    cumulative_sum = 0
    thresholds = [0.90, 0.95, 0.99, 0.995, 0.999]
    threshold_idx = 0
    
    results = []

    for rank, (token_id, count) in enumerate(sorted_tokens, 1):
        cumulative_sum += count
        cumulative_pct = cumulative_sum / total_tokens
        
        while threshold_idx < len(thresholds) and cumulative_pct >= thresholds[threshold_idx]:
            results.append({
                "coverage": thresholds[threshold_idx],
                "vocab_size": rank,
                "token_id": token_id,
                "count": count
            })
            threshold_idx += 1

    print("\n" + "=" * 60)
    print(" 🎯 RECOMMENDED PRUNING THRESHOLDS")
    print("=" * 60)
    print(f"{'Coverage Target':<20} | {'Required Vocab Size':<20} | {'Tokens Pruned':<15}")
    print("-" * 60)
    
    for res in results:
        target_pct = f"{res['coverage']*100:.1f}%"
        req_vocab = f"{res['vocab_size']:,}"
        pruned = f"{vocab_size - res['vocab_size']:,}"
        print(f"{target_pct:<20} | {req_vocab:<20} | {pruned:<15}")
    
    print("-" * 60)
    
    # Summary advice
    t99 = next((r for r in results if r['coverage'] == 0.99), None)
    if t99:
        recommended_size = (t99['vocab_size'] // 1000 + 1) * 1000 # round up to nearest thousand
        print(f"\n💡 CONCLUSION:")
        print(f"To achieve 99.0% coverage (the AraToken paper recommendation),")
        print(f"you only need the top ~{t99['vocab_size']:,} tokens.")
        print(f"You can safely train a new optimal tokenizer with vocab_size={recommended_size:,}")
        print(f"This will drop ~{vocab_size - recommended_size:,} useless tokens, shrinking the Gemma embedding matrix.")

if __name__ == "__main__":
    main()
