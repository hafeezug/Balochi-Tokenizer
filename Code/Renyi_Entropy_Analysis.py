# -*- coding: utf-8 -*-
"""
Rényi Entropy Analysis — Balochi Tokenizer Vocabulary Optimization
===================================================================
Analyzes vocabulary efficiency using information-theoretic measures
(Shannon entropy, Rényi entropy, effective vocabulary size).

Relies on Tokenizers_Comparison_Extended for tokenizer loading and tokenization.
"""

import os
import sys
import time
from collections import Counter
from datetime import datetime

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Import dependencies
try:
    import numpy as np
    from Tokenizers_Comparison import load_ablation_tokenizers, tokenize_text
except ImportError as e:
    print(f"❌ Error importing dependencies: {e}")
    sys.exit(1)

# ============================================
# 1. Configuration
# ============================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = SCRIPT_DIR
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "Output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

EVAL_TEXTS = [
    os.path.join(BASE_DIR, "..", "Tokens", "Tahir Hakim.txt"),
    os.path.join(BASE_DIR, "..", "Tokens", "liberal capitalism.txt"),
]

RENYI_ORDERS = [0, 0.5, 2, 3, 10, float('inf')]

# ============================================
# 2. Information-Theoretic Metrics
# ============================================

def compute_token_distribution(tokens):
    counter = Counter(tokens)
    total = len(tokens)
    probs = np.array([count / total for count in counter.values()])
    return counter, probs

def shannon_entropy(probs):
    p = probs[probs > 0]
    return -np.sum(p * np.log2(p))

def renyi_entropy(probs, alpha):
    p = probs[probs > 0]
    if alpha == 0:
        return np.log2(len(p))
    elif alpha == 1:
        return shannon_entropy(p)
    elif alpha == float('inf'):
        return -np.log2(np.max(p))
    else:
        return (1.0 / (1.0 - alpha)) * np.log2(np.sum(p ** alpha))

def zipf_fit(counter, top_n=100):
    freqs = sorted(counter.values(), reverse=True)[:top_n]
    if len(freqs) < 5: return 0.0, 0.0
    ranks = np.arange(1, len(freqs) + 1, dtype=float)
    log_ranks = np.log(ranks)
    log_freqs = np.log(np.array(freqs, dtype=float))

    n = len(log_ranks)
    sum_x = np.sum(log_ranks)
    sum_y = np.sum(log_freqs)
    sum_xy = np.sum(log_ranks * log_freqs)
    sum_x2 = np.sum(log_ranks ** 2)

    denominator = n * sum_x2 - sum_x ** 2
    if denominator == 0: return 0.0, 0.0

    slope = (n * sum_xy - sum_x * sum_y) / denominator
    intercept = (sum_y - slope * sum_x) / n

    y_pred = slope * log_ranks + intercept
    ss_res = np.sum((log_freqs - y_pred) ** 2)
    ss_tot = np.sum((log_freqs - np.mean(log_freqs)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    return -slope, r_squared

# ============================================
# 3. Core Analysis
# ============================================

def analyze_tokenizer_on_text(tok_name, tok_info, text, text_name):
    tokens, _ = tokenize_text(tok_name, tok_info, text)
    if not tokens: return None

    counter, probs = compute_token_distribution(tokens)
    h1 = shannon_entropy(probs)
    v_eff = 2.0 ** h1
    v_total = tok_info.get("vocab_size", None)

    renyi_vals = {}
    for alpha in RENYI_ORDERS:
        label = f"H_{alpha}" if alpha != float('inf') else "H_inf"
        renyi_vals[label] = renyi_entropy(probs, alpha)

    z_exp, z_r2 = zipf_fit(counter)
    
    # Identify algorithm
    algorithm = "Unknown"
    if "BPE" in tok_name: algorithm = "BPE"
    elif "WordPiece" in tok_name or "BERT" in tok_name: algorithm = "WordPiece"
    elif "SentencePiece" in tok_name or "Gemma" in tok_name: algorithm = "SentencePiece"

    return {
        "tokenizer": tok_name,
        "text": text_name,
        "algorithm": algorithm,
        "vocab_size": v_total,
        "total_tokens": len(tokens),
        "unique_types": len(counter),
        "type_token_ratio": len(counter) / len(tokens),
        "hapax_types": sum(1 for c in counter.values() if c == 1),
        "hapax_rate": sum(1 for c in counter.values() if c == 1) / len(counter),
        "shannon_entropy_H1": h1,
        "effective_vocab_V_eff": v_eff,
        "vocab_efficiency_eta": (v_eff / v_total) if v_total else None,
        "zipf_exponent": z_exp,
        "zipf_r_squared": z_r2,
        **renyi_vals,
    }

# ============================================
# 4. Report Generation
# ============================================

def generate_markdown_report(results, output_path):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# 📐 Rényi Entropy Analysis — Balochi Tokenizer Vocabulary Optimization\n",
        f"**Generated on:** {timestamp}\n\n",
        "## 1. Purpose\n",
        "Applies information-theoretic measures to formally evaluate vocabulary efficiency.\n"
    ]

    texts = sorted(set(r["text"] for r in results))
    for text_name in texts:
        text_results = [r for r in results if r["text"] == text_name]
        lines.append(f"## 2. Results — `{text_name}`\n")
        lines.append("### Shannon & Rényi Entropy Spectrum\n")
        
        # Sort results by effective vocab for cleaner display
        text_results.sort(key=lambda x: x["effective_vocab_V_eff"], reverse=True)
        
        lines.append("| Metric | " + " | ".join(r["tokenizer"] for r in text_results) + " |")
        lines.append("|--------|" + "|".join("-" * (len(r["tokenizer"]) + 2) for r in text_results) + "|")

        rows = [
            ("Vocab Size", "vocab_size", "{:,}"),
            ("Total Tokens", "total_tokens", "{:,}"),
            ("Unique Types", "unique_types", "{:,}"),
            ("Type-Token Ratio", "type_token_ratio", "{:.4f}"),
            ("**Shannon H₁**", "shannon_entropy_H1", "**{:.4f}**"),
            ("H₀ (Hartley)", "H_0", "{:.4f}"),
            ("H₂ (Collision)", "H_2", "{:.4f}"),
            ("H∞ (Min-entropy)", "H_inf", "{:.4f}"),
            ("**Effective Vocab (V_eff)**", "effective_vocab_V_eff", "**{:.1f}**"),
            ("**Vocab Efficiency (η)**", "vocab_efficiency_eta", "**{:.4f}**"),
            ("Zipf Exponent (s)", "zipf_exponent", "{:.3f}"),
        ]

        for label, key, fmt in rows:
            vals = []
            for r in text_results:
                v = r.get(key, None)
                if v is None:
                    vals.append("—")
                else:
                    vals.append(fmt.format(v))
            lines.append(f"| {label} | " + " | ".join(vals) + " |")
        lines.append("")

    report_text = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    return report_text

def generate_csv(results, output_path):
    if not results: return
    keys = results[0].keys()
    with open(output_path, "w", encoding="utf-8-sig") as f:
        f.write(",".join(keys) + "\n")
        for row in results:
            vals = []
            for k in keys:
                v = row.get(k, "")
                if isinstance(v, float):
                    if v == float('inf'): vals.append("inf")
                    else: vals.append(f"{v:.6f}")
                elif v is None: vals.append("")
                else: vals.append(str(v))
            f.write(",".join(vals) + "\n")

# ============================================
# 5. Main Execution
# ============================================

def main():
    print("=" * 80)
    print("  📐 Rényi Entropy Analysis — Balochi Tokenizers")
    print("=" * 80)

    # 1. Load evaluation texts
    texts = {}
    for path in EVAL_TEXTS:
        if os.path.exists(path):
            name = os.path.basename(path)
            with open(path, "r", encoding="utf-8") as f:
                texts[name] = f.read()
    if not texts:
        print("❌ No evaluation texts found.")
        return

    # 2. Load ablation tokenizers
    print("\n  Loading ablation tokenizers...")
    tokenizers = load_ablation_tokenizers()

    if not tokenizers:
        print("\n  ❌ Required ablation tokenizers not found. Exiting.")
        return

    # 3. Analyze
    print(f"\n  Running entropy analysis...")
    results = []
    for tok_name, tok_info in tokenizers.items():
        for text_name, text_content in texts.items():
            print(f"    Analyzing {tok_name} on {text_name}...", end="")
            start = time.perf_counter()
            res = analyze_tokenizer_on_text(tok_name, tok_info, text_content, text_name)
            if res:
                results.append(res)
            elapsed = time.perf_counter() - start
            print(f" done ({elapsed:.3f}s)")

    # 4. Reporting
    report_path = os.path.join(OUTPUT_DIR, "Renyi_Entropy_Report.md")
    csv_path = os.path.join(OUTPUT_DIR, "renyi_entropy_data.csv")

    generate_markdown_report(results, report_path)
    generate_csv(results, csv_path)

    print(f"\n  ✅ ANALYSIS COMPLETE")
    print(f"  Report: {report_path}")

if __name__ == "__main__":
    main()
