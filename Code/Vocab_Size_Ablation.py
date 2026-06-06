# -*- coding: utf-8 -*-
"""
Vocabulary Size Ablation Analysis — Balochi Tokenizers
======================================================
Analyzes diminishing returns and optimal vocabulary sizes for the 
ablation tokenizers (32K, 64K, 80K, 128K).

Relies on Tokenizers_Comparison_Extended for tokenization engine.
"""

import os
import sys
import time
from datetime import datetime

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Import core evaluation engine
try:
    from Tokenizers_Comparison_Extended import (
        load_ablation_tokenizers,
        tokenize_text,
        compute_metrics
    )
except ImportError:
    print("❌ Error: Tokenizers_Comparison_Extended.py must be in the same directory.")
    sys.exit(1)

# ============================================
# 1. Configuration
# ============================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = SCRIPT_DIR
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "Output", "Ablation")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CONFIG = {
    "ALGORITHMS": ["bpe", "wordpiece", "sentencepiece"],
    "VOCAB_SIZES": [32000, 64000, 80000, 128000],
    "EVAL_TEXTS": [
        os.path.join(BASE_DIR, "Sample Tokenized Text", "Tahir Hakim.txt"),
        os.path.join(BASE_DIR, "Sample Tokenized Text", "liberal capitalism.txt"),
    ],
}

# ============================================
# 2. Diminishing Returns Analysis
# ============================================

def compute_diminishing_returns(results):
    dr_rows = []

    for algo in CONFIG["ALGORITHMS"]:
        # Average metrics across texts for each vocab size
        for vs in CONFIG["VOCAB_SIZES"]:
            matching = [r for r in results if r["algorithm"] == algo and r["vocab_size_target"] == vs]
            if matching:
                avg_comp = sum(r["compression_ratio"] for r in matching) / len(matching)
                avg_fert = sum(r["fertility"] for r in matching) / len(matching)
                avg_tokens = sum(r["token_count"] for r in matching) / len(matching)
                dr_rows.append({
                    "algorithm": algo,
                    "vocab_size": vs,
                    "avg_compression": round(avg_comp, 4),
                    "avg_fertility": round(avg_fert, 4),
                    "avg_tokens": round(avg_tokens, 1),
                })

    # Compute marginal gains
    for algo in CONFIG["ALGORITHMS"]:
        algo_rows = sorted([r for r in dr_rows if r["algorithm"] == algo],
                           key=lambda x: x["vocab_size"])
        for i in range(1, len(algo_rows)):
            prev = algo_rows[i - 1]
            curr = algo_rows[i]
            delta_vocab = curr["vocab_size"] - prev["vocab_size"]
            delta_comp = curr["avg_compression"] - prev["avg_compression"]
            delta_fert = curr["avg_fertility"] - prev["avg_fertility"]
            delta_tokens = curr["avg_tokens"] - prev["avg_tokens"]

            curr["marginal_compression_per_1K"] = round(
                delta_comp / (delta_vocab / 1000) if delta_vocab > 0 else 0, 6)
            curr["marginal_fertility_per_1K"] = round(
                delta_fert / (delta_vocab / 1000) if delta_vocab > 0 else 0, 6)
            curr["marginal_tokens_per_1K"] = round(
                delta_tokens / (delta_vocab / 1000) if delta_vocab > 0 else 0, 4)

    return dr_rows


def find_optimal_vocab(dr_rows):
    recommendations = []
    for algo in CONFIG["ALGORITHMS"]:
        algo_rows = sorted([r for r in dr_rows if r["algorithm"] == algo],
                           key=lambda x: x["vocab_size"])
        if len(algo_rows) < 2:
            continue

        best = algo_rows[0]
        for r in algo_rows:
            if r["avg_compression"] > best["avg_compression"]:
                best = r

        marginal = best.get("marginal_compression_per_1K", None)
        verdict = "optimal" if marginal is None or abs(marginal) > 0.001 else "diminishing returns"

        recommendations.append({
            "algorithm": algo,
            "recommended_vocab": best["vocab_size"],
            "compression": best["avg_compression"],
            "fertility": best["avg_fertility"],
            "verdict": verdict,
        })
    return recommendations


def generate_ablation_report(results, dr_rows, recommendations, output_path):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = []
    lines.append("# 🔬 Vocabulary Size Ablation — Balochi Tokenizer Optimization Report\n")
    lines.append(f"**Generated on:** {timestamp}\n")
    lines.append("## 1. Experiment Design\n")
    lines.append("| Parameter | Value |")
    lines.append("|-----------|-------|")
    lines.append(f"| **Algorithms** | {', '.join(CONFIG['ALGORITHMS'])} |")
    lines.append(f"| **Vocab Sizes** | {', '.join(f'{v:,}' for v in CONFIG['VOCAB_SIZES'])} |")
    lines.append(f"| **Eval Texts** | {len(CONFIG['EVAL_TEXTS'])} |")
    lines.append("")

    texts = sorted(set(r["text"] for r in results))
    for text_name in texts:
        lines.append(f"## 2. Results — `{text_name}`\n")
        for algo in CONFIG["ALGORITHMS"]:
            algo_results = sorted(
                [r for r in results if r["algorithm"] == algo and r["text"] == text_name],
                key=lambda x: x["vocab_size_target"]
            )
            if not algo_results:
                continue

            lines.append(f"### {algo.upper()} Algorithm\n")
            lines.append("| Metric | " + " | ".join(f"{r['vocab_size_target']//1000}K" for r in algo_results) + " |")
            lines.append("|--------|" + "|".join("------" for _ in algo_results) + "|")

            metric_rows = [
                ("Actual Vocab", "vocab_size_actual", "{:,}"),
                ("Token Count", "token_count", "{:,}"),
                ("Unique Tokens", "unique_tokens", "{:,}"),
                ("**Compression Ratio**", "compression_ratio", "**{:.4f}**"),
                ("**Fertility**", "fertility", "**{:.4f}**"),
                ("Avg Token Length", "avg_token_length", "{:.4f}"),
                ("UNK Count", "unk_count", "{:,}"),
                ("UNK Rate (%)", "unk_rate", "{:.4f}%"),
                ("Continuation Rate (%)", "continuation_rate", "{:.2f}%"),
                ("Vocab Utilization (%)", "vocab_utilization", "{:.2f}%"),
                ("Speed (tok/s)", "speed", "{:,.0f}"),
                ("Roundtrip Fidelity", "roundtrip_fidelity", "{}"),
            ]

            for label, key, fmt in metric_rows:
                vals = []
                for r in algo_results:
                    v = r.get(key, None)
                    if v is None:
                        vals.append("—")
                    elif isinstance(v, bool):
                        vals.append("✅ Yes" if v else "✗ No")
                    else:
                        vals.append(fmt.format(v))
                lines.append(f"| {label} | " + " | ".join(vals) + " |")
            lines.append("")

    lines.append("## 3. Diminishing Returns Analysis\n")
    for algo in CONFIG["ALGORITHMS"]:
        algo_dr = sorted([r for r in dr_rows if r["algorithm"] == algo],
                         key=lambda x: x["vocab_size"])
        if not algo_dr:
            continue

        lines.append(f"### {algo.upper()}\n")
        lines.append("| Vocab Size | Avg Compression | Avg Fertility | Avg Tokens | "
                      "Δ Compression/1K | Δ Fertility/1K | Δ Tokens/1K |")
        lines.append("|:----------:|:---------------:|:-------------:|:----------:|"
                      ":-----------------:|:--------------:|:-----------:|")

        for r in algo_dr:
            mc = r.get("marginal_compression_per_1K", "—")
            mf = r.get("marginal_fertility_per_1K", "—")
            mt = r.get("marginal_tokens_per_1K", "—")
            mc_str = f"{mc}" if mc == "—" else f"{mc:.6f}"
            mf_str = f"{mf}" if mf == "—" else f"{mf:.6f}"
            mt_str = f"{mt}" if mt == "—" else f"{mt:.4f}"
            lines.append(f"| {r['vocab_size']:,} | {r['avg_compression']:.4f} | "
                          f"{r['avg_fertility']:.4f} | {r['avg_tokens']:.0f} | "
                          f"{mc_str} | {mf_str} | {mt_str} |")
        lines.append("")

    lines.append("## 4. Optimal Vocabulary Size Recommendations\n")
    if recommendations:
        lines.append("| Algorithm | Recommended Vocab | Compression | Fertility | Verdict |")
        lines.append("|-----------|:-----------------:|:-----------:|:---------:|:-------:|")
        for rec in recommendations:
            lines.append(f"| {rec['algorithm'].upper()} | "
                          f"**{rec['recommended_vocab']:,}** | "
                          f"{rec['compression']:.4f} | "
                          f"{rec['fertility']:.4f} | "
                          f"{rec['verdict']} |")
        lines.append("")

    report_text = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    return report_text

def generate_csv(results, output_path):
    if not results: return
    keys = list(results[0].keys())
    with open(output_path, "w", encoding="utf-8-sig") as f:
        f.write(",".join(keys) + "\n")
        for row in results:
            vals = []
            for k in keys:
                v = row.get(k, "")
                if isinstance(v, float):
                    vals.append(f"{v}")
                elif isinstance(v, bool):
                    vals.append("Yes" if v else "No")
                elif v is None:
                    vals.append("")
                else:
                    vals.append(str(v))
            f.write(",".join(vals) + "\n")

# ============================================
# 3. Main Execution
# ============================================

def main():
    print("=" * 80)
    print("  🔬 Vocabulary Size Ablation Analysis")
    print("=" * 80)

    # 1. Load evaluation texts
    eval_texts = {}
    for path in CONFIG["EVAL_TEXTS"]:
        if os.path.exists(path):
            name = os.path.basename(path)
            with open(path, "r", encoding="utf-8") as f:
                eval_texts[name] = f.read()
    if not eval_texts:
        print("❌ No evaluation texts found.")
        return

    # 2. Load ablation tokenizers using imported function
    print("\n  Loading ablation tokenizers...")
    ablation_toks = load_ablation_tokenizers()
    if not ablation_toks:
        print("  ❌ No ablation tokenizers found. Please run Train_Tokenizers.py first.")
        return

    # 3. Evaluate each loaded tokenizer
    print("\n  Evaluating tokenizers...")
    all_results = []
    
    # Map the labels from ABLATION_MODELS to standard algorithm/size names
    for name, tok_info in ablation_toks.items():
        # name is something like 'Balochi_BPE_64K'
        parts = name.split('_')
        if len(parts) >= 3:
            algo_raw = parts[1].lower()
            if algo_raw == "wp": algo = "wordpiece"
            elif algo_raw == "sp": algo = "sentencepiece"
            else: algo = "bpe"
            
            vs_raw = parts[2].replace("K", "")
            if vs_raw.isdigit():
                vs = int(vs_raw) * 1000
            else:
                vs = 0
        else:
            algo = "unknown"
            vs = 0

        for text_name, text_content in eval_texts.items():
            print(f"    Evaluating {algo.upper()} {vs//1000}K on {text_name}...", end="")
            tokens, elapsed = tokenize_text(name, tok_info, text_content)
            metrics = compute_metrics(name, tok_info, tokens, elapsed, text_content)
            
            # Combine dicts
            result = {
                "algorithm": algo,
                "vocab_size_target": vs,
                "vocab_size_actual": metrics.get("vocab_size"),
                "text": text_name,
            }
            result.update(metrics)
            all_results.append(result)
            print(f" comp={metrics['compression_ratio']:.3f}")

    # 4. Diminishing returns & reporting
    print("\n  Computing diminishing returns...")
    dr_rows = compute_diminishing_returns(all_results)
    recommendations = find_optimal_vocab(dr_rows)

    report_path = os.path.join(OUTPUT_DIR, "Vocab_Ablation_Report.md")
    csv_path = os.path.join(OUTPUT_DIR, "ablation_results.csv")
    summary_path = os.path.join(OUTPUT_DIR, "ablation_summary.csv")

    print(f"  Generating report: {report_path}")
    generate_ablation_report(all_results, dr_rows, recommendations, report_path)
    
    print(f"  Generating CSV: {csv_path}")
    generate_csv(all_results, csv_path)
    generate_csv(dr_rows, summary_path)

    print("\n  ✅ Ablation analysis complete!")

if __name__ == "__main__":
    main()
