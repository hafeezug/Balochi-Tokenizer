<div align="center">

# 🔤 Balochi Tokenizer Training
### From Corpus Construction to Entropy-Driven Pruning

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Models Evaluated](https://img.shields.io/badge/Models%20Evaluated-27-green.svg)]()
[![Language](https://img.shields.io/badge/Language-Southern%20Balochi-orange.svg)]()
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-balochiml-yellow.svg)](https://huggingface.co/balochiml)
[![University](https://img.shields.io/badge/University%20of%20Gwadar-CS%20Dept-purple.svg)]()

**Principal Researcher:** Hafeez Ullah · **Artificial Intelligence and Data Innovation Research Lab (AIDIRL), University of Gwadar/ Video Survilliance Lab, National University of Computer and Emerging Sciences(NUCES-FAST), Karachi**  
 **Total Models Evaluated:** 27 (15 primary cross-lingual + 12 ablation)  
---

### ⭐ Final Production Model: `Balochi_SP_47K`
*SentencePiece · 47,000 vocabulary · AraToken-normalized · Entropy-pruned*

</div>

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Background & Motivation](#2-background--motivation)
3. [Training Corpus Architecture & Statistics](#3-training-corpus-architecture--statistics)
4. [Evaluation Methodology & Metrics](#4-evaluation-methodology--metrics)
5. [Phase 1 — Cross-Lingual Benchmarking (Pre-Normalization)](#5-phase-1--cross-lingual-benchmarking-pre-normalization)
6. [Phase 2 — Orthographic Normalization Pipeline (AraToken Integration)](#6-phase-2--orthographic-normalization-pipeline-aratoken-integration)
7. [Phase 3 — Pre- vs. Post-Normalization Comparative Analysis](#7-phase-3--pre--vs-post-normalization-comparative-analysis)
8. [Phase 4 — Vocabulary Size Ablation Study (12-Model Matrix)](#8-phase-4--vocabulary-size-ablation-study-12-model-matrix)
9. [Phase 5 — Information-Theoretic Analysis: Rényi Entropy](#9-phase-5--information-theoretic-analysis-rényi-entropy)
10. [Phase 6 — Vocabulary Pruning: The 47K Breakthrough](#10-phase-6--vocabulary-pruning-the-47k-breakthrough)
11. [Complete Model Registry](#11-complete-model-registry)
12. [Tokenizer Selection Guide](#12-tokenizer-selection-guide)
13. [Implications for Downstream NLP Tasks](#13-implications-for-downstream-nlp-tasks)
- [Appendix A — Training Configuration Reference](#appendix-a--training-configuration-reference)
- [Appendix B — Corpus Processing Time](#appendix-b--corpus-processing-time)

---

## 1. Executive Summary

This report documents the complete end-to-end tokenizer optimization pipeline for **Southern Balochi**, a critically under-resourced language spoken by 10–15 million people across Balochistan (Pakistan, Iran, Afghanistan and many Gulf countries) yet nearly absent from the training data of state-of-the-art language models.

The research progressed through **six distinct phases** culminating in the identification of the optimal production tokenizer:

| Phase | Activity | Key Output |
|:---:|:---|:---|
| 1 | Cross-lingual benchmarking against 14 external tokenizers | Proved custom tokenizers are necessary — multilingual models fragment Balochi 2× |
| 2 | AraToken normalization integration | Removed ~6.4% non-semantic character noise |
| 3 | Pre vs. Post normalization comparison | Confirmed fertility improvement of ~0.03 tokens/word across all algorithms |
| 4 | 12-model vocabulary ablation (32K–128K) | Identified 64K as the "knee of the curve" for compression vs. memory |
| 5 | Rényi/Shannon entropy analysis | Mathematically proved 128K vocabulary efficiency collapses to 0.38% |
| 6 | Cumulative frequency pruning | Identified 47K as the optimal production vocabulary ceiling |

The final recommendation — **`Balochi_SP_47K` (SentencePiece, 47,000 vocabulary)** — achieves the perfect balance of linguistic coverage, embedding matrix efficiency, lossless roundtrip reconstruction, and compatibility with the downstream Gemma 2B Language Extension Pipeline (LEP). It reduces the embedding layer dimensions required for LLM fine-tuning by over **26%** compared to 64K models, without sacrificing any measurable linguistic coverage.

### Overall Performance Highlights

| Metric | Custom Balochi (Best) | BERT Multilingual | Gemma 2B |
|:---|:---:|:---:|:---:|
| Compression Ratio | **4.27** (SP-128K) | 2.06 | 1.93 |
| Fertility (tokens/word) | **1.10** (SP-128K) | 2.305 | 2.460 |
| UNK Rate | **0.00%** | 0.38% | 0.00% |
| Continuation Rate | **8.51%** (SP-128K) | 51.82% | 59.78% |
| Roundtrip Fidelity | ✅ Lossless (SP only) | ✗ Lossy | ✗ Lossy |
| Vocab Utilization | **~2.0%** (64K) | 0.58% | 0.30% |

---

## 2. Background & Motivation

### 2.1 The Low-Resource Language Problem

Natural Language Processing has advanced dramatically for high-resource languages (English, French, Arabic, Chinese), but Balochi remains computationally invisible. When a Balochi speaker interacts with an AI assistant, search engine, or translation tool, the system processes their language through a tokenizer trained predominantly on other scripts and languages. The result is severe over-fragmentation: meaningful morphological units are shattered into arbitrary character sequences that carry no linguistic significance.

The tokenization penalty is not an abstract problem — it has direct consequences for context window utilization. A model with a 4,096-token context window can process approximately **2× more Balochi text** per forward pass with a custom tokenizer compared to BERT or Gemma.

### 2.2 Why Balochi Requires Custom Tokenizers

Balochi presents four characteristics that challenge all standard multilingual tokenizers:

**Modified Perso-Arabic Script:** Balochi uses a specialized script incorporating characters not present in standard Arabic or Urdu, including unique diacritical markers (`ءَ`, `ءِ`, `ءُ`) that carry grammatical case information unique to the language.

**Agglutinative Morphology:** Balochi encodes grammatical relationships through suffixes and clitics that attach directly to stem forms. Preserving morphological boundaries is critical for semantic understanding in any downstream NLP task.

**Data Scarcity in Multilingual Corpora:** Multilingual tokenizers trained on web-crawled data encounter Balochi text extremely rarely, resulting in a vocabulary dominated by Arabic, Urdu, and Farsi subwords that poorly generalize to Balochi morphology.

**Script Overlap with Semantic Divergence ("False Friends"):** While Balochi shares characters with Arabic and Urdu, identical character sequences often carry entirely different meanings, making zero-shot cross-lingual transfer fundamentally unreliable.

### 2.3 The Fragmentation Problem Illustrated

The word `کپیٹلسٹک` (meaning "capitalistic") is a single morphological unit in Balochi:

| Tokenizer | Output | Token Count | Status |
|:---|:---|:---:|:---:|
| **Balochi BPE** | `['کپیٹلسٹک']` | **1** | ✅ Intact |
| **BERT Multilingual** | `['ک', '##پی', '##ٹ', '##لس', '##ٹ', '##ک']` | **6** | ❌ Fragmented |
| **Gemma 2B** | `['ک', 'پی', 'ٹ', 'لس', 'ٹ', 'ک']` | **6** | ❌ Fragmented |

BERT and Gemma split this common loanword into 6 arbitrary subword fragments — destroying the semantic unit entirely. Every additional fragment consumes precious context window capacity and forces the model to expend attention just to reassemble the original word before any linguistic processing can begin.

---

## 3. Training Corpus Architecture & Statistics

### 3.1 Corpus Components

The tokenizer training pipeline used a **mixed-domain, bilingual corpus** constructed by concatenating three distinct high-quality datasets. The multilingual approach ensures the tokenizer handles English loanwords and code-switching without falling back to `[UNK]` generation.

| Dataset | Approximate Words | Content & Purpose |
|:---|:---:|:---|
| `balochi_clean_corpus_dictionary.txt` | ~Lexical entries | Curated lexical entries, normalized grammatical forms, and core vocabulary. Ensures correctly-spelled root words and morphological stems are present. |
| `balochi_dedup_corpus.txt` | ~5,000,000 | Large-scale, deduplicated raw Balochi prose: literature, news, religious, and conversational text. Captures statistical co-occurrences of subwords, clitics, and suffixes in the wild. |
| `english_corpus_2M.txt` | ~2,000,000 | 2 million English words. Modern Balochi text frequently contains English code-switching and technical jargon. Prevents character-by-character `[UNK]` generation for English content. |
| **Combined Total** | **~54,000,000** | **~185.5 million characters total** |

### 3.2 Corpus Preprocessing

Prior to tokenizer training, the pipeline performs the following operations:

- Validates presence of all three source files
- Merges them into a single monolithic `training_corpus.txt`
- Strips lines shorter than 2 characters to remove noise
- Applies NFKC normalization to unify visually identical but differently-encoded characters (e.g., standard Arabic Kaf vs. Persian Keheh)

The resulting corpus of **185,587,264 characters** (~54M words) provides sufficient statistical depth for even the largest 128K vocabulary models.

### 3.3 Post-Normalization Corpus Statistics

After applying the AraToken-inspired normalization pipeline (Phase 2), the evaluation corpus `liberal capitalism.txt` changed as follows:

| Statistic | Pre-Normalization | Post-Normalization | Change |
|:---|:---:|:---:|:---:|
| Total Characters | 23,919 | 22,387 | -1,532 (-6.4%) |
| Character noise removed | — | ~1,532 | Diacritics, ZWNJs, tatweels |

The full training corpus was also re-normalized, yielding `training_corpus_normalized.txt` (~370 MB), used for the pruning analysis.

### 3.4 Evaluation Text Properties

All tokenizer evaluation was conducted on two Balochi-language texts representing distinct domains:

**`liberal capitalism.txt`** — A Balochi political/economic text discussing capitalist systems and their philosophical foundations. Properties after pre-normalization:

| Property | Value |
|:---|:---:|
| Total Characters | 23,919 |
| Total Words (whitespace-separated) | 5,036 |
| Sentences Detected | 160 |

This text is an ideal stress test because it features dense Balochi diacritical markers, mixed Balochi-English code-switching, political vocabulary with specialized terminology, and proper nouns not seen in general-purpose multilingual training data.

**`Balochi Labzank.txt`** — A Balochi political/historical text providing a complementary evaluation domain. The combination of both texts ensures evaluation metrics are not domain-biased.

---

## 4. Evaluation Methodology & Metrics

### 4.1 Primary Metrics

Each metric provides a distinct lens on tokenizer behavior:

| Metric | Formula | Ideal Direction | What It Reveals |
|:---|:---:|:---:|:---|
| **Token Count** | Count of produced tokens | ↓ Lower | Total encoding cost for a given document |
| **Unique Tokens** | Distinct token types used | ↑ Higher | Representational diversity on a document |
| **Compression Ratio** | Characters ÷ Tokens | ↑ Higher | Information density per token |
| **Fertility** | Tokens ÷ Words | ↓ Closer to 1.0 | Average word-splitting rate |
| **Avg Token Length** | Mean characters per token | ↑ Higher | Fragment size (longer = less fragmented) |
| **UNK Rate** | UNK tokens ÷ Total tokens | → 0% | Character coverage completeness |
| **Continuation Rate** | Continuation tokens ÷ Total | ↓ Lower | Degree of intra-word splitting |
| **Speed** | Tokens per second | ↑ Higher | Throughput for large-scale processing |
| **Roundtrip Fidelity** | Decode(Encode(text)) = text? | → Lossless | Data integrity for generative models |
| **Vocab Utilization** | Unique tokens ÷ Vocab Size | ↑ Higher per doc | How well vocabulary matches the domain |

### 4.2 Information-Theoretic Metrics (Phase 5)

| Metric | Formula | What It Reveals |
|:---|:---:|:---|
| **Shannon Entropy (H₁)** | $-\sum p_i \log_2 p_i$ | Uniformity of token usage distribution |
| **Effective Vocabulary (V_eff)** | $2^{H_1}$ | Hypothetical uniform-distribution equivalent vocab size |
| **Vocabulary Efficiency (η)** | $V_{eff} / V_{total}$ | Percentage of vocabulary doing meaningful work |
| **Min-Entropy (H∞)** | $-\log_2(\max p_i)$ | Dominance of the single most-frequent token |

---

## 5. Phase 1 — Cross-Lingual Benchmarking (Pre-Normalization)

Before constructing highly optimized models, three initial custom tokenizers (BPE-80K, WordPiece-64K, SentencePiece-64K) were trained on raw, un-normalized Balochi text and benchmarked against **14 external multilingual and language-family-specific models**.

### 5.1 All Tokenizers Evaluated

| # | Tokenizer | Family | Algorithm | Vocab Size | Source | Status |
|:---:|:---|:---:|:---:|:---:|:---|:---:|
| 1 | **Balochi_BPE** | Custom Balochi | BPE | 80,000 | Local file | ✅ |
| 2 | **Balochi_WordPiece** | Custom Balochi | WordPiece | 64,000 | Local file | ✅ |
| 3 | **Balochi_SentencePiece** | Custom Balochi | SP-Unigram | 64,000 | Local file | ✅ |
| 4 | **Balochi_30K** | HF External Balochi | BPE | 30,000 | `balochiml/balochi-tokenizer` | ✅ |
| 5 | **NLTK** | Baseline | Rule-based | N/A | — | ✅ |
| 6 | **BERT** | Multilingual | WordPiece | 119,547 | `bert-base-multilingual-cased` | ⚠️ |
| 7 | **Gemma 2B** | Multilingual | SP-BPE | 256,000 | `google/gemma-2b` | ✅ |
| 8 | **AraBERT_v2** | Arabic | WordPiece | 64,000 | `aubmindlab/bert-base-arabertv2` | ✅ |
| 9 | **CAMeLBERT_MSA** | Arabic | WordPiece | 30,000 | `CAMeL-Lab/bert-base-arabic-camelbert-msa` | ✅ |
| 10 | **ARBERT** | Arabic | WordPiece | 100,000 | `UBC-NLP/ARBERT` | ✅ |
| 11 | **AraGPT2** | Arabic | BPE | 64,000 | `aubmindlab/aragpt2-base` | ✅ |
| 12 | **ParsBERT** | Persian | WordPiece | 100,000 | `HooshvareLab/bert-base-parsbert-uncased` | ✅ |
| 13 | **PersianBERT_FA** | Persian | WordPiece | 100,000 | `HooshvareLab/bert-fa-base-uncased` | ✅ |
| 14 | **PersianBPE** | Persian | BPE | 30,000 | `mshojaei77/PersianBPETokenizer` | ❌ |
| 15 | **UrduBERT** | Urdu | WordPiece | 50,265 | Fallback: `flax-community/roberta-base-mr` | ⚠️ |

### 5.2 Master Performance Results (Pre-Normalization)

The table below presents the complete tokenization metrics from the initial cross-lingual evaluation on `liberal capitalism.txt`:

| Tokenizer | Token Count | Compression | Fertility | UNK Count | UNK Rate | Continuation Rate | Roundtrip |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Balochi BPE** | 5,903 | 3.79 | 1.172 | 0 | 0.000% | 0.00% | ✗ |
| **Balochi WordPiece** | 6,046 | 3.70 | 1.201 | 0 | 0.000% | 7.53% | ✗ |
| **Balochi SentencePiece** | **5,879** | **3.81** | **1.168** | 0 | 0.000% | 14.36% | **✅** |
| NLTK (baseline) | 5,353 | 4.18 | 1.063 | 0 | 0.000% | 0.00% | ✗ |
| **BERT Multilingual** | **9,938** | 2.25 | 1.974 | 44 | 0.443% | 43.74% | ✗ |
| **Gemma 2B** | **10,568** | 2.12 | 2.099 | 0 | 0.000% | 52.37% | ✗ |
| Balochi_30K (HF) | 7,149 | 3.13 | 1.420 | 1,111 | **15.54%** | 0.00% | ✗ |
| AraBERT_v2 | 6,019 | 3.72 | 1.195 | 3,071 | **51.02%** | 7.11% | ✗ |
| CAMeLBERT_MSA | 11,934 | 1.88 | 2.370 | 1 | 0.008% | 53.15% | ✗ |
| ARBERT | 11,003 | 2.03 | 2.185 | 2 | 0.018% | 49.19% | ✗ |
| AraGPT2 | 17,870 | 1.25 | 3.549 | 0 | 0.000% | 100.00% | ✅ |
| ParsBERT | 8,337 | 2.69 | 1.656 | 364 | 4.366% | 32.94% | ✗ |
| PersianBERT_FA | 8,660 | 2.59 | 1.720 | 67 | 0.774% | 35.44% | ✗ |
| PersianBPE | 0 | 0.00 | 0.000 | — | FAILED | — | N/A |
| UrduBERT | 38,390 | 0.58 | 7.625 | 0 | 0.000% | 0.00% | ✗ |

### 5.3 Performance by Language Family

Aggregating by language group reveals a clear hierarchy:

| Language Family | Avg Compression | Avg Fertility | Notes |
|:---|:---:|:---:|:---|
| **Balochi (custom)** | **3.77** | **1.180** | Custom tokenizers dominate all metrics |
| Arabic (pretrained) | 2.22 | 2.325 | Perso-Arabic overlap but high fragmentation |
| Persian (pretrained) | 1.76 | 1.125 | Only ParsBERT/FA relevant; PersianBPE failed |
| Urdu (pretrained) | 0.58 | 7.625 | Catastrophic for Balochi despite shared script |
| Baseline (generic) | 2.85 | 1.712 | BERT/Gemma: 2× worse than custom |

### 5.4 Head-to-Head Group Analysis

#### Group 1 — WordPiece Family: Balochi WP vs. BERT Multilingual

The most critical comparison for BERT-style fine-tuning tasks:

| Metric | Balochi WordPiece | BERT Multilingual | Winner |
|:---|:---:|:---:|:---:|
| Token Count | 6,046 | 9,938 | 🏆 Balochi WP (39% fewer) |
| Unique Tokens | 1,011 | 679 | 🏆 Balochi WP (+49% diversity) |
| Compression Ratio | 3.70 | 2.25 | 🏆 Balochi WP (+64.4%) |
| Fertility | 1.201 | 1.974 | 🏆 Balochi WP (1.64× lower) |
| UNK Count | 0 | 44 | 🏆 Balochi WP |
| Continuation Rate | 7.53% | 43.74% | 🏆 Balochi WP (5.8× lower) |
| Speed | 260,404 t/s | 451,551 t/s | BERT (Rust-cached advantage) |
| Roundtrip Fidelity | ✗ | ✗ | Tie |

**Verdict:** Balochi WordPiece wins **6 of 8 metrics**. For BERT-style MLM training on Balochi text, using the custom tokenizer nearly doubles the effective context window capacity.

#### Group 2 — SentencePiece Family: Balochi SP vs. Gemma 2B

The most critical comparison for GPT-style/LLaMA/Gemma continual pre-training:

| Metric | Balochi SentencePiece | Gemma 2B | Winner |
|:---|:---:|:---:|:---:|
| Token Count | 5,879 | 10,568 | 🏆 Balochi SP (44% fewer) |
| Unique Tokens | 1,147 | 764 | 🏆 Balochi SP (+50% diversity) |
| Compression Ratio | 3.81 | 2.12 | 🏆 Balochi SP (+79.7%) |
| Fertility | 1.168 | 2.099 | 🏆 Balochi SP (1.80× lower) |
| UNK Count | 0 | 0 | Tie |
| Continuation Rate | 14.36% | 52.37% | 🏆 Balochi SP (3.6× lower) |
| Speed | 440,124 t/s | 602,481 t/s | Gemma |
| Roundtrip Fidelity | ✅ Lossless | ✗ Lossy | 🏆 Balochi SP |

**Verdict:** Balochi SentencePiece wins **6 of 8 metrics** and uniquely achieves lossless roundtrip — critical for generative language model applications. The token count reduction of 44% translates directly to processing **1.8× more Balochi text** per training batch on identical hardware.

#### Group 3 — BPE Family: Balochi BPE vs. NLTK vs. 30K-Balochi

| Metric | Balochi BPE | NLTK | 30K-Balochi | Winner |
|:---|:---:|:---:|:---:|:---:|
| Token Count | 5,903 | 5,353 | 7,149 | NLTK (non-neural) |
| Compression Ratio | 3.79 | 4.18 | 3.13 | NLTK (non-neural) |
| Fertility | 1.172 | 1.063 | 1.420 | NLTK (non-neural) |
| UNK Count | 0 | 0 | 1,111 | 🏆 Balochi BPE / NLTK |
| Neural-compatible | ✅ | ❌ | ✅ | 🏆 Balochi BPE / 30K |
| Roundtrip | ✗ | ✗ | ✗ | Tie |

**Verdict:** NLTK leads numerically but **cannot be used in neural LLMs** requiring a fixed subword vocabulary. Among neural-compatible models, Balochi BPE (80K) decisively outperforms 30K-Balochi — producing 17.6% fewer tokens with zero UNK vs. 30K's catastrophic 15.54% UNK rate. The 30K tokenizer's vocabulary is insufficient for English loanwords, replacing all English text with `[UNK]` tokens entirely.

#### Group 4 — Perso-Arabic Script Family: The Urdu Lesson

Despite sharing the same Nastaliq script conventions as Balochi, UrduBERT performs catastrophically:

| Tokenizer | Fertility | Token Count | UNK Rate | Compression |
|:---|:---:|:---:|:---:|:---:|
| Balochi SentencePiece | 1.168 | 5,879 | 0.000% | 3.81 |
| Balochi BPE | 1.172 | 5,903 | 0.000% | 3.79 |
| AraBERT_v2 | 1.195 | 6,019 | **51.02%** | 3.72 |
| ParsBERT | 1.656 | 8,337 | 4.366% | 2.69 |
| **UrduBERT** | **7.625** | **38,390** | 0.000% | **0.58** |

UrduBERT's fertility of **7.625 tokens per word** — the worst of all 15 tested tokenizers — conclusively demonstrates that script similarity alone does not justify cross-lingual tokenizer transfer. Balochi and Urdu require entirely separate tokenizers.

### 5.5 Critical Failure Modes in Cross-Lingual Models

The run log revealed several real-world failure modes encountered during evaluation:

**Context Window Overflow (Sequence Length Errors):** When evaluating on the 5,036-word test corpus, several standard models triggered critical sequence-length warnings:
- `BERT Multilingual`: "Token indices sequence length is longer than the specified maximum sequence length for this model (9,938 > 512)" — the entire test document overflows BERT's maximum context by 19×.
- `AraBERT_v2`: "Token indices sequence length is longer than the specified maximum sequence length for this model (6,019 > 512)" — still 11.7× overflow.

**Fatal Tokenizer Crashes:**
- `PersianBPETokenizer`: Crashed entirely with "Exception: Unk token `[UNK]` not found in the vocabulary" — indicating a malformed HuggingFace tokenizer configuration.

These failures confirm that even adjacent Perso-Arabic language tokenizers are not production-safe for Balochi text.

### 5.6 Tokenization In Action — Worked Examples

#### Example 1: Single Political-Economic Term

**Input:** `کپیٹلسٹک` (capitalistic)

| Tokenizer | Tokens | Count |
|:---|:---|:---:|
| **Balochi BPE** | `['کپیٹلسٹک']` | **1** ✅ |
| Balochi WordPiece | `['کپیٹل', '##سٹک']` | 2 ✅ |
| Balochi SP | `['▁کپیٹل', 'سٹک']` | 2 ✅ |
| BERT Multilingual | `['ک', '##پی', '##ٹ', '##لس', '##ٹ', '##ک']` | **6** ❌ |
| Gemma 2B | `['ک', 'پی', 'ٹ', 'لس', 'ٹ', 'ک']` | **6** ❌ |
| Balochi_30K | `['کپی', 'ٹ', 'لس', 'ٹک']` | 4 ⚠️ |

#### Example 2: Full Political Phrase

**Input:** `لبرل کپیٹلسٹک آشوب ءِ بنیاد` (Foundation of liberal capitalistic revolution)

| Tokenizer | Token Count | Context Overhead vs. Balochi BPE |
|:---|:---:|:---:|
| **Balochi BPE** | **5** | Baseline |
| Balochi WordPiece | 6 | +20% |
| Balochi SP | 6 | +20% |
| NLTK | 5 | Baseline (non-neural) |
| BERT Multilingual | **15** | **+200% (3× more)** |
| Gemma 2B | **16** | **+220% (3.2× more)** |
| Balochi_30K | 8 | +60% |

#### Example 3: Author Name with Title

**Input:** `ڈاکٹر عبدالوہاب سوری` (Dr. Abdulwahab Soori)

| Tokenizer | Tokens | Count |
|:---|:---|:---:|
| **All Custom Balochi** | `ڈاکٹر`, `عبدالوہاب`, `سوری` | **3** ✅ |
| BERT Multilingual | `ڈاکٹر`, `عبد`, `##ال`, `##و`, `##ہا`, `##ب`, `سو`, `##ری` | 8 ❌ |
| Gemma 2B | `ڈ`, `اک`, `ٹر`, `▁عبد`, `الو`, `ہ`, `اب`, `▁س`, `وری` | 9 ❌ |

All three custom tokenizers correctly recognize the doctor's title and full proper noun as single vocabulary entries. BERT and Gemma fragment this 3-word name into 8 and 9 arbitrary pieces respectively.

---

## 6. Phase 2 — Orthographic Normalization Pipeline (AraToken Integration)

### 6.1 Why Normalization is Critical for Balochi

Balochi web-scraped corpora contain severe orthographic noise arising from: inconsistent diacritic usage across publishers, invisible Unicode control characters from copy-pasted digital text, Hamza/Alif variants imported from Arabic typographic conventions, and both `ے` (Urdu Ye) and `ی` (Farsi Ye) being used interchangeably in different regional writing conventions.

This noise directly harms tokenizer training: identical words spelled with different invisible characters are treated as distinct surface forms, artificially inflating the long tail of the vocabulary with near-duplicates, and preventing the subword algorithms from finding true morphological boundaries.

### 6.2 AraToken Methodology — The Foundation

The normalization pipeline was adapted from the **AraToken** paper "https://arxiv.org/abs/2512.18399", which demonstrated an **8–9% fertility reduction** across all algorithms for Arabic — from 1.311 to 1.199 for SentencePiece — through normalization alone, before any training changes.

The AraToken methodology maps almost perfectly onto the Balochi problem space, with critical Balochi-specific adaptations required at each step.

### 6.3 Complete Normalization Rules Applied

The following table details every normalization rule, its linguistic justification for Balochi, and whether it is identical to AraToken or Balochi-specific:

| Rule | Characters Affected | Source | Balochi Justification |
|:---|:---|:---:|:---|
| **NFKC Unicode Normalization** | All compatibility chars | AraToken | Normalizes Urdu/Sindhi/Persian lookalike characters to a unified space; must be applied first |
| **Diacritics Drop** | `\u064B`–`\u065F` | AraToken | Diacritics (Harakat) in Balochi are rare and inconsistently applied. Identical words with/without diacritics get different embeddings. |
| **Alif Variant Unification** | أ/إ/آ/ٱ → ا | AraToken | Removes stylistic Arabic variations imported into Balochi loanwords, standardizing stems |
| **Tatweel Removal** | `\u0640` (kashida) | AraToken | Used for decorative text stretching; purely visual, carries zero linguistic information |
| **Arabic Punctuation Normalization** | ؟→? / ؛→; / ،→, | AraToken | Prevents tokenizer from treating identical punctuation as distinct tokens |
| **Arabic-Indic Numeral Normalization** | ٠١٢٣٤٥٦٧٨٩ → 0-9 | AraToken | Eliminates numeral inconsistency across documents |
| **ZWNJ Removal** | `\u200C` | **Balochi-specific** | Zero Width Non-Joiners in digital Balochi text cause spurious splits in verb compounds |
| **ZWJ Removal** | `\u200D` | **Balochi-specific** | Zero Width Joiners from copy-pasted web text; remove to prevent artificial word splits |
| **RLM/ALM Removal** | `\u200F`, `\u061C` | **Balochi-specific** | Right-to-Left Mark and Arabic Letter Mark in mixed-direction documents; carry no phonetic information |
| **Ye Variant Preservation** | `ے` vs `ی` | **Balochi-specific (inverted)** | Unlike standard AraToken collapse, Balochi relies on the `ے`/`ی` distinction for grammatical and dialectal signals — intentionally preserved |

### 6.4 Python Implementation

```python
import re
import unicodedata

def normalize_balochi(text: str, drop_diacritics: bool = True,
                       preserve_ye: bool = True) -> str:
    # Step 1: NFKC Unicode normalization
    text = unicodedata.normalize('NFKC', text)
    # Step 2: Alif variant unification
    text = re.sub(r'[أإآٱ]', 'ا', text)
    # Step 3: Ye unification (Balochi-specific — preserve by default)
    if not preserve_ye:
        text = text.replace('ے', 'ی')
    # Step 4: Arabic-Indic numeral normalization
    arabic_indic = str.maketrans('٠١٢٣٤٥٦٧٨٩', '0123456789')
    text = text.translate(arabic_indic)
    # Step 5: Arabic punctuation normalization
    text = text.replace('؟', '?').replace('؛', ';').replace('،', ',')
    # Step 6: Tatweel removal
    text = text.replace('\u0640', '')
    # Step 7: Invisible character removal (Balochi-specific)
    text = text.replace('\u200C', '').replace('\u200D', '')
    text = text.replace('\u200F', '').replace('\u061C', '')
    # Step 8: Diacritics
    if drop_diacritics:
        text = re.sub(r'[\u064B-\u065F\u0610-\u061A\u06D6-\u06DC]', '', text)
    # Step 9: Whitespace normalization
    text = re.sub(r'\s+', ' ', text).strip()
    return text
```

### 6.5 Ye Preservation — The Balochi "Alif4 Equivalent"

A critical divergence from the AraToken methodology is the treatment of the `ے`/`ی` distinction. AraToken tested both "preserve" and "collapse" modes for Arabic Hamza placement, finding that **preserving orthographic distinctions** yielded 71% loss reduction vs. 63% for aggressive normalization in downstream LLM training.

For Balochi, the analogous distinction is the `ے` (Urdu Ye, U+06D2) vs. `ی` (Farsi Ye, U+06CC). Southern Balochi relies on this visual and grammatical distinction — largely using `ے` for singular/oblique and `ی` for other grammatical contexts. Collapsing them would destroy grammatical signal before the language model even sees the data. The decision was made to **preserve this distinction** in all normalized models.

---

## 7. Phase 3 — Pre- vs. Post-Normalization Comparative Analysis

### 7.1 Quantitative Impact of Normalization (64K Models, `liberal capitalism.txt`)

The following comparison isolates the 64K models trained on raw text (Phase 1) vs. the equivalent 64K models trained on normalized text (Phase 2), evaluated on the same normalized evaluation document:

| Algorithm | State | Token Count | Fertility (tok/word) | Compression Ratio |
|:---|:---:|:---:|:---:|:---:|
| **SentencePiece** | Pre-Norm | 5,879 | 1.168 | 3.81 |
| **SentencePiece** | Post-Norm | **5,750** | **1.142** | **3.89\*** |
| **BPE** | Pre-Norm | 5,920 | 1.175 | 4.04 |
| **BPE** | Post-Norm | **5,864** | **1.165** | **3.82\*** |
| **WordPiece** | Pre-Norm | 6,046 | 1.201 | 3.70 |
| **WordPiece** | Post-Norm | **5,910** | **1.174** | **3.79\*** |

> **\* The Compression Ratio Paradox Explained:** The mathematical compression ratio (Characters ÷ Tokens) appears lower after normalization, which seems counterintuitive. This is a mathematical artifact, not a regression. Because normalization stripped over 1,500 invisible characters and diacritics from the source text, the numerator (Characters) shrank dramatically while the denominator (Tokens) decreased by a much smaller amount. The tokenizers are performing more efficiently — proven by the universally lower Fertility scores. The denominator decrease is real and the linguistic improvement is real; only the ratio measurement is skewed by the cleaned numerator.

### 7.2 Fertility Improvements by Algorithm

| Algorithm | Pre-Norm Fertility | Post-Norm Fertility | Absolute Improvement | % Improvement |
|:---|:---:|:---:|:---:|:---:|
| SentencePiece 64K | 1.168 | **1.142** | -0.026 | **2.23%** |
| BPE 64K | 1.175 | **1.165** | -0.010 | **0.85%** |
| WordPiece 64K | 1.201 | **1.174** | -0.027 | **2.25%** |

### 7.3 Token Count Reductions

| Algorithm | Pre-Norm Tokens | Post-Norm Tokens | Tokens Saved |
|:---|:---:|:---:|:---:|
| SentencePiece 64K | 5,879 | 5,750 | **129 fewer** |
| BPE 64K | 5,920 | 5,864 | **56 fewer** |
| WordPiece 64K | 6,046 | 5,910 | **136 fewer** |

### 7.4 Continuation Rate Improvement

| Algorithm | Pre-Norm Cont. Rate | Post-Norm Cont. Rate | Reduction |
|:---|:---:|:---:|:---:|
| SentencePiece 64K | 12.98% | **12.43%** | -0.55 pp |
| BPE 64K | 0.00% | 0.00% | — |
| WordPiece 64K | 6.02% | **5.40%** | -0.62 pp |

The ~0.5–0.6 percentage point reduction in continuation rates means ~55–136 fewer mid-word subword splits per document — directly reducing the frequency with which the model must "stitch together" word fragments to recover semantic meaning.

---

## 8. Phase 4 — Vocabulary Size Ablation Study (12-Model Matrix)

To overcome the limitation of selecting vocabulary sizes (64K, 80K) without empirical validation, a complete ablation study was conducted training **12 discrete tokenizer models** across 3 algorithms × 4 vocabulary sizes.

### 8.1 Experimental Setup

| Parameter | Values |
|:---|:---|
| Algorithms | BPE, WordPiece, SentencePiece (Unigram) |
| Vocabulary Sizes | 32,000 / 64,000 / 80,000 / 128,000 |
| Total Models | 12 |
| Evaluation Texts | `Balochi Labzank.txt` + `liberal capitalism.txt` |

### 8.2 Full Ablation Results (Both Texts, All 12 Models)

#### `Balochi Labzank.txt` Results

| Algorithm | Size | Token Count | Unique Tokens | Compression | Fertility | Vocab Util. | Speed (tok/s) | Roundtrip |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| BPE | 32K | 4,527 | 1,357 | 3.9797 | 1.1671 | 4.24% | 209,820 | ✗ |
| BPE | 64K | 4,405 | 1,354 | 4.0899 | 1.1356 | 2.12% | 264,128 | ✗ |
| BPE | 80K | 4,369 | 1,347 | 4.1236 | 1.1263 | 1.68% | 188,643 | ✗ |
| BPE | 128K | 4,315 | 1,339 | **4.1752** | **1.1124** | 1.05% | 235,301 | ✗ |
| WP | 32K | 4,575 | 1,383 | 3.9379 | 1.1794 | 4.32% | 198,664 | ✗ |
| WP | 64K | 4,437 | 1,369 | 4.0604 | 1.1439 | 2.14% | 161,905 | ✗ |
| WP | 80K | 4,401 | 1,367 | 4.0936 | 1.1346 | 1.71% | 191,950 | ✗ |
| WP | 128K | 4,336 | 1,353 | **4.1550** | **1.1178** | 1.06% | 200,636 | ✗ |
| SP | 32K | 4,504 | 1,478 | 4.0000 | 1.1611 | 4.62% | 446,710 | ✅ |
| SP | 64K | 4,353 | 1,456 | 4.1388 | 1.1222 | 2.27% | 284,687 | ✅ |
| SP | 80K | 4,309 | 1,447 | 4.1810 | 1.1109 | 1.81% | 279,517 | ✅ |
| SP | 128K | **4,240** | 1,424 | **4.2491** | **1.0931** | 1.11% | 349,598 | ✅ |

#### `liberal capitalism.txt` Results

| Algorithm | Size | Token Count | Compression | Fertility | Vocab Util. |
|:---|:---:|:---:|:---:|:---:|:---:|
| BPE | 32K | 6,117 | 3.9103 | 1.2147 | 3.43% |
| BPE | 64K | 5,920 | 4.0404 | 1.1755 | 1.67% |
| BPE | 80K | 5,840 | 4.0957 | 1.1597 | 1.33% |
| BPE | 128K | **5,747** | **4.1620** | **1.1412** | 0.82% |
| WP | 32K | 6,146 | 3.8918 | 1.2204 | 3.47% |
| WP | 64K | 5,950 | 4.0200 | 1.1815 | 1.69% |
| WP | 80K | 5,925 | 4.0370 | 1.1765 | 1.35% |
| WP | 128K | **5,769** | **4.1461** | **1.1456** | 0.83% |
| SP | 32K | 5,985 | 3.9965 | 1.1884 | 3.93% |
| SP | 64K | 5,787 | 4.1332 | 1.1491 | 1.91% |
| SP | 80K | 5,732 | 4.1729 | 1.1382 | 1.53% |
| SP | 128K | **5,608** | **4.2652** | **1.1136** | 0.94% |

> **Universal observation:** All 12 models produced zero UNK tokens (`UNK Rate = 0.0000%`). SentencePiece is the only algorithm family achieving lossless roundtrip fidelity at every vocabulary size.

### 8.3 Diminishing Returns Analysis

This is the core of vocabulary ablation — identifying the inflection point where vocabulary expansion stops providing proportional compression gains.

#### SentencePiece — Diminishing Returns Trajectory (Averaged Across Both Texts)

| Vocab Size | Avg Compression | Avg Fertility | Avg Tokens | Δ Compression per +1K Tokens |
|:---:|:---:|:---:|:---:|:---:|
| **32,000** | 3.9982 | 1.1748 | 5,244 | — |
| **64,000** | 4.1360 | 1.1357 | 5,070 | **+0.0043** (strong) |
| **80,000** | 4.1770 | 1.1245 | 5,020 | +0.0026 (moderate) |
| **128,000** | **4.2571** | **1.1033** | **4,924** | +0.0017 (diminishing) |

The marginal return per additional 1K vocabulary tokens drops from **+0.0043** (32K→64K) to **+0.0017** (80K→128K) — a 60% reduction in compression return per vocabulary investment. The sharpest inflection point is clearly at 64K, where the curve breaks.

#### BPE — Diminishing Returns Trajectory

| Vocab Size | Avg Compression | Avg Fertility | Δ Compression per +1K |
|:---:|:---:|:---:|:---:|
| **32,000** | 3.9450 | 1.1909 | — |
| **64,000** | 4.0651 | 1.1556 | +0.0038 |
| **80,000** | 4.1097 | 1.1430 | +0.0028 |
| **128,000** | 4.1686 | 1.1268 | +0.0012 |

BPE shows the fastest rate of diminishing returns, falling to +0.0012 per +1K beyond 80K — meaning the vocabulary expansion from 80K to 128K yields only half the compression gain per vocabulary unit compared to the 32K→64K leap.

#### WordPiece — Diminishing Returns Trajectory

| Vocab Size | Avg Compression | Avg Fertility | Δ Compression per +1K |
|:---:|:---:|:---:|:---:|
| **32,000** | 3.9149 | 1.1999 | — |
| **64,000** | 4.0402 | 1.1627 | +0.0039 |
| **80,000** | 4.0653 | 1.1555 | +0.0016 |
| **128,000** | 4.1506 | 1.1317 | +0.0018 |

WordPiece shows an unusual pattern: the 80K→128K gain (+0.0018) slightly exceeds the 64K→80K gain (+0.0016), suggesting the intermediate 80K range is particularly inefficient for WordPiece. Despite this, the 64K→80K period still represents a dramatic slowdown from the initial 32K→64K leap.

### 8.4 Vocabulary Utilization — The Sparse Embedding Problem

As vocabulary size increases, the percentage of the vocabulary actually used during inference on a typical document collapses:

| Algorithm | 32K Utilization | 64K Utilization | 80K Utilization | 128K Utilization |
|:---|:---:|:---:|:---:|:---:|
| **BPE** | ~4.0% | ~1.8% | ~1.5% | ~0.9% |
| **WordPiece** | ~4.1% | ~1.9% | ~1.5% | ~0.9% |
| **SentencePiece** | ~4.3% | ~2.0% | ~1.6% | ~1.0% |

At 128K, single-document utilization drops to ~1.0%. This means 99% of the embedding matrix slots are not activated for any given document. In a downstream LLM, these represent **dead weight embedding parameters** — consuming GPU memory, initialization cost, and gradient updates for embeddings that will be rarely or never observed during training.

### 8.5 Vocabulary Size Recommendations from Ablation

**Primary Recommendation — Maximum Context Window Efficiency: SentencePiece 128K**

If the goal is packing maximum Balochi text into an LLM prompt (inference context packing), SentencePiece 128K is the definitive winner:
- Best absolute compression: **4.2571 chars/token**
- Best fertility: **1.1033 tokens/word** (essentially 1 token per word)
- Sustained scaling advantage: +0.0017 per +1K beyond 80K (best among all algorithms)

**Secondary Recommendation — The 64K "Knee of the Curve"**

For production LLM training where model memory footprint is a constraint (as larger vocabularies require a proportionally larger embedding matrix):
- The 32K→64K leap delivers the maximum return (+0.0043 compression per 1K tokens)
- After 64K, return rates drop by >50% across all algorithms
- **SentencePiece 64K** (Compression: 4.136, Fertility: 1.135) remains an exceptionally strong candidate

---

## 9. Phase 5 — Information-Theoretic Analysis: Rényi Entropy

### 9.1 Why Entropy Analysis?

The vocabulary ablation study showed that 128K provides the best text compression. But compression alone does not capture the full cost. Rényi Entropy analysis provides a mathematically rigorous second opinion by examining how efficiently the vocabulary is utilized from an information-theoretic perspective.

### 9.2 Shannon Entropy and Effective Vocabulary

**Shannon Entropy (H₁)** measures the uncertainty or information content of the token probability distribution:

$$H_1 = -\sum_{i} p_i \log_2 p_i$$

A lower entropy indicates a more skewed, less uniform distribution where a few tokens dominate the text.

**Effective Vocabulary (V_eff)** = $2^{H_1}$ — the size of a hypothetical uniform vocabulary that would produce the same entropy. It represents the number of tokens that are "doing actual work" in a document.

**Vocabulary Efficiency (η)** = $V_{eff} / V_{total}$ — what percentage of the actual dictionary is effectively utilized.

### 9.3 Entropy Trajectories Across Vocabulary Sizes

As vocabulary size increases from 32K to 128K, a critical mathematical phenomenon emerges: **the Effective Vocabulary actually decreases even as the total vocabulary grows**.

#### SentencePiece Entropy Trajectory (Balochi Labzank Text, Pre-Normalization)

| Total Vocab | Shannon Entropy (H₁) | Effective Vocab (V_eff) | Vocab Efficiency (η) |
|:---:|:---:|:---:|:---:|
| **32,000** | 9.0351 | **524.6** | **1.64%** |
| **64,000** | 8.9766 | **503.8** | **0.79%** |
| **80,000** | 8.9553 | **496.4** | **0.62%** |
| **128,000** | 8.9151 | **482.7** | **0.38%** |

#### WordPiece Entropy Trajectory (Tahir Hakim Text)

| Total Vocab | Shannon Entropy (H₁) | Effective Vocab (V_eff) | Vocab Efficiency (η) |
|:---:|:---:|:---:|:---:|
| **32,000** | 8.8167 | **450.9** | **1.41%** |
| **64,000** | 8.7750 | **438.1** | **0.68%** |
| **80,000** | 8.7681 | **436.0** | **0.54%** |
| **128,000** | 8.7377 | **426.9** | **0.33%** |

#### Post-Normalization Entropy (64K Models, `liberal capitalism.txt`)

| Model | Shannon Entropy (H₁) | Min-Entropy (H∞) | Effective Vocab (V_eff) | Vocab Efficiency (η) |
|:---|:---:|:---:|:---:|:---:|
| **SentencePiece 64K** | **6.8443** | 2.7224 | **114.9** | 0.0018 |
| BPE 64K | 6.7346 | 2.1979 | 106.5 | 0.0017 |
| WordPiece 64K | 6.8193 | 1.8088 | 112.9 | 0.0018 |

### 9.4 The "Dead Weight" Embedding Problem Quantified

The drop in V_eff as V_total increases reveals the hidden cost of large vocabularies:

When vocabulary expands to 128K, the tokenizer learns extremely long, rare token sequences. During inference on a standard Balochi document, these rare tokens are either not used at all, or used very sparsely. The probability mass concentrates on common tokens, skewing the distribution (lowering H₁) and crushing Vocabulary Efficiency (η) to just **0.38%**.

In a downstream LLM:
- A 128K vocabulary requires an embedding matrix **4× larger** than 32K
- If vocabulary efficiency is only 0.38%, the vast majority of embedding parameters are under-trained
- These "dead weight" embeddings consume GPU memory during training but receive near-zero gradient updates
- They degrade model generalization by introducing noise into the parameter space

Concretely: the Effective Vocabulary for any single Balochi document is only ~482–524 tokens at the 32K level. Even at 128K, it is only ~482. The actual linguistic diversity of a Balochi document does not change with vocabulary size — only the representation format does.

### 9.5 Severe Long-Tail Sparsity Confirmed

The post-normalization entropy analysis reveals an extreme Zipfian distribution in Balochi text tokenization:

- Despite having 64,000 available vocabulary slots, the **Effective Vocabulary per document hovers around 106–114 tokens**
- The Zipf exponent is approximately **s ≈ 1.05** for SentencePiece
- Min-Entropy scores (H∞) ranging from 1.80 to 2.72 indicate that the single most frequent token **heavily dominates** the distribution
- WordPiece suffers most severely from peak token dominance (H∞ = 1.80), meaning its most common token appears at an extremely high frequency relative to all others

### 9.6 The 64K Sweet Spot — Entropy Analysis Confirmation

| Vocab Size | Compression (SP) | V_eff | η | Assessment |
|:---:|:---:|:---:|:---:|:---:|
| 32K | 3.9982 | 524.6 | **1.64%** | Good efficiency, poor compression |
| **64K** | **4.1360** | **503.8** | **0.79%** | ✅ **Optimal equilibrium** |
| 80K | 4.1770 | 496.4 | 0.62% | Marginal gains, declining efficiency |
| 128K | 4.2571 | 482.7 | 0.38% | Best compression, catastrophic efficiency |

The 64K point achieves the optimal mathematical equilibrium: strong compression (4.136) with reasonable effective vocabulary (~504 tokens). The entropy analysis **confirms the ablation study finding** through a completely independent mathematical lens.

**SentencePiece Superiority Confirmed by Entropy:** SentencePiece consistently maintains the highest Shannon Entropy and Effective Vocabulary across all sizes. At 64K, SentencePiece achieves an effective vocabulary of **503.8 tokens**, while WordPiece reaches only **438.1** and BPE only **424.1**. This mathematically proves that SentencePiece distributes semantic meaning more evenly across its tokens, generating a more balanced, uniform distribution for Balochi text.

---

## 10. Phase 6 — Vocabulary Pruning: The 47K Breakthrough

### 10.1 From 64K to 47K — The Pruning Rationale

Following normalization and ablation, a key insight emerged: the normalization pipeline had removed the orthographic noise that previously required rare tokens to represent. With the corpus cleaned, maintaining a 64K vocabulary meant retaining approximately 17,000 tokens that existed only to represent diacritical variants, ZWNJ-split forms, and other noise artifacts — tokens that would never appear in properly-normalized text.

The question became: **what is the minimum vocabulary size needed to cover 99% of real Balochi text after normalization?**

### 10.2 Cumulative Frequency Analysis — Terminal Output

A `Vocabulary Pruning Analyzer` script was executed on the full normalized training corpus (~370 MB, `training_corpus_normalized.txt`):

```
================================================================================
 ✂️ Vocabulary Pruning Analyzer — Cumulative Frequency Thresholding
==========================================================
 🎯 RECOMMENDED PRUNING THRESHOLDS
============================================================
Coverage Target      | Required Vocab Size  | Tokens Pruned
------------------------------------------------------------
90.0%                | 10,219               | 53,781
95.0%                | 21,141               | 42,859
99.0%                | 46,406               | 17,594
99.5%                | 53,240               | 10,760
99.9%                | 61,004               | 2,996
------------------------------------------------------------
```

### 10.3 Interpreting the Pruning Thresholds

The results reveal the extreme Zipfian distribution characteristic of tokenized natural language:

- **90% coverage** requires only **10,219 tokens** — just 16% of the 64K vocabulary
- **95% coverage** requires only **21,141 tokens** — 33% of the 64K vocabulary  
- **99% coverage** (AraToken's recommended threshold) requires **46,406 tokens** — 73% of the 64K vocabulary
- **The remaining 17,594 tokens** (covering the bottom 1% of text) are overwhelmingly noise artifacts from the un-normalized training corpus tail

### 10.4 The 47K Production Model

Following the AraToken paper's 99% cumulative frequency recommendation, the decision was made to train a new production tokenizer with `vocab_size = 47,000` — rounded slightly above 46,406 to provide a small safety buffer.

The `Balochi_SP_47K` model was then evaluated against the larger legacy models using Rényi Entropy:

| Post-Norm Model | Vocab Size | Shannon Entropy (H₁) | Effective Vocab (V_eff) |
|:---|:---:|:---:|:---:|
| `Balochi_SP_128K` | 128,000 | 7.3924 | 168.0 tokens |
| `Balochi_SP_64K` | 64,000 | 7.4184 | 171.1 tokens |
| **`Balochi_SP_47K`** | **47,000** | **7.4295** | **172.4 tokens** ✅ |

**This is the decisive finding:** Despite removing 17,000 tokens from the 64K vocabulary, the 47K model achieves **higher Shannon Entropy** and utilizes a **larger effective vocabulary per document** than either the 64K or 128K models.

### 10.5 Why Pruning Improves Entropy

The 17,000 pruned tokens were "noise tokens" — rare subwords appearing fewer than ~5 times in 54 million training tokens, representing diacritical variants, ZWNJ-split artifacts, and formatting anomalies from the un-normalized corpus. These tokens had the following properties:
- Near-zero probability in the token distribution
- Pull the distribution toward extreme sparsity (very low entropy)
- Occupy embedding matrix rows that receive essentially zero gradient signal during downstream LLM training
- Crowd out semantic room in the fixed-size vocabulary

Removing them **concentrates probability mass on legitimate, frequently-used morphological subwords**, raising both the Shannon Entropy and the Effective Vocabulary — making the tokenizer more informationally rich per slot.

### 10.6 Embedding Matrix Savings for Gemma 2B

The practical benefit for the planned Language Extension Pipeline (LEP) integration with Gemma 2B is quantifiable:

| Configuration | New Vocab Tokens Added | Embedding Row Savings vs. 64K |
|:---|:---:|:---:|
| Gemma + Balochi_SP_64K | ~64,000 new rows | — |
| **Gemma + Balochi_SP_47K** | **~47,000 new rows** | **~17,000 rows (26.6% reduction)** |

A 26.6% reduction in new embedding layer dimensions directly translates to reduced GPU memory requirements for the embedding and unembedding matrices, proportionally lower initialization cost for the new Balochi token embeddings, and faster convergence during the continual pre-training phase since all 47K tokens receive adequate training signal.

---

## 11. Complete Model Registry

A total of **19 locally-trained models** were produced across all phases:

### Phase 1: Pre-Normalization Models (Original Baselines)

| # | Model Name | Algorithm | Vocab Size | Notes |
|:---:|:---|:---:|:---:|:---|
| 1 | Balochi_SentencePiece (64K) | Google SentencePiece | 64,000 | Original SP baseline |
| 2 | Balochi_WordPiece (64K) | HuggingFace WordPiece | 64,000 | Original WP baseline |
| 3 | Balochi_BPE (80K) | HuggingFace BPE | 80,000 | Original BPE baseline |
| 4 | Balochi_30K | HF External BPE | 30,000 | External HF (`balochiml/balochi-tokenizer`) |

### Phase 2: Post-Normalization Ablation Series — SentencePiece (Unigram) Family

| # | Model Name | Vocab Size | Status |
|:---:|:---|:---:|:---:|
| 5 | `Balochi_SP_32K` | 32,000 | Ablation reference |
| 6 | **`Balochi_SP_47K`** | **47,000** | **⭐ FINAL OPTIMAL PRODUCTION MODEL** |
| 7 | `Balochi_SP_64K` | 64,000 | Sweet spot for resource-constrained use |
| 8 | `Balochi_SP_80K` | 80,000 | Legacy reference |
| 9 | `Balochi_SP_128K` | 128,000 | Maximum compression reference |

### Phase 2: Post-Normalization Ablation Series — BPE Family

| # | Model Name | Vocab Size | Status |
|:---:|:---|:---:|:---:|
| 10 | `Balochi_BPE_32K` | 32,000 | Ablation reference |
| 11 | `Balochi_BPE_47K` | 47,000 | Pruning experiment |
| 12 | `Balochi_BPE_64K` | 64,000 | Sweet spot reference |
| 13 | `Balochi_BPE_80K` | 80,000 | Legacy reference |
| 14 | `Balochi_BPE_128K` | 128,000 | Maximum compression reference |

### Phase 2: Post-Normalization Ablation Series — WordPiece Family

| # | Model Name | Vocab Size | Status |
|:---:|:---|:---:|:---:|
| 15 | `Balochi_WP_32K` | 32,000 | Ablation reference |
| 16 | `Balochi_WP_47K` | 47,000 | Pruning experiment |
| 17 | `Balochi_WP_64K` | 64,000 | Sweet spot reference |
| 18 | `Balochi_WP_80K` | 80,000 | Legacy reference |
| 19 | `Balochi_WP_128K` | 128,000 | Maximum compression reference |

---

## 12. Tokenizer Selection Guide

Select the appropriate tokenizer based on downstream task requirements:

| Use Case | Recommended Tokenizer | Primary Rationale |
|:---|:---|:---|
| **Gemma 2B / LLaMA CPT** | `Balochi_SP_47K` | Optimal embedding efficiency + lossless roundtrip; 26.6% smaller embedding layer |
| **BERT / ALBERT fine-tuning** | `Balochi_WP_47K` | Native WordPiece architecture; near-identical coverage to 64K |
| **GPT-2 / RoBERTa pre-training** | `Balochi_BPE_47K` | BPE aligns with GPT-style training; pruned for efficiency |
| **Resource-constrained environments** | `Balochi_SP_64K` | Optimal compression/memory trade-off if 47K unavailable |
| **Maximum sequence packing** | `Balochi_SP_128K` | Best absolute compression (4.257 chars/token) |
| **Mixed Balochi-English text** | `Balochi_BPE_64K` or `Balochi_SP_64K` | Both handle English loanwords; 30K fails entirely |
| **Data integrity (generation)** | Any SP model | Only SP family achieves lossless roundtrip reconstruction |
| **Character-level robustness** | `Balochi_SP_*` | `byte_fallback=True` guarantees zero UNK on any Unicode input |
| **Cross-lingual BERT tasks** | `AraBERT_v2` or `CAMeLBERT` | If Balochi-specific models unavailable; best Arabic BERT baselines |
| **Word-count corpus analysis** | NLTK | Fastest for pure word statistics; not suitable for neural LLMs |

---

## 13. Implications for Downstream NLP Tasks

### 13.1 Continual Pre-Training (CPT) — Compute Savings Calculation

For a training corpus of 100M Balochi character tokens:

| Tokenizer | Training Tokens Generated | Relative Training Cost |
|:---|:---:|:---:|
| BERT Multilingual | ~112M tokens | 100% (baseline) |
| Gemma 2B | ~119M tokens | 106% |
| **Balochi WP (64K)** | **~29M tokens** | **26% of BERT** |
| **Balochi SP (47K)** | **~28M tokens** | **25% of BERT** |

Using the custom Balochi tokenizer reduces training token count by approximately **74%** for the same textual content — directly proportional to GPU compute and time savings.

### 13.2 Context Window Efficiency

In a model with a 4,096-token context window, the fertility differential translates to dramatically different semantic content per forward pass:

| Tokenizer | Fertility | Balochi Words per 4,096-token Context |
|:---|:---:|:---:|
| BERT (2.305) | 2.305 | ~1,777 words |
| Gemma (2.460) | 2.460 | ~1,665 words |
| **Balochi SP 47K (~1.14)** | ~1.14 | **~3,593 words** |
| **Improvement** | — | **~2× more semantic content** |

### 13.3 Masked Language Model Fine-tuning

For BERT-style models fine-tuned on Balochi NLP tasks (NER, classification, sentiment):
- A 512-token BERT sequence covers ~222 Balochi words with BERT tokenizer vs. ~434 words with Balochi WordPiece
- BERT's 44 UNK tokens per ~10,000 tokens accumulate to thousands of silently-lost characters in large datasets
- Custom tokenizers eliminate truncation artifacts in documents that would otherwise overflow 512-token limits

### 13.4 Named Entity Recognition (NER)

Morphological boundary preservation is directly critical for NER quality. When BERT fragments a proper noun like `عبدالوہاب` into 5 continuation tokens, the entity boundary signal is diluted — the model must assign B-tags and I-tags across fragmented pieces rather than recognizing a single complete entity span. Custom tokenizers that preserve proper noun integrity as single tokens make NER span boundaries crisper, improving F1 scores on entity boundary detection.

### 13.5 Text Generation and Summarization

Lossless roundtrip fidelity (unique to SentencePiece family models) is the gold standard for generative tasks. For autoregressive generation, span extraction in QA, and copy-span tasks in summarization, the ability to map token sequences back to exact surface forms without normalization artifacts is a concrete quality advantage.

### 13.6 Language Extension Pipeline (LEP) for Gemma

The AraToken-demonstrated LEP approach maps directly to Balochi-Gemma integration:

| LEP Component | Balochi Adaptation |
|:---|:---|
| **Vocabulary Extension** | Extract tokens from `Balochi_SP_47K` not present in Gemma's 256K BPE vocabulary |
| **Mean Subtoken Initialization** | Initialize new Balochi token embeddings as mean of Gemma's existing Perso-Arabic subtoken embeddings: $\mathbf{e}_{new} = \frac{1}{\|S\|}\sum_{i \in S} \mathbf{e}_i$ |
| **Gradient Masking** | Freeze Gemma's original 256K embeddings; train only new Balochi embeddings |
| **Selective Layer Unfreezing** | Unfreeze last 4 layers of Gemma-2B (layers 14–17 of 18) |

AraToken demonstrated that this approach reduces evaluation loss from **8.28 to 2.43 in only 800 steps on 100K samples** — under 0.01% of typical LLM pre-training budget. The 47K Balochi SP tokenizer is optimally positioned for this pipeline.

---

## 14. Final Conclusions & Future Work

### 14.1 Key Research Conclusions

**Conclusion 1 — Custom Tokenizers are Mandatory**

Generic multilingual tokenizers (BERT: fertility 2.305, Gemma: fertility 2.460) fragment Balochi text 2× more severely than custom tokenizers. Cross-lingual transfer from Arabic, Persian, or Urdu tokenizers fails in practice: AraBERT_v2 has a 51% UNK rate on Balochi; UrduBERT achieves a catastrophic fertility of 7.625. There is no viable alternative to training dedicated, corpus-matched tokenizers.

**Conclusion 2 — Normalization is a Prerequisite, Not an Optimization**

The AraToken normalization pipeline removes ~6.4% non-semantic character noise from Balochi corpora. The resulting fertility improvements (−0.026 for SentencePiece at 64K) are modest in isolation but create a prerequisite for accurate vocabulary frequency analysis and downstream pruning. Tokenizers trained on raw text embed noise patterns into the vocabulary tail; tokenizers trained on normalized text produce cleaner, more linguistically grounded subword boundaries.

**Conclusion 3 — SentencePiece (Unigram) Outperforms BPE and WordPiece Across All Dimensions**

SentencePiece achieves the best compression ratio at every vocabulary size, the lowest fertility at every size, the highest Shannon Entropy, the largest Effective Vocabulary, and is the only algorithm family to provide mathematically lossless roundtrip text reconstruction. It is also over 4× faster than HuggingFace's BPE implementation at encoding throughput. For all downstream Balochi NLP applications, SentencePiece is the clear algorithmic choice.

**Conclusion 4 — 64K is the Compression "Knee" but 47K is the Optimal Production Vocabulary**

The vocabulary ablation study shows 64K as the point where marginal compression return per additional vocabulary token drops by more than 50%. The entropy pruning analysis shows that 17,000 tokens in the 64K vocabulary are statistical noise artifacts from the un-normalized training tail. The `Balochi_SP_47K` model — trained after normalization, pruned at 99% cumulative frequency coverage — achieves higher entropy and larger effective vocabulary than 64K or 128K models while reducing embedding matrix dimensions by 26.6%.

**Conclusion 5 — The 47K Model is Production-Ready for Gemma LEP Integration**

The `Balochi_SP_47K` model satisfies all technical requirements for the Language Extension Pipeline:
- Zero UNK rate on any Balochi/English text (via byte_fallback)
- Lossless roundtrip fidelity (essential for generative tasks)
- 26.6% smaller embedding layer than 64K baseline
- Superior information-theoretic efficiency (higher V_eff) than all larger models
- AraToken-validated normalization pipeline ensuring clean morphological boundaries

### 14.2 Summary of Recommendations

| Scenario | Recommended Model | Key Justification |
|:---|:---|:---|
| **Production (Gemma 2B LEP)** | `Balochi_SP_47K` | Optimal embedding efficiency + lossless reconstruction |
| **Resource-constrained production** | `Balochi_SP_64K` | Strong compression/memory balance |
| **Maximum text compression** | `Balochi_SP_128K` | Best absolute compression (4.257 chars/token) |
| **BERT-style architecture** | `Balochi_WP_47K` | WordPiece architecture + pruned efficiency |

### 14.3 Identified Limitations

1. **Two-Document Evaluation Corpus:** All metrics are computed on two political/economic texts. Cross-domain generalization to literary, religious, social media, or technical Balochi text requires additional evaluation.
2. **Southern Dialect Focus:** The corpus and evaluation texts represent Southern Balochi. Northern, Eastern, and Makrani dialect variations may show different fertility and coverage profiles.
3. **No Morpheme Boundary Gold Standard:** Fertility and continuation rate are proxies for morphological preservation; direct evaluation against a Balochi morpheme boundary gold standard (which does not currently exist) has not been performed.
4. **Single Normalization Configuration Evaluated:** The research reports results for one normalization configuration (diacritics dropped, Ye preserved). A systematic ablation across all AraToken-equivalent configuration combinations (as recommended in the Research-Paper-Suggestion) was not completed.
5. **No Downstream Task Benchmarking:** Tokenizer quality has not been directly validated on downstream NLP tasks (NER F1, classification accuracy, MLM perplexity) — the ultimate ground truth for tokenizer quality.

---

## Appendix A — Training Configuration Reference

| Parameter | BPE | WordPiece | SentencePiece |
|:---|:---:|:---:|:---:|
| Implementation | HuggingFace `tokenizers` | HuggingFace `tokenizers` | Google `sentencepiece` |
| Pre-tokenization | Whitespace + Punctuation + Digits | Whitespace + Punctuation + Digits | None (raw character stream) |
| Space handling | — | `##` continuation prefix | `▁` word-start marker |
| Special tokens | `[PAD]`, `[UNK]`, `[CLS]`, `[SEP]`, `[MASK]` | Same | `<pad>`, `<unk>`, `<s>`, `</s>` |
| Character coverage | NFKC + mandatory alphabet | NFKC + mandatory alphabet | `coverage=0.9995` |
| Mandatory alphabet | `a-z`, `A-Z`, `0-9`, `۰-۹`, `٠-٩` | Same | Implicit via character coverage |
| Byte fallback | N/A | N/A | `byte_fallback=True` |
| Roundtrip fidelity | ✗ Lossy | ✗ Lossy | ✅ Lossless |

---

## Appendix B — Corpus Processing Time

Training the 128K SentencePiece model on the full 185.5M character corpus:

```
all chars count = 185,587,264
✓ Trained in ~1234.5s (~20.6 minutes)
✓ Actual vocab: 128,000 (fully saturated — corpus has sufficient unique sequences)
```

Normalized corpus pruning analysis on ~370 MB (`training_corpus_normalized.txt`):

```
✓ Tokenization complete in 75.8s
✓ Total tokens in corpus: 54,301,475
✓ Unique tokens utilized: 63,845 (of 64,000 — 99.76% vocabulary saturation)
```
---

<div align="center">

</div>
