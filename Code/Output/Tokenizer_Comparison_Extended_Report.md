# Comprehensive Balochi + Cross-Language Tokenizer Comparison

**Generated:** 2026-06-07 15:06:32  

**Script:** `Tokenizers_Comparison_Extended.py`

## 1. Input Text Summary

| Property | Value |
|----------|-------|
| **File** | `liberal capitalism.txt` |
| **Characters** | 23,440 |
| **Words** | 5,035 |
| **Sentences** | 155 |

## 2. Tokenizers Loaded

| # | Tokenizer | Family | Script | Vocab Size | HuggingFace ID |
|---|-----------|--------|--------|------------|----------------|
| 1 | **Balochi_BPE** | Balochi | BPE | 80,000 | `Local file` |
| 2 | **Balochi_WordPiece** | Balochi | WordPiece | 64,000 | `Local file` |
| 3 | **Balochi_SentencePiece** | Balochi | SentencePiece | 64,000 | `Local file` |
| 4 | **Balochi_30K** | Balochi | BPE | 30,000 | `balochiml/balochi-tokenizer` |
| 5 | **NLTK** | Baseline | Rule-based | N/A | `—` |
| 6 | **BERT** | Generic | WordPiece | 119,547 | `bert-base-multilingual-cased` |
| 7 | **Gemma** | Generic | SP-BPE | 256,000 | `google/gemma-2b` |
| 8 | **AraBERT_v2** | Arabic | WordPiece | 64,000 | `aubmindlab/bert-base-arabertv2` |
| 9 | **CAMeLBERT_MSA** | Arabic | WordPiece | 30,000 | `CAMeL-Lab/bert-base-arabic-camelbert-msa` |
| 10 | **ARBERT** | Arabic | WordPiece | 100,000 | `UBC-NLP/ARBERT` |
| 11 | **AraGPT2** | Arabic | BPE | 64,000 | `aubmindlab/aragpt2-base` |
| 12 | **ParsBERT** | Persian | WordPiece | 100,000 | `HooshvareLab/bert-base-parsbert-uncased` |
| 13 | **PersianBERT_FA** | Persian | WordPiece | 100,000 | `HooshvareLab/bert-fa-base-uncased` |
| 14 | **PersianBPE** | Persian | BPE | 30,000 | `mshojaei77/PersianBPETokenizer` |
| 15 | **UrduBERT** | Urdu | WordPiece | 50,265 | `urduhack/UrduBERT` |

## 3. Tokenization Results

| Tokenizer | Language | Tokens | Speed (tok/s) | Time (s) |
|-----------|----------|--------|---------------|----------|
| Balochi_BPE | Balochi | 5,767 | 159,222 | 0.0362 |
| Balochi_WordPiece | Balochi | 5,912 | 318,555 | 0.0186 |
| Balochi_SentencePiece | Balochi | 5,746 | 672,228 | 0.0085 |
| NLTK | Baseline | 5,353 | 8,420 | 0.6357 |
| BERT | Multilingual | 10,991 | 422,580 | 0.0260 |
| Gemma | Multilingual | 11,621 | 621,693 | 0.0187 |
| Balochi_30K | Balochi | 7,129 | 512,874 | 0.0139 |
| AraBERT_v2 | Arabic | 5,990 | 326,833 | 0.0183 |
| CAMeLBERT_MSA | Arabic | 11,837 | 559,958 | 0.0211 |
| ARBERT | Arabic | 11,003 | 480,613 | 0.0229 |
| AraGPT2 | Arabic | 19,976 | 695,221 | 0.0287 |
| ParsBERT | Persian | 8,337 | 408,246 | 0.0204 |
| PersianBERT_FA | Persian | 8,660 | 415,745 | 0.0208 |
| PersianBPE | Persian | 0 | 0 | 0.0086 |
| UrduBERT | Urdu | 40,496 | 965,195 | 0.0420 |

## 4. Master Metrics Table

| Metric | Balochi_BPE | Balochi_WordPiece | Balochi_SentencePiece | NLTK | BERT | Gemma | Balochi_30K | AraBERT_v2 | CAMeLBERT_MSA | ARBERT | AraGPT2 | ParsBERT | PersianBERT_FA | PersianBPE | UrduBERT |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| **Token Count** | 5,767 | 5,912 | 5,746 | 5,353 | 10,991 | 11,621 | 7,129 | 5,990 | 11,837 | 11,003 | 19,976 | 8,337 | 8,660 | 0 | 40,496 |
| **Unique Tokens** | 1,006 | 1,029 | 1,105 | 1,013 | 682 | 767 | 894 | 384 | 648 | 704 | 699 | 900 | 914 | 0 | 94 |
| **Vocab Size** | 80000 | 64000 | 64000 | N/A | 119547 | 256000 | 30000 | 64000 | 30000 | 100000 | 64000 | 100000 | 100000 | 30000 | 50265 |
| **Vocab Util. (%)** | 1.26% | 1.61% | 1.73% | N/A | 0.57% | 0.30% | 2.98% | 0.60% | 2.16% | 0.70% | 1.09% | 0.90% | 0.91% | 0.00% | 0.19% |
| **Compression Ratio** | 4.06 | 3.96 | 4.08 | 4.38 | 2.13 | 2.02 | 3.29 | 3.91 | 1.98 | 2.13 | 1.17 | 2.81 | 2.71 | 0.00 | 0.58 |
| **Fertility** | 1.145 | 1.174 | 1.141 | 1.063 | 2.183 | 2.308 | 1.416 | 1.190 | 2.351 | 2.185 | 3.967 | 1.656 | 1.720 | 0.000 | 8.043 |
| **Avg Token Length** | 3.19 | 3.22 | 4.08 | 3.44 | 2.67 | 2.02 | 3.21 | 4.33 | 2.86 | 2.56 | 2.03 | 2.84 | 2.74 | 0.00 | 1.00 |
| **UNK Count** | 0 | 0 | 0 | 0 | 44 | 0 | 1,111 | 4,103 | 1,054 | 2 | 0 | 364 | 67 | 0 | 0 |
| **UNK Rate (%)** | 0.0000% | 0.0000% | 0.0000% | 0.0000% | 0.4003% | 0.0000% | 15.5842% | 68.4975% | 8.9043% | 0.0182% | 0.0000% | 4.3661% | 0.7737% | 0.0000% | 0.0000% |
| **Continuation Rate** | 0.00% | 5.43% | 12.37% | 0.00% | 49.13% | 56.68% | 0.00% | 6.66% | 52.77% | 49.19% | 100.00% | 32.94% | 35.44% | 0.00% | 0.00% |
| **Speed (tok/s)** | 159,222 | 318,555 | 672,228 | 8,420 | 422,580 | 621,693 | 512,874 | 326,833 | 559,958 | 480,613 | 695,221 | 408,246 | 415,745 | 0 | 965,195 |
| **Time (s)** | 0.0362 | 0.0186 | 0.0085 | 0.6357 | 0.0260 | 0.0187 | 0.0139 | 0.0183 | 0.0211 | 0.0229 | 0.0287 | 0.0204 | 0.0208 | 0.0086 | 0.0420 |
| **Roundtrip Fidelity** | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | N/A | ✗ |

## 5. Group A: WordPiece Family

| Metric | Balochi_WordPiece | AraBERT_v2 | CAMeLBERT_MSA | ARBERT | BERT |
|--------|--------|--------|--------|--------|--------|
| **Token Count** | 5,912 | 5,990 | 11,837 | 11,003 | 10,991 |
| **Unique Tokens** | 1,029 | 384 | 648 | 704 | 682 |
| **Vocab Size** | 64000 | 64000 | 30000 | 100000 | 119547 |
| **Vocab Util. (%)** | 1.61% | 0.60% | 2.16% | 0.70% | 0.57% |
| **Compression Ratio** | 3.96 | 3.91 | 1.98 | 2.13 | 2.13 |
| **Fertility** | 1.174 | 1.190 | 2.351 | 2.185 | 2.183 |
| **Avg Token Length** | 3.22 | 4.33 | 2.86 | 2.56 | 2.67 |
| **UNK Count** | 0 | 4,103 | 1,054 | 2 | 44 |
| **UNK Rate (%)** | 0.0000% | 68.4975% | 8.9043% | 0.0182% | 0.4003% |
| **Continuation Rate** | 5.43% | 6.66% | 52.77% | 49.19% | 49.13% |
| **Speed (tok/s)** | 318,555 | 326,833 | 559,958 | 480,613 | 422,580 |
| **Time (s)** | 0.0186 | 0.0183 | 0.0211 | 0.0229 | 0.0260 |
| **Roundtrip Fidelity** | ✗ | ✗ | ✗ | ✗ | ✗ |

## 6. Group B: SP/WP — Balochi vs Persian vs Gemma

| Metric | Balochi_SentencePiece | ParsBERT | PersianBERT_FA | Gemma |
|--------|--------|--------|--------|--------|
| **Token Count** | 5,746 | 8,337 | 8,660 | 11,621 |
| **Unique Tokens** | 1,105 | 900 | 914 | 767 |
| **Vocab Size** | 64000 | 100000 | 100000 | 256000 |
| **Vocab Util. (%)** | 1.73% | 0.90% | 0.91% | 0.30% |
| **Compression Ratio** | 4.08 | 2.81 | 2.71 | 2.02 |
| **Fertility** | 1.141 | 1.656 | 1.720 | 2.308 |
| **Avg Token Length** | 4.08 | 2.84 | 2.74 | 2.02 |
| **UNK Count** | 0 | 364 | 67 | 0 |
| **UNK Rate (%)** | 0.0000% | 4.3661% | 0.7737% | 0.0000% |
| **Continuation Rate** | 12.37% | 32.94% | 35.44% | 56.68% |
| **Speed (tok/s)** | 672,228 | 408,246 | 415,745 | 621,693 |
| **Time (s)** | 0.0085 | 0.0204 | 0.0208 | 0.0187 |
| **Roundtrip Fidelity** | ✓ | ✗ | ✗ | ✗ |

## 7. Group C: BPE Family

| Metric | Balochi_BPE | AraGPT2 | PersianBPE | Balochi_30K | NLTK |
|--------|--------|--------|--------|--------|--------|
| **Token Count** | 5,767 | 19,976 | 0 | 7,129 | 5,353 |
| **Unique Tokens** | 1,006 | 699 | 0 | 894 | 1,013 |
| **Vocab Size** | 80000 | 64000 | 30000 | 30000 | N/A |
| **Vocab Util. (%)** | 1.26% | 1.09% | 0.00% | 2.98% | N/A |
| **Compression Ratio** | 4.06 | 1.17 | 0.00 | 3.29 | 4.38 |
| **Fertility** | 1.145 | 3.967 | 0.000 | 1.416 | 1.063 |
| **Avg Token Length** | 3.19 | 2.03 | 0.00 | 3.21 | 3.44 |
| **UNK Count** | 0 | 0 | 0 | 1,111 | 0 |
| **UNK Rate (%)** | 0.0000% | 0.0000% | 0.0000% | 15.5842% | 0.0000% |
| **Continuation Rate** | 0.00% | 100.00% | 0.00% | 0.00% | 0.00% |
| **Speed (tok/s)** | 159,222 | 695,221 | 0 | 512,874 | 8,420 |
| **Time (s)** | 0.0362 | 0.0287 | 0.0086 | 0.0139 | 0.6357 |
| **Roundtrip Fidelity** | ✗ | ✓ | N/A | ✗ | ✗ |

## 8. Group D: Perso-Arabic Script

| Metric | UrduBERT | Balochi_BPE | Balochi_WordPiece | AraBERT_v2 | ParsBERT |
|--------|--------|--------|--------|--------|--------|
| **Token Count** | 40,496 | 5,767 | 5,912 | 5,990 | 8,337 |
| **Unique Tokens** | 94 | 1,006 | 1,029 | 384 | 900 |
| **Vocab Size** | 50265 | 80000 | 64000 | 64000 | 100000 |
| **Vocab Util. (%)** | 0.19% | 1.26% | 1.61% | 0.60% | 0.90% |
| **Compression Ratio** | 0.58 | 4.06 | 3.96 | 3.91 | 2.81 |
| **Fertility** | 8.043 | 1.145 | 1.174 | 1.190 | 1.656 |
| **Avg Token Length** | 1.00 | 3.19 | 3.22 | 4.33 | 2.84 |
| **UNK Count** | 0 | 0 | 0 | 4,103 | 364 |
| **UNK Rate (%)** | 0.0000% | 0.0000% | 0.0000% | 68.4975% | 4.3661% |
| **Continuation Rate** | 0.00% | 0.00% | 5.43% | 6.66% | 32.94% |
| **Speed (tok/s)** | 965,195 | 159,222 | 318,555 | 326,833 | 408,246 |
| **Time (s)** | 0.0420 | 0.0362 | 0.0186 | 0.0183 | 0.0204 |
| **Roundtrip Fidelity** | ✗ | ✗ | ✗ | ✗ | ✗ |

## 9. Group E1: Balochi WP vs BERT

| Metric | Balochi_WordPiece | BERT |
|--------|--------|--------|
| **Token Count** | 5,912 | 10,991 |
| **Unique Tokens** | 1,029 | 682 |
| **Vocab Size** | 64000 | 119547 |
| **Vocab Util. (%)** | 1.61% | 0.57% |
| **Compression Ratio** | 3.96 | 2.13 |
| **Fertility** | 1.174 | 2.183 |
| **Avg Token Length** | 3.22 | 2.67 |
| **UNK Count** | 0 | 44 |
| **UNK Rate (%)** | 0.0000% | 0.4003% |
| **Continuation Rate** | 5.43% | 49.13% |
| **Speed (tok/s)** | 318,555 | 422,580 |
| **Time (s)** | 0.0186 | 0.0260 |
| **Roundtrip Fidelity** | ✗ | ✗ |

## 10. Group E2: Balochi SP vs Gemma

| Metric | Balochi_SentencePiece | Gemma |
|--------|--------|--------|
| **Token Count** | 5,746 | 11,621 |
| **Unique Tokens** | 1,105 | 767 |
| **Vocab Size** | 64000 | 256000 |
| **Vocab Util. (%)** | 1.73% | 0.30% |
| **Compression Ratio** | 4.08 | 2.02 |
| **Fertility** | 1.141 | 2.308 |
| **Avg Token Length** | 4.08 | 2.02 |
| **UNK Count** | 0 | 0 |
| **UNK Rate (%)** | 0.0000% | 0.0000% |
| **Continuation Rate** | 12.37% | 56.68% |
| **Speed (tok/s)** | 672,228 | 621,693 |
| **Time (s)** | 0.0085 | 0.0187 |
| **Roundtrip Fidelity** | ✓ | ✗ |

## 11. Group E3: Balochi BPE vs NLTK vs 30K

| Metric | Balochi_BPE | NLTK | Balochi_30K |
|--------|--------|--------|--------|
| **Token Count** | 5,767 | 5,353 | 7,129 |
| **Unique Tokens** | 1,006 | 1,013 | 894 |
| **Vocab Size** | 80000 | N/A | 30000 |
| **Vocab Util. (%)** | 1.26% | N/A | 2.98% |
| **Compression Ratio** | 4.06 | 4.38 | 3.29 |
| **Fertility** | 1.145 | 1.063 | 1.416 |
| **Avg Token Length** | 3.19 | 3.44 | 3.21 |
| **UNK Count** | 0 | 0 | 1,111 |
| **UNK Rate (%)** | 0.0000% | 0.0000% | 15.5842% |
| **Continuation Rate** | 0.00% | 0.00% | 0.00% |
| **Speed (tok/s)** | 159,222 | 8,420 | 512,874 |
| **Time (s)** | 0.0362 | 0.6357 | 0.0139 |
| **Roundtrip Fidelity** | ✗ | ✗ | ✗ |

## 12. Vocabulary Size Ablation Results

> Tokenizers trained at different vocabulary sizes from the `Balochi_Tokenizer_Vocab_Ablation.ipynb` notebook.

| Metric | Balochi_BPE_32K | Balochi_BPE_47K | Balochi_BPE_64K | Balochi_BPE_80K | Balochi_BPE_128K | Balochi_WP_32K | Balochi_WP_47K | Balochi_WP_64K | Balochi_WP_80K | Balochi_WP_128K | Balochi_SP_32K | Balochi_SP_47K | Balochi_SP_64K | Balochi_SP_80K | Balochi_SP_128K |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| **Token Count** | 6,029 | 5,930 | 5,873 | 5,767 | 5,712 | 6,067 | 5,961 | 5,912 | 5,819 | 5,739 | 5,956 | 5,841 | 5,746 | 5,719 | 5,665 |
| **Unique Tokens** | 1,030 | 1,020 | 1,015 | 1,006 | 992 | 1,048 | 1,031 | 1,029 | 1,029 | 1,014 | 1,109 | 1,102 | 1,105 | 1,106 | 1,099 |
| **Vocab Size** | 32000 | 47000 | 64000 | 80000 | 128000 | 32000 | 47000 | 64000 | 80000 | 128000 | 32000 | 47000 | 64000 | 80000 | 128000 |
| **Vocab Util. (%)** | 3.22% | 2.17% | 1.59% | 1.26% | 0.78% | 3.28% | 2.19% | 1.61% | 1.29% | 0.79% | 3.47% | 2.34% | 1.73% | 1.38% | 0.86% |
| **Compression Ratio** | 3.89 | 3.95 | 3.99 | 4.06 | 4.10 | 3.86 | 3.93 | 3.96 | 4.03 | 4.08 | 3.94 | 4.01 | 4.08 | 4.10 | 4.14 |
| **Fertility** | 1.197 | 1.178 | 1.166 | 1.145 | 1.134 | 1.205 | 1.184 | 1.174 | 1.156 | 1.140 | 1.183 | 1.160 | 1.141 | 1.136 | 1.125 |
| **Avg Token Length** | 3.05 | 3.10 | 3.13 | 3.19 | 3.22 | 3.19 | 3.21 | 3.22 | 3.24 | 3.26 | 3.94 | 4.01 | 4.08 | 4.10 | 4.14 |
| **UNK Count** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **UNK Rate (%)** | 0.0000% | 0.0000% | 0.0000% | 0.0000% | 0.0000% | 0.0000% | 0.0000% | 0.0000% | 0.0000% | 0.0000% | 0.0000% | 0.0000% | 0.0000% | 0.0000% | 0.0000% |
| **Continuation Rate** | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 7.85% | 6.21% | 5.43% | 3.92% | 2.58% | 15.46% | 13.80% | 12.37% | 11.96% | 11.12% |
| **Speed (tok/s)** | 319,478 | 286,821 | 249,905 | 261,492 | 186,553 | 291,362 | 232,023 | 211,900 | 311,641 | 307,655 | 1,026,189 | 811,284 | 879,360 | 849,992 | 735,753 |
| **Time (s)** | 0.0189 | 0.0207 | 0.0235 | 0.0221 | 0.0306 | 0.0208 | 0.0257 | 0.0279 | 0.0187 | 0.0187 | 0.0058 | 0.0072 | 0.0065 | 0.0067 | 0.0077 |
| **Roundtrip Fidelity** | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ |

## 13. Analysis & Interpretation

### 13.1 Cross-Language Compression Comparison

| Language Group | Avg Compression | Avg Fertility | Tokenizers |
|----------------|-----------------|---------------|------------|
| Balochi (custom) | 4.04 | 1.154 | Balochi_BPE, Balochi_WordPiece, Balochi_SentencePiece |
| Arabic (custom) | 2.30 | 2.423 | AraBERT_v2, CAMeLBERT_MSA, ARBERT, AraGPT2 |
| Persian (custom) | 1.84 | 1.125 | ParsBERT, PersianBERT_FA, PersianBPE |
| Urdu (custom) | 0.58 | 8.043 | UrduBERT |
| Baseline (generic) | 2.84 | 1.851 | BERT, Gemma, NLTK |

### 13.2 Key Findings

- **Domain specificity advantage:** Custom Balochi tokenizers are expected to outperform generic multilingual tokenizers (BERT, Gemma) on Balochi text by producing fewer subword fragments and lower fertility.

- **Script-family proximity:** Arabic and Persian tokenizers share the same Perso-Arabic script family as Balochi, making their fertility and UNK rates the most meaningful external benchmarks — more so than mBERT or Gemma.

- **Vocabulary size effect:** Larger vocabulary (80K–128K) generally reduces fertility but increases memory overhead. The optimal point for Balochi is determined by the Rényi efficiency ablation in the companion notebook.

- **AraBERT v2 (64K WP)** and **ParsBERT (100K WP)** serve as the primary WordPiece upper-bound references — both trained on 70M+ token Perso-Arabic corpora.

- **AraGPT2 (BPE 50K)** provides the cleanest Arabic BPE reference for comparison against Balochi BPE (80K) in the BPE group.

- **UrduBERT** is the most linguistically proximate external tokenizer to Balochi due to shared Nastaliq script conventions and similar morphological complexity.

### 13.3 Roundtrip Fidelity

| Tokenizer | Fidelity |
|-----------|----------|
| Balochi_BPE | ✗ Lossy |
| Balochi_WordPiece | ✗ Lossy |
| Balochi_SentencePiece | ✓ Lossless |
| NLTK | ✗ Lossy |
| BERT | ✗ Lossy |
| Gemma | ✗ Lossy |
| Balochi_30K | ✗ Lossy |
| AraBERT_v2 | ✗ Lossy |
| CAMeLBERT_MSA | ✗ Lossy |
| ARBERT | ✗ Lossy |
| AraGPT2 | ✓ Lossless |
| ParsBERT | ✗ Lossy |
| PersianBERT_FA | ✗ Lossy |
| PersianBPE | — N/A |
| UrduBERT | ✗ Lossy |

## 14. Tokenizer Selection Guide

| Use Case | Recommended Tokenizer | Rationale |
|----------|----------------------|-----------|
| Balochi BERT fine-tuning | **Balochi_WordPiece 64K** | Native WP; matches BERT architecture exactly |
| Balochi GPT/Gemma CPT | **Balochi_BPE 80K** | BPE aligns with GPT-2/Gemma training conventions |
| Balochi SentencePiece pipeline | **Balochi_SP 64K** | Zero UNK via byte_fallback; SP models (T5, mT5) |
| Arabic NER / SA tasks | **AraBERT v2 or CAMeLBERT** | Proven Arabic BERT baselines |
| Arabic text generation | **AraGPT2** | Arabic GPT-2 with BPE tokenizer |
| Persian BERT tasks | **ParsBERT** | Standard Persian BERT baseline |
| Persian text analysis | **PersianBERT_FA** | HooshvareLab FA-base, widely used |
| Urdu NLP tasks | **UrduBERT** | Nastaliq script; same script family as Balochi |
| Cross-lingual baseline | **BERT Multilingual 119K** | Covers 104 languages for transfer comparison |
| Word-count baseline | **NLTK** | Raw word count before subword splitting |

## 15. Citation

```bibtex
@misc{hafeezullah2025balochi,
  title   = {Comprehensive Balochi Tokenizer Comparison: Custom vs. Cross-Language Baselines},
  author  = {Hafeez Ullah},
  year    = {2025},
  url     = {https://huggingface.co/balochiml},
  note    = {University of Gwadar, Department of Computer Science}
}
```
