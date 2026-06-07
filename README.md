# 🔤 Balochi Tokenizer — `Balochi_SP_47K`

> **Production tokenizer for Southern Balochi** · SentencePiece Unigram · 47,000 vocabulary · AraToken-normalized · Entropy-pruned
Evaluated against 27 tokenizers across 6 research phases. Optimized for integration with the **Gemma 2B Language Extension Pipeline (LEP)**.

---

## Why This Tokenizer Exists

Standard multilingual tokenizers fragment Balochi text at **2× the rate** of this custom model. BERT shatters the single word `کپیٹلسٹک` into 6 pieces; this tokenizer keeps it whole.

| Tokenizer | `کپیٹلسٹک` (capitalistic) | Tokens |
|:---|:---|:---:|
| **Balochi SP 47K** | `['▁کپیٹل', 'سٹک']` | **2** ✅ |
| BERT Multilingual | `['ک', '##پی', '##ٹ', '##لس', '##ٹ', '##ک']` | **6** ❌ |
| Gemma 2B | `['ک', 'پی', 'ٹ', 'لس', 'ٹ', 'ک']` | **6** ❌ |

In a 4,096-token context window, this tokenizer fits **~3,593 Balochi words** vs. ~1,665 for Gemma — nearly **2× more semantic content per forward pass**.

---

## Quick Start

```python
import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.Load("balochi_sp_47k.model")

text =
 "
باسک یک لبزانکی ءُ اِلمی دیوان انت ۔ پہ زبان ءُ لبزانک ءِ دیمروئ ءَ باسکءِ تند جاہ  سال 2005 ءَ بنگیج کنگ بیتگ ۔   اے جہدءِ بندات ءَ باسک ءَ  ھالتران ، سازءُ زیمل ءِ دیمروئ، دودءُ ربیدگ ءِ پجار ءِ سرھالانی سرءَ ھم کار کُتگ ۔ اے کاروان ءَ بازینے باسک اتکگ ءُ بازینے باسک شُتگ اَنت بلے ھرکس ءَ کہ اے دیوان ءَ کلہوکے ھم کارکُتگ آئ ءِ نام چَہ ادءَ گار نہ بیتگ۔
"
tokens = sp.EncodeAsPieces(text)
ids    = sp.EncodeAsIds(text)

# Lossless roundtrip (unique to SentencePiece family)
assert sp.Decode(ids) == text
```

```python
# HuggingFace-compatible wrapper
from transformers import PreTrainedTokenizerFast
tokenizer = PreTrainedTokenizerFast(tokenizer_file="balochi_sp_47k.json")
```

---

## Performance vs. Multilingual Baselines

Evaluated on `liberal capitalism.txt` (5,036 words of Southern Balochi political text):

| Tokenizer | Compression | Fertility | UNK Rate | Continuation Rate | Roundtrip |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Balochi SP 47K** | **~4.13** | **~1.14** | **0.00%** | **~12.4%** | **✅ Lossless** |
| Balochi SP 64K | 4.13 | 1.14 | 0.00% | 12.4% | ✅ Lossless |
| Balochi SP 128K | 4.27 | 1.10 | 0.00% | 8.5% | ✅ Lossless |
| BERT Multilingual | 2.25 | 1.97 | 0.44% | 43.7% | ✗ Lossy |
| Gemma 2B | 2.12 | 2.10 | 0.00% | 52.4% | ✗ Lossy |
| AraBERT v2 | 3.72 | 1.20 | **51.0%** | 7.1% | ✗ Lossy |
| UrduBERT | 0.58 | **7.63** | 0.00% | 0.0% | ✗ Lossy |

> **Fertility** = tokens per word (lower → better). **1.14** means nearly one token per Balochi word.  
> UrduBERT's fertility of 7.63 confirms script similarity ≠ tokenizer compatibility.

---

## Why 47K and Not 64K?

Pruning from 64K was driven by **entropy analysis on the full normalized corpus** (54.3M tokens):

| Vocab Size | Shannon Entropy (H₁) | Effective Vocab (V_eff) | Efficiency (η) |
|:---:|:---:|:---:|:---:|
| 128,000 | 7.3924 | 168.0 | 0.13% |
| 64,000 | 7.4184 | 171.1 | 0.27% |
| **47,000** | **7.4295** | **172.4** | **0.37%** ✅ |

**Removing 17,000 noise tokens raises both entropy and effective vocabulary.** Those tokens were diacritical variants and ZWNJ-split artifacts that only existed because the training corpus hadn't been normalized yet — they would never appear in clean text.

Cumulative frequency analysis showed that 99% of real Balochi text is covered by just **46,406 tokens**. The 47K model rounds up slightly for safety.

| Coverage Target | Vocab Required | Tokens Pruned |
|:---:|:---:|:---:|
| 90% | 10,219 | 53,781 |
| 95% | 21,141 | 42,859 |
| **99%** | **46,406** | **17,594** |
| 99.5% | 53,240 | 10,760 |

For **Gemma 2B LEP integration**, this means ~17,000 fewer new embedding rows — a **26.6% reduction** in embedding layer dimensions with no measurable coverage loss.

---


### Latest Performance Metrics (47K Vocabulary)

| Tokenizer | Token Count | Compression | Fertility | Roundtrip |
|:---|:---:|:---:|:---:|:---:|
| **Balochi SP 47K** | **5,746** | **4.08** | **1.141** | Lossless |
| **Balochi BPE 47K** | **5,767** | **4.06** | **1.145** | Lossless |
| **Balochi WP 47K** | **5,912** | **3.96** | **1.174** | Lossless |
| Gemma 2B | 11,621 | 2.02 | 2.308 | Fragmented |
| BERT Multilingual | 10,991 | 2.13 | 2.183 | Fragmented |

## Normalization Pipeline

This tokenizer was trained on text pre-processed with a Balochi-adapted version of the **AraToken** normalization methodology:

| Rule | Characters | Note |
|:---|:---|:---|
| NFKC Unicode normalization | All compatibility chars | Applied first |
| Diacritics removal | `\u064B`–`\u065F` | Inconsistently used in Balochi; causes duplicate embeddings |
| Alif variant unification | أ/إ/آ/ٱ → ا | Standardizes Arabic loanword stems |
| Tatweel removal | `\u0640` | Decorative only; no linguistic content |
| Arabic punctuation | ؟→? · ؛→; · ،→, | Prevents duplicate punctuation tokens |
| Arabic-Indic numerals | ٠١٢…٩ → 0-9 | Eliminates numeral inconsistency |
| ZWNJ/ZWJ removal | `\u200C`, `\u200D` | **Balochi-specific** — causes spurious verb-compound splits |
| RLM/ALM removal | `\u200F`, `\u061C` | **Balochi-specific** — copy-paste artifacts in mixed-direction text |
| **Ye variant preserved** | `ے` vs `ی` | **Inverted from AraToken** — this distinction carries grammatical case in Southern Balochi |

```python
import re, unicodedata

def normalize_balochi(text: str, drop_diacritics: bool = True,
                      preserve_ye: bool = True) -> str:
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'[أإآٱ]', 'ا', text)
    if not preserve_ye:
        text = text.replace('ے', 'ی')
    text = text.translate(str.maketrans('٠١٢٣٤٥٦٧٨٩', '0123456789'))
    text = text.replace('؟','?').replace('؛',';').replace('،',',')
    text = text.replace('\u0640', '')
    text = text.replace('\u200C','').replace('\u200D','')
    text = text.replace('\u200F','').replace('\u061C','')
    if drop_diacritics:
        text = re.sub(r'(?<!ء)[\u064B-\u065F\u0610-\u061A\u06D6-\u06DC]', '', text)
    return re.sub(r'\s+', ' ', text).strip()
```

---

## Training Corpus

These tokenizers were trained from scratch on a massive, deduplicated dataset comprising **~54 Million words**.

| Source | Size | Content |
|:---|:---:|:---|
| `balochi_dedup_corpus.txt` | ~52M words | Deduplicated Balochi prose: literature, news, religious, conversational |
| `balochi_clean_corpus_dictionary.txt` | ~lexical | Curated dictionary entries and normalized grammatical forms |
| `english_corpus_2M.txt` | ~2M words | English text for code-switching coverage |
| **Total** | **~54M words** | **185.6M characters** |

**The Mixed-Corpus Advantage:**
To prevent the models from destroying or over-fragmenting English loan-words or technical terminology, the training data strategically includes **~2 million English words** seamlessly mixed with the ~52 million Balochi words. This highly optimized ratio ensures that the tokenizers can flawlessly process English text (e.g., "war industry", mathematical symbols, code) without sacrificing the vocabulary slots dedicated strictly to core Balochi morphological roots.

---

## Model Variants

| Model | Vocab | Best For |
|:---|:---:|:---|
| ⭐ **`Balochi_SP_47K`** | 47,000 | Gemma/LLaMA CPT · optimal embedding efficiency · lossless roundtrip |
| `Balochi_SP_64K` | 64,000 | Resource-constrained production · compression/memory sweet spot |
| `Balochi_SP_128K` | 128,000 | Maximum sequence packing · best raw compression (4.26 chars/token) |
| `Balochi_WP_47K` | 47,000 | BERT / ALBERT fine-tuning · native WordPiece architecture · protects Latin alphabet and Arabic numerals natively |
| `Balochi_BPE_47K` | 47,000 | GPT-2 / RoBERTa pre-training · BPE-aligned training · relies on HF's fast tokenizers |

All SP models include `byte_fallback=True` — **zero UNK rate** on any Unicode input.

---

## 💻 Usage Code for Tokenizer Architectures

### 1. WordPiece Tokenizer (BERT-style)

```python
from tokenizers import Tokenizer

# Load the custom WordPiece tokenizer
wp_tokenizer = Tokenizer.from_file("Balochi_WP_47K.json")

text =
 "
باسک یک لبزانکی ءُ اِلمی دیوان انت ۔ پہ زبان ءُ لبزانک ءِ دیمروئ ءَ باسکءِ تند جاہ  سال 2005 ءَ بنگیج کنگ بیتگ ۔   اے جہدءِ بندات ءَ باسک ءَ  ھالتران ، سازءُ زیمل ءِ دیمروئ، دودءُ ربیدگ ءِ پجار ءِ سرھالانی سرءَ ھم کار کُتگ ۔ اے کاروان ءَ بازینے باسک اتکگ ءُ بازینے باسک شُتگ اَنت بلے ھرکس ءَ کہ اے دیوان ءَ کلہوکے ھم کارکُتگ آئ ءِ نام چَہ ادءَ گار نہ بیتگ۔
"
encoded = wp_tokenizer.encode(text)

print("WordPiece Tokens:")
print(encoded.tokens)
```

### 2. SentencePiece Tokenizer (Gemma / LLaMA-style)

```python
import sentencepiece as spm

# Load the custom SentencePiece model
spm_model = spm.SentencePieceProcessor()
spm_model.load("balochi_sp_47k.model")

text =
 "
باسک یک لبزانکی ءُ اِلمی دیوان انت ۔ پہ زبان ءُ لبزانک ءِ دیمروئ ءَ باسکءِ تند جاہ  سال 2005 ءَ بنگیج کنگ بیتگ ۔   اے جہدءِ بندات ءَ باسک ءَ  ھالتران ، سازءُ زیمل ءِ دیمروئ، دودءُ ربیدگ ءِ پجار ءِ سرھالانی سرءَ ھم کار کُتگ ۔ اے کاروان ءَ بازینے باسک اتکگ ءُ بازینے باسک شُتگ اَنت بلے ھرکس ءَ کہ اے دیوان ءَ کلہوکے ھم کارکُتگ آئ ءِ نام چَہ ادءَ گار نہ بیتگ۔
"
tokens = spm_model.encode_as_pieces(text)

print("SentencePiece Tokens:")
print(tokens)
```

### 3. HF BPE Tokenizer (RoBERTa / GPT-2-style)

```python
from tokenizers import Tokenizer

# Load the custom Hugging Face BPE tokenizer
bpe_tokenizer = Tokenizer.from_file("Balochi_BPE_47K.json")

text =
 "
باسک یک لبزانکی ءُ اِلمی دیوان انت ۔ پہ زبان ءُ لبزانک ءِ دیمروئ ءَ باسکءِ تند جاہ  سال 2005 ءَ بنگیج کنگ بیتگ ۔   اے جہدءِ بندات ءَ باسک ءَ  ھالتران ، سازءُ زیمل ءِ دیمروئ، دودءُ ربیدگ ءِ پجار ءِ سرھالانی سرءَ ھم کار کُتگ ۔ اے کاروان ءَ بازینے باسک اتکگ ءُ بازینے باسک شُتگ اَنت بلے ھرکس ءَ کہ اے دیوان ءَ کلہوکے ھم کارکُتگ آئ ءِ نام چَہ ادءَ گار نہ بیتگ۔
"
encoded = bpe_tokenizer.encode(text)

print("HF BPE Tokens:")
print(encoded.tokens)
```

---

## Training Configuration

| Parameter | Value |
|:---|:---|
| Algorithm | SentencePiece Unigram |
| Implementation | Google `sentencepiece` |
| Vocabulary size | 47,000 |
| Character coverage | `0.9995` |
| Byte fallback | `True` |
| Space handling | `▁` word-start marker |
| Special tokens | `<pad>`, `<unk>`, `<s>`, `</s>` |
| Roundtrip fidelity | ✅ Lossless |
| Normalization | AraToken-adapted (Balochi-specific rules) |

---

## Intended Uses & Limitations

**Suitable for:**
- Continual pre-training (CPT) of LLMs on Southern Balochi text
- BERT-style masked language modeling (use `Balochi_WP_47K` for WordPiece compatibility)
- Named entity recognition, text classification, sentiment analysis on Balochi
- Gemma 2B Language Extension Pipeline (LEP) via vocabulary extension + mean subtoken initialization

**Limitations:**
- Optimized for **Southern Balochi**; Northern, Eastern, and Makrani dialect performance is untested
- Evaluated on political/economic text only — cross-domain generalization (literary, social media, technical) requires validation
- No downstream task benchmarking (NER F1, MLM perplexity) has been performed yet
- English coverage is functional but not optimized; a bilingual Bal-Eng tokenizer is planned

---
</div>
