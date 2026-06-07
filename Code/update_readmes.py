import os
import re

# Paths to the READMEs
README_PATHS = [
    r"..\..\Balochi Tokenizer_HF\README.md",
    r"..\..\Final Tokenizers\README.md",
    r"..\..\Final Tokenizers-4\README.md",
    r"..\README_Git.md"
]

# Paths to the new reports
EXTENDED_REPORT_PATH = "Output/Tokenizer_Comparison_Extended_Report.md"
ABLATION_REPORT_PATH = "Output/Ablation/Vocab_Ablation_Report.md"

def parse_markdown_table(md_text, section_header):
    """Finds a table under a specific header and returns its rows."""
    pattern = re.compile(rf"{re.escape(section_header)}\n+.*?\|(.*?)\n\n", re.DOTALL)
    m = pattern.search(md_text)
    if not m:
        return []
    
    table_text = m.group(0)
    lines = [line.strip() for line in table_text.split('\n') if line.strip().startswith('|')]
    return lines

def extract_master_metrics():
    """Extracts the transposed master metrics table from the new report."""
    with open(EXTENDED_REPORT_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    
    lines = parse_markdown_table(content, "## 4. Master Metrics Table")
    if not lines or len(lines) < 3:
        return {}
    
    # Header row
    headers = [col.strip() for col in lines[0].split('|')[1:-1]]
    tokenizers = headers[1:] # First col is 'Metric'
    
    data = {tok: {} for tok in tokenizers}
    
    for row in lines[2:]:
        cols = [col.strip() for col in row.split('|')[1:-1]]
        if not cols:
            continue
        metric_name = cols[0].replace("**", "").strip()
        for i, val in enumerate(cols[1:]):
            if i < len(tokenizers):
                tok = tokenizers[i]
                data[tok][metric_name] = val
    
    return data

def extract_ablation_metrics(algo_header, eval_text_header):
    """Extracts ablation metrics for a specific algorithm and text."""
    with open(ABLATION_REPORT_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    
    # First find the text section
    text_split = content.split(eval_text_header)
    if len(text_split) < 2: return {}
    text_content = text_split[1]
    
    # Then find the algorithm section
    lines = parse_markdown_table(text_content, algo_header)
    if not lines or len(lines) < 3: return {}
    
    sizes = [col.strip() for col in lines[0].split('|')[1:-1]][1:]
    
    data = {size: {} for size in sizes}
    for row in lines[2:]:
        cols = [col.strip() for col in row.split('|')[1:-1]]
        if not cols: continue
        metric_name = cols[0].replace("**", "").strip()
        for i, val in enumerate(cols[1:]):
            if i < len(sizes):
                size = sizes[i]
                data[size][metric_name] = val
                
    return data

def extract_diminishing_returns(algo_header):
    with open(ABLATION_REPORT_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    
    text_split = content.split("## 3. Diminishing Returns Analysis")
    if len(text_split) < 2: return {}
    
    lines = parse_markdown_table(text_split[1], algo_header)
    if not lines or len(lines) < 3: return {}
    
    data = {}
    for row in lines[2:]:
        cols = [col.strip() for col in row.split('|')[1:-1]]
        if len(cols) >= 4:
            size = cols[0]
            data[size] = {
                "Avg Compression": cols[1],
                "Avg Fertility": cols[2],
                "Avg Tokens": cols[3]
            }
    return data

def process_readme(filepath, master_data, labzank_ablation, lib_cap_ablation, dr_data):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
        
    # Mapping between README tokenizer names and new report names
    tok_map = {
        "Balochi BPE": "Balochi_BPE",
        "Balochi WordPiece": "Balochi_WordPiece",
        "Balochi SentencePiece": "Balochi_SentencePiece",
        "NLTK (baseline)": "NLTK",
        "BERT Multilingual": "BERT",
        "Gemma 2B": "Gemma",
        "Balochi_30K (HF)": "Balochi_30K",
        "AraBERT_v2": "AraBERT_v2",
        "CAMeLBERT_MSA": "CAMeLBERT_MSA",
        "ARBERT": "ARBERT",
        "AraGPT2": "AraGPT2",
        "ParsBERT": "ParsBERT",
        "PersianBERT_FA": "PersianBERT_FA",
        "PersianBPE": "PersianBPE",
        "UrduBERT": "UrduBERT"
    }

    # 1. Update Master Performance Results
    def replace_master_row(m):
        row = m.group(0)
        cols = row.split('|')
        tok_display = cols[1].replace("**", "").strip()
        
        if tok_display in tok_map:
            tok_key = tok_map[tok_display]
            if tok_key in master_data:
                d = master_data[tok_key]
                
                # Format exactly as original
                is_bold = "**" in cols[1]
                
                def fmt(val, bold=False):
                    return f" **{val}** " if bold else f" {val} "
                
                try:
                    cols[2] = fmt(d.get("Token Count", ""), is_bold)
                    cols[3] = fmt(d.get("Compression Ratio", ""), is_bold)
                    cols[4] = fmt(d.get("Fertility", ""), is_bold)
                    cols[5] = fmt(d.get("UNK Count", ""), False)
                    cols[6] = fmt(d.get("UNK Rate (%)", ""), "**" in cols[6])
                    if len(cols) > 7:
                        cols[7] = fmt(d.get("Continuation Rate", ""), False)
                except Exception as e:
                    print(f"Error on row: {row}")
                    print(cols)
                    raise e
                
                return "|".join(cols)
        return row

    # Find the master table lines and replace
    lines = content.split('\n')
    in_master_table = False
    for i, line in enumerate(lines):
        if "### 5.2 Master Performance Results" in line:
            in_master_table = True
        elif in_master_table and (line.startswith("## ") or line.startswith("### ") and "5.2" not in line):
            in_master_table = False
            
        if in_master_table and line.strip().startswith('|') and not line.strip().startswith('|:') and not line.strip().startswith('| Tokenizer'):
            lines[i] = replace_master_row(re.match(r'.*', line))

    # Rejoin lines
    content = '\n'.join(lines)

    # 2. Update Ablation Results
    def update_ablation_table(content, header, data_source):
        in_table = False
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if header in line:
                in_table = True
            elif in_table and line.startswith("##") and not line.startswith("####"):
                in_table = False
                
            if in_table and line.strip().startswith('|') and not line.strip().startswith('|:') and not line.strip().startswith('| Algorithm'):
                cols = line.split('|')
                algo = cols[1].strip()
                size = cols[2].strip()
                size_key = size.replace("K", "") + "K"
                
                algo_key = "BPE" if "BPE" in algo else ("WORDPIECE" if "WP" in algo else "SENTENCEPIECE")
                if algo_key in data_source and size_key in data_source[algo_key]:
                    d = data_source[algo_key][size_key]
                    
                    is_bold = "**" in cols[3]
                    def fmt(val, bold=False): return f" **{val}** " if bold else f" {val} "
                    
                    # Token Count | Compression | Fertility | Vocab Util.
                    # Some tables have Unique Tokens, some don't.
                    if "Unique Tokens" in content.split(header)[1].split('\n')[2]: # Check header row for this table
                         if len(cols) > 5:
                             cols[3] = fmt(d.get("Token Count", ""), is_bold)
                             # cols[4] = Unique Tokens
                             cols[5] = fmt(d.get("Compression Ratio", ""), is_bold)
                             cols[6] = fmt(d.get("Fertility", ""), is_bold)
                             if len(cols) > 7 and "Vocab Util." in cols[7]:
                                 pass
                    else:
                         if len(cols) > 5:
                             cols[3] = fmt(d.get("Token Count", ""), is_bold)
                             cols[4] = fmt(d.get("Compression Ratio", ""), is_bold)
                             cols[5] = fmt(d.get("Fertility", ""), is_bold)
                    
                    lines[i] = "|".join(cols)
        return '\n'.join(lines)
    
    content = update_ablation_table(content, "#### `liberal capitalism.txt` Results", lib_cap_ablation)
    
    # 3. Update Diminishing Returns
    # Doing a simpler regex replacement for Diminishing returns numbers
    for algo, dr_algo_name in [("SentencePiece", "### SENTENCEPIECE"), ("BPE", "### BPE"), ("WordPiece", "### WORDPIECE")]:
        algo_data = dr_data.get(dr_algo_name, {})
        for size_key, vals in algo_data.items():
            # e.g., | **32,000** | 3.9982 | 1.1748 | 5,244 | — |
            pattern = re.compile(rf"\|\s*\**{size_key}\**\s*\|\s*[\d\.]+\s*\|\s*[\d\.]+\s*\|\s*[\d,]+\s*\|")
            def repl(m):
                s = m.group(0)
                parts = s.split('|')
                parts[2] = f" {vals['Avg Compression']} "
                parts[3] = f" {vals['Avg Fertility']} "
                parts[4] = f" {vals['Avg Tokens']} "
                return "|".join(parts)
            content = pattern.sub(repl, content)

    # Save
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
        print(f"Updated {filepath}")


def main():
    print("Extracting new data...")
    master_data = extract_master_metrics()
    
    lib_cap_ablation = {
        "BPE": extract_ablation_metrics("### BPE Algorithm", "## 2. Results — `liberal capitalism.txt`"),
        "WORDPIECE": extract_ablation_metrics("### WORDPIECE Algorithm", "## 2. Results — `liberal capitalism.txt`"),
        "SENTENCEPIECE": extract_ablation_metrics("### SENTENCEPIECE Algorithm", "## 2. Results — `liberal capitalism.txt`")
    }
    
    dr_data = {
        "### BPE": extract_diminishing_returns("### BPE"),
        "### WORDPIECE": extract_diminishing_returns("### WORDPIECE"),
        "### SENTENCEPIECE": extract_diminishing_returns("### SENTENCEPIECE")
    }

    print("Updating README files...")
    for path in README_PATHS:
        full_path = os.path.abspath(path)
        if os.path.exists(full_path):
            process_readme(full_path, master_data, {}, lib_cap_ablation, dr_data)
        else:
            print(f"Warning: Could not find {full_path}")

if __name__ == "__main__":
    main()
