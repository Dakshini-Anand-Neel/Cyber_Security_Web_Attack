# Data Card — Web Attacks Long Dataset

## Dataset Summary

| Field | Value |
|-------|-------|
| **Name** | `shengqin/web-attacks-long` |
| **Source** | [Hugging Face Datasets](https://huggingface.co/datasets/shengqin/web-attacks-long) |
| **Task** | Multi-class Text Classification (Web Attack Detection) |
| **Language** | English / Mixed (HTTP payloads, URL-encoded, HTML, SQL) |
| **License** | Public (see HF dataset page) |
| **Total Samples** | 16 401 |
| **Splits** | Train only (full dataset used, split manually 80/20) |
| **Format** | CSV → Parquet on HF; columns: `Payload`, `Label`, `text_label`, `ID` |

---

## Schema

| Column | Type | Description |
|--------|------|-------------|
| `Payload` | `string` | Raw HTTP request payload / query string |
| `Label` | `int64` | Numeric class label (1 = XSS, 2 = SQLi, 3 = Normal) |
| `text_label` | `string` | Human-readable label ("XSS", "SQLi", "normal") |
| `ID` | `int64` | Unique sample identifier |

---

## Class Distribution

| Class | Label | Count | Share |
|-------|-------|-------|-------|
| SQLi  | 2     | 7 567 | 46.1% |
| XSS   | 1     | 5 932 | 36.2% |
| Normal| 3     | 2 902 | 17.7% |

> **Note:** The dataset is moderately imbalanced — Normal samples are underrepresented. Stratified train/test splits are used to preserve proportions.

---

## Sample Payloads

| text_label | Payload Example |
|------------|-----------------|
| Normal     | `Select a paint color for the room.` |
| Normal     | `{"id":null,"name":"Slowbro"}` |
| XSS        | `<cite onpointerup=alert(1)>XSS</cite>` |
| XSS        | `<style>:target {transform: rotate(180deg);}</style>` |
| SQLi       | `1%" ) ) ) union all select null,null,null#` |
| SQLi       | `1 AND (SELECT 7028 FROM(SELECT COUNT(*),CONCAT(0x716a707871,...)` |

---

## Preprocessing Applied

1. **Deduplication** — 0 duplicate rows found in this dataset.
2. **Null Handling** — No null values present; logic included for robustness.
3. **Whitespace Stripping** — Applied to `Payload` and `text_label` columns.
4. **Payload Feature Engineering** — 7 derived numeric features:
   - `payload_len` — character count
   - `payload_entropy` — Shannon entropy (character level)
   - `special_char_cnt` — count of `< > " ' ( )`
   - `digit_ratio` — fraction of digit characters
   - `upper_ratio` — fraction of upper-case characters
   - `kw_sql` — binary flag for SQL keywords
   - `kw_xss` — binary flag for XSS tokens
5. **Encoding** — `Payload` and `text_label` label-encoded for ML; raw text preserved in `data/raw/`.
6. **Scaling** — StandardScaler (Z-score) applied to numeric features.

---

## Caveats & Limitations

- **Syntactic bias:** The dataset uses highly distinguishable patterns, leading to near-perfect classification scores. Real-world adversarial payloads may evade simple keyword-based features.
- **Language:** Payloads are primarily ASCII / URL-encoded. Multi-byte or obfuscated attacks may underperform.
- **Static dataset:** No concept drift modeling — production use requires periodic retraining.
- **Imbalance:** Normal class is ~18% of data; may affect threshold-sensitive use cases.

---

## Citation

```
Dataset: shengqin/web-attacks-long
Platform: Hugging Face Datasets
URL: https://huggingface.co/datasets/shengqin/web-attacks-long
```

---

## Local Paths

| Purpose | Path |
|---------|------|
| Raw CSV cache | `E:\CYBERSECURITY\data\raw_train.csv` |
| Processed features | `E:\CYBERSECURITY\data\processed\features.csv` |
| Interim cleaned | `E:\CYBERSECURITY\data\interim\cleaned.csv` |
