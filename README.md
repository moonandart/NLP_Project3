# IndoBERT Summarization — BERT2BERT (Liputan6 News Dataset)

This project applies **Indonesian BERT2BERT** (`cahya/bert2bert-indonesian-summarization`) to summarize long-form news articles from **Liputan6**.  
It uses a **token-safe summarization approach** for BERT2BERT (max 512 tokens) and is optimized for **Google Colab (T4 GPU)**.

---

## 🧠 Model Overview

| Model | Type | Language | Framework |
|--------|------|-----------|-----------|
| [`cahya/bert2bert-indonesian-summarization`](https://huggingface.co/cahya/bert2bert-indonesian-summarization) | Seq2Seq | Bahasa Indonesia | 🤗 Transformers |

### Why BERT2BERT?
- Fine-tuned specifically for **Indonesian summarization**  
- Lightweight and stable on **Colab T4 GPU**  
- Ideal for abstractive summaries (Liputan6-style news)

---

## 🗂 Dataset
Dataset used: **Liputan6 News Articles (Cleaned)**  
File: `data/liputan6_clean_ready.csv`

| Column | Description |
|---------|-------------|
| `clean_article_text` | Full news article text |
| `clean_summary_text` | Reference human-written summary |

---

## 🚀 Notebook
Notebook path:  
[`notebooks/IndoBERT_Summarization_Bert2Bert_ID.ipynb`](notebooks/IndoBERT_Summarization_Bert2Bert_ID.ipynb)

### Colab Badge
You can open and run directly in Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/NLP_project3/blob/main/notebooks/IndoBERT_Summarization_Bert2Bert_ID.ipynb)

*(Replace `yourusername` with your GitHub username.)*

---

## ⚙️ Folder Structure

```
NLP_project3/
├── data/
│   └── liputan6_clean_ready.csv
├── notebooks/
│   └── IndoBERT_Summarization_Bert2Bert_ID.ipynb
├── outputs/
│   ├── summarized_results.csv
│   ├── rouge_scores.csv
│   └── examples/
│       ├── sample_article_1_summary.txt
│       └── ...
├── models/
│   └── cahya_bert2bert/
└── README.md
```

---

## 🧾 Key Features
- ✅ Token-safe summarization (auto-chunking for ≤512 tokens)
- ⚡ T4 GPU-ready, float16 precision
- 📊 Automatic ROUGE evaluation
- 💾 Saves summaries and examples in `/outputs/`
- 🧹 Clean Colab-Drive workflow (auto-mount and cache)

---

## 🧩 Example Output

| Input (Article Snippet) | Generated Summary |
|--------------------------|------------------|
| *"Pemerintah Indonesia hari ini mengumumkan..."* | *"Pemerintah Indonesia resmi mengumumkan kebijakan baru terkait..."* |

---

## 📈 Evaluation
The notebook uses **ROUGE** (via `evaluate` library) to measure summary quality vs reference summaries (`clean_summary_text`).

Results are stored in:
```
outputs/rouge_scores.csv
outputs/rouge_scores_<timestamp>.json
```

---

## 🧰 Requirements

- Python ≥ 3.9  
- `transformers >= 4.44.0`  
- `torch (CUDA 12.1)`  
- `sentencepiece`, `evaluate`, `rouge-score`

For Colab: these are auto-installed by the notebook.

---

## 🤝 Acknowledgements
- [Hugging Face Hub — @cahya](https://huggingface.co/cahya) for `bert2bert-indonesian-summarization`  
- [IndoNLP](https://github.com/IndoNLP) for the IndoBERT family  
- Liputan6 dataset preprocessed from [sum_liputan6](https://github.com/fajri91/sum_liputan6)

---

## 📚 Citation & Credits

If you use this notebook, model, or dataset in your research or projects, please cite the following works.

### 🧩 Model Reference
**Model:** [`cahya/bert2bert-indonesian-summarization`](https://huggingface.co/cahya/bert2bert-indonesian-summarization)  
**Author:** [Cahya Wirawan](https://github.com/cahya-wirawan)  
**License:** Apache 2.0  

```
@misc{wirawan2021bert2bert,
  author       = {Cahya Wirawan},
  title        = {BERT2BERT Indonesian Summarization},
  year         = {2021},
  publisher    = {Hugging Face},
  howpublished = {\url{https://huggingface.co/cahya/bert2bert-indonesian-summarization}},
}
```

---

### 🗂 Dataset Reference
**Dataset:** Liputan6: A Large-scale Indonesian Dataset for Text Summarization  
**Authors:** Fajri Koto, Jey Han Lau, and Timothy Baldwin  
**Conference:** AACL-IJCNLP 2020  
**Paper:** [ACL Anthology Link](https://aclanthology.org/2020.aacl-main.85/)

```
@inproceedings{koto2020liputan6,
  title     = {{Liputan6: A Large-scale Indonesian Dataset for Text Summarization}},
  author    = {Koto, Fajri and Lau, Jey Han and Baldwin, Timothy},
  booktitle = {Proceedings of the 1st Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics and the 10th International Joint Conference on Natural Language Processing (AACL-IJCNLP 2020)},
  year      = {2020},
  address   = {Suzhou, China},
  publisher = {Association for Computational Linguistics},
  pages     = {802--810},
  url       = {https://aclanthology.org/2020.aacl-main.85}
}
```

---

> 💡 **Acknowledgement:**  
> This project builds upon the open-source work of [Cahya Wirawan](https://github.com/cahya-wirawan) and the Liputan6 dataset creators — Fajri Koto, Jey Han Lau, and Timothy Baldwin. Their contributions to Indonesian NLP and summarization research are invaluable to the community.
