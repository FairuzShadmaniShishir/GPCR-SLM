# GPCR-SLM 🧬

**Small Language Model-Based Classification of GPCRs using Knowledge Distillation**

[![Paper](https://img.shields.io/badge/Paper-Under%20Review%20IEEE%20TCBB%202026-blue)](#citation)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-green)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

> A scalable, computationally efficient framework for classifying G protein-coupled receptors (GPCRs) across 86 families — targeting the superfamily behind ~35% of FDA-approved drugs — with 99% accuracy and 33.5× speedup over large protein language models.

---

## 📌 Overview

G protein-coupled receptors (GPCRs) represent the largest and most therapeutically important protein superfamily in eukaryotes, yet accurate classification remains challenging due to:

- **Low sequence homology** between closely related GPCR families
- **Fixed-size architectures** in existing deep learning tools that cannot recognize novel families
- **Computational cost** of large protein language models at proteome scale

GPCR-SLM solves all three problems with a **lightweight transformer** trained via **knowledge distillation**, combining the accuracy of large PLMs with the efficiency required for high-throughput drug discovery pipelines.

---

## 📊 Performance

| Method | Accuracy | Speed |
|---|---|---|
| **GPCR-SLM (ours)** | **99.0%** | **33.5× faster than large PLMs** |
| HMMER | 91.0% | Baseline |
| BLAST | 86.4% | Baseline |
| Classical ML on embeddings | <90% | Fast but inaccurate |

Evaluated across **86 GPCR families** with 5-fold cross-validation.

---

## ✅ External Validation

GPCR-SLM was validated on three real-world datasets beyond the training benchmark:

| Dataset | Description | Result |
|---|---|---|
| **Clinical GPCR variants** | Patient-derived sequence variants | Robust classification |
| **Metagenomic sequences** | Gut microbiome GPCR candidates | High-confidence annotations |
| **Tumor-derived sequences** | Cancer genomics GPCR data | Utility in oncology pipelines |

This confirms the model's generalizability beyond curated databases — critical for translational drug discovery applications.

---

## 🔑 Key Features

- **99% classification accuracy** across 86 GPCR families
- **Open-set framework** — handles novel GPCR families without retraining
- **Knowledge distillation** — student model captures teacher PLM knowledge at a fraction of the cost
- **33.5× computational speedup** over large protein language models
- **Rigorous baselines** — BLAST, HMMER, PHMMER, and classical ML comparisons included
- **Externally validated** on clinical, metagenomic, and tumor-derived data

---

## 🗂️ Repository Structure

```
GPCR-SLM/
│
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── main.py                          # Main entry point
│
├── src/
│   ├── model.py                     # Student transformer architecture
│   ├── train.py                     # Knowledge distillation training
│   ├── predict.py                   # Inference on new sequences
│   ├── embeddings.py                # PLM embedding generation
│   └── interpretability.py         # Attention-based interpretability
│
├── baselines/                       # Competitor method implementations
│   ├── blast_experiment.py          # BLAST classification pipeline
│   ├── blast_results.py             # BLAST result parsing
│   ├── hmmer3_experiment.py         # HMMER3 classification pipeline
│   ├── phmmer.py                    # PHMMER pipeline
│   ├── phmmer_evaluation.py         # PHMMER evaluation
│   ├── classical_classifiers.py     # ML baselines (SVM, RF, XGBoost)
│   └── auc_comparison.py            # ROC/AUC comparison across methods
│
├── validation/                      # External validation scripts
│   ├── gut_metagenome.py            # Metagenomic GPCR validation
│   └── visualize_gut_embedding.py   # Embedding visualization
│
└── results/
    └── all_results_cv_test_split.pkl
```

---

## ⚙️ Installation

```bash
git clone https://github.com/FairuzShadmaniShishir/GPCR-SLM.git
cd GPCR-SLM
pip install -r requirements.txt
```

**Requirements:** Python 3.8+, PyTorch, ESM-2/ProtGPT2, scikit-learn, BioPython (see `requirements.txt`)

---

## 🚀 Quick Start

**Classify GPCR sequences:**
```bash
python main.py --input your_sequences.fasta --output predictions.csv
```

**Train from scratch with knowledge distillation:**
```bash
python src/train.py --data data/gpcr_families.csv --teacher esm2 --epochs 50
```

**Run baseline comparisons:**
```bash
python baselines/blast_experiment.py --input your_sequences.fasta
python baselines/hmmer3_experiment.py --input your_sequences.fasta
```

**Validate on metagenomic data:**
```bash
python validation/gut_metagenome.py --input metagenome.fasta
```

---

## 🧠 Method

GPCR-SLM uses a two-stage training pipeline:

**1. Teacher Model**
A large pretrained protein language model (ESM-2 / ProtGPT2) generates rich sequence embeddings capturing evolutionary and functional information across GPCR families.

**2. Student Model (GPCR-SLM)**
A lightweight transformer is trained via knowledge distillation to mimic the teacher's representations — achieving near-identical accuracy at 33.5× lower computational cost.

**3. Open-Set Classification**
Unlike fixed-size softmax classifiers, GPCR-SLM uses a flexible classification head that can accommodate novel GPCR families discovered after training — essential as the protein universe continues to expand.

---

## 📄 Citation

If you use GPCR-SLM in your research, please cite:

```bibtex
@article{shishir2026gpcrslm,
  title   = {GPCR-SLM: Small Language Model-Based Classification of GPCRs
             using Knowledge Distillation},
  author  = {Fairuz Shadmani Shishir and Cuncong Zhong and Sumaiya Shomaji},
  journal = {IEEE Transactions on Computational Biology and Bioinformatics},
  year    = {2026},
  note    = {Under Review}
}
```

---

## 🔗 Related Work

- [NbBayesLM](https://github.com/FairuzShadmaniShishir/NbBayesLM) — Bayesian nanobody thermostability prediction · *Frontiers in Bioinformatics*
- [MetaLLM](https://github.com/FairuzShadmaniShishir/A-Deep-Learning-Framework-for-Protein-to-Metal-Binding-Prediction-Using-Protein-Language-Models) — Protein metal binding site prediction · *IEEE Trans. Comput. Biol. Bioinform.*, 2025
- [CIgFlow](#) — Antigen-specific antibody design via conditional flow matching · *Under Review, IEEE TCBB 2026*

---

## 📬 Contact

**Fairuz Shadmani Shishir**
PhD Candidate, University of Kansas
✉️ shishir@ku.edu
🔗 [Google Scholar](#) · [LinkedIn](https://www.linkedin.com/in/fairuz-shadmani-shishir-558a13142/) · [GitHub](https://github.com/FairuzShadmaniShishir)
