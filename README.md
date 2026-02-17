# GPCR-SLM

**GPCR-SLM**: Small Language Model-Based Classification of GPCRs using Knowledge Distillation Technique

Accurate protein family classification is crucial, as proteins within the same family share conserved structural domains and biochemical functions that determine their biological roles. GPCR-SLM provides a scalable and efficient framework to classify **G-protein coupled receptors (GPCRs)** across 86 distinct families using a lightweight transformer model optimized through knowledge distillation.

---


## Abstract

GPCR-SLM addresses the limitations of traditional sequence alignment tools like BLAST and HMMER, which struggle to distinguish closely related GPCR families with low sequence homology. Deep learning methods often use fixed-size architectures, preventing recognition of novel families.  

Our framework uses a **lightweight transformer model** optimized through **knowledge distillation**, achieving:

- **Accuracy**: 99%  
- **BLAST comparison**: 86.4%  
- **HMMER comparison**: 91%  
- **Computational efficiency**: ~33.5× faster than large protein language models  

GPCR-SLM demonstrates that distilled protein language models combined with flexible classification frameworks enable **high-resolution functional annotation** while remaining computationally efficient.

---

## Features

- Classification across **86 GPCR families**  
- Lightweight transformer architecture  
- Knowledge distillation for improved efficiency  
- Scalable framework for newly discovered protein families  
- High accuracy and computational efficiency  
- Easy-to-use scripts for training, evaluation, and prediction  

---

## Installation

1. **Clone the repository**:  
```bash
git clone https://github.com/username/GPCR-SLM.git
cd GPCR-SLM
