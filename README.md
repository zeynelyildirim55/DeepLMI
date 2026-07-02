# DeepLMI: Rigorous lncRNA-miRNA Interaction Prediction

This repository provides a strict, leak-free implementation for predicting interactions between long non-coding RNAs (lncRNAs) and microRNAs (miRNAs) using Graph Convolutional Networks (GCN) and NLP-based biological embeddings.

This implementation specifically addresses and rectifies **Data Leakage** issues commonly found in interaction prediction literature. It is designed to evaluate the true limits of transductive GCN architectures on highly sparse biological graphs, particularly focusing on **Cold-Start (Blind)** scenarios.

---

## 🔬 Dataset & Preprocessing

Unlike conventional dense interaction databases, this repository utilizes a highly specific and sparse interaction network.
- **`custom_dataset/node_link.csv`**: The core ground-truth graph, heavily filtered from raw interaction chunks to contain only validated pairs for specific targeted RNAs.
- **`fasta_files/`**: Raw nucleotide sequences used to extract node features via `Doc2Vec` and `RNA-FM`.
- **`data_with_negatives/`**: Since deep learning requires counter-examples, this module generates synthetic non-interacting pairs (Label 0) at a 1:1 ratio with true interactions (Label 1) to build robust train/test splits.

> **Note:** The final, ready-to-use splits (incorporating negatives) are located in `processed_data/splits_indexed/`.

---

## 🚀 How to Run

### 1. Standard 5-Fold Cross Validation
Evaluates the model natively on the graph. This script utilizes K-Fold CV, dynamically hiding 20% of the edges during training.
```bash
python main.py
```
*Note: Due to the extreme sparsity of the custom dataset, breaking 20% of the edges causes heavy network fragmentation, resulting in a realistic Val AUC of ~0.67.*

### 2. Cold-Start (Blind) Tests
Evaluates the model's ability to predict interactions for completely unseen, novel RNAs that have no initial connections in the training graph.

```bash
# Test where miRNAs are unseen
python main_cold_start.py --mode blind_miRNA

# Test where lncRNAs are unseen
python main_cold_start.py --mode blind_lncRNA

# Test where BOTH RNAs are completely unseen
python main_cold_start.py --mode blind_both
```
*Note: Since standard GCNs are Transductive and rely solely on topological message-passing, these tests mathematically result in an AUC of ~0.47 (random guessing). This strictly proves the inherent limitation of GCNs on zero-degree cold-start nodes.*

---

## 🛠️ Environment Setup

```bash
conda create -n deeplmi python=3.10
conda activate deeplmi
pip install -r requirements.txt
```

---

## 📝 Findings & Conclusion
The experiments conducted in this repository explicitly demonstrate that without data leakage:
1. GCN performance drops significantly on fragmented/sparse topologies.
2. Predicting links for Cold-Start (Blind) nodes using standard Transductive GCNs is mathematically impossible without incorporating Inductive branches (e.g., GraphSAGE) or relying purely on sequence-to-sequence matching models.
