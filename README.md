# CAFA-6 Protein Function Prediction (AI-driven Development)

This repository is designed for **AI-driven development** of a protein function
prediction system for the Kaggle CAFA-6 competition.

**Competition Link**: https://www.kaggle.com/competitions/cafa-6-protein-function-prediction/overview

---

## 🎯 Project Goal

Predict Gene Ontology (GO) terms for proteins using:

- Amino acid sequences
- (Optional) taxonomy / species information


---

## ❌ Non-Goals (Important)

This repository is **NOT** intended to:

- Train very large end-to-end models from scratch
- Perform heavy hyperparameter searches inside Kaggle
- Mix experimental notebook code with production code

---

## 🧠 AI Development Policy

When modifying or extending this repository, **AI agents must follow these rules**:

1. All reusable logic must be implemented under `src/`
2. Notebooks are for exploration only (no core logic)
3. File paths and parameters must be configurable via `configs/`
4. Cached artifacts (embeddings, models) must be saved explicitly
5. Code should favor clarity and debuggability over extreme optimization

---

## 📦 Dataset Overview

The dataset is designed for **multi-label protein function prediction** using Gene Ontology (GO) terms.

### Key Files

| File | Description |
|------|-------------|
| `train_sequences.fasta` | Protein amino acid sequences (training) |
| `train_terms.tsv` | Ground truth GO term annotations |
| `train_taxonomy.tsv` | Taxonomy (species) information for proteins |
| `go-basic.obo` | Gene Ontology hierarchy structure |
| `testsuperset.fasta` | Test protein sequences for prediction |
| `testsuperset-taxon-list.tsv` | Taxonomy IDs for test proteins |
| `IA.tsv` | Information Accretion weights for GO terms |
| `sample_submission.tsv` | Submission format template |

### Important Notes

- **Multi-label problem**: Each protein can have multiple GO terms
- **Hierarchical structure**: GO terms are organized in an ontology
- **Taxonomy is optional**: Species information may improve predictions

### Data Statistics (from `src/main.ipynb` analysis)

**Training Sequences:**
- Number of sequences: 82,403
- Sequence length distribution:
  - Min: 16 amino acids
  - Max: 10,000+ amino acids
  - Mean: ~400-500 amino acids
  - 25th percentile: ~200 aa
  - 50th percentile (median): ~350 aa
  - 75th percentile: ~550 aa
  - 90th percentile: ~800 aa
  - 95th percentile: ~1,100 aa
  - 99th percentile: ~2,000 aa
- Long sequences (>10,000 aa): A small number exist

**Test Data:**
- Species distribution (from `testsuperset-taxon-list.tsv`):
  - Human (Homo sapiens, 9606)
  - Rat (Rattus norvegicus, 10116)
  - Rice (Oryza sativa subsp. japonica, 39947)
  - Zebrafish (Danio rerio, 7955)
  - Fruit fly (Drosophila melanogaster, 7227)

**GO Terms:**
- GO ontology structure includes:
  - `id`: GO term identifier (e.g., GO:0000001)
  - `name`: Term name (e.g., "mitochondrion inheritance")
  - `namespace`: biological_process, molecular_function, or cellular_component
  - `is_a`: Parent-child relationships in the ontology
  - `def`: Definition with references

**Submission Format:**
- The submission file contains both GO term predictions and free-text descriptions
- Two types of rows per protein:
  - GO term rows: `protein_id`, `GO:XXXXXXX`, `score` (0-1)
  - Text description rows: `protein_id`, `Text`, `score`, `description`


## 📁 Project Structure

```
Kaggle_CAFA6/
├── input/                          # Competition data files
│   └── cafa-6-protein-function-prediction/
│       ├── Train/                  # Training data
│       │   ├── train_sequences.fasta
│       │   ├── train_terms.tsv
│       │   ├── train_taxonomy.tsv
│       │   └── go-basic.obo
│       ├── Test/                   # Test data
│       │   ├── testsuperset.fasta
│       │   └── testsuperset-taxon-list.tsv
│       ├── IA.tsv                  # GO term weights
│       └── sample_submission.tsv
├── src/                            # Source code (reusable logic)
│   └── main.ipynb                  # Main notebook for Kaggle submission
├── model/                          # Pre-trained model weights
│   ├── esm2_t12_35M_UR50D/        # ESM2 protein embedding model
│   ├── esm2_t30_150M_UR50D/
│   ├── esm2_t33_650M_UR50D/
│   └── BiomedNLP-BiomedBERT-base-uncased-abstract/  # BiomedBERT for GO text
├── output/                         # Generated embeddings and predictions
├── pyproject.toml                  # Project dependencies (uv)
└── README.md
```

---

## 🧠 Modeling Approach

This repository adopts a **dual-embedding (dual-encoder) approach** for protein
function prediction.

Instead of directly classifying proteins into a fixed set of GO labels,
we embed **proteins** and **GO terms** into a shared latent space and train a model
to align them.

---

### 1. Protein Embedding

- Each protein is represented by its amino acid sequence.
- Sequences are encoded using a pretrained protein language model
  (e.g. ESM-2).
- The output is a fixed-dimensional vector representing the protein.

This embedding captures:
- sequence patterns
- evolutionary and structural signals
- biochemical properties learned from large protein databases

---

### 2. GO Term Embedding

- Each Gene Ontology (GO) term is also represented as a vector.
- GO embeddings are obtained separately (e.g. from text descriptions,
  ontology structure, or a pretrained language model).
- All GO terms are embedded into the same dimensional space.

This allows the model to:
- reason about similarities between GO terms
- generalize across related biological functions

---

### 3. Shared Latent Space

Both protein embeddings and GO embeddings are projected into a **shared latent space**
using lightweight neural networks (typically linear layers).

Let:
- `h_p` be the protein embedding
- `h_go` be the GO term embedding

Then:
- `z_p = f_p(h_p)`  
- `z_go = f_go(h_go)`

where `f_p` and `f_go` are learnable projection functions.

---

### 4. Training Objective

The model is trained to **bring related protein–GO pairs closer together**
in the latent space, while pushing unrelated pairs further apart.

Intuitively:
- If a protein is annotated with a GO term, their embeddings should be similar.
- If not, their embeddings should be dissimilar.

This alignment objective can be implemented using:
- similarity scores (e.g. dot product or cosine similarity)
- multi-label losses (e.g. BCE over protein–GO pairs)

---

### 5. Inference

At inference time:
1. Embed the protein sequence.
2. Compare it against all GO term embeddings.
3. Rank GO terms by similarity score.
4. Output GO terms and confidence scores in Kaggle submission format.

---

### Design Rationale

This embedding-based formulation offers several advantages:

- Scalability to large GO vocabularies
- Flexibility to incorporate GO semantics
- Efficient reuse of cached embeddings
- Clear separation between representation learning and prediction logic

This design is particularly suitable for:
- AI-assisted iterative development
- constrained execution environments (e.g. Kaggle notebooks)

---

## 🛠️ セットアップ手順

### 1. 環境構築

1. **uvをインストール**
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   uv --version
   ```

2. **仮想環境作成**
   ```bash
   uv venv
   ```

3. **仮想環境有効化**
   ```bash
   source .venv/bin/activate
   ```

4. **依存関係のインストール**
   ```bash
   uv pip install -e .
   ```

### 2. データセットのダウンロード

Kaggleからコンペティションデータをダウンロードし、`input/cafa-6-protein-function-prediction/`に配置してください。

```bash
# Kaggle CLIを使用する場合
kaggle competitions download -c cafa-6-protein-function-prediction
unzip cafa-6-protein-function-prediction.zip -d input/cafa-6-protein-function-prediction/
```

### 3. モデルファイルのセットアップ

このレポジトリでは、事前学習済みモデルは**Gitで管理していません**（サイズが大きいため）。
以下の手順でモデルファイルを準備してください。

#### オプションA: Hugging Face Hubから直接ダウンロード（推奨）

```python
from transformers import AutoModel, AutoTokenizer

# ESM2モデル（タンパク質埋め込み用）
model_name = "facebook/esm2_t12_35M_UR50D"
model = AutoModel.from_pretrained(model_name, cache_dir="./model")
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="./model")

# BiomedBERTモデル（GO term埋め込み用）
model_name = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
model = AutoModel.from_pretrained(model_name, cache_dir="./model")
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="./model")
```

#### オプションB: ローカルのzipファイルから解凍

`model/zips/`にモデルのzipファイルがある場合：

```bash
# model/zipsディレクトリに移動
cd model/zips

# 各モデルを解凍
unzip esm2_t12_35M_UR50D.zip -d ../esm2_t12_35M_UR50D/
unzip BiomedNLP-BiomedBERT-base-uncased-abstract.zip -d ../BiomedNLP-BiomedBERT-base-uncased-abstract/

# 必要に応じて他のモデルも解凍
# unzip esm2_t30_150M_UR50D.zip -d ../esm2_t30_150M_UR50D/
# unzip esm2_t33_650M_UR50D.zip -d ../esm2_t33_650M_UR50D/

cd ../..
```

#### 必要なモデル

現在のコードで使用しているモデル：
- `facebook/esm2_t12_35M_UR50D` - タンパク質配列の埋め込み用
- `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract` - GO term埋め込み用

**注意**: `model/`ディレクトリ全体は`.gitignore`で除外されています。モデルファイルはGitにコミットされません。

## To Do

### フォルダ整理

main.ipynbに機能が集中しすぎている
stepごとに.pyファイルを作成し、関数やクラスを管理する

---

### 
