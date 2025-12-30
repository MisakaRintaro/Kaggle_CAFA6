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
│       ├── Test (Targets)/         # Test data
│       │   ├── testsuperset.fasta
│       │   └── testsuperset-taxon-list.tsv
│       ├── IA.tsv                  # GO term Information Accretion weights
│       └── sample_submission.tsv
├── src/                            # Source code (modularized)
│   ├── main.py                     # Main pipeline script
│   ├── config.py                   # Configuration and paths
│   ├── data_loader.py              # Data loading utilities
│   ├── protein_embedding.py        # ESM-2 protein embedding
│   ├── go_embedding.py             # BiomedBERT GO term embedding
│   ├── model.py                    # JointModel architecture
│   ├── training.py                 # Training loop and checkpointing
│   ├── prediction.py               # Inference and submission
│   ├── hierarchical_postprocess.py # Hierarchical GO postprocessing
│   ├── evaluation.py               # Evaluation metrics (IA-weighted Fmax)
│   └── main.ipynb                  # Legacy notebook (archived)
├── docs/                           # Documentation
│   ├── data_loading.md             # Data loading details
│   ├── embeddings.md               # Embedding generation
│   ├── model_architecture.md       # JointModel structure
│   ├── hierarchical_postprocessing.md  # Postprocessing details
│   └── evaluation.md               # Evaluation metrics (CAFA-6)
├── model/                          # Pre-trained model weights (gitignored)
│   ├── esm2_t12_35M_UR50D/        # ESM-2 protein embedding model
│   └── BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext/
├── output/                         # Generated files
│   ├── embeddings/                 # Cached embeddings
│   │   ├── train_protein_embeddings.pt
│   │   ├── test_protein_embeddings.pt
│   │   └── go_embeddings.pt
│   ├── models/                     # Trained models
│   │   └── joint_model.pt
│   └── submission.tsv              # Final submission file
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

## 🚀 Implementation Status & Pipeline

### Current Pipeline (7 Steps, 3 Phases)

The implementation is fully modularized into `src/*.py` files. The main pipeline in [src/main.py](src/main.py) is organized into **3 distinct phases** with **7 steps total**:

---

#### **Phase 1: Data Preparation (Steps 1-4)**

**Step 1: Load GO Ontology and Training Labels**
- Parse GO ontology from `go-basic.obo` to extract hierarchy (`child_to_parents`, `go_terms`)
- Load training labels from `train_terms.tsv`
- Load Information Accretion (IA) weights
- Implementation: [data_loader.py](src/data_loader.py)

**Step 2: Create GO Embeddings**
- Encode GO term names and definitions using BiomedBERT
- Model: `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext`
- Implementation: [go_embedding.py](src/go_embedding.py)
- Output: 768-dimensional vectors
- Cached to disk for reuse

**Step 3: Load and Embed Training Protein Sequences**
- Encode protein sequences using ESM-2 (`facebook/esm2_t12_35M_UR50D`)
- Implementation: [protein_embedding.py](src/protein_embedding.py)
- Output: 480-dimensional vectors per protein
- Cached to disk for reuse (`.pt` files in `output/embeddings/`)

**Step 4: Create Training and Validation Datasets**
- Stratified split by label count (default: 80/20 split)
- Implementation: [evaluation.py](src/evaluation.py) → `split_train_validation()`
- Ensures balanced distribution of protein annotation complexity
- Creates PyTorch DataLoaders for training and validation

---

#### **Phase 2: Training and Evaluation (Step 5)**

**Step 5: Model Training and Validation**
- **Architecture**: JointModel (Dual-Encoder)
  - Protein encoder: Linear(480 → 256)
  - GO encoder: Linear(768 → 256)
  - Scoring: Dot product in joint space
- **Loss**: BCEWithLogitsLoss (multi-label classification)
- **Optimizer**: Adam (lr=1e-3, default 10 epochs)
- **Validation Evaluation**:
  - Evaluates on held-out validation set
  - **Compares before/after hierarchical postprocessing**:
    - Baseline metrics (without postprocessing)
    - Post-processed metrics (with hierarchical corrections)
    - Shows improvement delta (Δ)
  - Metrics: Precision, Recall, F1, Average Precision, **IA-weighted Fmax** (official CAFA-6 metric)
  - Saves comparison results to JSON file
- Implementation: [model.py](src/model.py), [training.py](src/training.py), [evaluation.py](src/evaluation.py)

---

#### **Phase 3: Inference and Submission (Steps 6-7)**

**Step 6: Test Inference**
- Load and encode test protein sequences using ESM-2
- Generate predictions for all test proteins
- Compute scores for all GO terms
- Select top-K predictions per protein (default: K=100)
- Implementation: [protein_embedding.py](src/protein_embedding.py), [prediction.py](src/prediction.py)

**Step 7: Postprocessing and Submission**
- **Hierarchical Postprocessing** (always enabled):
  - Enforces GO hierarchy constraints using hybrid approach:
    1. **Bottom-up propagation**: If child has high score, increase parent score (α=0.3)
    2. **Top-down suppression**: If parent has low score, decrease child score (threshold=0.3, β=0.5)
  - Parameters are fixed in [hierarchical_postprocess.py](src/hierarchical_postprocess.py)
  - Implementation: [hierarchical_postprocess.py](src/hierarchical_postprocess.py)
- **Submission File Creation**:
  - Generate `submission.tsv` in Kaggle submission format
  - Format: `protein_id\tGO:term\tscore` (one per line)

### Development Mode (DEV_TEST)

For faster iteration during development, set `DEV_TEST = True` in [config.py](src/config.py):

```python
# config.py
DEV_TEST = True  # Enables development mode
DEV_TEST_MAX_BATCHES = 100  # Process only first 100 batches
```

**Effects:**
- Limits ESM-2 encoding to first 100 batches (faster testing)
- Uses separate output files with `_dev` suffix to avoid overwriting production outputs
- Ideal for rapid prototyping and debugging

**Usage:**
```bash
# Development mode - quick testing
python src/main.py  # with DEV_TEST=True in config.py

# Production mode - full dataset
python src/main.py  # with DEV_TEST=False in config.py
```

### Configuration

All parameters are centralized in [src/config.py](src/config.py):

```python
# Model parameters
JOINT_DIM = 256
TRAIN_BATCH_SIZE = 16
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3

# Evaluation
ENABLE_VALIDATION = True
VAL_SPLIT_RATIO = 0.2
VAL_STRATIFY_BY_LABEL_COUNT = True
```

**Hierarchical Postprocessing Parameters:**

Fixed parameters are defined in [src/hierarchical_postprocess.py](src/hierarchical_postprocess.py) and always enabled:
- `ALPHA = 0.3` (Bottom-up propagation coefficient)
- `THRESHOLD = 0.3` (Top-down suppression threshold)
- `BETA = 0.5` (Top-down suppression relaxation coefficient)

### Documentation

Detailed documentation for each component is available in the `docs/` directory:

- [Data Loading](docs/data_loading.md): FASTA parsing, GO OBO parsing, label loading
- [Embeddings](docs/embeddings.md): ESM-2 protein encoding, BiomedBERT GO encoding
- [Model Architecture](docs/model_architecture.md): JointModel structure, training, inference
- [Hierarchical Postprocessing](docs/hierarchical_postprocessing.md): Bottom-up/Top-down algorithms, parameters
- [Evaluation](docs/evaluation.md): IA-weighted Fmax, validation metrics (CAFA-6 official)

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

---

## 📋 To Do

### ✅ 完了済みタスク

以下のコア機能は実装完了しています。詳細は各ドキュメントを参照してください。

1. **モジュール化** - 9つの独立モジュールに機能分割完了
2. **7ステップパイプライン** - 3フェーズ構造（データ準備→訓練/評価→推論/提出）
3. **階層的後処理** - Bottom-up/Top-down手法実装、常時有効化（固定パラメータ）
4. **評価指標** - IA-weighted Fmax（CAFA-6公式）、train/val split、before/after後処理比較
5. **ドキュメント** - 全モジュールの詳細ドキュメント整備

---

### 未完了タスク

以下のタスクは今後の改善候補です。

#### 1. JointModelの改善

JointModelは二つのEmbeddingを単純な線型結合層一つで変換するかなり単純なモデルである。
もう少し複雑なモデルにすることで精度改善が期待できる。
具体的にはTransformerのAttention機構を一層でいいので追加できると良いと考えている。
何故なら、本タスクの目的は本質的にアミノ酸配列とそれに対応する辞書を作成することに近似でき、それはAttention機構のkey, valueに対応すると考えられるからである。
しかし、TransformerのAttention機構は計算が重たいので、要検討である。

#### 2. ESM-2推論の高速化

特にESM-2による推論部分に大きな時間がかかっている。この部分の推論時間短縮に成功すれば、その分のリソースを他に回せる。

---

### 追加データの活用

現在はinputの中でも使っていない情報が多数ある。
これらを使用することで最終的なoutputの精度向上が期待できる。
使用方法は必ずしも機械学習や深層学習的手法に限らず、前処理や後述する後処理への使用も検討される。
ただし、test時に使えない情報の扱いには注意が必要である。

#### 1. `IA.tsv` (Information Accretion weights) ✅ 一部実装済み
- **内容**: 各GO termに対する重要度スコア(Information Accretion)
- **実装済み**:
  - ✅ 評価指標として使用 ([evaluation.py](src/evaluation.py)で`compute_ia_weighted_fmax()`)
  - ✅ IA-weighted Precision/Recall/Fmaxの計算（CAFA-6公式メトリクス）
- **未実装** (今後の改善候補):
  - 予測時の閾値調整: IA値が高いGO termは予測を慎重に行う
  - 損失関数の重み付け: IA値に応じて損失に重みを付ける
  - スコアキャリブレーション: 重要なGO termの予測確率を調整
- **注意**: trainとtestの両方で使用可能

#### 2. `testsuperset-taxon-list.tsv` (テストデータの生物種情報)
- **内容**: テストセットの各タンパク質の生物種(taxonomy ID)
- **用途**:
  - 生物種別のモデル選択やアンサンブル
  - 生物種固有のGO term傾向を活用した後処理
  - 生物種情報を条件付け変数として使用(conditional prediction)
- **注意**: test時に使用可能なので積極的に活用すべき

#### 3. `train_taxonomy.tsv` (訓練データの生物種情報)
- **内容**: 訓練セットの各タンパク質の生物種(taxonomy ID)
- **用途**:
  - 生物種情報をタンパク質embeddingに追加(concatenateまたはcross-attention)
  - 生物種ごとのGO term分布の学習
  - データ拡張: 生物種情報をマスクしたり摂動させる
  - 層別サンプリング: 生物種のバランスを考慮した学習
- **注意**: trainのみで使用可能。testでは`testsuperset-taxon-list.tsv`を使用

#### 4. `train_terms.tsv`のaspect列
- **内容**: 各GO termの種類(biological_process / molecular_function / cellular_component)
- **用途**:
  - aspect別のモデル構築: 3つの異なるモデルを訓練
  - aspect別の損失計算: 各aspectで異なる重みや閾値を使用
  - 予測の制約: 生物学的に矛盾するaspectの組み合わせを排除
  - アンサンブル: aspect別予測を統合
- **注意**: `go-basic.obo`のnamespace情報と対応している

#### 5. FASTA形式のアミノ酸配列以外の情報
- **内容**: FASTAヘッダーに含まれるメタデータ(配列ID、アノテーション等)
- **用途**:
  - 配列IDを使用してtaxonomy情報やGO term情報と結合
  - ヘッダー内の追加情報(もしあれば)の活用
- **注意**: 現在は配列のみを使用しているが、ヘッダー情報の解析も検討

#### 6. `go-basic.obo`のidとname以外の情報 ✅ 一部実装済み
- **内容**: GO ontologyの詳細構造
- **実装済み**:
  - ✅ **is_a関係**: 階層的な後処理 ([hierarchical_postprocess.py](src/hierarchical_postprocess.py))
    - Bottom-up伝播とTop-down抑制による整合性保証
  - ✅ **def(定義文)**: GO termのnameとdefを結合してテキストembeddingを生成 ([go_embedding.py](src/go_embedding.py))
  - ✅ **namespace**: parse時に取得済み（aspect別モデリングへの活用は未実装）
- **未実装** (今後の改善候補):
  - **is_a関係**: Graph Neural Network (GNN)による埋め込み学習
  - **def(定義文)**: より高度な前処理方法の最適化
  - **namespace**: aspect別モデルの構築
  - **relationship**: その他の関係性(part_of, regulates等)を用いた制約
  - **is_obsolete**: 廃止されたGO termの除外処理

**実装の優先順位(推奨):**

1. **高優先度**: `namespace`/`aspect`を用いたaspect別モデリング
2. **中優先度**: 生物種情報の活用 (`train_taxonomy.tsv`, `testsuperset-taxon-list.tsv`)
3. **中優先度**: IA重みを用いた損失関数の改善
4. **低優先度**: その他のGO relationship、FASTAヘッダー解析

---

