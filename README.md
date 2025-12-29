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

### 後処理の追加

GOは階層的な構造を持つ。
よって、モデルの予測でとあるGO, GO_aである確率が1に近い時、その上位概念(is_a)の確率も1に近いのが自然である。
逆もまた然りで、GO_aの確率が0に近い時、その下位概念も0に近いのが自然である。
この補正の実装には、以下の2つの設計のどちらかを選択する必要がある。
1. とある下位GOの確率が高い時,その上位GOの確率を引き上げる
2. とある上位GOの確率が低い時,その下位GOの確率を引き下げる

---

### JointModelの改善

JointModelは二つのEmbeddingを単純な線型結合層一つで変換するかなり単純なモデルである。
もう少し複雑なモデルにすることで精度改善が期待できる。
具体的にはTransformerのAttention機構を一層でいいので追加できると良いと考えている。
何故なら、本タスクの目的は本質的にアミノ酸配列とそれに対応する辞書を作成することに近似でき、それはAttention機構のkey, valueに対応すると考えられるからである。

---

### 推論時間短縮

特にesm2による推論部分に大きな時間がかかっている。
この部分の推論時間短縮に成功すれば、その分のリソースを他に回せる

--- 

### 使っていないinputの使用

現在はinputの中でも使っていない情報が多数ある。
これらを使用することで最終的なoutputの精度向上が期待できる。
使用方法は必ずしも機械学習や深層学習的手法に限らず、前処理や後述する後処理への使用も検討される。
ただし、test時に使えない情報の扱いには注意が必要である。

#### 1. `IA.tsv` (Information Accretion weights)
- **内容**: 各GO termに対する重要度スコア(Information Accretion)
- **用途**:
  - 評価指標として使用される(コンペのメトリクスに関連)
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

#### 6. `go-basic.obo`のidとname以外の情報
- **内容**: GO ontologyの詳細構造
  - `is_a`: 親子関係(階層構造)
  - `def`: GO termの定義文
  - `namespace`: biological_process / molecular_function / cellular_component
  - `relationship`: その他の関係性(part_of, regulates等)
  - `alt_id`: 代替ID
  - `is_obsolete`: 廃止されたGO term
- **用途**:
  - **is_a関係**:
    - 階層的な後処理(前述の「後処理の追加」セクション参照)
    - Graph Neural Network (GNN)による埋め込み学習
    - 予測の伝播: 子ノードの予測から親ノードの予測を補正
  - **def(定義文)**:
    - ✅ 現在使用中: GO termのnameとdefを結合してテキストembeddingを生成
    - 改善の余地: defの前処理方法の最適化(引用符除去以外の手法も検討)
  - **namespace**:
    - aspect別モデルの構築に使用
  - **relationship**:
    - より複雑なグラフ構造の学習
    - 異なる関係性に基づく制約の追加
  - **is_obsolete**:
    - 廃止されたGO termを予測候補から除外

#### 実装の優先順位(推奨)
1. **高優先度**(未実装):
   - `is_a`関係: 階層的後処理
   - `namespace`/`aspect`: aspect別モデリング
2. **中優先度**(未実装):
   - `IA.tsv`: 損失関数の重み付けやスコアキャリブレーション
   - `testsuperset-taxon-list.tsv`: 生物種条件付き予測
   - `train_taxonomy.tsv`: 生物種情報のembedding統合
3. **低優先度**(未実装):
   - その他の`relationship`: より複雑なグラフ構造
   - FASTAヘッダーの詳細解析

---

