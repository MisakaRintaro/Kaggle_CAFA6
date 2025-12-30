# Embeddings

このドキュメントでは、タンパク質配列とGO termのテキスト情報を埋め込みベクトルに変換する方法について説明します。

## 概要

埋め込み生成は以下の2つのモジュールで実装されています:

1. **タンパク質埋め込み**: `src/protein_embedding.py` (ESM-2使用)
2. **GO埋め込み**: `src/go_embedding.py` (BiomedBERT使用)

---

## 1. タンパク質埋め込み (ESM-2)

### モデル

- **ESM-2** (Evolutionary Scale Modeling 2)
- Meta AI が開発したタンパク質言語モデル
- アミノ酸配列をベクトルに変換

### 使用モデル

```python
facebook/esm2_t12_35M_UR50D
```

- 12層、3500万パラメータ
- 出力次元: 480次元

### 主な関数

#### `encode_all_sequences_with_esm(seq_dict, tokenizer, model, device, batch_size=64, max_batches=None)`

全タンパク質配列を埋め込みベクトルに変換します。

**パラメータ:**
- `seq_dict`: {(protein_id, taxon_id): sequence}
- `tokenizer`: ESM-2トークナイザ
- `model`: ESM-2モデル
- `device`: 使用デバイス (cuda/mps/cpu)
- `batch_size`: バッチサイズ (デフォルト: 64)
- `max_batches`: 処理する最大バッチ数 (DEV_TESTモード用)

**戻り値:**
```python
Dict[Tuple[str, str], np.ndarray]
# {(protein_id, taxon_id): embedding (480,)}
```

**処理フロー:**
1. 配列をバッチに分割
2. 各バッチをトークン化
3. ESM-2でエンコード
4. [CLS]と[EOS]を除いてmean pooling
5. 結果を辞書に格納

### 使用例

```python
from transformers import AutoTokenizer, AutoModel
from protein_embedding import encode_all_sequences_with_esm
import torch

# モデルとトークナイザの読み込み
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")
model = AutoModel.from_pretrained("facebook/esm2_t12_35M_UR50D")
model = model.to(device)

# 埋め込み生成
protein_emb_dict = encode_all_sequences_with_esm(
    seq_dict,
    tokenizer,
    model,
    device,
    batch_size=64
)
```

### 保存と読み込み

```python
from protein_embedding import save_emb_dict, load_emb_dict

# 保存
save_emb_dict(protein_emb_dict, "protein_embeddings.pt")

# 読み込み
protein_emb_dict = load_emb_dict("protein_embeddings.pt")
```

---

## 2. GO埋め込み (BiomedBERT)

### モデル

- **BiomedBERT**
- 生医学テキストで事前学習されたBERTモデル
- GO termの名前と定義をベクトルに変換

### 使用モデル

```python
microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext
```

- BERT-base アーキテクチャ
- 出力次元: 768次元

### 主な関数

#### `create_go_embedding_matrix(terms, train_label_df, tokenizer, model, device)`

訓練データに登場するGO termの埋め込み行列を作成します。

**パラメータ:**
- `terms`: parse_go_obo()から取得したGO term辞書
- `train_label_df`: 訓練ラベルのDataFrame
- `tokenizer`: BiomedBERTトークナイザ
- `model`: BiomedBERTモデル
- `device`: 使用デバイス

**戻り値:**
```python
Tuple[torch.Tensor, List[str], Dict[str, np.ndarray]]
# (go_emb_matrix, go_id_list, go_embeddings_dict)
#
# go_emb_matrix: (num_go, 768) の行列
# go_id_list: 各行に対応するGO ID（ソート済み）
# go_embeddings_dict: {GO_ID: embedding}
```

**処理フロー:**
1. 訓練データに登場するユニークなGO termを抽出
2. 各GO termについて:
   - nameとdefを結合してテキスト化
   - BiomedBERTでエンコード
   - CLSトークンの出力を取得
3. ソート済みのGO IDリストで行列を構築

### 使用例

```python
from transformers import AutoTokenizer, AutoModel
from go_embedding import create_go_embedding_matrix

# モデルとトークナイザの読み込み
tokenizer = AutoTokenizer.from_pretrained(
    "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
)
model = AutoModel.from_pretrained(
    "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
)
model = model.to(device)

# GO埋め込み生成
go_emb_matrix, go_id_list, go_embeddings_dict = create_go_embedding_matrix(
    go_terms,
    train_label_df,
    tokenizer,
    model,
    device
)
```

### 保存と読み込み

```python
from go_embedding import (
    save_go_embeddings,
    load_go_embeddings,
    reconstruct_go_matrix_from_dict
)

# 保存（辞書形式）
save_go_embeddings(go_embeddings_dict, "go_embeddings.pt")

# 読み込み
go_embeddings_dict = load_go_embeddings("go_embeddings.pt")

# 行列とリストを再構築
go_emb_matrix, go_id_list = reconstruct_go_matrix_from_dict(go_embeddings_dict)
```

---

## データフロー

```
タンパク質配列
    ↓
ESM-2 Tokenizer → ESM-2 Model → Mean Pooling
    ↓
タンパク質埋め込み (480次元)


GO term (name + def)
    ↓
BiomedBERT Tokenizer → BiomedBERT Model → CLS token
    ↓
GO埋め込み (768次元)
```

---

## メモリとパフォーマンスの考慮事項

### 1. バッチ処理

- GPUメモリに応じてバッチサイズを調整
- 推奨値: 64 (ESM-2), 1 (BiomedBERT, GO termは数千個程度)

### 2. DEV_TESTモード

- `max_batches`パラメータで処理バッチ数を制限
- 開発時の動作確認に便利

### 3. キャッシング

- 埋め込みは一度計算したら保存し、再利用する
- main.pyでは自動的に既存ファイルをチェック

### 4. デバイス選択

- CUDA > MPS > CPUの優先順位で自動選択
- `config.get_device()`で取得

---

## 埋め込みの用途

これらの埋め込みは次のステップで使用されます:

1. **モデル入力**: JointModelの入力として使用
2. **類似度計算**: タンパク質とGO termの関連性をスコア化
3. **予測**: テストタンパク質に対するGO term予測

---

## 関連ファイル

- `src/protein_embedding.py`: タンパク質埋め込みの実装
- `src/go_embedding.py`: GO埋め込みの実装
- `src/config.py`: モデル名とパラメータの設定
- `docs/model_architecture.md`: 次のステップ（モデル学習）
