# Model Architecture

このドキュメントでは、CAFA-6で使用するJointModelの構造と学習方法について説明します。

## 概要

JointModelは「Dual-Encoder」アーキテクチャを採用しています。タンパク質埋め込みとGO埋め込みを共通の「joint空間」にマッピングし、内積でスコアを計算します。

実装は以下のモジュールに分かれています:
- `src/model.py`: モデル定義とデータセット
- `src/training.py`: 学習ループと保存/読み込み
- `src/prediction.py`: 推論とsubmission file生成

---

## アーキテクチャ

### Dual-Encoder構造

```
タンパク質埋め込み (480次元)
    ↓
protein_fc: Linear(480 → 256)
    ↓
z_p ∈ R^256 (joint space)


GO埋め込み (768次元)
    ↓
go_fc: Linear(768 → 256)
    ↓
e_t ∈ R^256 (joint space)


スコア計算:
score(protein, GO) = z_p · e_t  (内積)
```

### JointModelクラス

```python
class JointModel(nn.Module):
    def __init__(
        self,
        input_dim_protein,   # 480
        input_dim_go,        # 768
        joint_dim,           # 256
        go_raw_emb           # (num_go, 768)
    ):
        self.protein_fc = nn.Linear(input_dim_protein, joint_dim)
        self.go_fc = nn.Linear(input_dim_go, joint_dim)
        self.register_buffer("go_raw_emb", go_raw_emb)
```

#### 主なメソッド

**`forward(h_batch)`**
- タンパク質埋め込みから全GO termに対するスコアを計算
- 戻り値: (batch_size, num_go) のlogits

**`predict_proba(h_batch)`**
- forwardの出力にシグモイドを適用
- 戻り値: (batch_size, num_go) の確率 (0〜1)

---

## データセット

### ProteinGODataset

訓練用のデータセット。タンパク質埋め込みとマルチラベルを管理します。

```python
class ProteinGODataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        # X: (num_samples, 480)
        # y: (num_samples, num_go)
```

### create_protein_go_label_matrix()

タンパク質埋め込み辞書とラベル情報から訓練データを構築します。

```python
X, y, protein_ids = create_protein_go_label_matrix(
    protein_emb_dict,
    train_label_df,
    go_id_list
)
# X: (num_proteins, 480)
# y: (num_proteins, num_go) - マルチラベル行列
```

---

## 学習プロセス

### 損失関数

**Binary Cross Entropy with Logits Loss (BCEWithLogitsLoss)**

マルチラベル分類に適した損失関数。各GO termについて独立にBCEを計算します。

```python
criterion = nn.BCEWithLogitsLoss()
loss = criterion(logits, y_batch)
```

### オプティマイザ

**Adam**

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
```

### 学習ループ

```python
from training import train_model

history = train_model(
    model,
    train_loader,
    device,
    num_epochs=10,
    learning_rate=1e-3
)
```

**出力:**
```python
history = {
    'epoch': [1, 2, ..., 10],
    'loss': [各バッチの損失値],
    'avg_loss': [各エポックの平均損失]
}
```

---

## モデルの保存と読み込み

### 保存

```python
from training import save_model

save_model(
    model,
    filepath="joint_model.pt",
    go_id_list=go_id_list,
    history=history
)
```

**保存内容:**
- `model_state_dict`: モデルの重み
- `go_id_list`: GO IDのリスト
- `history`: 訓練履歴
- `model_config`: モデルの設定

### 読み込み

```python
from training import load_model

model, go_id_list, history = load_model(
    filepath="joint_model.pt",
    go_raw_emb_matrix=go_emb_matrix,
    device=device
)
```

---

## 推論

### TestProteinDataset

テスト用のデータセット（ラベルなし）。

```python
class TestProteinDataset(Dataset):
    def __init__(self, protein_emb_dict):
        # ラベルは不要、protein_idと埋め込みのみ
```

### predict()

テストデータに対して推論を実行します。

```python
from prediction import predict

predictions = predict(
    model,
    test_loader,
    go_id_list,
    device,
    top_k=100
)
```

**戻り値:**
```python
Dict[Tuple[str, str], List[Tuple[str, float]]]
# {(protein_id, taxon_id): [(go_id, score), ...]}
# 各タンパク質について上位K個のGO termとスコア
```

---

## 提出ファイルの作成

```python
from prediction import create_submission_file

create_submission_file(
    predictions,
    output_path="submission.tsv",
    add_text_descriptions=False
)
```

**フォーマット:**
```
protein_id    GO:0005737    0.856432
protein_id    GO:0005634    0.742108
...
```

---

## パフォーマンスチューニング

### 1. ハイパーパラメータ

config.pyで調整可能:
- `JOINT_DIM`: joint空間の次元 (推奨: 256)
- `TRAIN_BATCH_SIZE`: バッチサイズ (推奨: 16)
- `NUM_EPOCHS`: エポック数 (推奨: 10)
- `LEARNING_RATE`: 学習率 (推奨: 1e-3)

### 2. モデルの改善案

#### 現在の制限
- 単純な線形変換のみ
- タンパク質とGO term間の複雑な関係を捉えきれない可能性

#### 改善の方向性
- **Attention機構の追加**: Transformerの1層を追加
- **深いネットワーク**: Multi-layer Perceptron (MLP)
- **正則化**: Dropout, Batch Normalization

---

## データフロー（全体）

```
1. データ準備
   protein_emb_dict + train_label_df + go_id_list
   → create_protein_go_label_matrix()
   → X, y

2. Dataset作成
   X, y → ProteinGODataset → DataLoader

3. モデル初期化
   JointModel(input_dim_protein=480, input_dim_go=768, joint_dim=256)

4. 学習
   train_model() → history

5. 保存
   save_model() → joint_model.pt

6. 推論
   test_loader → predict() → predictions

7. 提出ファイル
   predictions → create_submission_file() → submission.tsv
```

---

## 関連ファイル

- `src/model.py`: モデルとデータセットの定義
- `src/training.py`: 学習と保存/読み込み
- `src/prediction.py`: 推論と提出ファイル生成
- `src/config.py`: ハイパーパラメータの設定
- `docs/hierarchical_postprocessing.md`: 次のステップ（後処理）
