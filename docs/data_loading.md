# Data Loading

このドキュメントでは、CAFA-6コンペティションで使用するデータの読み込み方法について説明します。

## 概要

データ読み込みは `src/data_loader.py` モジュールで実装されています。

### 主な機能

1. **FASTAファイルの読み込み** (`load_fasta`)
2. **GO Ontologyの解析** (`parse_go_obo`)
3. **訓練ラベルの読み込み** (`load_train_labels`)

---

## 1. FASTAファイルの読み込み

### 関数: `load_fasta(path, mode, max_aa_len=1024)`

タンパク質配列を FASTA 形式から読み込みます。

#### パラメータ

- `path`: FASTAファイルのパス
- `mode`: `"train"` または `"test"`
  - `"train"`: EntryID のみ使用 (taxon_id は train_taxonomy.tsv から取得可能)
  - `"test"`: ヘッダーから protein_id と taxon_id を抽出
- `max_aa_len`: 最大アミノ酸配列長（デフォルト: 1024）

#### 戻り値

```python
Dict[Tuple[str, str], str]
# {(protein_id, taxon_id): sequence}
```

#### 使用例

```python
from data_loader import load_fasta

# 訓練データ
train_seq_dict = load_fasta(
    "/path/to/train_sequences.fasta",
    mode="train"
)

# テストデータ
test_seq_dict = load_fasta(
    "/path/to/testsuperset.fasta",
    mode="test"
)
```

---

## 2. GO Ontologyの解析

### 関数: `parse_go_obo(filepath)`

Gene Ontology の OBO 形式ファイルを解析し、階層構造を取得します。

#### パラメータ

- `filepath`: go-basic.obo のパス

#### 戻り値

```python
Tuple[Dict[str, List[str]], Dict[str, Dict[str, Any]]]

# (child_to_parents, go_terms)
#
# child_to_parents: {GO_ID: [parent_GO_IDs]}
# go_terms: {GO_ID: {'name': ..., 'def': ..., 'namespace': ...}}
```

#### 使用例

```python
from data_loader import parse_go_obo

child_to_parents, go_terms = parse_go_obo("/path/to/go-basic.obo")

# 特定のGO termの親を取得
parents = child_to_parents.get("GO:0005737", [])

# GO termの情報を取得
term_info = go_terms["GO:0005737"]
print(term_info['name'])  # "cytoplasm"
print(term_info['namespace'])  # "cellular_component"
```

---

## 3. 訓練ラベルの読み込み

### 関数: `load_train_labels(filepath)`

train_terms.tsv から訓練ラベルを読み込みます。

#### パラメータ

- `filepath`: train_terms.tsv のパス

#### 戻り値

```python
pd.DataFrame
# columns: ['EntryID', 'term', 'aspect']
```

#### 使用例

```python
from data_loader import load_train_labels

train_label_df = load_train_labels("/path/to/train_terms.tsv")

# 特定のタンパク質のGO termsを取得
protein_labels = train_label_df[train_label_df['EntryID'] == 'A0A009IHW8']
```

---

## データフロー

```
1. FASTAファイル → load_fasta() → {(protein_id, taxon_id): sequence}
2. GO OBO → parse_go_obo() → (child_to_parents, go_terms)
3. train_terms.tsv → load_train_labels() → DataFrame
```

これらのデータは後続のステップで使用されます:
- タンパク質配列 → ESM-2で埋め込みベクトル化
- GO term情報 → BiomedBERTでテキスト埋め込み化
- 訓練ラベル → モデルの教師信号

---

## 注意事項

### 1. 配列長の制限

- ESM-2モデルの制約により、配列は最大1024アミノ酸に制限されます
- より長い配列は自動的に切り詰められます

### 2. Taxon ID の取得

- **訓練データ**: `train_taxonomy.tsv` から別途取得可能
- **テストデータ**: FASTAヘッダーから直接抽出

### 3. メモリ効率

- 全配列をメモリに読み込むため、大規模データセットでは注意が必要
- DEV_TESTモードで部分的なデータで動作確認することを推奨

---

## 関連ファイル

- `src/data_loader.py`: 実装コード
- `src/config.py`: データパスの設定
- `docs/embeddings.md`: 次のステップ（埋め込み生成）
