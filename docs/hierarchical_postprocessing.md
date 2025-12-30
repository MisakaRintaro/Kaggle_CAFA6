# Hierarchical Postprocessing

このドキュメントでは、GO階層の生物学的制約を考慮した階層的後処理について説明します。

## 背景と動機

### GO Ontologyの階層構造

Gene Ontology (GO) は **有向非巡回グラフ (DAG)** 構造を持ちます。

```
                  [GO Root]
                      |
          +-----------+-----------+
          |           |           |
    [Parent A]   [Parent B]  [Parent C]
       |  |          |
       |  +---+------+
       |      |
    [Child X] [Child Y]
```

### 生物学的制約

1. **子がtrueなら親もtrue**
   - タンパク質がある特定の機能(子GO)を持つ場合、その上位概念(親GO)も必ず持つ

2. **親がfalseなら子もfalse**
   - タンパク質が上位概念(親GO)を持たない場合、より特定的な機能(子GO)も持たない

### 問題

機械学習モデルの予測は、この階層的制約を必ずしも満たしません:
- 子GOのスコアが高いのに、親GOのスコアが低い
- 親GOのスコアが低いのに、子GOのスコアが高い

→ **階層的後処理で矛盾を解消**

---

## 実装アプローチ

### ハイブリッド手法

2つの補正を順番に適用します:

1. **Bottom-up伝播** (子→親方向)
2. **Top-down抑制** (親→子方向)

実装: `src/hierarchical_postprocess.py`

---

## 1. Bottom-up伝播

### 目的

子GOのスコアが高い場合、親GOのスコアを引き上げる

### アルゴリズム

```python
# トポロジカルソート順の逆順（葉→根）で処理
for go_id in reversed(topological_order):
    if go_id has children:
        max_child_score = max(score[child] for child in children)
        score[go_id] = max(score[go_id], max_child_score * alpha)
```

### パラメータ

- **`alpha`** (伝播係数): 推奨値 0.3
  - 子のスコアをどの程度親に伝播させるか
  - 0.0 = 伝播なし, 1.0 = 完全伝播

### 効果

- 子がtrueなら親もtrueという制約を保証
- より具体的な予測から一般的な予測への伝播

### 実装

```python
from hierarchical_postprocess import bottom_up_propagation

scores_after_bottom_up = bottom_up_propagation(
    scores,
    parent_to_children,
    topological_order,
    alpha=0.3
)
```

---

## 2. Top-down抑制

### 目的

親GOのスコアが低い場合、子GOのスコアを抑制する

### アルゴリズム

```python
# トポロジカルソート順（根→葉）で処理
for go_id in topological_order:
    if score[go_id] < threshold:
        for child in children:
            suppressed_score = score[go_id] * (1 + beta)
            score[child] = min(score[child], suppressed_score)
```

### パラメータ

- **`threshold`** (抑制開始閾値): 推奨値 0.3
  - この値以下の親GOに対して抑制を適用
- **`beta`** (抑制緩和係数): 推奨値 0.5
  - 抑制の強さを調整
  - 0.0 = 強い抑制, 1.0 = 弱い抑制

### 効果

- 親がfalseなのに子がtrueという矛盾を解消
- より一般的な予測から具体的な予測への影響

### 実装

```python
from hierarchical_postprocess import top_down_suppression

scores_after_top_down = top_down_suppression(
    scores,
    parent_to_children,
    topological_order,
    threshold=0.3,
    beta=0.5
)
```

---

## ハイブリッド後処理

### 統合関数

```python
from hierarchical_postprocess import hierarchical_postprocess

corrected_predictions = hierarchical_postprocess(
    predictions,
    child_to_parents,
    alpha=0.3,
    threshold=0.3,
    beta=0.5
)
```

### 処理フロー

```
入力: predictions = {GO_id: probability}

1. グラフ構造構築
   - child_to_parents から parent_to_children を構築
   - トポロジカルソートを計算

2. Bottom-up伝播（葉→根）
   - 各GOについて子の最大スコアでスコアを引き上げ

3. Top-down抑制（根→葉）
   - 親スコアが閾値以下なら子スコアを抑制

出力: 補正後のpredictions
```

---

## 全タンパク質への適用

```python
from hierarchical_postprocess import apply_hierarchical_postprocess_to_all_proteins

corrected_predictions = apply_hierarchical_postprocess_to_all_proteins(
    predictions,
    child_to_parents,
    alpha=0.3,
    threshold=0.3,
    beta=0.5
)
```

入力:
```python
Dict[Tuple[str, str], List[Tuple[str, float]]]
# {(protein_id, taxon_id): [(go_id, score), ...]}
```

出力:
```python
Dict[Tuple[str, str], List[Tuple[str, float]]]
# 階層的後処理を適用した予測結果
```

---

## 階層的整合性の評価

### 整合性スコア

親子間の矛盾率を測定します:

```python
from hierarchical_postprocess import compute_hierarchical_consistency_score

inconsistency_rate = compute_hierarchical_consistency_score(
    predictions,
    child_to_parents,
    threshold=0.5
)
```

**矛盾の定義:**
- 子が陽性(≥threshold)なのに親が陰性(<threshold)

**戻り値:**
- 0.0〜1.0 の範囲
- 低いほど整合性が高い

---

## 設定

### config.pyでの設定

```python
# 階層的後処理を有効化
ENABLE_HIERARCHICAL_POSTPROCESS = True

# パラメータ
HIERARCHICAL_ALPHA = 0.3
HIERARCHICAL_THRESHOLD = 0.3
HIERARCHICAL_BETA = 0.5
```

### パラメータチューニング

validation setで最適なパラメータを探索:

| パラメータ | 範囲 | 効果 |
|----------|------|------|
| `alpha` | 0.1〜0.5 | 小: 弱い伝播, 大: 強い伝播 |
| `threshold` | 0.2〜0.5 | 抑制を開始するスコア |
| `beta` | 0.3〜0.7 | 小: 強い抑制, 大: 弱い抑制 |

---

## パフォーマンスへの影響

### 期待される効果

1. **階層的整合性の向上**
   - 矛盾率が大幅に減少

2. **Precision/Recallのバランス**
   - Bottom-up: Recallが向上
   - Top-down: Precisionが向上

3. **IA-weighted Fmax**
   - より深いGO termの予測精度が向上
   - 最終スコアの改善

### 計算コスト

- トポロジカルソート: O(V + E)
- Bottom-up伝播: O(V)
- Top-down抑制: O(V)
- **全体: 線形時間で高速**

---

## データ構造

### child_to_parents

```python
Dict[str, List[str]]
# {GO_ID: [parent_GO_IDs]}

# 例:
{
    "GO:0005737": ["GO:0005622", "GO:0044424"],
    "GO:0005622": ["GO:0005623"],
    ...
}
```

`parse_go_obo()`から取得

### parent_to_children

```python
Dict[str, List[str]]
# {GO_ID: [child_GO_IDs]}
```

`build_parent_to_children()`で構築

### topological_order

```python
List[str]
# トポロジカルソート済みのGO ID順序（根→葉）
```

Kahn's algorithmで計算

---

## 実装例（main.pyでの使用）

```python
# Step 7: 予測
predictions = predict(model, test_loader, go_id_list, device)

# Step 7.5: 階層的後処理
if ENABLE_HIERARCHICAL_POSTPROCESS:
    predictions = apply_hierarchical_postprocess_to_all_proteins(
        predictions,
        child_to_parents,
        alpha=HIERARCHICAL_ALPHA,
        threshold=HIERARCHICAL_THRESHOLD,
        beta=HIERARCHICAL_BETA
    )

# Step 8: 提出ファイル作成
create_submission_file(predictions, PATH_SUBMISSION)
```

---

## 関連ファイル

- `src/hierarchical_postprocess.py`: 実装コード
- `src/config.py`: パラメータ設定
- `src/main.py`: パイプラインでの使用例
- `docs/evaluation.md`: 評価指標（整合性スコア含む）
