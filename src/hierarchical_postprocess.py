#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Hierarchical postprocessing for CAFA-6 competition.

This module provides functions for:
- Building GO term hierarchical graph structure
- Computing topological order
- Bottom-up propagation (child → parent)
- Top-down suppression (parent → child)
- Hybrid hierarchical postprocessing

Fixed Parameters:
- ALPHA = 0.3 (Bottom-up propagation coefficient)
- THRESHOLD = 0.3 (Top-down suppression threshold)
- BETA = 0.5 (Top-down suppression relaxation coefficient)
"""

from typing import Dict, List, Tuple
from collections import deque, defaultdict
import numpy as np
import gc


# ============================================================================
# Fixed Parameters for Hierarchical Postprocessing
# ============================================================================

# Bottom-up propagation coefficient
# Controls how much child scores propagate to parent scores
ALPHA = 0.3

# Top-down suppression threshold
# Parents below this threshold will suppress their children
THRESHOLD = 0.3

# Top-down suppression relaxation coefficient
# Controls the strength of suppression
BETA = 0.5


def build_parent_to_children(
    child_to_parents: Dict[str, List[str]]
) -> Dict[str, List[str]]:
    """
    child_to_parentsの逆引き辞書を作成する

    Parameters
    ----------
    child_to_parents : Dict[str, List[str]]
        各GO IDに対する親GO IDのリスト

    Returns
    -------
    parent_to_children : Dict[str, List[str]]
        各GO IDに対する子GO IDのリスト
    """
    parent_to_children: Dict[str, List[str]] = defaultdict(list)

    for child_id, parent_ids in child_to_parents.items():
        for parent_id in parent_ids:
            parent_to_children[parent_id].append(child_id)

    # defaultdictを通常のdictに変換
    return dict(parent_to_children)


def compute_topological_order(
    child_to_parents: Dict[str, List[str]],
    all_go_ids: List[str]
) -> List[str]:
    """
    トポロジカルソート (Kahn's algorithm) でGO IDの順序を計算する

    Parameters
    ----------
    child_to_parents : Dict[str, List[str]]
        各GO IDに対する親GO IDのリスト
    all_go_ids : List[str]
        全てのGO IDのリスト（ソート対象）

    Returns
    -------
    topological_order : List[str]
        トポロジカルソート済みのGO ID順序（根→葉）
    """
    # 入次数（親の数）を計算
    in_degree: Dict[str, int] = {}
    for go_id in all_go_ids:
        in_degree[go_id] = len(child_to_parents.get(go_id, []))

    # 入次数が0のノード（根ノード）をキューに追加
    queue = deque([go_id for go_id in all_go_ids if in_degree[go_id] == 0])

    # parent_to_childrenを構築
    parent_to_children = build_parent_to_children(child_to_parents)

    topological_order: List[str] = []

    while queue:
        # キューから1つ取り出す
        current = queue.popleft()
        topological_order.append(current)

        # 現在のノードの子ノードの入次数を減らす
        for child in parent_to_children.get(current, []):
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)

    # 全てのノードが処理されたか確認（サイクルがないか）
    if len(topological_order) != len(all_go_ids):
        print(f"Warning: Topological sort incomplete. "
              f"Processed {len(topological_order)}/{len(all_go_ids)} nodes. "
              f"Graph may contain cycles or isolated nodes.")

    return topological_order


def bottom_up_propagation(
    scores: Dict[str, float],
    parent_to_children: Dict[str, List[str]],
    topological_order: List[str],
    alpha: float = ALPHA
) -> Dict[str, float]:
    """
    Bottom-up伝播: 子の最大スコアで親のスコアを引き上げる

    Parameters
    ----------
    scores : Dict[str, float]
        各GO IDに対する予測スコア（0.0〜1.0）
    parent_to_children : Dict[str, List[str]]
        各GO IDに対する子GO IDのリスト
    topological_order : List[str]
        トポロジカルソート済みのGO ID順序
    alpha : float
        伝播係数（推奨値: 0.3）

    Returns
    -------
    updated_scores : Dict[str, float]
        Bottom-up伝播後のスコア
    """
    # スコアのコピーを作成
    updated_scores = scores.copy()

    # 逆順（葉→根）で処理
    for go_id in reversed(topological_order):
        # 子ノードが存在する場合
        children = parent_to_children.get(go_id, [])
        if children:
            # 子ノードのうち、スコアが存在するものだけを考慮
            child_scores = [updated_scores[child] for child in children if child in updated_scores]

            if child_scores:
                max_child_score = max(child_scores)
                # 現在のスコアと、子の最大スコア * alpha の大きい方を採用
                updated_scores[go_id] = max(
                    updated_scores.get(go_id, 0.0),
                    max_child_score * alpha
                )

    return updated_scores


def top_down_suppression(
    scores: Dict[str, float],
    parent_to_children: Dict[str, List[str]],
    topological_order: List[str],
    threshold: float = THRESHOLD,
    beta: float = BETA
) -> Dict[str, float]:
    """
    Top-down抑制: 親のスコアが低い場合、子のスコアを抑制する

    Parameters
    ----------
    scores : Dict[str, float]
        各GO IDに対する予測スコア（0.0〜1.0）
    parent_to_children : Dict[str, List[str]]
        各GO IDに対する子GO IDのリスト
    topological_order : List[str]
        トポロジカルソート済みのGO ID順序
    threshold : float
        抑制開始閾値（推奨値: 0.3）
    beta : float
        抑制緩和係数（推奨値: 0.5）

    Returns
    -------
    updated_scores : Dict[str, float]
        Top-down抑制後のスコア
    """
    # スコアのコピーを作成
    updated_scores = scores.copy()

    # 順序通り（根→葉）で処理
    for go_id in topological_order:
        current_score = updated_scores.get(go_id, 0.0)

        # 親のスコアが閾値以下の場合
        if current_score < threshold:
            # 子ノードのスコアを抑制
            children = parent_to_children.get(go_id, [])
            for child in children:
                if child in updated_scores:
                    # 子のスコアを、親のスコア * (1 + beta) 以下に抑制
                    suppressed_score = current_score * (1.0 + beta)
                    updated_scores[child] = min(
                        updated_scores[child],
                        suppressed_score
                    )

    return updated_scores


def hierarchical_postprocess(
    predictions: Dict[str, float],
    child_to_parents: Dict[str, List[str]],
    alpha: float = ALPHA,
    threshold: float = THRESHOLD,
    beta: float = BETA
) -> Dict[str, float]:
    """
    階層的後処理のハイブリッドアプローチ
    Bottom-up伝播とTop-down抑制を順番に適用する

    Parameters
    ----------
    predictions : Dict[str, float]
        各GO IDに対する予測スコア（0.0〜1.0）
    child_to_parents : Dict[str, List[str]]
        各GO IDに対する親GO IDのリスト (parse_go_obo()から取得)
    alpha : float
        Bottom-up伝播係数（推奨値: 0.3）
    threshold : float
        Top-down抑制の閾値（推奨値: 0.3）
    beta : float
        Top-down抑制の緩和係数（推奨値: 0.5）

    Returns
    -------
    corrected_predictions : Dict[str, float]
        階層的後処理を適用した予測スコア
    """
    # 1. グラフ構造の構築
    all_go_ids = list(predictions.keys())
    parent_to_children = build_parent_to_children(child_to_parents)

    # 2. トポロジカルソートの計算
    topological_order = compute_topological_order(child_to_parents, all_go_ids)

    # 3. Bottom-up伝播
    scores_after_bottom_up = bottom_up_propagation(
        predictions,
        parent_to_children,
        topological_order,
        alpha=alpha
    )

    # 4. Top-down抑制
    corrected_predictions = top_down_suppression(
        scores_after_bottom_up,
        parent_to_children,
        topological_order,
        threshold=threshold,
        beta=beta
    )

    return corrected_predictions


def apply_hierarchical_postprocess_to_all_proteins(
    predictions: Dict[Tuple[str, str], List[Tuple[str, float]]],
    child_to_parents: Dict[str, List[str]],
    alpha: float = ALPHA,
    threshold: float = THRESHOLD,
    beta: float = BETA
) -> Dict[Tuple[str, str], List[Tuple[str, float]]]:
    """
    全タンパク質の予測結果に階層的後処理を適用する

    Parameters
    ----------
    predictions : Dict[Tuple[str, str], List[Tuple[str, float]]]
        {(protein_id, taxon_id): [(go_id, score), ...]} 形式の予測結果
    child_to_parents : Dict[str, List[str]]
        各GO IDに対する親GO IDのリスト
    alpha : float
        Bottom-up伝播係数
    threshold : float
        Top-down抑制の閾値
    beta : float
        Top-down抑制の緩和係数

    Returns
    -------
    corrected_predictions : Dict[Tuple[str, str], List[Tuple[str, float]]]
        階層的後処理を適用した予測結果
    """
    corrected_predictions = {}

    total_proteins = len(predictions)
    print(f"Applying hierarchical postprocessing to {total_proteins} proteins...")

    for idx, (protein_id, go_predictions) in enumerate(predictions.items()):
        # リスト形式から辞書形式に変換
        prediction_dict = {go_id: score for go_id, score in go_predictions}

        # 階層的後処理を適用
        corrected_dict = hierarchical_postprocess(
            prediction_dict,
            child_to_parents,
            alpha=alpha,
            threshold=threshold,
            beta=beta
        )

        # 辞書形式からリスト形式に変換し、スコアでソート
        corrected_list = sorted(
            corrected_dict.items(),
            key=lambda x: x[1],
            reverse=True
        )

        corrected_predictions[protein_id] = corrected_list

        # 進捗表示（1000件ごとに表示してI/O削減）
        if (idx + 1) % 1000 == 0 or (idx + 1) == total_proteins:
            print(f"Processed [{idx+1}/{total_proteins}] proteins")
            # 定期的にガベージコレクションを実行してメモリを解放
            gc.collect()

        # メモリ管理: 中間データを削除
        del prediction_dict, corrected_dict, corrected_list

    print("Hierarchical postprocessing completed!")
    return corrected_predictions


def compute_hierarchical_consistency_score(
    predictions: Dict[str, float],
    child_to_parents: Dict[str, List[str]],
    threshold: float = 0.5
) -> float:
    """
    階層的整合性スコアを計算: 親子間の矛盾率を測定

    矛盾とは:
    - 子がthreshold以上なのに、親がthreshold未満
    - または、親がthreshold未満なのに、子がthreshold以上

    Parameters
    ----------
    predictions : Dict[str, float]
        各GO IDに対する予測スコア
    child_to_parents : Dict[str, List[str]]
        各GO IDに対する親GO IDのリスト
    threshold : float
        陽性判定の閾値

    Returns
    -------
    inconsistency_rate : float
        矛盾率（0.0〜1.0、低いほど整合性が高い）
    """
    total_edges = 0
    inconsistent_edges = 0

    for child_id, parent_ids in child_to_parents.items():
        if child_id not in predictions:
            continue

        child_score = predictions[child_id]
        child_positive = child_score >= threshold

        for parent_id in parent_ids:
            if parent_id not in predictions:
                continue

            total_edges += 1
            parent_score = predictions[parent_id]
            parent_positive = parent_score >= threshold

            # 子がPositiveなのに親がNegative → 矛盾
            if child_positive and not parent_positive:
                inconsistent_edges += 1

    if total_edges == 0:
        return 0.0

    inconsistency_rate = inconsistent_edges / total_edges
    return inconsistency_rate
