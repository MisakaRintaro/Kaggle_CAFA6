#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluation utilities for CAFA-6 competition.

This module provides functions for:
- Train/validation split with stratification
- Computing basic classification metrics (Precision, Recall, F1, AP)
- Computing IA-weighted Fmax (official CAFA-6 metric)
- Computing hierarchical consistency score
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score
)


def split_train_validation(
    protein_ids: List[Tuple[str, str]],
    train_label_df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify_by_label_count: bool = True
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    タンパク質をtrain/validationに分割する

    Parameters
    ----------
    protein_ids : List[Tuple[str, str]]
        全タンパク質のID [(protein_id, taxon_id), ...]
    train_label_df : pd.DataFrame
        訓練ラベルのDataFrame (columns: ['EntryID', 'term', 'aspect'])
    test_size : float
        validation setの割合 (default: 0.2)
    random_state : int
        ランダムシード
    stratify_by_label_count : bool
        ラベル数で層別化するかどうか

    Returns
    -------
    train_ids : List[Tuple[str, str]]
        訓練用のタンパク質ID
    val_ids : List[Tuple[str, str]]
        検証用のタンパク質ID
    """
    # protein_id (タプルの最初の要素) のみを取得
    protein_id_strs = [pid[0] for pid in protein_ids]

    if stratify_by_label_count:
        # 各タンパク質のラベル数を計算
        label_counts = train_label_df['EntryID'].value_counts().to_dict()
        protein_label_counts = [label_counts.get(pid, 0) for pid in protein_id_strs]

        # ラベル数を離散化（層別化用）
        # ラベル数の分位数で層を作る
        quantiles = pd.qcut(protein_label_counts, q=5, labels=False, duplicates='drop')
        stratify = quantiles
    else:
        stratify = None

    # train/validation分割
    train_ids, val_ids = train_test_split(
        protein_ids,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify
    )

    print(f"Train/Validation split:")
    print(f"  Train proteins: {len(train_ids)}")
    print(f"  Validation proteins: {len(val_ids)}")
    print(f"  Split ratio: {1-test_size:.1%} / {test_size:.1%}")

    if stratify_by_label_count:
        # 分割後の統計を表示
        train_id_strs = [pid[0] for pid in train_ids]
        val_id_strs = [pid[0] for pid in val_ids]

        train_label_counts = [label_counts.get(pid, 0) for pid in train_id_strs]
        val_label_counts = [label_counts.get(pid, 0) for pid in val_id_strs]

        print(f"  Train avg labels per protein: {np.mean(train_label_counts):.2f}")
        print(f"  Val avg labels per protein: {np.mean(val_label_counts):.2f}")

    return train_ids, val_ids


def compute_basic_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    基本的な分類指標を計算

    Parameters
    ----------
    y_true : np.ndarray
        真のラベル (shape: [num_samples, num_classes])
    y_pred_proba : np.ndarray
        予測確率 (shape: [num_samples, num_classes])
    threshold : float
        二値化の閾値

    Returns
    -------
    metrics : Dict[str, float]
        各種評価指標
    """
    # 閾値で二値化
    y_pred_binary = (y_pred_proba >= threshold).astype(int)

    # average='samples': 各サンプル（タンパク質）ごとに計算して平均
    precision = precision_score(y_true, y_pred_binary, average='samples', zero_division=0)
    recall = recall_score(y_true, y_pred_binary, average='samples', zero_division=0)
    f1 = f1_score(y_true, y_pred_binary, average='samples', zero_division=0)

    # Average Precision (閾値非依存)
    ap = average_precision_score(y_true, y_pred_proba, average='samples')

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'average_precision': ap
    }


def load_ia_weights(ia_path: str) -> Dict[str, float]:
    """
    IA.tsvからInformation Accretion weightsを読み込む

    Parameters
    ----------
    ia_path : str
        IA.tsvのファイルパス

    Returns
    -------
    ia_weights : Dict[str, float]
        {GO_ID: IA_weight}
    """
    ia_df = pd.read_csv(ia_path, sep='\t', header=None, names=['GO', 'weight'])
    ia_weights = dict(zip(ia_df['GO'], ia_df['weight']))

    print(f"Loaded IA weights for {len(ia_weights)} GO terms")
    return ia_weights


def compute_ia_weighted_precision_recall(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    go_id_list: List[str],
    ia_weights: Dict[str, float],
    threshold: float
) -> Tuple[float, float]:
    """
    IA-weighted Precision and Recallを計算

    Parameters
    ----------
    y_true : np.ndarray
        真のラベル (shape: [num_samples, num_classes])
    y_pred_proba : np.ndarray
        予測確率 (shape: [num_samples, num_classes])
    go_id_list : List[str]
        各列に対応するGO IDのリスト
    ia_weights : Dict[str, float]
        各GO termのIA weight
    threshold : float
        予測の閾値

    Returns
    -------
    precision : float
        IA-weighted precision
    recall : float
        IA-weighted recall
    """
    # 閾値で二値化
    y_pred_binary = (y_pred_proba >= threshold).astype(int)

    # 各GO termのIA weightを取得（存在しない場合は0）
    ia_weight_array = np.array([ia_weights.get(go_id, 0.0) for go_id in go_id_list])

    # True Positive, False Positive, False Negativeを計算
    tp = (y_true * y_pred_binary).astype(float)  # (num_samples, num_classes)
    fp = ((1 - y_true) * y_pred_binary).astype(float)
    fn = (y_true * (1 - y_pred_binary)).astype(float)

    # IA weightで重み付け
    tp_weighted = tp * ia_weight_array  # (num_samples, num_classes)
    fp_weighted = fp * ia_weight_array
    fn_weighted = fn * ia_weight_array

    # 全サンプルで合計
    total_tp = tp_weighted.sum()
    total_fp = fp_weighted.sum()
    total_fn = fn_weighted.sum()

    # Precision = TP / (TP + FP)
    # Recall = TP / (TP + FN)
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0

    return precision, recall


def compute_ia_weighted_fmax(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    go_id_list: List[str],
    ia_weights: Dict[str, float],
    thresholds: Optional[np.ndarray] = None
) -> Tuple[float, float]:
    """
    IA-weighted Fmaxを計算（複数の閾値でF1を計算し、最大値を取る）

    Parameters
    ----------
    y_true : np.ndarray
        真のラベル (shape: [num_samples, num_classes])
    y_pred_proba : np.ndarray
        予測確率 (shape: [num_samples, num_classes])
    go_id_list : List[str]
        各列に対応するGO IDのリスト
    ia_weights : Dict[str, float]
        各GO termのIA weight
    thresholds : Optional[np.ndarray]
        評価する閾値のリスト（Noneの場合は0.01〜0.99を100分割）

    Returns
    -------
    fmax : float
        IA-weighted maximum F1-score
    best_threshold : float
        Fmaxを達成した閾値
    """
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)

    f1_scores = []

    for threshold in thresholds:
        precision, recall = compute_ia_weighted_precision_recall(
            y_true, y_pred_proba, go_id_list, ia_weights, threshold
        )

        # F1 = 2 * (precision * recall) / (precision + recall)
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        f1_scores.append(f1)

    # 最大F1スコアとその閾値
    fmax = max(f1_scores)
    best_threshold = thresholds[np.argmax(f1_scores)]

    return fmax, best_threshold


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    go_id_list: List[str],
    ia_weights: Dict[str, float],
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    包括的な評価指標を計算

    Parameters
    ----------
    y_true : np.ndarray
        真のラベル
    y_pred_proba : np.ndarray
        予測確率
    go_id_list : List[str]
        GO IDのリスト
    ia_weights : Dict[str, float]
        IA weights
    threshold : float
        基本指標の計算に使用する閾値

    Returns
    -------
    metrics : Dict[str, float]
        全ての評価指標
    """
    # 基本指標
    basic_metrics = compute_basic_metrics(y_true, y_pred_proba, threshold=threshold)

    # IA-weighted Fmax
    fmax, best_threshold = compute_ia_weighted_fmax(
        y_true, y_pred_proba, go_id_list, ia_weights
    )

    # 結果をまとめる
    metrics = {
        **basic_metrics,
        'ia_weighted_fmax': fmax,
        'ia_weighted_best_threshold': best_threshold
    }

    return metrics


def print_evaluation_results(metrics: Dict[str, float]) -> None:
    """
    評価結果を見やすく表示

    Parameters
    ----------
    metrics : Dict[str, float]
        評価指標の辞書
    """
    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)

    print("\nBasic Metrics (threshold=0.5):")
    print(f"  Precision:         {metrics['precision']:.4f}")
    print(f"  Recall:            {metrics['recall']:.4f}")
    print(f"  F1-score:          {metrics['f1']:.4f}")
    print(f"  Average Precision: {metrics['average_precision']:.4f}")

    print("\nIA-weighted Metrics:")
    print(f"  Fmax:              {metrics['ia_weighted_fmax']:.4f}")
    print(f"  Best threshold:    {metrics['ia_weighted_best_threshold']:.4f}")

    if 'hierarchical_consistency' in metrics:
        print("\nHierarchical Consistency:")
        print(f"  Inconsistency rate: {metrics['hierarchical_consistency']:.4f}")

    print("="*60)
