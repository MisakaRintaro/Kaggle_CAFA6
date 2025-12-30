#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Model definitions for CAFA-6 competition.

This module provides:
- JointModel: Dual-encoder model for protein-GO term matching
- ProteinGODataset: Dataset class for training
- Helper functions for creating training data
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class ProteinGODataset(Dataset):
    """
    タンパク質埋め込みとGOラベルのデータセット
    """
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Parameters
        ----------
        X : np.ndarray
            shape = (num_samples, protein_emb_dim)
            タンパク質埋め込み
        y : np.ndarray
            shape = (num_samples, num_go_terms)
            マルチラベル行列
        """
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {
            'x': self.X[idx],  # (protein_emb_dim,)
            'y': self.y[idx]   # (num_go_terms,)
        }


def create_protein_go_label_matrix(
    protein_emb_dict: Dict[Tuple[str, str], np.ndarray],
    train_label_df: pd.DataFrame,
    go_id_list: List[str]
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[str, str]]]:
    """
    タンパク質埋め込みとGOラベルの対応行列を作成する関数

    Parameters
    ----------
    protein_emb_dict : Dict[Tuple[str, str], np.ndarray]
        {(protein_id, taxon_id): embedding} 形式の辞書
    train_label_df : pd.DataFrame
        train_terms.tsv から読み込んだDataFrame
        columns: ['EntryID', 'term', 'aspect']
    go_id_list : List[str]
        GO IDのリスト（ソート済み）
        go_raw_emb_matrix の行と対応

    Returns
    -------
    X : np.ndarray
        shape = (num_proteins, protein_emb_dim)
        タンパク質埋め込み行列
    y : np.ndarray
        shape = (num_proteins, num_go_terms)
        マルチラベル行列（0 or 1）
    protein_ids : List[Tuple[str, str]]
        各行に対応する (protein_id, taxon_id) のリスト
    """
    # 1. GO ID -> インデックスのマッピングを作成
    go_id_to_idx = {go_id: idx for idx, go_id in enumerate(go_id_list)}
    num_go = len(go_id_list)

    # 2. EntryID -> GO terms のマッピングを作成
    protein_to_go_terms = {}
    for _, row in train_label_df.iterrows():
        entry_id = row['EntryID']
        go_term = row['term']

        if entry_id not in protein_to_go_terms:
            protein_to_go_terms[entry_id] = []

        # go_id_listに含まれるGO termのみ追加
        if go_term in go_id_to_idx:
            protein_to_go_terms[entry_id].append(go_term)

    # 3. protein_emb_dict から訓練データを構築
    #    ラベル情報があるタンパク質のみを使用
    X_list = []
    y_list = []
    protein_ids = []

    for (protein_id, taxon_id), emb in protein_emb_dict.items():
        # このタンパク質のラベルが存在するかチェック
        if protein_id in protein_to_go_terms:
            # 埋め込みベクトルを追加
            X_list.append(emb)

            # マルチラベルベクトルを作成（全GO termに対して0/1）
            label_vec = np.zeros(num_go, dtype=np.float32)
            for go_term in protein_to_go_terms[protein_id]:
                idx = go_id_to_idx[go_term]
                label_vec[idx] = 1.0

            y_list.append(label_vec)
            protein_ids.append((protein_id, taxon_id))

    # 4. numpy配列に変換
    X = np.stack(X_list, axis=0)  # (num_proteins, protein_emb_dim)
    y = np.stack(y_list, axis=0)  # (num_proteins, num_go_terms)

    print(f"Created label matrix:")
    print(f"  Number of proteins: {len(protein_ids)}")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  Average labels per protein: {y.sum(axis=1).mean():.2f}")
    print(f"  Min labels per protein: {y.sum(axis=1).min():.0f}")
    print(f"  Max labels per protein: {y.sum(axis=1).max():.0f}")

    return X, y, protein_ids


class JointModel(nn.Module):
    """
    Dual-encoder model for protein-GO term matching.

    - タンパク質埋め込み (ESM) h_p ∈ R^640
    - GO埋め込み (LLM)     g_t ∈ R^d_llm

    の両方に線形変換をかけて、同じ joint 空間 (R^d_joint) にマッピングするモデル。

        z_p = protein_fc(h_p)         ∈ R^d_joint
        e_t = go_fc(g_t)              ∈ R^d_joint
        score(p, t) = <z_p, e_t>      （内積）

    このクラスでは:
    - ESM本体、LLM本体の重みは更新しない（事前に計算済みの埋め込みを使う）
    - 学習するのは protein_fc と go_fc のみ
    """

    def __init__(
        self,
        input_dim_protein: int,   # 例: 480 (ESM埋め込み次元)
        input_dim_go: int,        # 例: 768 (LLM埋め込み次元)
        joint_dim: int,           # 例: 256 (joint 空間の次元)
        go_raw_emb: torch.Tensor, # shape = (num_go, input_dim_go)
    ) -> None:
        super().__init__()

        # タンパク質側: input_dim_protein → joint_dim
        self.protein_fc = nn.Linear(input_dim_protein, joint_dim)

        # GO側: input_dim_go → joint_dim
        self.go_fc = nn.Linear(input_dim_go, joint_dim)

        # 事前計算済みの GO 生埋め込み（LLM の出力）をモデル内に持たせる
        # 学習対象ではないので register_buffer を使う
        # go_raw_emb の shape は (num_go, input_dim_go) を想定
        self.register_buffer("go_raw_emb", go_raw_emb)

    @property
    def num_go(self) -> int:
        """GO の種類数を返すプロパティ。"""
        return self.go_raw_emb.size(0)

    def encode_protein(self, h_p: torch.Tensor) -> torch.Tensor:
        """
        タンパク質埋め込みを joint 空間に写像する関数。

        Parameters
        ----------
        h_p : torch.Tensor
            shape = (batch_size, input_dim_protein)
            事前に計算された ESM 埋め込みのミニバッチ。

        Returns
        -------
        torch.Tensor
            shape = (batch_size, joint_dim)
            joint 空間でのタンパク質ベクトル z_p。
        """
        z_p = self.protein_fc(h_p)
        return z_p

    def encode_go(self) -> torch.Tensor:
        """
        全ての GO を joint 空間に写像した行列を返す。

        Returns
        -------
        torch.Tensor
            shape = (num_go, joint_dim)
            各行が 1 つの GO に対応する joint 空間でのベクトル e_t。
        """
        # go_raw_emb: (num_go, input_dim_go)
        e_go = self.go_fc(self.go_raw_emb)  # (num_go, joint_dim)
        return e_go

    def forward(self, h_batch: torch.Tensor) -> torch.Tensor:
        """
        タンパク質埋め込みのバッチから、全 GO に対するスコア行列を計算する。

        Parameters
        ----------
        h_batch : torch.Tensor
            shape = (batch_size, input_dim_protein)
            ミニバッチのタンパク質埋め込み。

        Returns
        -------
        torch.Tensor
            shape = (batch_size, num_go)
            各サンプル・各 GO に対するスコア（logits）。
            損失関数 BCEWithLogitsLoss にそのまま渡せる。
        """
        # タンパク質を joint 空間へ
        z_p = self.encode_protein(h_batch)  # (B, D_joint)

        # 全 GO を joint 空間へ
        e_go = self.encode_go()             # (G, D_joint)

        # 内積でスコア行列を計算:
        #   (B, D) @ (D, G) -> (B, G)
        scores = z_p @ e_go.t()
        return scores

    def predict_proba(self, h_batch: torch.Tensor) -> torch.Tensor:
        """
        各タンパク質・各 GO に対する「確率」(0〜1) を返すユーティリティ。

        Parameters
        ----------
        h_batch : torch.Tensor
            shape = (batch_size, input_dim_protein)

        Returns
        -------
        torch.Tensor
            shape = (batch_size, num_go)
            シグモイドをかけた後の確率。
        """
        logits = self.forward(h_batch)
        probs = torch.sigmoid(logits)
        return probs
