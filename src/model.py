#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Model definitions for CAFA-6 competition.

This module provides:
- JointModel: Dual-encoder model with cross-attention for protein-GO term matching
- ProteinGODataset: Dataset class for training
- Helper functions for creating training data
"""

from typing import Dict, List, Tuple, Optional
import math
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

    # デバッグ情報: サンプルIDを確認
    emb_sample_ids = list(protein_emb_dict.keys())[:5]
    label_sample_ids = list(protein_to_go_terms.keys())[:5]

    print(f"DEBUG: Protein embedding dict size: {len(protein_emb_dict)}")
    print(f"DEBUG: Protein labels dict size: {len(protein_to_go_terms)}")
    print(f"DEBUG: Sample embedding IDs: {emb_sample_ids}")
    print(f"DEBUG: Sample label IDs: {label_sample_ids}")

    for (protein_id, taxon_id), emb in protein_emb_dict.items():
        # Extract UniProt accession from full header if needed
        # Format: "sp|A0A0C5B5G6|MOTSC_HUMAN" -> "A0A0C5B5G6"
        if '|' in protein_id:
            accession = protein_id.split('|')[1]
        else:
            accession = protein_id

        # このタンパク質のラベルが存在するかチェック
        if accession in protein_to_go_terms:
            # 埋め込みベクトルを追加
            X_list.append(emb)

            # マルチラベルベクトルを作成（全GO termに対して0/1）
            label_vec = np.zeros(num_go, dtype=np.float32)
            for go_term in protein_to_go_terms[accession]:
                idx = go_id_to_idx[go_term]
                label_vec[idx] = 1.0

            y_list.append(label_vec)
            protein_ids.append((protein_id, taxon_id))

    # データが空でないかチェック
    if len(X_list) == 0:
        raise ValueError(
            f"No matching proteins found between embeddings and labels!\n"
            f"  Embedding dict size: {len(protein_emb_dict)}\n"
            f"  Label dict size: {len(protein_to_go_terms)}\n"
            f"  Sample embedding IDs: {emb_sample_ids}\n"
            f"  Sample label IDs: {label_sample_ids}\n"
            f"This usually means the protein IDs in the embedding file don't match "
            f"the IDs in the label file. Check the ID format (e.g., with/without taxon)."
        )

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
    Dual-encoder model with Cross-Attention mechanism for protein-GO term matching.

    This model uses cross-attention to allow the protein embedding to "attend to"
    different GO terms, learning which GO terms are most relevant for each protein.

    Architecture:
        1. タンパク質埋め込み → Linear(480 → 256) → z_p
        2. GO埋め込み群 → Linear(768 → 256) → E_go
        3. Cross-Attention:
           - Query = Linear(z_p)
           - Key = Linear(E_go)
           - Attention Score = (Q @ K.T) / sqrt(d_k)
        4. Dropout for regularization

    Parameters
    ----------
    input_dim_protein : int
        Dimension of protein embeddings (e.g., 480 for ESM-2)
    input_dim_go : int
        Dimension of GO embeddings (e.g., 768 for BiomedBERT)
    joint_dim : int
        Dimension of the joint embedding space (e.g., 256)
    go_raw_emb : torch.Tensor
        Pre-computed GO embeddings, shape = (num_go, input_dim_go)
    dropout : float
        Dropout probability for regularization (default: 0.1)
    """

    def __init__(
        self,
        input_dim_protein: int,
        input_dim_go: int,
        joint_dim: int,
        go_raw_emb: torch.Tensor,
        dropout: float = 0.1
    ) -> None:
        super().__init__()

        # Protein encoder: input_dim_protein → joint_dim
        self.protein_fc = nn.Linear(input_dim_protein, joint_dim)

        # GO encoder: input_dim_go → joint_dim
        self.go_fc = nn.Linear(input_dim_go, joint_dim)

        # Cross-Attention components
        self.attn_query = nn.Linear(joint_dim, joint_dim)
        self.attn_key = nn.Linear(joint_dim, joint_dim)

        # Regularization
        self.dropout = nn.Dropout(dropout)

        # Scaling factor for attention scores
        self.scale = math.sqrt(joint_dim)

        # Pre-computed GO embeddings (not trainable)
        self.register_buffer("go_raw_emb", go_raw_emb)

    @property
    def num_go(self) -> int:
        """Number of GO terms."""
        return self.go_raw_emb.size(0)

    def encode_protein(self, h_p: torch.Tensor) -> torch.Tensor:
        """
        Encode protein embeddings into joint space.

        Parameters
        ----------
        h_p : torch.Tensor
            shape = (batch_size, input_dim_protein)
            Pre-computed protein embeddings (ESM-2)

        Returns
        -------
        torch.Tensor
            shape = (batch_size, joint_dim)
            Protein vectors in joint space
        """
        z_p = self.protein_fc(h_p)
        return z_p

    def encode_go(self) -> torch.Tensor:
        """
        Encode all GO terms into joint space.

        Returns
        -------
        torch.Tensor
            shape = (num_go, joint_dim)
            GO term vectors in joint space
        """
        e_go = self.go_fc(self.go_raw_emb)
        return e_go

    def forward(self, h_batch: torch.Tensor) -> torch.Tensor:
        """
        Compute attention-based scores for all GO terms.

        Uses cross-attention mechanism where protein embeddings act as
        queries and GO embeddings act as keys. The attention scores
        directly represent the relevance of each GO term to the protein.

        Parameters
        ----------
        h_batch : torch.Tensor
            shape = (batch_size, input_dim_protein)
            Batch of protein embeddings

        Returns
        -------
        torch.Tensor
            shape = (batch_size, num_go)
            Attention scores (logits) for each protein-GO pair
        """
        # Encode protein and GO into joint space
        z_p = self.encode_protein(h_batch)  # (B, D)
        e_go = self.encode_go()              # (G, D)

        # Apply dropout to joint embeddings
        z_p = self.dropout(z_p)
        e_go = self.dropout(e_go)

        # Cross-Attention: Protein → Query, GO → Key
        Q = self.attn_query(z_p)   # (B, D)
        K = self.attn_key(e_go)    # (G, D)

        # Compute attention scores: (B, D) @ (D, G) → (B, G)
        # Scaled dot-product attention
        scores = (Q @ K.t()) / self.scale

        return scores

    def predict_proba(self, h_batch: torch.Tensor) -> torch.Tensor:
        """
        Predict probabilities for each protein-GO pair.

        Parameters
        ----------
        h_batch : torch.Tensor
            shape = (batch_size, input_dim_protein)

        Returns
        -------
        torch.Tensor
            shape = (batch_size, num_go)
            Probabilities after sigmoid activation
        """
        logits = self.forward(h_batch)
        probs = torch.sigmoid(logits)
        return probs


def compute_pos_weight(
    y_train: np.ndarray,
    clip_max: Optional[float] = None
) -> torch.Tensor:
    """
    Compute pos_weight for BCEWithLogitsLoss to handle class imbalance.

    pos_weight[i] = (num_negative_samples[i]) / (num_positive_samples[i])
                  = number of negatives / number of positives for each GO term

    This gives higher weight to positive samples of rare GO terms.

    Parameters
    ----------
    y_train : np.ndarray
        Training labels, shape = (num_proteins, num_go_terms)
        Binary matrix where y_train[i, j] = 1 if protein i has GO term j
    clip_max : float, optional
        Maximum value to clip pos_weight (prevents extreme values)
        Default: None (no clipping)

    Returns
    -------
    torch.Tensor
        pos_weight tensor, shape = (num_go_terms,)
        Weight for positive samples of each GO term

    Notes
    -----
    - For GO terms with no positive samples, pos_weight is set to 1.0
    - Clipping prevents extremely large weights for very rare GO terms
    - BCEWithLogitsLoss uses this to weight the positive class loss:
      loss = -pos_weight * y * log(sigmoid(x)) - (1 - y) * log(1 - sigmoid(x))
    """
    # Count positive and negative samples for each GO term
    pos_counts = y_train.sum(axis=0)  # (num_go,)
    num_proteins = y_train.shape[0]
    neg_counts = num_proteins - pos_counts

    # Compute pos_weight = neg_count / pos_count
    # Add small epsilon to avoid division by zero
    pos_weight = neg_counts / (pos_counts + 1e-8)

    # Handle edge case: if a GO term has no positive samples, set weight to 1.0
    pos_weight[pos_counts == 0] = 1.0

    # Clip to prevent extreme values
    if clip_max is not None:
        pos_weight = np.minimum(pos_weight, clip_max)

    # Convert to torch tensor
    pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float32)

    # Print statistics
    print(f"\nPos_weight statistics:")
    print(f"  Min:    {pos_weight.min():.2f}")
    print(f"  Max:    {pos_weight.max():.2f}")
    print(f"  Mean:   {pos_weight.mean():.2f}")
    print(f"  Median: {np.median(pos_weight):.2f}")
    print(f"  Number of GO terms: {len(pos_weight)}")
    print(f"  GO terms with no positives: {(pos_counts == 0).sum()}")
    if clip_max is not None:
        print(f"  Clipped to max: {clip_max}")

    return pos_weight_tensor
