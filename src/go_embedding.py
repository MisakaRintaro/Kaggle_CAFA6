#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GO term embedding utilities for CAFA-6 competition.

This module provides functions for:
- Creating GO term embeddings using BiomedBERT
- Saving/loading GO embeddings
- Converting between dictionary and matrix formats
"""

from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel


def embed_go_text(text: str, tokenizer: AutoTokenizer, model: AutoModel, device: torch.device) -> np.ndarray:
    """
    BiomedBERT を使って、文字列をベクトルに変換する関数。
    出力は shape = (hidden_dim,) の1つのベクトル。

    標準的には CLS ベクトル（文頭トークン）を使う。

    Parameters
    ----------
    text : str
        埋め込みたいテキスト（GO termの名前や定義など）
    tokenizer : AutoTokenizer
        BiomedBERT用のトークナイザ
    model : AutoModel
        BiomedBERTモデル
    device : torch.device
        使用するデバイス

    Returns
    -------
    np.ndarray
        shape = (hidden_dim,) のベクトル
    """
    # 1. テキストをモデル入力に変換（tokenize）
    tokens = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    )

    # 2. デバイスに移動
    tokens = {k: v.to(device) for k, v in tokens.items()}

    # 3. モデルに通す
    with torch.no_grad():
        output = model(**tokens)

    # 4. CLS トークンの出力を取り出す（文全体の意味）
    cls_vector = output.last_hidden_state[:, 0, :]   # shape = (1, hidden_dim)

    # 5. shape を (hidden_dim,) に変えてnumpyに変換
    return cls_vector.squeeze(0).cpu().numpy()


def create_go_embedding_matrix(
    terms: Dict[str, Dict[str, Any]],
    train_label_df: pd.DataFrame,
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device
) -> Tuple[torch.Tensor, List[str], Dict[str, np.ndarray]]:
    """
    学習データに登場するGO termの埋め込み行列を作成する関数

    Parameters
    ----------
    terms : Dict[str, Dict[str, Any]]
        parse_go_obo関数で取得したGO term辞書
    train_label_df : pd.DataFrame
        train_terms.tsv から読み込んだDataFrame
    tokenizer : AutoTokenizer
        BiomedBERT用のトークナイザ
    model : AutoModel
        BiomedBERTモデル
    device : torch.device
        使用するデバイス

    Returns
    -------
    go_emb_matrix : torch.Tensor
        shape = (num_go, hidden_dim) のGO埋め込み行列
        行の順序は go_id_list と一致（ソート済み）
    go_id_list : List[str]
        各行に対応するGO IDのリスト（ソート済み）
    go_embeddings_dict : Dict[str, np.ndarray]
        GO ID -> 埋め込みベクトルの辞書
        これを保持することで、順序に依存しない安全な管理が可能
    """
    # 1. 学習データに登場するユニークなGO termを取得
    unique_go_terms = train_label_df['term'].unique().tolist()
    print(f"Number of unique GO terms in training data: {len(unique_go_terms)}")

    # 2. 各GO termの埋め込みを辞書として計算
    go_embeddings_dict: Dict[str, np.ndarray] = {}

    for go_id in unique_go_terms:
        if go_id in terms:
            # GO termの名前と定義を結合してテキスト化
            term_info = terms[go_id]
            name = term_info.get('name', '')
            definition = term_info.get('def', '')

            # 定義から引用符とリファレンスを除去
            if definition:
                # "..." [ref] 形式から "..." 部分だけ取り出す
                definition = definition.split('[')[0].strip().strip('"')

            # 名前と定義を結合
            text = f"{name}. {definition}" if definition else name

            # 埋め込みを計算して辞書に格納
            emb = embed_go_text(text, tokenizer, model, device)
            go_embeddings_dict[go_id] = emb
        else:
            print(f"Warning: GO term {go_id} not found in go-basic.obo")

    # 3. モデル用に確定的な順序でソート
    #    ソートすることで、常に同じ順序が保証される（再現性）
    go_id_list = sorted(go_embeddings_dict.keys())

    # 4. ソートされた順序で行列を構築
    go_emb_matrix = torch.from_numpy(
        np.stack([go_embeddings_dict[go_id] for go_id in go_id_list], axis=0)
    ).float()

    print(f"GO embedding matrix shape: {go_emb_matrix.shape}")
    print(f"GO IDs are sorted alphabetically for reproducibility")

    return go_emb_matrix, go_id_list, go_embeddings_dict


def save_go_embeddings(
    go_embeddings_dict: Dict[str, np.ndarray],
    filepath: str
) -> None:
    """
    GO埋め込み辞書をファイルに保存する関数

    Parameters
    ----------
    go_embeddings_dict : Dict[str, np.ndarray]
        GO ID -> 埋め込みベクトルの辞書
    filepath : str
        保存先ファイルパス
    """
    path = Path(filepath)
    torch.save(go_embeddings_dict, path)
    print(f"GO embeddings saved to {filepath}")


def load_go_embeddings(filepath: str) -> Dict[str, np.ndarray]:
    """
    保存したGO埋め込み辞書を読み込む関数

    Parameters
    ----------
    filepath : str
        読み込むファイルのパス

    Returns
    -------
    Dict[str, np.ndarray]
        GO ID -> 埋め込みベクトルの辞書
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    go_embeddings_dict = torch.load(path, map_location="cpu", weights_only=False)
    print(f"GO embeddings loaded from {filepath}")
    print(f"Number of GO terms: {len(go_embeddings_dict)}")

    return go_embeddings_dict


def reconstruct_go_matrix_from_dict(
    go_embeddings_dict: Dict[str, np.ndarray]
) -> Tuple[torch.Tensor, List[str]]:
    """
    GO埋め込み辞書から行列とIDリストを再構築する関数

    Parameters
    ----------
    go_embeddings_dict : Dict[str, np.ndarray]
        GO ID -> 埋め込みベクトルの辞書

    Returns
    -------
    go_emb_matrix : torch.Tensor
        shape = (num_go, hidden_dim) のGO埋め込み行列
    go_id_list : List[str]
        各行に対応するGO IDのリスト（ソート済み）
    """
    # ソートされた順序でIDリストを作成（再現性のため）
    go_id_list = sorted(go_embeddings_dict.keys())

    # 行列を構築
    go_emb_matrix = torch.from_numpy(
        np.stack([go_embeddings_dict[go_id] for go_id in go_id_list], axis=0)
    ).float()

    return go_emb_matrix, go_id_list
