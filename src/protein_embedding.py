#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Protein embedding utilities for CAFA-6 competition.

This module provides functions for:
- Creating protein embeddings using ESM-2
- Batching sequences for efficient processing
- Saving/loading protein embeddings
"""

from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel


def make_batches_from_seq_dict(
    seq_dict: Dict[Tuple[str, str], str],
    batch_size: int
) -> List[List[Tuple[Tuple[str, str], str]]]:
    """
    {(protein_id, taxon_id): seq} 形式の辞書を、
    指定したバッチサイズごとに分割する関数です。

    Parameters
    ----------
    seq_dict : Dict[Tuple[str, str], str]
        キー: (protein_id, taxon_id)
        値:   アミノ酸配列文字列
    batch_size : int
        1バッチに含める最大件数

    Returns
    -------
    batches : List[List[Tuple[Tuple[str, str], str]]]
        各要素が「バッチ」のリスト。
        各バッチは [((protein_id, taxon_id), seq), ...] という形のリスト。
    """
    items = list(seq_dict.items())  # [ ((protein_id, taxon_id), seq), ... ]
    batches: List[List[Tuple[Tuple[str, str], str]]] = []

    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batches.append(batch)

    return batches


def encode_batch_with_esm(
    batch: List[Tuple[Tuple[str, str], str]],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
    max_aa_len: int = 1024
) -> List[Tuple[Tuple[str, str], np.ndarray]]:
    """
    1バッチ分の [( (protein_id, taxon_id), seq ), ...] を入力として、
    各配列を ESM-2 でベクトル化し、
    [ ( (protein_id, taxon_id), embedding ), ... ] を返す関数です。

    各配列について:
      - tokenizer で padding / truncation / special token 付与
      - model から hidden_states (B, L, D) を取得
      - attention_mask から実長 true_len を求める
      - [CLS] と [EOS] を除いたトークンを mean pooling して 1ベクトル(D,) にする

    Parameters
    ----------
    batch : List[Tuple[Tuple[str, str], str]]
        1バッチ分のリスト。
        各要素は ((protein_id, taxon_id), seq)。
    tokenizer : AutoTokenizer
        ESM-2 用トークナイザ。
    model : AutoModel
        ESM-2 モデル本体。
    device : torch.device
        使用するデバイス (cuda / cpu / mps)。
    max_aa_len : int
        最大アミノ酸配列長

    Returns
    -------
    result_list : List[Tuple[Tuple[str, str], np.ndarray]]
        [ ( (protein_id, taxon_id), embedding ), ... ] のリスト。
        embedding は numpy.ndarray で shape = (D,)。
    """
    # ID と配列を別リストに分解
    id_list = [item[0] for item in batch]   # [(protein_id, taxon_id), ...]
    seq_list = [item[1] for item in batch]  # ["MKT...", "AAA...", ...]

    # 1) tokenizer でトークナイズ + padding
    inputs = tokenizer(
        seq_list,
        return_tensors="pt",
        padding=True,        # バッチ内の最長長さに合わせて padding
        truncation=True,     # モデルの最大長を超える場合は切り捨て
        max_length=max_aa_len + 2,
        add_special_tokens=True
    )

    # 2) device に載せる
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 3) モデルで hidden states を計算
    with torch.no_grad():
        outputs = model(**inputs)

    hidden_states: torch.Tensor = outputs.last_hidden_state  # (B, L, D)
    attention_mask: torch.Tensor = inputs["attention_mask"]  # (B, L)

    batch_size_actual = hidden_states.size(0)
    result_list: List[Tuple[Tuple[str, str], np.ndarray]] = []

    for i in range(batch_size_actual):
        seq_id = id_list[i]          # (protein_id, taxon_id)
        mask_i = attention_mask[i]   # (L,)
        hs_i = hidden_states[i]      # (L, D)

        # 有効トークン数（CLS/EOS 含む）
        true_len = mask_i.sum().item()

        # 通常は [CLS] と [EOS] を除いた [1 : true_len-1] を使う
        if true_len > 2:
            token_vecs = hs_i[1:true_len - 1, :]  # (L_true-2, D)
        else:
            # 極端に短い場合の保険
            token_vecs = hs_i[:true_len, :]

        # mean pooling で 1 ベクトルに集約
        protein_vec = token_vecs.mean(dim=0)          # (D,)
        protein_vec_np = protein_vec.cpu().numpy()    # numpy に変換

        result_list.append((seq_id, protein_vec_np))

    return result_list


def encode_all_sequences_with_esm(
    seq_dict: Dict[Tuple[str, str], str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
    batch_size: int = 64,
    max_batches: int | None = None
) -> Dict[Tuple[str, str], np.ndarray]:
    """
    {(protein_id, taxon_id): seq} 形式の辞書を入力として、
    ESM-2 で全配列をベクトル化し、
    {(protein_id, taxon_id): embedding} の辞書を返す高レベル関数です。

    内部では:
      - make_batches_from_seq_dict でバッチに分割
      - encode_batch_with_esm で各バッチを処理
      - 結果を1つの辞書にまとめる

    Parameters
    ----------
    seq_dict : Dict[Tuple[str, str], str]
        {(protein_id, taxon_id): seq} 形式の辞書。
    tokenizer : AutoTokenizer
        ESM-2 用トークナイザ。
    model : AutoModel
        ESM-2 モデル本体。
    device : torch.device
        使用するデバイス。
    batch_size : int
        1バッチあたりの件数。GPUメモリに応じて調整。
    max_batches : int | None
        処理する最大バッチ数。Noneの場合は全バッチを処理。
        開発・テスト時にデータ量を制限するために使用。

    Returns
    -------
    emb_dict : Dict[Tuple[str, str], np.ndarray]
        {(protein_id, taxon_id): embedding} 形式の辞書。
        embedding は numpy.ndarray (D,)。
    """
    batches = make_batches_from_seq_dict(seq_dict, batch_size=batch_size)
    emb_dict: Dict[Tuple[str, str], np.ndarray] = {}

    model.eval()

    # max_batchesが指定されている場合は、その数だけに制限
    batches_to_process = batches[:max_batches] if max_batches is not None else batches

    total_batches = len(batches)
    processing_batches = len(batches_to_process)

    if max_batches is not None:
        print(f"DEV_TEST mode: Processing {processing_batches}/{total_batches} batches")

    for batch_idx, batch in enumerate(batches_to_process):
        batch_results = encode_batch_with_esm(batch, tokenizer, model, device)
        for seq_id, emb in batch_results:
            emb_dict[seq_id] = emb

        # 進捗表示
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == processing_batches:
            print(f"Processed batch [{batch_idx+1}/{processing_batches}]")

        # 一時変数を消してキャッシュを捨てる
        del batch_results
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return emb_dict


def save_emb_dict(emb_dict: Dict[Tuple[str, str], Any], filepath: str) -> None:
    """
    emb_dict をファイルに保存する関数

    Parameters
    ----------
    emb_dict : Dict[Tuple[str, str], Any]
        キーが (protein_id, taxon_id)、値が埋め込みベクトル
        （torch.Tensor や numpy.ndarray）である辞書。
    filepath : str
        保存先ファイルパス。

    Returns
    -------
    None
        指定パスにファイルが作成されます。
    """
    path = Path(filepath)

    # GPU 上の Tensor が含まれていた場合に備えて CPU に移しておく
    cpu_dict = {}
    for key, value in emb_dict.items():
        # torch.Tensor なら .cpu() で CPU 転送
        if isinstance(value, torch.Tensor):
            cpu_dict[key] = value.cpu()
        else:
            # numpy.ndarray などはそのまま入れてOK
            cpu_dict[key] = value

    # torch.save は Python の任意オブジェクトを保存できる
    torch.save(cpu_dict, path)


def load_emb_dict(filepath: str, map_location: str = "cpu") -> Dict[Tuple[str, str], Any]:
    """
    保存しておいた emb_dict を読み込む関数

    Parameters
    ----------
    filepath : str
        読み込むファイルのパス。
    map_location : str, default "cpu"
        torch.load の map_location 引数。
        GPU 環境で「最初から GPU に載せたい」場合は "cuda" などを指定。

    Returns
    -------
    Dict[Tuple[str, str], Any]
        保存されていた emb_dict と同じ形式の辞書。
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    emb_dict = torch.load(path, map_location=map_location, weights_only=False)
    return emb_dict
