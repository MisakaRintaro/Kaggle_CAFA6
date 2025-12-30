#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Prediction utilities for CAFA-6 competition.

This module provides functions for:
- Creating test DataLoader
- Running inference
- Creating submission files
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class TestProteinDataset(Dataset):
    """
    テストデータ用のデータセット（ラベルなし）
    """
    def __init__(
        self,
        protein_emb_dict: Dict[Tuple[str, str], np.ndarray]
    ):
        """
        Parameters
        ----------
        protein_emb_dict : Dict[Tuple[str, str], np.ndarray]
            {(protein_id, taxon_id): embedding} 形式の辞書
        """
        # 辞書をリストに変換（順序を固定）
        self.protein_ids = list(protein_emb_dict.keys())
        self.embeddings = [protein_emb_dict[pid] for pid in self.protein_ids]

        # Tensorに変換
        self.X = torch.from_numpy(np.stack(self.embeddings, axis=0)).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {
            'x': self.X[idx],                    # (protein_emb_dim,)
            'protein_id': self.protein_ids[idx]  # (protein_id, taxon_id)
        }


def test_collate_fn(batch):
    """
    テストデータ用のカスタムcollate_fn
    protein_idがタプルなので、デフォルトのcollate_fnでは処理できない
    """
    x_list = [item['x'] for item in batch]
    protein_ids = [item['protein_id'] for item in batch]

    return {
        'x': torch.stack(x_list),
        'protein_id': protein_ids  # リストのまま返す
    }


def create_test_loader(
    protein_emb_dict: Dict[Tuple[str, str], np.ndarray],
    batch_size: int = 16
) -> DataLoader:
    """
    テストデータのDataLoaderを作成する関数

    Parameters
    ----------
    protein_emb_dict : Dict[Tuple[str, str], np.ndarray]
        テストタンパク質の埋め込み辞書
    batch_size : int
        バッチサイズ

    Returns
    -------
    DataLoader
        テストデータのDataLoader
    """
    test_dataset = TestProteinDataset(protein_emb_dict)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # テストデータはシャッフルしない
        num_workers=0,
        collate_fn=test_collate_fn  # カスタムcollate_fnを使用
    )

    print(f"Test DataLoader created:")
    print(f"  Number of samples: {len(test_dataset)}")
    print(f"  Batch size: {batch_size}")
    print(f"  Number of batches: {len(test_loader)}")

    return test_loader


def predict(
    model: nn.Module,
    test_loader: DataLoader,
    go_id_list: List[str],
    device: torch.device,
    top_k: int = 100
) -> Dict[Tuple[str, str], List[Tuple[str, float]]]:
    """
    テストデータに対して推論を実行する関数

    Parameters
    ----------
    model : nn.Module
        訓練済みモデル
    test_loader : DataLoader
        テストデータのDataLoader
    go_id_list : List[str]
        GO IDのリスト（モデルの出力と対応）
    device : torch.device
        使用するデバイス
    top_k : int
        各タンパク質に対して予測する上位K個のGO term

    Returns
    -------
    predictions : Dict[Tuple[str, str], List[Tuple[str, float]]]
        {(protein_id, taxon_id): [(go_id, score), ...]} 形式の予測結果
    """
    model.eval()
    predictions = {}

    print(f"Starting inference...")
    print(f"  Device: {device}")
    print(f"  Top-K predictions per protein: {top_k}")
    print(f"  Number of batches: {len(test_loader)}")
    print()

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            # データをデバイスに移動
            x_batch = batch['x'].to(device)  # (B, protein_emb_dim)
            protein_ids = batch['protein_id']  # タプルのリスト

            # Forward pass
            probs = model.predict_proba(x_batch)  # (B, num_go)

            # 各サンプルについて上位K個のGO termを取得
            for i in range(len(protein_ids)):
                # 確率の高い順にソート
                scores = probs[i].cpu().numpy()
                top_indices = np.argsort(scores)[::-1][:top_k]

                # (GO ID, スコア) のリストを作成
                protein_id = protein_ids[i]
                predictions[protein_id] = [
                    (go_id_list[idx], float(scores[idx]))
                    for idx in top_indices
                ]

            # 進捗表示
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(test_loader):
                print(f"Processed batch [{batch_idx+1}/{len(test_loader)}]")

    print(f"\nInference completed!")
    print(f"  Total proteins predicted: {len(predictions)}")

    return predictions


def create_submission_file(
    predictions: Dict[Tuple[str, str], List[Tuple[str, float]]],
    output_path: str,
    add_text_descriptions: bool = False
) -> None:
    """
    予測結果から提出ファイルを作成する関数

    Parameters
    ----------
    predictions : Dict[Tuple[str, str], List[Tuple[str, float]]]
        {(protein_id, taxon_id): [(go_id, score), ...]} 形式の予測結果
    output_path : str
        出力ファイルパス
    add_text_descriptions : bool
        テキスト説明行を追加するかどうか
    """
    rows = []

    print(f"Creating submission file...")
    print(f"  Number of proteins: {len(predictions)}")

    for (protein_id, taxon_id), go_predictions in predictions.items():
        # GO term predictions
        for go_id, score in go_predictions:
            rows.append({
                'protein_id': protein_id,
                'kind_or_term': go_id,
                'score': f"{score:.6f}",
                'description': None
            })

        # Text descriptions (optional)
        if add_text_descriptions:
            # ダミーのテキスト説明を3つ追加
            # 実際にはLLMなどで生成する必要がある
            for i in range(3):
                rows.append({
                    'protein_id': protein_id,
                    'kind_or_term': 'Text',
                    'score': f"{0.5:.6f}",  # ダミースコア
                    'description': f"Predicted function description {i+1} for {protein_id}"
                })

    # DataFrameに変換
    submission_df = pd.DataFrame(rows)

    # TSVファイルとして保存
    submission_df.to_csv(
        output_path,
        sep='\t',
        index=False,
        header=False
    )

    print(f"Submission file saved to {output_path}")
    print(f"  Total rows: {len(submission_df)}")
    print(f"  GO term predictions: {len([r for r in rows if r['kind_or_term'] != 'Text'])}")
    if add_text_descriptions:
        print(f"  Text descriptions: {len([r for r in rows if r['kind_or_term'] == 'Text'])}")

    # サンプル表示
    print(f"\nFirst 10 rows of submission:")
    print(submission_df.head(10).to_string(index=False))
