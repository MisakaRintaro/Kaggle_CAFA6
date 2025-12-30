#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Training utilities for CAFA-6 competition.

This module provides functions for:
- Training the JointModel
- Saving/loading trained models
"""

from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    num_epochs: int = 10,
    learning_rate: float = 1e-3
) -> Dict[str, List[float]]:
    """
    JointModelを訓練する関数

    Parameters
    ----------
    model : nn.Module
        訓練するモデル（JointModel）
    train_loader : DataLoader
        訓練データのDataLoader
    device : torch.device
        使用するデバイス
    num_epochs : int
        エポック数
    learning_rate : float
        学習率

    Returns
    -------
    history : Dict[str, List[float]]
        訓練履歴（損失値など）
    """
    # モデルをデバイスに移動
    model = model.to(device)

    # 損失関数とオプティマイザ
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 訓練履歴を記録
    history = {
        'epoch': [],
        'loss': [],
        'avg_loss': []
    }

    print(f"Starting training...")
    print(f"  Device: {device}")
    print(f"  Number of epochs: {num_epochs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Batch size: {train_loader.batch_size}")
    print(f"  Number of batches per epoch: {len(train_loader)}")
    print(f"  Total training samples: {len(train_loader.dataset)}")
    print()

    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []

        for batch_idx, batch in enumerate(train_loader):
            # データをデバイスに移動
            x_batch = batch['x'].to(device)  # (B, protein_emb_dim)
            y_batch = batch['y'].to(device)  # (B, num_go)

            # Forward pass
            optimizer.zero_grad()
            logits = model(x_batch)  # (B, num_go)

            # Loss計算
            loss = criterion(logits, y_batch)

            # Backward pass
            loss.backward()
            optimizer.step()

            # 損失を記録
            epoch_losses.append(loss.item())

            # 進捗表示（10バッチごと）
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
                avg_loss = np.mean(epoch_losses[-10:])
                print(f"Epoch [{epoch+1}/{num_epochs}] "
                      f"Batch [{batch_idx+1}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f} "
                      f"Avg Loss (last 10): {avg_loss:.4f}")

        # エポック終了時の平均損失
        avg_epoch_loss = np.mean(epoch_losses)
        history['epoch'].append(epoch + 1)
        history['loss'].extend(epoch_losses)
        history['avg_loss'].append(avg_epoch_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}] completed. "
              f"Average Loss: {avg_epoch_loss:.4f}")
        print("-" * 60)

    print("Training completed!")
    return history


def save_model(
    model: nn.Module,
    filepath: str,
    go_id_list: List[str],
    history: Dict[str, List[float]]
) -> None:
    """
    訓練済みモデルを保存する関数

    Parameters
    ----------
    model : nn.Module
        保存するモデル
    filepath : str
        保存先ファイルパス
    go_id_list : List[str]
        GO IDのリスト（推論時に必要）
    history : Dict[str, List[float]]
        訓練履歴
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'go_id_list': go_id_list,
        'history': history,
        'model_config': {
            'input_dim_protein': model.protein_fc.in_features,
            'input_dim_go': model.go_fc.in_features,
            'joint_dim': model.protein_fc.out_features,
            'num_go': model.num_go
        }
    }

    torch.save(checkpoint, filepath)
    print(f"Model saved to {filepath}")
    print(f"  - Model state dict")
    print(f"  - GO ID list ({len(go_id_list)} terms)")
    print(f"  - Training history")
    print(f"  - Model configuration")


def load_model(
    filepath: str,
    go_raw_emb_matrix: torch.Tensor,
    device: torch.device = torch.device('cpu')
) -> Tuple[nn.Module, List[str], Dict[str, List[float]]]:
    """
    保存したモデルを読み込む関数

    Parameters
    ----------
    filepath : str
        モデルファイルのパス
    go_raw_emb_matrix : torch.Tensor
        GO埋め込み行列（モデル初期化に必要）
    device : torch.device
        使用するデバイス

    Returns
    -------
    model : nn.Module
        読み込んだモデル
    go_id_list : List[str]
        GO IDのリスト
    history : Dict[str, List[float]]
        訓練履歴
    """
    from .model import JointModel

    checkpoint = torch.load(filepath, map_location=device, weights_only=False)

    # モデルを再構築
    config = checkpoint['model_config']
    model = JointModel(
        input_dim_protein=config['input_dim_protein'],
        input_dim_go=config['input_dim_go'],
        joint_dim=config['joint_dim'],
        go_raw_emb=go_raw_emb_matrix
    )

    # 重みを読み込み
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Model loaded from {filepath}")
    print(f"  - Input dim (protein): {config['input_dim_protein']}")
    print(f"  - Input dim (GO): {config['input_dim_go']}")
    print(f"  - Joint dim: {config['joint_dim']}")
    print(f"  - Number of GO terms: {config['num_go']}")

    return model, checkpoint['go_id_list'], checkpoint['history']
