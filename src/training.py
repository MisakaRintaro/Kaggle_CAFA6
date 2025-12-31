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
    learning_rate: float = 1e-3,
    val_loader: DataLoader = None,
    patience: int = 3,
    min_delta: float = 0.01,
    pos_weight: torch.Tensor = None
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
    val_loader : DataLoader, optional
        検証データのDataLoader。Noneの場合はearly stoppingなし
    patience : int
        Early stoppingのpatience（改善が見られないエポック数の許容値）
    min_delta : float
        改善と見なすための最小相対改善率（例: 0.01 = 1%改善）
    pos_weight : torch.Tensor, optional
        Weight for positive samples to handle class imbalance
        shape = (num_go_terms,)
        If None, all classes are weighted equally

    Returns
    -------
    history : Dict[str, List[float]]
        訓練履歴（損失値など）
    """
    # モデルをデバイスに移動
    model = model.to(device)

    # 損失関数とオプティマイザ
    # pos_weightがある場合はデバイスに移動してから使用
    if pos_weight is not None:
        pos_weight = pos_weight.to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 訓練履歴を記録
    history = {
        'epoch': [],
        'loss': [],
        'avg_loss': [],
        'val_loss': []
    }

    # Early stopping用の変数
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    print(f"Starting training...")
    print(f"  Device: {device}")
    print(f"  Number of epochs: {num_epochs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Batch size: {train_loader.batch_size}")
    print(f"  Number of batches per epoch: {len(train_loader)}")
    print(f"  Total training samples: {len(train_loader.dataset)}")
    print(f"  Pos_weight enabled: {'Yes' if pos_weight is not None else 'No'}")
    if val_loader is not None:
        print(f"  Validation enabled: Yes")
        print(f"  Validation samples: {len(val_loader.dataset)}")
        print(f"  Early stopping patience: {patience}")
    else:
        print(f"  Validation enabled: No")
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

        # Validation loss計算
        val_loss = None
        if val_loader is not None:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch in val_loader:
                    x_batch = batch['x'].to(device)
                    y_batch = batch['y'].to(device)
                    logits = model(x_batch)
                    loss = criterion(logits, y_batch)
                    val_losses.append(loss.item())

            val_loss = np.mean(val_losses)
            history['val_loss'].append(val_loss)

            print(f"Epoch [{epoch+1}/{num_epochs}] completed. "
                  f"Train Loss: {avg_epoch_loss:.4f}, Val Loss: {val_loss:.4f}")

            # Early stopping チェック
            # 相対的改善率 = (best_val_loss - val_loss) / best_val_loss
            if best_val_loss == float('inf'):
                # 初回は無条件で記録
                improvement_rate = float('inf')
            else:
                improvement_rate = (best_val_loss - val_loss) / best_val_loss

            # min_delta（デフォルト0.001 = 0.1%）以上の相対改善があれば記録更新
            if improvement_rate > min_delta:
                best_val_loss = val_loss
                patience_counter = 0
                # ベストモデルを保存
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                print(f"  → New best validation loss! (improvement rate: {improvement_rate*100:.2f}%)")
            else:
                patience_counter += 1
                if best_val_loss == float('inf'):
                    print(f"  → No improvement. Patience: {patience_counter}/{patience}")
                else:
                    print(f"  → No improvement (rate: {improvement_rate*100:.2f}% < {min_delta*100:.2f}%). "
                          f"Patience: {patience_counter}/{patience}")

                if patience_counter >= patience:
                    print(f"\nEarly stopping triggered! No improvement for {patience} epochs.")
                    print(f"Best validation loss: {best_val_loss:.4f}")
                    # ベストモデルの重みを復元
                    if best_model_state is not None:
                        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
                        print("Restored best model weights.")
                    break
        else:
            print(f"Epoch [{epoch+1}/{num_epochs}] completed. "
                  f"Average Loss: {avg_epoch_loss:.4f}")

        print("-" * 60)

    print("Training completed!")
    if val_loader is not None and best_val_loss < float('inf'):
        print(f"Best validation loss: {best_val_loss:.4f}")
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
    from model import JointModel

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
