#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Custom loss functions for CAFA-6 competition.

This module provides:
- SampledBCEWithLogitsLoss: BCE loss with negative sampling for class imbalance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SampledBCEWithLogitsLoss(nn.Module):
    """
    Binary Cross Entropy with Logits Loss using negative sampling.

    Instead of computing loss over all GO terms (which can be 30k+), this loss:
    1. Uses ALL positive GO terms for each protein
    2. Randomly samples K negative GO terms for each protein
    3. Computes BCE loss only on the sampled subset

    This addresses class imbalance by reducing the dominance of negative samples.

    Parameters
    ----------
    num_negative_samples : int
        Number of negative samples to use per protein (default: 1000)
        Recommended range: 500-5000
    reduction : str
        Specifies the reduction to apply: 'mean' | 'sum' | 'none' (default: 'mean')

    Notes
    -----
    - Positive samples are always included (no sampling)
    - Negative sampling is random and changes each forward pass
    - More efficient than full BCE when num_go_terms >> num_positive_per_protein
    """

    def __init__(self, num_negative_samples: int = 1000, reduction: str = 'mean'):
        super().__init__()
        self.num_negative_samples = num_negative_samples
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute sampled BCE loss.

        Parameters
        ----------
        logits : torch.Tensor
            Predicted logits, shape = (batch_size, num_go_terms)
        targets : torch.Tensor
            Ground truth labels (0 or 1), shape = (batch_size, num_go_terms)

        Returns
        -------
        torch.Tensor
            Computed loss (scalar if reduction='mean' or 'sum')
        """
        batch_size, num_go = logits.shape
        device = logits.device

        # Storage for batch losses
        batch_losses = []

        # Process each sample in the batch
        for i in range(batch_size):
            sample_logits = logits[i]  # (num_go,)
            sample_targets = targets[i]  # (num_go,)

            # Find positive and negative indices
            pos_indices = torch.where(sample_targets == 1)[0]
            neg_indices = torch.where(sample_targets == 0)[0]

            num_positives = len(pos_indices)
            num_negatives = len(neg_indices)

            # Sample negative indices
            if num_negatives > self.num_negative_samples:
                # Randomly sample K negatives
                sampled_neg_indices = neg_indices[
                    torch.randperm(num_negatives, device=device)[:self.num_negative_samples]
                ]
            else:
                # Use all negatives if fewer than K
                sampled_neg_indices = neg_indices

            # Combine positive and sampled negative indices
            selected_indices = torch.cat([pos_indices, sampled_neg_indices])

            # Get logits and targets for selected indices
            selected_logits = sample_logits[selected_indices]
            selected_targets = sample_targets[selected_indices]

            # Compute BCE loss for this sample
            sample_loss = F.binary_cross_entropy_with_logits(
                selected_logits,
                selected_targets,
                reduction='mean'
            )

            batch_losses.append(sample_loss)

        # Stack batch losses
        loss_tensor = torch.stack(batch_losses)

        # Apply reduction
        if self.reduction == 'mean':
            return loss_tensor.mean()
        elif self.reduction == 'sum':
            return loss_tensor.sum()
        else:  # 'none'
            return loss_tensor

    def extra_repr(self) -> str:
        """String representation for debugging."""
        return f'num_negative_samples={self.num_negative_samples}, reduction={self.reduction}'
