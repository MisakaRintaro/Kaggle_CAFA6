#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Configuration file for CAFA-6 competition.

This module provides centralized configuration for:
- Data paths
- Model paths
- Output paths
- Hyperparameters
"""

from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent.parent
INPUT_DIR = BASE_DIR / "input" / "cafa-6-protein-function-prediction"
OUTPUT_DIR = BASE_DIR / "output"
MODEL_DIR = OUTPUT_DIR / "models"
EMBEDDING_DIR = OUTPUT_DIR / "embeddings"

# Create output directories if they don't exist
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
MODEL_DIR.mkdir(exist_ok=True, parents=True)
EMBEDDING_DIR.mkdir(exist_ok=True, parents=True)

# ============================================================================
# Development Mode
# ============================================================================

# Set to True for quick testing with limited data
DEV_TEST = False

# Number of batches to process in DEV_TEST mode (for protein embedding)
DEV_TEST_MAX_BATCHES = 100


# ============================================================================
# Input Data Paths
# ============================================================================

# Sample submission and IA weights
PATH_SAMPLE_SUB = INPUT_DIR / "sample_submission.tsv"
PATH_IA = INPUT_DIR / "IA.tsv"  # Information Accretion weights

# GO ontology
PATH_GO_OBO = INPUT_DIR / "Train" / "go-basic.obo"

# Training data
PATH_TRAIN_FASTA = INPUT_DIR / "Train" / "train_sequences.fasta"
PATH_TRAIN_TERMS = INPUT_DIR / "Train" / "train_terms.tsv"
PATH_TRAIN_TAXONOMY = INPUT_DIR / "Train" / "train_taxonomy.tsv"

# Test data
PATH_TEST_FASTA = INPUT_DIR / "Test (Targets)" / "testsuperset.fasta"
PATH_TEST_TAXON = INPUT_DIR / "Test (Targets)" / "testsuperset-taxon-list.tsv"


# ============================================================================
# Model Paths (Hugging Face)
# ============================================================================

# ESM-2 model for protein sequence embedding
ESM_MODEL_NAME = "facebook/esm2_t12_35M_UR50D"

# BiomedBERT model for GO term text embedding
BIOMEDBERT_MODEL_NAME = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"


# ============================================================================
# Output Paths
# ============================================================================

def _get_output_paths():
    """
    Get output paths based on DEV_TEST mode.

    Returns paths with '_dev' suffix when DEV_TEST is True.
    """
    suffix = "_dev" if DEV_TEST else ""

    return {
        'train_protein_emb': EMBEDDING_DIR / f"train_protein_embeddings{suffix}.pt",
        'test_protein_emb': EMBEDDING_DIR / f"test_protein_embeddings{suffix}.pt",
        'go_embeddings': EMBEDDING_DIR / f"go_embeddings{suffix}.pt",
        'trained_model': MODEL_DIR / f"joint_model{suffix}.pt",
        'submission': OUTPUT_DIR / f"submission{suffix}.tsv"
    }

# Get paths based on DEV_TEST mode
_paths = _get_output_paths()

# Protein embeddings
PATH_TRAIN_PROTEIN_EMB = _paths['train_protein_emb']
PATH_TEST_PROTEIN_EMB = _paths['test_protein_emb']

# GO embeddings
PATH_GO_EMBEDDINGS = _paths['go_embeddings']

# Trained model
PATH_TRAINED_MODEL = _paths['trained_model']

# Submission file
PATH_SUBMISSION = _paths['submission']


# ============================================================================
# Model Hyperparameters
# ============================================================================

# Protein embedding parameters
MAX_AA_LENGTH = 1024
PROTEIN_BATCH_SIZE = 64  # For ESM-2 inference

# GO embedding parameters
GO_MAX_TEXT_LENGTH = 256

# Joint model architecture
JOINT_DIM = 256  # Dimension of joint embedding space
DROPOUT = 0.1  # Dropout rate for regularization in cross-attention layers

# Training parameters
ENABLE_TRAINING = True  # Whether to train model (False = load existing model)
TRAIN_BATCH_SIZE = 16
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3
EARLY_STOPPING_PATIENCE = 3  # Number of epochs with no improvement to wait before stopping
EARLY_STOPPING_MIN_DELTA = 0.01  # Minimum relative improvement rate (1% = 0.01)

# Class imbalance handling
USE_POS_WEIGHT = True  # Whether to use pos_weight for handling class imbalance
POS_WEIGHT_CLIP_MAX = 100.0  # Maximum value for pos_weight to prevent extreme values

# Negative sampling
USE_NEGATIVE_SAMPLING = True  # Whether to use negative sampling (alternative to pos_weight)
NUM_NEGATIVE_SAMPLES = 100  # Number of negative samples per protein (recommended: 500-5000)
# Note: USE_NEGATIVE_SAMPLING and USE_POS_WEIGHT are mutually exclusive
# If both are True, USE_NEGATIVE_SAMPLING takes precedence

# Prediction parameters
TOP_K_PREDICTIONS = 100  # Number of GO terms to predict per protein
ADD_TEXT_DESCRIPTIONS = False  # Whether to add text descriptions in submission

# Evaluation parameters
ENABLE_VALIDATION = True  # Whether to perform validation during training
VAL_SPLIT_RATIO = 0.2  # Validation set ratio
VAL_RANDOM_SEED = 42  # Random seed for train/val split
VAL_STRATIFY_BY_LABEL_COUNT = True  # Whether to stratify by label count


# ============================================================================
# Device Configuration
# ============================================================================

import torch

def get_device() -> torch.device:
    """
    Get the best available device (cuda > mps > cpu)

    Returns
    -------
    torch.device
        The device to use for computation
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


# ============================================================================
# Helper Functions
# ============================================================================

def get_config_summary() -> str:
    """
    Get a summary of the current configuration

    Returns
    -------
    str
        Configuration summary
    """
    device = get_device()

    summary = f"""
CAFA-6 Configuration Summary
{'='*60}

Mode: {"DEVELOPMENT (Limited Data)" if DEV_TEST else "PRODUCTION (Full Data)"}
Device: {device}

Input Paths:
  - GO OBO: {PATH_GO_OBO}
  - Train FASTA: {PATH_TRAIN_FASTA}
  - Train Terms: {PATH_TRAIN_TERMS}
  - Test FASTA: {PATH_TEST_FASTA}

Output Paths:
  - Train Protein Embeddings: {PATH_TRAIN_PROTEIN_EMB}
  - Test Protein Embeddings: {PATH_TEST_PROTEIN_EMB}
  - GO Embeddings: {PATH_GO_EMBEDDINGS}
  - Trained Model: {PATH_TRAINED_MODEL}
  - Submission File: {PATH_SUBMISSION}

Model Configuration:
  - ESM Model: {ESM_MODEL_NAME}
  - BiomedBERT Model: {BIOMEDBERT_MODEL_NAME}
  - Joint Embedding Dimension: {JOINT_DIM}

Training Configuration:
  - Batch Size: {TRAIN_BATCH_SIZE}
  - Number of Epochs: {NUM_EPOCHS}
  - Learning Rate: {LEARNING_RATE}

Prediction Configuration:
  - Top-K Predictions: {TOP_K_PREDICTIONS}
  - Add Text Descriptions: {ADD_TEXT_DESCRIPTIONS}

Development Mode:
  - DEV_TEST: {DEV_TEST}
  - Max Batches (if DEV_TEST): {DEV_TEST_MAX_BATCHES}

{'='*60}
"""
    return summary


if __name__ == "__main__":
    # Print configuration summary when run as script
    print(get_config_summary())
