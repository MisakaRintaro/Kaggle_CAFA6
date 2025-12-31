#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main script for CAFA-6 competition.

This script orchestrates the entire pipeline:
1. Load data (FASTA, GO OBO, labels)
2. Create embeddings (protein and GO)
3. Train the JointModel
4. Make predictions on test data
5. Generate submission file
"""

import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel

# Import our modules
from config import (
    PATH_GO_OBO,
    PATH_TRAIN_FASTA,
    PATH_TRAIN_TERMS,
    PATH_TEST_FASTA,
    PATH_IA,
    PATH_TRAIN_PROTEIN_EMB,
    PATH_TEST_PROTEIN_EMB,
    PATH_GO_EMBEDDINGS,
    PATH_TRAINED_MODEL,
    PATH_SUBMISSION,
    ESM_MODEL_NAME,
    BIOMEDBERT_MODEL_NAME,
    PROTEIN_BATCH_SIZE,
    TRAIN_BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    JOINT_DIM,
    DROPOUT,
    TOP_K_PREDICTIONS,
    ADD_TEXT_DESCRIPTIONS,
    DEV_TEST,
    DEV_TEST_MAX_BATCHES,
    ENABLE_VALIDATION,
    VAL_SPLIT_RATIO,
    VAL_RANDOM_SEED,
    VAL_STRATIFY_BY_LABEL_COUNT,
    ENABLE_TRAINING,
    EARLY_STOPPING_PATIENCE,
    EARLY_STOPPING_MIN_DELTA,
    USE_POS_WEIGHT,
    POS_WEIGHT_CLIP_MAX,
    USE_NEGATIVE_SAMPLING,
    NUM_NEGATIVE_SAMPLES,
    get_device,
    get_config_summary
)

from data_loader import (
    load_fasta,
    parse_go_obo,
    load_train_labels
)

from protein_embedding import (
    encode_all_sequences_with_esm,
    save_emb_dict,
    load_emb_dict
)

from go_embedding import (
    create_go_embedding_matrix,
    save_go_embeddings,
    load_go_embeddings,
    reconstruct_go_matrix_from_dict
)

from model import (
    create_protein_go_label_matrix,
    ProteinGODataset,
    JointModel,
    compute_pos_weight
)

from training import (
    train_model,
    save_model,
    load_model
)

from prediction import (
    create_test_loader,
    predict,
    create_submission_file
)

from hierarchical_postprocess import (
    apply_hierarchical_postprocess_to_all_proteins
)

from evaluation import (
    split_train_validation,
    load_ia_weights,
    evaluate_predictions,
    print_evaluation_results
)


def main():
    """
    Main pipeline for CAFA-6 competition
    """
    # Print configuration
    print(get_config_summary())

    # Get device
    device = get_device()
    print(f"\nUsing device: {device}\n")

    # ========================================================================
    # Step 1: Load GO ontology and training labels
    # ========================================================================
    print("="*80)
    print("Step 1: Loading GO ontology and training labels")
    print("="*80)

    child_to_parents, go_terms = parse_go_obo(str(PATH_GO_OBO))
    print(f"Loaded {len(go_terms)} GO terms from OBO file")

    train_label_df = load_train_labels(str(PATH_TRAIN_TERMS))
    print(f"Loaded {len(train_label_df)} training labels")

    # ========================================================================
    # Step 2: Create GO embeddings using BiomedBERT
    # ========================================================================
    print("\n" + "="*80)
    print("Step 2: Creating GO embeddings")
    print("="*80)

    # Check if GO embeddings already exist
    if PATH_GO_EMBEDDINGS.exists():
        print(f"Loading existing GO embeddings from {PATH_GO_EMBEDDINGS}")
        go_embeddings_dict = load_go_embeddings(str(PATH_GO_EMBEDDINGS))
        go_emb_matrix, go_id_list = reconstruct_go_matrix_from_dict(go_embeddings_dict)
    else:
        print("Creating new GO embeddings...")
        # Load BiomedBERT
        bert_tokenizer = AutoTokenizer.from_pretrained(BIOMEDBERT_MODEL_NAME)
        bert_model = AutoModel.from_pretrained(BIOMEDBERT_MODEL_NAME)
        bert_model = bert_model.to(device)
        bert_model.eval()

        # Create GO embeddings
        go_emb_matrix, go_id_list, go_embeddings_dict = create_go_embedding_matrix(
            go_terms,
            train_label_df,
            bert_tokenizer,
            bert_model,
            device
        )

        # Save GO embeddings
        save_go_embeddings(go_embeddings_dict, str(PATH_GO_EMBEDDINGS))

        # Free memory
        del bert_tokenizer, bert_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"GO embedding matrix shape: {go_emb_matrix.shape}")

    # ========================================================================
    # Step 3: Load and embed training protein sequences
    # ========================================================================
    print("\n" + "="*80)
    print("Step 3: Loading and embedding training sequences")
    print("="*80)

    # Check if training protein embeddings already exist
    if PATH_TRAIN_PROTEIN_EMB.exists():
        print(f"Loading existing training protein embeddings from {PATH_TRAIN_PROTEIN_EMB}")
        train_protein_emb_dict = load_emb_dict(str(PATH_TRAIN_PROTEIN_EMB))
    else:
        print("Creating new training protein embeddings...")
        # Load training FASTA
        train_seq_dict = load_fasta(str(PATH_TRAIN_FASTA), mode="train")

        # Load ESM-2 model
        esm_tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL_NAME)
        esm_model = AutoModel.from_pretrained(ESM_MODEL_NAME)
        esm_model = esm_model.to(device)
        esm_model.eval()

        # Create protein embeddings
        train_protein_emb_dict = encode_all_sequences_with_esm(
            train_seq_dict,
            esm_tokenizer,
            esm_model,
            device,
            batch_size=PROTEIN_BATCH_SIZE,
            max_batches=DEV_TEST_MAX_BATCHES if DEV_TEST else None
        )

        # Save protein embeddings
        save_emb_dict(train_protein_emb_dict, str(PATH_TRAIN_PROTEIN_EMB))

        # Free memory
        del esm_tokenizer, esm_model, train_seq_dict
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"Number of training proteins: {len(train_protein_emb_dict)}")

    # ========================================================================
    # Step 4: Create training and validation datasets
    # ========================================================================
    print("\n" + "="*80)
    print("Step 4: Creating training and validation datasets")
    print("="*80)

    # Create full label matrix
    X_full, y_full, protein_ids = create_protein_go_label_matrix(
        train_protein_emb_dict,
        train_label_df,
        go_id_list
    )

    # Split into train/validation if enabled
    if ENABLE_VALIDATION:
        print("Splitting data into train/validation sets...")
        train_protein_ids, val_protein_ids = split_train_validation(
            protein_ids,
            train_label_df,
            test_size=VAL_SPLIT_RATIO,
            random_state=VAL_RANDOM_SEED,
            stratify_by_label_count=VAL_STRATIFY_BY_LABEL_COUNT
        )

        # Create indices for train/val split
        train_indices = [i for i, pid in enumerate(protein_ids) if pid in train_protein_ids]
        val_indices = [i for i, pid in enumerate(protein_ids) if pid in val_protein_ids]

        X_train, y_train = X_full[train_indices], y_full[train_indices]
        X_val, y_val = X_full[val_indices], y_full[val_indices]

        print(f"Train set: {len(train_indices)} proteins")
        print(f"Validation set: {len(val_indices)} proteins")

        # Create validation dataset and loader
        val_dataset = ProteinGODataset(X_val, y_val)
        val_loader = DataLoader(
            val_dataset,
            batch_size=TRAIN_BATCH_SIZE,
            shuffle=False,
            num_workers=0
        )
        val_protein_ids_list = [protein_ids[i] for i in val_indices]
    else:
        print("Validation disabled - using all data for training")
        X_train, y_train = X_full, y_full
        val_loader = None
        val_protein_ids_list = None
        y_val = None

    # Create training dataset and loader
    train_dataset = ProteinGODataset(X_train, y_train)
    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    print(f"Training dataset created with {len(train_dataset)} samples")

    # ========================================================================
    # Step 5: Model Training and Validation
    # ========================================================================
    print("\n" + "="*80)
    print("Step 5: Model Training and Validation")
    print("="*80)

    # Check training mode
    if ENABLE_TRAINING:
        print("Training mode: ENABLED")
        # Check if trained model already exists
        if PATH_TRAINED_MODEL.exists():
            print(f"Warning: Model file already exists at {PATH_TRAINED_MODEL}")
            print("Training will overwrite the existing model.")

        print("Training new model...")
        # Get dimensions
        input_dim_protein = X_train.shape[1]
        input_dim_go = go_emb_matrix.shape[1]

        # Initialize model
        model = JointModel(
            input_dim_protein=input_dim_protein,
            input_dim_go=input_dim_go,
            joint_dim=JOINT_DIM,
            go_raw_emb=go_emb_matrix
        )

        # Compute pos_weight for class imbalance handling (if not using negative sampling)
        pos_weight = None
        if USE_POS_WEIGHT and not USE_NEGATIVE_SAMPLING:
            print("\nComputing pos_weight for class imbalance handling...")
            pos_weight = compute_pos_weight(
                y_train,
                clip_max=POS_WEIGHT_CLIP_MAX
            )

        # Train model
        history = train_model(
            model,
            train_loader,
            device,
            num_epochs=NUM_EPOCHS,
            learning_rate=LEARNING_RATE,
            val_loader=val_loader,
            patience=EARLY_STOPPING_PATIENCE,
            min_delta=EARLY_STOPPING_MIN_DELTA,
            pos_weight=pos_weight,
            use_negative_sampling=USE_NEGATIVE_SAMPLING,
            num_negative_samples=NUM_NEGATIVE_SAMPLES
        )

        # Save model
        save_model(
            model,
            str(PATH_TRAINED_MODEL),
            go_id_list,
            history
        )
    else:
        print("Training mode: DISABLED")
        print("Loading existing trained model...")

        # Check if model file exists
        if not PATH_TRAINED_MODEL.exists():
            raise FileNotFoundError(
                f"Model file not found at {PATH_TRAINED_MODEL}\n"
                f"Please set ENABLE_TRAINING=True in config to train a new model."
            )

        # Load existing model
        model, loaded_go_id_list, history = load_model(
            str(PATH_TRAINED_MODEL),
            go_emb_matrix,
            device
        )
        # Verify GO ID list matches
        if loaded_go_id_list != go_id_list:
            raise ValueError(
                f"GO ID list mismatch!\n"
                f"Expected {len(go_id_list)} terms, but loaded model has {len(loaded_go_id_list)} terms.\n"
                f"Please retrain the model with ENABLE_TRAINING=True."
            )

    # ========================================================================
    # Step 5 (continued): Validation Evaluation
    # ========================================================================
    if ENABLE_VALIDATION and val_loader is not None:
        print("\n" + "-"*80)
        print("Evaluating on validation set")
        print("-"*80)

        # Make predictions on validation set
        model.eval()
        y_val_pred_list = []

        with torch.no_grad():
            for batch in val_loader:
                x_batch = batch['x'].to(device)
                # Forward pass to get logits
                logits = model(x_batch)
                # Convert logits to probabilities using sigmoid
                y_pred_proba = torch.sigmoid(logits)
                y_val_pred_list.append(y_pred_proba.cpu().numpy())

        y_val_pred = np.vstack(y_val_pred_list)
        print(f"Validation predictions shape: {y_val_pred.shape}")

        # Load IA weights
        print(f"Loading IA weights from {PATH_IA}...")
        ia_weights = load_ia_weights(str(PATH_IA))

        # --- Evaluate without hierarchical postprocessing ---
        print("\n" + "="*60)
        print("Evaluating WITHOUT hierarchical postprocessing...")
        print("="*60)

        metrics_before = evaluate_predictions(
            y_val,
            y_val_pred,
            go_id_list,
            ia_weights,
            threshold=0.5
        )

        print("\n" + "-"*60)
        print("VALIDATION RESULTS (Before Hierarchical Postprocessing)")
        print("-"*60)
        print_evaluation_results(metrics_before)
        print("-"*60)

        # --- Apply hierarchical postprocessing ---
        print("\n" + "="*60)
        print("Applying hierarchical postprocessing to validation predictions...")
        print("="*60)

        # Convert validation predictions to dict format for postprocessing
        val_predictions_dict = {}
        for i, protein_id in enumerate(val_protein_ids_list):
            # Get top predictions for this protein
            protein_scores = y_val_pred[i]
            # Create list of (go_id, score) tuples
            go_scores = [(go_id_list[j], float(protein_scores[j])) for j in range(len(go_id_list))]
            val_predictions_dict[protein_id] = go_scores

        # Apply hierarchical postprocessing (with fixed parameters)
        val_predictions_dict = apply_hierarchical_postprocess_to_all_proteins(
            val_predictions_dict,
            child_to_parents
        )

        # Convert back to matrix format
        y_val_pred_post = np.zeros_like(y_val_pred)
        for i, protein_id in enumerate(val_protein_ids_list):
            go_scores_dict = {go_id: score for go_id, score in val_predictions_dict[protein_id]}
            for j, go_id in enumerate(go_id_list):
                y_val_pred_post[i, j] = go_scores_dict.get(go_id, 0.0)

        # Evaluate after postprocessing
        print("Evaluating WITH hierarchical postprocessing...")
        metrics_after = evaluate_predictions(
            y_val,
            y_val_pred_post,
            go_id_list,
            ia_weights,
            threshold=0.5
        )

        print("\n" + "-"*60)
        print("VALIDATION RESULTS (After Hierarchical Postprocessing)")
        print("-"*60)
        print_evaluation_results(metrics_after)
        print("-"*60)

        # Compare metrics
        print("\n" + "="*60)
        print("COMPARISON: Impact of Hierarchical Postprocessing")
        print("="*60)
        print(f"IA-weighted Fmax: {metrics_before['ia_fmax']:.4f} → {metrics_after['ia_fmax']:.4f} "
              f"(Δ = {metrics_after['ia_fmax'] - metrics_before['ia_fmax']:+.4f})")
        print(f"F1 Score:         {metrics_before['f1']:.4f} → {metrics_after['f1']:.4f} "
              f"(Δ = {metrics_after['f1'] - metrics_before['f1']:+.4f})")
        print("="*60)

        # Save both metrics
        all_metrics = {
            'before_postprocessing': metrics_before,
            'after_postprocessing': metrics_after,
            'improvement': {
                'ia_fmax': float(metrics_after['ia_fmax'] - metrics_before['ia_fmax']),
                'f1': float(metrics_after['f1'] - metrics_before['f1'])
            }
        }
        metrics_path = PATH_TRAINED_MODEL.parent / f"{PATH_TRAINED_MODEL.stem}_val_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        print(f"\nValidation metrics saved to {metrics_path}")

    # ========================================================================
    # Step 6: Test Inference
    # ========================================================================
    print("\n" + "="*80)
    print("Step 6: Test Inference (Load, Embed, Predict)")
    print("="*80)
    print("\n" + "-"*80)
    print("Loading and embedding test sequences")
    print("-"*80)

    # Check if test protein embeddings already exist
    if PATH_TEST_PROTEIN_EMB.exists():
        print(f"Loading existing test protein embeddings from {PATH_TEST_PROTEIN_EMB}")
        test_protein_emb_dict = load_emb_dict(str(PATH_TEST_PROTEIN_EMB))
    else:
        print("Creating new test protein embeddings...")
        # Load test FASTA
        test_seq_dict = load_fasta(str(PATH_TEST_FASTA), mode="test")

        # Load ESM-2 model
        esm_tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL_NAME)
        esm_model = AutoModel.from_pretrained(ESM_MODEL_NAME)
        esm_model = esm_model.to(device)
        esm_model.eval()

        # Create protein embeddings
        test_protein_emb_dict = encode_all_sequences_with_esm(
            test_seq_dict,
            esm_tokenizer,
            esm_model,
            device,
            batch_size=PROTEIN_BATCH_SIZE,
            max_batches=DEV_TEST_MAX_BATCHES if DEV_TEST else None
        )

        # Save protein embeddings
        save_emb_dict(test_protein_emb_dict, str(PATH_TEST_PROTEIN_EMB))

        # Free memory
        del esm_tokenizer, esm_model, test_seq_dict
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"Number of test proteins: {len(test_protein_emb_dict)}")

    print("\n" + "-"*80)
    print("Making predictions on test data")
    print("-"*80)

    # Create test DataLoader
    test_loader = create_test_loader(
        test_protein_emb_dict,
        batch_size=TRAIN_BATCH_SIZE
    )

    # Run inference
    predictions = predict(
        model,
        test_loader,
        go_id_list,
        device,
        top_k=TOP_K_PREDICTIONS
    )

    # ========================================================================
    # Step 7: Postprocessing and Submission
    # ========================================================================
    print("\n" + "="*80)
    print("Step 7: Postprocessing and Submission")
    print("="*80)
    print("\n" + "-"*80)
    print("Applying hierarchical postprocessing")
    print("-"*80)

    predictions = apply_hierarchical_postprocess_to_all_proteins(
        predictions,
        child_to_parents
    )

    print("\n" + "-"*80)
    print("Creating submission file")
    print("-"*80)

    create_submission_file(
        predictions,
        str(PATH_SUBMISSION),
        add_text_descriptions=ADD_TEXT_DESCRIPTIONS
    )

    print("\n" + "="*80)
    print("Pipeline completed successfully!")
    print("="*80)
    print(f"Submission file saved to: {PATH_SUBMISSION}")


if __name__ == "__main__":
    main()
