#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data loading utilities for CAFA-6 competition.

This module provides functions for loading:
- FASTA files (protein sequences)
- GO ontology (go-basic.obo)
- Training labels
"""

from pathlib import Path
from typing import Dict, List, Tuple, Any
import pandas as pd


def preview_fasta(path, max_records=3):
    print(f"=== {path} ===")
    count = 0
    with open(path) as f:
        for line in f:
            if line.startswith(">"):
                if count >= max_records:
                    break
                print(line.strip())
                count += 1
            elif count > 0 and count <= max_records:
                print(line.strip()[:80], "..." if len(line.strip()) > 80 else "")
    print(f"Total records previewed: {count}")
    print()


def load_fasta(path, mode, max_aa_len=1024):
    print(f"=== {path} ===")
    protainID_to_sequence = {}

    current_header = None
    current_seq = []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith(">"):
                # Save previous sequence if exists
                if current_header is not None:
                    seq_str = "".join(current_seq)
                    if len(seq_str) <= max_aa_len:
                        if mode == "train":
                            protein_id = current_header.split()[0]
                            taxon_id = current_header.split("TaxID=")[1] if "TaxID=" in current_header else "0"
                            protainID_to_sequence[(protein_id, taxon_id)] = seq_str
                        elif mode == "test":
                            protein_id = current_header.split()[0]
                            taxon_id = current_header.split("TaxID=")[1] if "TaxID=" in current_header else "0"
                            protainID_to_sequence[(protein_id, taxon_id)] = seq_str

                # Start new sequence
                current_header = line[1:]  # Remove ">"
                current_seq = []
            else:
                current_seq.append(line)

        # Save last sequence
        if current_header is not None:
            seq_str = "".join(current_seq)
            if len(seq_str) <= max_aa_len:
                if mode == "train":
                    protein_id = current_header.split()[0]
                    taxon_id = current_header.split("TaxID=")[1] if "TaxID=" in current_header else "0"
                    protainID_to_sequence[(protein_id, taxon_id)] = seq_str
                elif mode == "test":
                    protein_id = current_header.split()[0]
                    taxon_id = current_header.split("TaxID=")[1] if "TaxID=" in current_header else "0"
                    protainID_to_sequence[(protein_id, taxon_id)] = seq_str

    print(f"Loaded {len(protainID_to_sequence)} sequences")
    return protainID_to_sequence


def parse_go_obo(filepath: str) -> Tuple[Dict[str, List[str]], Dict[str, Dict[str, Any]]]:
    """
    go-basic.obo ファイルを読み込んで、ヘッダー情報と Term 情報をパースする関数。

    Parameters
    ----------
    filepath : str
        go-basic.obo ファイルのパス

    Returns
    -------
    meta : Dict[str, List[str]]
        ヘッダー部分の key: value をまとめた辞書
        例: {"format-version": ["1.2"], "data-version": ["..."], ...}

    terms : Dict[str, Dict[str, Any]]
        各 GO term の情報を格納した辞書
        キー: GO term ID (例: "GO:0000001")
        値: {
              "id": "GO:0000001",
              "name": "mitochondrion inheritance",
              "namespace": "biological_process",
              "def": "...",
              "alt_id": [...],
              "synonym": [...],
              "is_a": [...],
              "is_obsolete": False,
              "replaced_by": [...],
              "raw": {元の key: value を全部入れた辞書}
          }
    """
    meta = {}
    terms = {}

    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    current_section = "header"
    current_term = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # [Term] または [Typedef] などのセクション開始
        if line.startswith("["):
            if line == "[Term]":
                current_section = "term"
                # 前の Term があれば保存
                if current_term and current_term["id"]:
                    terms[current_term["id"]] = current_term

                # 新しい Term の初期化
                current_term = {
                    "id": None,
                    "name": None,
                    "namespace": None,
                    "def": None,
                    "alt_id": [],
                    "synonym": [],
                    "is_a": [],
                    "is_obsolete": False,
                    "replaced_by": [],
                    # すべての key: value をそのまま保持しておきたい場合
                    "raw": {}
                }
            else:
                current_section = "other"
            continue

        # key: value 形式の行をパース
        if ":" not in line:
            continue

        key, _, val = line.partition(":")
        key = key.strip()
        value = val.strip()

        if current_section == "header":
            # ヘッダー部分
            if key not in meta:
                meta[key] = []
            meta[key].append(value)

        elif current_section == "term":
            # Term 部分
            if current_term is not None:
                # raw に全て格納
                if key not in current_term["raw"]:
                    current_term["raw"][key] = []
                current_term["raw"][key].append(value)

                # よく使う項目を個別に抽出
                if key == "id":
                    current_term["id"] = value
                elif key == "name":
                    current_term["name"] = value
                elif key == "namespace":
                    current_term["namespace"] = value
                elif key == "def":
                    current_term["def"] = value
                elif key == "alt_id":
                    current_term["alt_id"].append(value)
                elif key == "synonym":
                    current_term["synonym"].append(value)
                elif key == "is_a":
                    current_term["is_a"].append(value)
                elif key == "is_obsolete":
                    # "true" / "false" が来るので bool に変換
                    current_term["is_obsolete"] = value.lower() == "true"
                elif key == "replaced_by":
                    current_term["replaced_by"].append(value)

    # ファイル末尾の Term も保存
    if current_term and current_term["id"]:
        terms[current_term["id"]] = current_term

    return meta, terms


def load_train_labels(filepath: str) -> pd.DataFrame:
    """Load training labels from train_terms.tsv"""
    return pd.read_csv(filepath, sep="\t")
