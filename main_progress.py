# ライブラリのインポート

import numpy as np
import pandas as pd
import os
import display
from collections import Counter
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Dict, List, Tuple, Any

def main():
    # グローバル変数の定義
    # 各種パスはkaggleアップロード時に適宜変更すること
    BATCH_SIZE = 16
    MAX_ESM2_TIME = 5

    ESM2_MODEL_DIR = "../model/esm2_t12_35M_UR50D"
    BERT_MODEL_DIR = "../model/BiomedNLP-BiomedBERT-base-uncased-abstract"

    # Embedding save paths
    TRAIN_SAVE_PATH = "../output/train_embeddings.pt"
    TEST_SAVE_PATH = "../output/test_embeddings.pt"
    GO_EMBEDDINGS_SAVE_PATH = "../output/go_embeddings.pt"  # GO埋め込み辞書の保存先

    # Model save paths
    MODEL_SAVE_PATH = "../output/joint_model.pth"

    # Training hyperparameters
    NUM_EPOCHS = 10
    LEARNING_RATE = 1e-3

    # Step0: Check the data
    # ==== File paths ====

    # Base directory
    INPUT_DIR = "../input/cafa-6-protein-function-prediction"

    # TSV files
    PATH_SAMPLE_SUB   = f"{INPUT_DIR}/sample_submission.tsv"
    PATH_IA           = f"{INPUT_DIR}/IA.tsv"
    PATH_TEST_TAXON   = f"{INPUT_DIR}/Test/testsuperset-taxon-list.tsv"
    PATH_TRAIN_TERMS  = f"{INPUT_DIR}/Train/train_terms.tsv"
    PATH_TRAIN_TAXON  = f"{INPUT_DIR}/Train/train_taxonomy.tsv"

    # FASTA files
    PATH_TEST_FASTA   = f"{INPUT_DIR}/Test/testsuperset.fasta"
    PATH_TRAIN_FASTA  = f"{INPUT_DIR}/Train/train_sequences.fasta"

    # OBO file
    PATH_GO_OBO       = f"{INPUT_DIR}/Train/go-basic.obo"

    print("=== sample_submission.tsv ===")

    # 列が3つの行も4つの行もあるので、列名をあらかじめ4つ定義しておく
    cols = ["protein_id", "kind_or_term", "score", "description"]

    sample_df = pd.read_csv(
        PATH_SAMPLE_SUB,
        sep="\t",          # 区切りはタブ
        header=None,       # ファイル先頭行をヘッダとして使わない
        names=cols,        # 自分で列名を指定
        engine="python",   # 行ごとに列数が違っても柔軟に解釈してくれる
    )

    display(sample_df.head(10))

    print("\n=== IA.tsv ===")
    IA_df = pd.read_csv(
        PATH_IA, 
        sep="\t", 
        header=None,
        names=["GO", "weight"]
    )
    display(IA_df.head())

    print("\n=== testsuperset-taxon-list.tsv ===")
    test_ID2Species_df = pd.read_csv(PATH_TEST_TAXON, sep="\t")
    display(test_ID2Species_df.head())

    print("\n=== train_terms.tsv ===")
    train_label_df = pd.read_csv(PATH_TRAIN_TERMS, sep="\t")
    display(train_label_df.head())

    print("\n=== train_taxonomy.tsv ===")
    train_Protain2TaxonID_df = pd.read_csv(
        PATH_TRAIN_TAXON, 
        sep="\t", 
        header=None,
        names=["EntryID", "TaxonID"]
    )
    display(train_Protain2TaxonID_df.head())

    def preview_fasta(path, max_records=3):
        print(f"=== {path} ===")
        count = 0
        with open(path) as f:
            header = None
            seq = ""
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    if header is not None:
                        print(header)
                        print(seq[:80], "\n")
                        count += 1
                        if count >= max_records:
                            break
                    header = line
                    seq = ""
                else:
                    seq += line

            if count < max_records and header:
                print(header)
                print(seq[:80], "\n")
    preview_fasta(PATH_TRAIN_FASTA)
    preview_fasta(PATH_TEST_FASTA)

    def load_fasta(path, mode, max_aa_len=1024):
        print(f"=== {path} ===")
        protainID_to_sequence = {}
        
        with open(path) as f:
            protain_info = None
            seq = ""
            
            for line in f:
                line = line.strip()
                
                if line.startswith(">"):
                    # 新しい>から始まる行が登場したら、
                    # それまでのpritain_id, taxon_id, seqをprotainID_to_sequenceに記録
                    # そしてseqをリセット
                    if protain_info is not None:
                        # max_aa_lenより大きい場合はカット
                        if len(seq) <= max_aa_len:
                            seq = seq[:max_aa_len]
                        protainID_to_sequence[protain_info] = seq
                        seq = ""

                    # trainとtestで微妙に形式が違うので場合分け
                    # 正規表現で取り出した方がいいかも
                    if mode == "train":
                        protain_id = line.split("|")[1]
                        taxon_id = line.split("OX=")[1].split()[0]
                    elif mode == "test":
                        protain_id, taxon_id = line.split()
                        protain_id = protain_id[1:]
                        
                    protain_info = (protain_id, taxon_id)
                else:
                    seq += line
                    
        return protainID_to_sequence
    
    train_IDs_to_Seq = load_fasta(path=PATH_TRAIN_FASTA, mode="train")
    test_IDs_to_Seq = load_fasta(path=PATH_TEST_FASTA, mode="test")
    
    count = 0
    for k, v in train_IDs_to_Seq.items():
        print(f"protain_info: {k}")
        print(f"seq_head: {v[:20]}")
        count += 1

        if count == 3:
            break

    count = 0
    for k, v in test_IDs_to_Seq.items():
        print(f"protain_info: {k}")
        print(f"seq_head: {v[:20]}")
        count += 1

        if count == 3:
            break
    
    # アミノ酸配列の長さの分布を確認
    from collections import Counter
    import numpy as np
    import matplotlib.pyplot as plt

    # ============================================
    # 前提:
    #   sequences: アミノ酸配列のリスト
    #   例: ["MKTLLI...", "AAAA...", ...]  # 長さ 82403 を想定
    # ============================================
    # ０) sequenceだけのリスト用意
    train_sequences = list(train_IDs_to_Seq.values())
    test_sequences = list(test_IDs_to_Seq.values())

    # 1) 各配列の長さを計算
    lengths = [len(seq) for seq in train_sequences]

    print(f"配列本数 (N): {len(lengths)}")
    print(f"最小長: {min(lengths)}")
    print(f"最大長: {max(lengths)}")
    print(f"平均長: {np.mean(lengths):.2f}")
    print(f"中央値: {np.median(lengths):.2f}")

    # 分位点も見てみる（25%, 50%, 75%）
    for q in [0.25, 0.5, 0.75, 0.9, 0.95, 0.99]:
        print(f"{int(q*100)}% 分位: {np.quantile(lengths, q):.2f}")

    # 長さ10000を超える配列の数
    over_10000_lengths = [length for length in lengths if length >= 10000]
    print(f"長さ10000を超える配列の数: {len(over_10000_lengths)}")

    # 2) 長さごとの本数を集計（分布を見る
    length_counter = Counter(lengths)


    # 3) ヒストグラムを描いてみる（ざっくり分布を確認）
    plt.figure(figsize=(8, 4))
    plt.hist(lengths, bins=50)  # ビン数はお好みで調整
    plt.xlabel("sequence length (amino acids)")
    plt.ylabel("count")
    plt.title("Length distribution of protein sequences")
    plt.show()

    from pathlib import Path
    from typing import Dict, List, Tuple, Any

    def parse_go_obo(filepath: str) -> Tuple[Dict[str, List[str]], Dict[str, Dict[str, Any]]]:
        """
        go-basic.obo ファイルを読み込んで、ヘッダー情報と Term 情報をパースする関数。

        Parameters
        ----------
        filepath : str
            読み込む go-basic.obo ファイルのパス

        Returns
        -------
        meta : Dict[str, List[str]]
            ファイル先頭部分（[Term] より前）のメタデータ。
            キー: "format-version" や "subsetdef" などの項目名
            値  : その行の値を格納した文字列のリスト
                （同じキーが複数回出てくることがあるのでリスト）

        terms : Dict[str, Dict[str, Any]]
            GO term を格納した辞書。
            キー: "GO:0000001" のような id
            値  : その term の情報をまとめた辞書。例:
                {
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
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        # ヘッダー部分を入れる辞書
        meta: Dict[str, List[str]] = {}

        # Term を id -> term_dict の形で入れる辞書
        terms: Dict[str, Dict[str, Any]] = {}

        # 現在読み取り中の term（[Term] ブロック）の情報を一時的に格納する変数
        current_term: Dict[str, Any] | None = None

        # まだ [Term] に入っていないヘッダー行を読んでいるかどうか
        in_header = True

        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")

                # コメント行（! で始まる）はスキップ
                if line.startswith("!"):
                    continue

                # 空行は「ブロックの区切り」を意味する
                if not line.strip():
                    # [Term] ブロックの終端だった場合、現在の term を保存する
                    if current_term is not None and "id" in current_term and current_term["id"]:
                        term_id = current_term["id"]
                        terms[term_id] = current_term
                        current_term = None
                    continue

                # [Term] という行が来たら、新しい term ブロックの開始
                if line == "[Term]":
                    in_header = False  # ここから先はヘッダーではない
                    # 今までの current_term が残っていたら保存（EOF直前の空行がないケースの保険）
                    if current_term is not None and "id" in current_term and current_term["id"]:
                        term_id = current_term["id"]
                        terms[term_id] = current_term
                    # 新しい term 用の辞書を初期化
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
                    continue

                # ここから下は「key: value」形式の行を処理する
                if ":" not in line:
                    # 想定外の行はそのまま無視してよい（必要ならログ出力など）
                    continue

                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip()

                if in_header:
                    # ヘッダー部分 -> meta に格納
                    meta.setdefault(key, []).append(value)
                else:
                    # Term ブロックの中 -> current_term を更新
                    if current_term is None:
                        # ありえないが、安全のため
                        continue

                    # 「生の key:value も残しておきたい」場合
                    current_term["raw"].setdefault(key, []).append(value)

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
                    else:
                        # その他の key（comment, relationship など）は必要に応じて later use
                        # ここでは raw にだけ入れておく
                        pass

        # ファイルが空行で終わっていない場合、最後の term を保存
        if current_term is not None and "id" in current_term and current_term["id"]:
            term_id = current_term["id"]
            terms[term_id] = current_term

        return meta, terms
    
    meta, terms = parse_go_obo(PATH_GO_OBO)

    print("meta exmple")
    for k, v in meta.items():
        print(f"k: {k}")
        print(f"v: {v}")
        break

    print("========================================================================================")

    print("terms example")
    for k, v in terms.items():
        print(f"k: {k}")
        print(f"v: {v}")
        break

    # Step1: Convert amino acid seaquence to vector
    ## Step1-1: load esm2 model
    import torch
    from transformers import AutoTokenizer, AutoModel
    esm2_tokenizer = AutoTokenizer.from_pretrained(ESM2_MODEL_DIR, do_lower_case=False)
    esm2_model = AutoModel.from_pretrained(ESM2_MODEL_DIR)

    import torch
    # GPUが使えるかどうか確認
    # kaggleではmps->cudaに変更すること
    # device = torch.device("mps" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("device:", device)

    # モデルをGPUまたはCPUに移動
    esm2_model = esm2_model.to(device)

    # 学習ではなく推論なので eval() にしておく（dropoutなどをオフ）
    esm2_model.eval()
    from typing import Dict, Tuple, List


    def make_batches_from_seq_dict(
        seq_dict: Dict[Tuple[str, str], str],
        batch_size: int
    ) -> List[List[Tuple[Tuple[str, str], str]]]:
        """
        説明:
            {(protein_id, taxon_id): seq} 形式の辞書を、
            指定したバッチサイズごとに分割する関数です。

        引数:
            seq_dict:
                キー: (protein_id, taxon_id)
                値:   アミノ酸配列文字列
            batch_size:
                1バッチに含める最大件数。

        戻り値:
            batches:
                各要素が「バッチ」のリスト。
                各バッチは [((protein_id, taxon_id), seq), ...] という形のリスト。
        
        使用例:
            seq_dict = {("P12345", "9606"): "MKT...", ("Q8ABC9", "10090"): "AAA..."}
            batches = make_batches_from_seq_dict(seq_dict, batch_size=256)
        """
        items = list(seq_dict.items())  # [ ((protein_id, taxon_id), seq), ... ]
        batches: List[List[Tuple[Tuple[str, str], str]]] = []

        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            batches.append(batch)

        return batches

    from typing import Dict


    def encode_all_sequences_with_esm(
        seq_dict: Dict[Tuple[str, str], str],
        tokenizer: AutoTokenizer,
        model: AutoModel,
        device: torch.device,
        batch_size: int = 64
    ) -> Dict[Tuple[str, str], np.ndarray]:
        """
        説明:
            {(protein_id, taxon_id): seq} 形式の辞書を入力として、
            ESM-2 で全配列をベクトル化し、
            {(protein_id, taxon_id): embedding} の辞書を返す高レベル関数です。

            内部では:
            - make_batches_from_seq_dict でバッチに分割
            - encode_batch_with_esm で各バッチを処理
            - 結果を1つの辞書にまとめる

        引数:
            seq_dict:
                {(protein_id, taxon_id): seq} 形式の辞書。
            tokenizer:
                ESM-2 用トークナイザ。
            model:
                ESM-2 モデル本体。
            device:
                使用するデバイス。
            batch_size:
                1バッチあたりの件数。GPUメモリに応じて調整。

        戻り値:
            emb_dict:
                {(protein_id, taxon_id): embedding} 形式の辞書。
                embedding は numpy.ndarray (D,)。
        """
        batches = make_batches_from_seq_dict(seq_dict, batch_size=batch_size)
        emb_dict: Dict[Tuple[str, str], np.ndarray] = {}

        model.eval()

        for batch in batches:
            batch_results = encode_batch_with_esm(batch, tokenizer, model, device)
            for seq_id, emb in batch_results:
                emb_dict[seq_id] = emb

            # 一時変数を消してキャッシュを捨てる
            del batch_results
            torch.cuda.empty_cache()

        return emb_dict
    
    # mini_batchで動作検証
    from itertools import islice

    # train_IDs_to_Seqから先頭BATCH_SIZE個だけ抽出
    mini_IDs_to_Seq = dict(islice(train_IDs_to_Seq.items(), 0, BATCH_SIZE+1))

    # mini_batch作成 かつ 実行時間測定
    import time
    start = time.perf_counter()

    mini_emb_dict = encode_all_sequences_with_esm(
        seq_dict=mini_IDs_to_Seq,
        tokenizer=esm2_tokenizer,
        model=esm2_model,
        device=device,
        batch_size=BATCH_SIZE
    )

    end = time.perf_counter()

    # 全体の実行時間推定
    # 予測実行時間が想定実行時間を超えていた場合エラー発生
    ex_time_per_batch = end - start
    print(f"実行時間: {ex_time_per_batch}")

    whole_ex_time = (len(train_IDs_to_Seq) + len(test_IDs_to_Seq)) / BATCH_SIZE * ex_time_per_batch / 3600
    print(f"whole_ex_time is {whole_ex_time} hours")


    # 
    print(f"想定実行時間: {MAX_ESM2_TIME}")
    print(f"予測実行時間: {whole_ex_time}")
    if whole_ex_time >= MAX_ESM2_TIME:
        print(f"予測実行時間が想定実行時間を超えています")

    """
    train_emb_dict = encode_all_sequences_with_esm(
        seq_dict=train_IDs_to_Seq,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=16
    )
    """

    """
    test_emb_dict = encode_all_sequences_with_esm(
        seq_dict=test_IDs_to_Seq,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=16
    )
    """

    ## step1-3: save amimno acid vectors

if __name__ == "__main__":
    main()
