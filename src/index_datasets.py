import os
import json
import argparse
import logging
import pandas as pd
import numpy as np
import chardet
import faiss
from sentence_transformers import SentenceTransformer

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def ensure_directories(paths):
    for path in paths:
        os.makedirs(path, exist_ok=True)
        logging.info(f"Ensured directory exists: {path}")

def read_csv_with_encoding_detection(csv_path, encoding_log_path):
    with open(csv_path, 'rb') as f:
        result = chardet.detect(f.read(10000))
        encoding = result['encoding']

    with open(encoding_log_path, "a") as log_f:
        log_f.write(f"{os.path.basename(csv_path)}: {encoding}\n")

    logging.info(f"Detected encoding for {csv_path}: {encoding}")
    df = pd.read_csv(csv_path, delimiter=";", encoding=encoding)
    return df

def load_or_create_index(index_path, embedding_dim):
    if os.path.exists(index_path):
        logging.info(f"Loading existing FAISS index from {index_path}")
        return faiss.read_index(index_path)
    else:
        logging.info("Creating new FAISS index.")
        base_index = faiss.IndexFlatL2(embedding_dim)
        return faiss.IndexIDMap(base_index)

def load_or_create_id_tracker(id_tracker_path):
    if os.path.exists(id_tracker_path):
        logging.info(f"Loading ID tracker from {id_tracker_path}")
        with open(id_tracker_path, "r") as f:
            return json.load(f)
    logging.info("No ID tracker found. Starting fresh.")
    return []

def save_id_tracker(id_tracker_path, id_tracker):
    with open(id_tracker_path, "w") as f:
        json.dump(id_tracker, f, indent=4)
    logging.info(f"Saved ID tracker with {len(id_tracker)} entries.")

def process_datasets(dataset_dir, model, index, id_tracker, encoding_log_path):
    processed = 0
    total = 0
    for file in os.listdir(dataset_dir):
        if not file.endswith(".csv"):
            continue
        total += 1
        dataset_id = file.replace(".csv", "")
        csv_path = os.path.join(dataset_dir, file)
        json_path = os.path.join(dataset_dir, f"{dataset_id}.json")

        if dataset_id in id_tracker:
            logging.info(f"Skipping {dataset_id}: already indexed.")
            continue
        if not os.path.exists(json_path):
            logging.warning(f"Skipping {dataset_id}: missing JSON metadata.")
            continue

        try:
            df = read_csv_with_encoding_detection(csv_path, encoding_log_path)
        except Exception as e:
            logging.warning(f"Failed to read {dataset_id}: {e}")
            continue

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception as e:
            logging.warning(f"Failed to parse JSON for {dataset_id}: {e}")
            continue

        combined_text = (
            meta.get("title", "") + "\n" +
            meta.get("notes", "") + "\n" +
            df.head(5).to_string()
        )

        try:
            embedding = model.encode(combined_text).reshape(1, -1)
        except Exception as e:
            logging.warning(f"Failed to encode text for {dataset_id}: {e}")
            continue

        id_numeric = abs(hash(dataset_id)) % (10 ** 12)
        index.add_with_ids(embedding, np.array([id_numeric]))
        id_tracker.append(dataset_id)
        processed += 1
        logging.info(f"Indexed {dataset_id} (ID: {id_numeric})")

    logging.info(f"Processed {processed} of {total} datasets.")
    return processed

def main(args):
    logging.info("Starting dataset indexing script.")
    ensure_directories([args.dataset_dir, os.path.dirname(args.index_path)])
    open(args.encoding_log, "w").close()
    logging.info("Encoding log cleared.")

    logging.info(f"Loading embedding model: {args.embedding_model}")
    embedding_model = SentenceTransformer(args.embedding_model)

    index = load_or_create_index(args.index_path, embedding_model.get_sentence_embedding_dimension())
    id_tracker = load_or_create_id_tracker(args.id_tracker)

    count = process_datasets(args.dataset_dir, embedding_model, index, id_tracker, args.encoding_log)

    faiss.write_index(index, args.index_path)
    logging.info(f"FAISS index written to {args.index_path}")

    save_id_tracker(args.id_tracker, id_tracker)

    logging.info(f"Finished. {count} new datasets were added.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index CSV datasets into faiss using semantic embeddings.")
    parser.add_argument("--dataset_dir", type=str, default="./data/datasets/opisi-train", help="Directory with CSV and JSON dataset pairs.")
    parser.add_argument("--index_path", type=str, default="./data/models/faiss-index-general.index", help="Path to store faiss index.")
    parser.add_argument("--encoding_log", type=str, default="./data/models/encoding_log.txt", help="Log file for detected encodings.")
    parser.add_argument("--id_tracker", type=str, default="./data/models/id_tracker.json", help="Path to track indexed dataset IDs.")
    parser.add_argument("--embedding_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="HuggingFace model to use for embeddings.")
    
    args = parser.parse_args()
    main(args)
