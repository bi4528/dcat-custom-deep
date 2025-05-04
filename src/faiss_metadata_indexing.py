import os
import json
import logging
import sys
import pandas as pd
import numpy as np
import faiss
import chardet
from sentence_transformers import SentenceTransformer
import yaml

# -------------------- Load Config --------------------
CONFIG_PATH = "./config.yaml"
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

DATASET_DIR = config["paths"]["dataset_for_train"]
FAISS_INDEX_PATH = config["paths"]["faiss_index_metadata_path"]
ID_TRACKER_JSON = config["paths"]["id_metadata_tracker_json"]
EMBEDDING_MODEL_NAME = config["models"]["embedding_model"]
BATCH_SIZE = config["batch_size"]

# -------------------- Logging Setup --------------------
log_level = getattr(logging, config.get("logging", {}).get("level", "INFO").upper(), logging.INFO)
log_file = config.get("logging", {}).get("log_file", None)

if log_file:
    logging.basicConfig(
        level=log_level,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='a'
    )
else:
    logging.basicConfig(
        level=log_level,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        stream=sys.stdout,
    )

# -------------------- Utils --------------------
def read_csv_with_encoding_detection(csv_path):
    try:
        with open(csv_path, 'rb') as f:
            result = chardet.detect(f.read(10000))
            encoding = result['encoding']
        df = pd.read_csv(csv_path, delimiter=";", encoding=encoding)
        logging.info(f"Loaded CSV {os.path.basename(csv_path)} with encoding {encoding}")
        return df
    except Exception as e:
        logging.error(f"Failed to read CSV {csv_path}: {e}")
        return None

def normalize_embeddings(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norms

# -------------------- Main --------------------
def main():
    if not os.path.exists(DATASET_DIR):
        logging.error(f"Dataset folder not found: {DATASET_DIR}")
        sys.exit(1)

    os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)

    logging.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embedding_dim = model.get_sentence_embedding_dimension()

    base_index = faiss.IndexFlatIP(embedding_dim)
    index = faiss.IndexIDMap(base_index)

    all_texts = []
    all_ids = []

    for file in os.listdir(DATASET_DIR):
        if file.endswith(".csv"):
            dataset_id = file.replace(".csv", "")
            csv_path = os.path.join(DATASET_DIR, file)
            json_path = os.path.join(DATASET_DIR, f"{dataset_id}.json")

            if not os.path.exists(json_path):
                logging.warning(f"Skipping {dataset_id}: JSON metadata missing.")
                continue

            df = read_csv_with_encoding_detection(csv_path)
            if df is None:
                continue

            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception as e:
                logging.warning(f"Skipping {dataset_id}: failed to read JSON: {e}")
                continue

            combined_text = (
                meta.get("title", "") + "\n" +
                meta.get("notes", "") + "\n" +
                df.head(5).to_string()
            )

            all_texts.append(combined_text)
            all_ids.append(dataset_id)

    logging.info(f"Total datasets collected: {len(all_texts)}")

    if not all_texts:
        logging.error("No datasets found to index. Exiting.")
        sys.exit(1)

    embeddings = []
    for start in range(0, len(all_texts), BATCH_SIZE):
        batch = all_texts[start:start+BATCH_SIZE]
        batch_embeddings = model.encode(batch, batch_size=BATCH_SIZE, show_progress_bar=True)
        embeddings.append(batch_embeddings)

    embeddings = np.vstack(embeddings)
    embeddings = normalize_embeddings(embeddings)

    ids = [abs(hash(x)) % (10 ** 12) for x in all_ids]
    index.add_with_ids(embeddings, np.array(ids))

    faiss.write_index(index, FAISS_INDEX_PATH)
    logging.info(f"FAISS index saved at: {FAISS_INDEX_PATH}")

    with open(ID_TRACKER_JSON, "w", encoding="utf-8") as f:
        json.dump(all_ids, f, indent=4, ensure_ascii=False)
    logging.info(f"ID tracker saved at: {ID_TRACKER_JSON}")

if __name__ == "__main__":
    logging.info("Welcome to metadata indexing...")
    main()
