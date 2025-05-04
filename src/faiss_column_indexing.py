import os
import json
import logging
import sys
import pandas as pd
import numpy as np
import faiss
import chardet
from sentence_transformers import SentenceTransformer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout,
    force=True
)

# Config
DATASET_DIR = "./data/datasets/opsi-train"
FAISS_INDEX_PATH = "./data/models/faiss-index-columns.index"
ID_TRACKER_JSON = "./data/models/id_tracker_columns.json"
BATCH_SIZE = 16
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"

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

def main():
    if not os.path.exists(DATASET_DIR):
        logging.error(f"Dataset folder not found: {DATASET_DIR}")
        sys.exit(1)
    
    os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)

    logging.info("Loading embedding model...")
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

            df = read_csv_with_encoding_detection(csv_path)
            if df is None:
                continue

            for col in df.columns:
                samples = df[col].dropna().astype(str).head(5).tolist()
                sample_text = "\n".join(samples)
                combined_text = f"Column: {col}\nSample values:\n{sample_text}"

                all_texts.append(combined_text)
                all_ids.append(f"{dataset_id}::{col}")

    logging.info(f"Total columns collected: {len(all_texts)}")

    if not all_texts:
        logging.error("No columns found to index. Exiting.")
        sys.exit(1)

    # Encode texts
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
    logging.info("Welocme to column indexing...")
    main()
