import os
import sys
import json
import logging
import yaml
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# -------------------- Load Config --------------------
CONFIG_PATH = "./config.yaml"
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

INPUT_EXAMPLES_CSV = config["paths"]["opsi_column_examples_csv"]
FAISS_INDEX_PATH = config["paths"]["faiss_index_column_path"]
ID_TRACKER_JSON = config["paths"]["id_column_tracker_json"]
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

# -------------------- Main Script --------------------
def normalize_embeddings(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norms

def main():
    if not os.path.exists(INPUT_EXAMPLES_CSV):
        logging.error(f"Missing input file: {INPUT_EXAMPLES_CSV}")
        sys.exit(1)

    os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)

    df = pd.read_csv(INPUT_EXAMPLES_CSV)
    if not {"column", "sample_values", "description"}.issubset(df.columns):
        logging.error("Input CSV must contain columns: column, sample_values, description")
        sys.exit(1)

    texts = []
    ids = []

    for i, row in df.iterrows():
        col = row["column"]
        samples = row["sample_values"]
        combined_text = f"Column: {col}\nSample values:\n{samples}"
        texts.append(combined_text)
        ids.append(f"opsi2024::{col}")

    logging.info(f"Encoding {len(texts)} column entries...")

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embedding_dim = model.get_sentence_embedding_dimension()

    embeddings = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start:start+BATCH_SIZE]
        batch_embeddings = model.encode(batch, batch_size=BATCH_SIZE, show_progress_bar=True)
        embeddings.append(batch_embeddings)

    embeddings = np.vstack(embeddings)
    embeddings = normalize_embeddings(embeddings)
    id_hashes = [abs(hash(x)) % (10 ** 12) for x in ids]

    base_index = faiss.IndexFlatIP(embedding_dim)
    index = faiss.IndexIDMap(base_index)
    index.add_with_ids(embeddings, np.array(id_hashes))

    faiss.write_index(index, FAISS_INDEX_PATH)
    logging.info(f"FAISS index saved to: {FAISS_INDEX_PATH}")

    with open(ID_TRACKER_JSON, "w", encoding="utf-8") as f:
        json.dump(ids, f, indent=2, ensure_ascii=False)
    logging.info(f"ID tracker saved to: {ID_TRACKER_JSON}")

if __name__ == "__main__":
    logging.info("Starting OPSI column indexing...")
    main()
