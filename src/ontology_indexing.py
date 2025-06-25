import os
import json
import logging
import sys
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import yaml

# -------------------- Load Config --------------------
CONFIG_PATH = "./config.yaml"
try:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
except Exception as e:
    print(f"[FATAL] Failed to read config.yaml: {e}")
    sys.exit(1)

ONTOLOGY_JSON = config["paths"]["ontology_terms_json"]
FAISS_INDEX_PATH = config["paths"]["faiss_index_ontology_path"]
ID_TRACKER_JSON = config["paths"]["id_ontology_tracker_json"]
EMBEDDING_MODEL_NAME = config["models"]["embedding_model"]

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

def normalize_embeddings(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norms

def ontology_indexing():
    logging.info("Starting ontology FAISS indexing...")

    if not os.path.exists(ONTOLOGY_JSON):
        logging.error(f"Ontology JSON file not found: {ONTOLOGY_JSON}")
        sys.exit(1)

    try:
        with open(ONTOLOGY_JSON, "r", encoding="utf-8") as f:
            ontology_terms = json.load(f)
    except Exception as e:
        logging.error(f"Failed to load ontology JSON: {e}")
        sys.exit(1)

    texts = []
    ids = []
    for i, term in enumerate(ontology_terms):
        try:
            label = term["label"]
            description = term.get("description", "")
            text = f"{label}: {description}"
            texts.append(text)
            ids.append(abs(hash(term["uri"])) % (10**12))
        except Exception as e:
            logging.warning(f"Skipping malformed entry {i}: {e}")

    if not texts:
        logging.error("No valid ontology terms found to index.")
        sys.exit(1)

    logging.info(f"Loaded {len(texts)} ontology terms for embedding.")

    # Embedding
    try:
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        embeddings = model.encode(texts, batch_size=16, show_progress_bar=True)
        embeddings = normalize_embeddings(np.array(embeddings))
    except Exception as e:
        logging.error(f"Embedding model failed: {e}")
        sys.exit(1)

    # FAISS indexing
    try:
        dimension = embeddings.shape[1]
        base_index = faiss.IndexFlatIP(dimension)
        index = faiss.IndexIDMap(base_index)
        index.add_with_ids(embeddings, np.array(ids))
        os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)
        faiss.write_index(index, FAISS_INDEX_PATH)
        logging.info(f"FAISS ontology index saved to: {FAISS_INDEX_PATH}")
    except Exception as e:
        logging.error(f"Failed to build or save FAISS index: {e}")
        sys.exit(1)

    # Save ID tracker
    try:
        id_map = {str(id_): term for id_, term in zip(ids, ontology_terms)}
        with open(ID_TRACKER_JSON, "w", encoding="utf-8") as f:
            json.dump(id_map, f, indent=4, ensure_ascii=False)
        logging.info(f"Ontology ID tracker saved to: {ID_TRACKER_JSON}")
    except Exception as e:
        logging.error(f"Failed to write ID tracker: {e}")
        sys.exit(1)
