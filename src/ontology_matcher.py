import os
import sys
import json
import yaml
import faiss
import logging
import numpy as np
from sentence_transformers import SentenceTransformer

# -------------------- Configuration --------------------
CONFIG_PATH = "./config.yaml"
try:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
except Exception as e:
    print(f"[FATAL] Failed to read config.yaml: {e}")
    sys.exit(1)

ONTOLOGY_INDEX_PATH = config["paths"]["faiss_index_ontology_path"]
ONTOLOGY_TRACKER_PATH = config["paths"]["id_ontology_tracker_json"]
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

# -------------------- Load Resources --------------------
try:
    logging.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
except Exception as e:
    logging.error(f"Failed to load embedding model: {e}")
    sys.exit(1)

if not os.path.exists(ONTOLOGY_INDEX_PATH):
    logging.error(f"Ontology FAISS index not found: {ONTOLOGY_INDEX_PATH}")
    sys.exit(1)

if not os.path.exists(ONTOLOGY_TRACKER_PATH):
    logging.error(f"Ontology tracker file not found: {ONTOLOGY_TRACKER_PATH}")
    sys.exit(1)

try:
    ontology_index = faiss.read_index(ONTOLOGY_INDEX_PATH)
except Exception as e:
    logging.error(f"Failed to load FAISS index: {e}")
    sys.exit(1)

try:
    with open(ONTOLOGY_TRACKER_PATH, "r", encoding="utf-8") as f:
        ontology_id_map = json.load(f)
    ontology_id_map = {int(k): v for k, v in ontology_id_map.items()}
except Exception as e:
    logging.error(f"Failed to load ontology tracker: {e}")
    sys.exit(1)

logging.info("Ontology matcher initialized successfully.")

# -------------------- Utility --------------------
def normalize_embeddings(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norms

# -------------------- Core Function --------------------
def match_to_ontology(text, top_k=1):
    """
    Matches an input text (column description) to the best ontology term(s).

    Args:
        text (str): The input string to match.
        top_k (int): Number of top candidates to return.

    Returns:
        List[Dict]: A list of dictionaries with keys: uri, label, score, description.
    """
    logging.info(f"Matching text to ontology: {text}")
    try:
        embedding = embedding_model.encode([text])
        embedding = normalize_embeddings(np.array(embedding)).astype(np.float32)
        D, I = ontology_index.search(embedding, top_k)
    except Exception as e:
        logging.error(f"Error during ontology search: {e}")
        return []

    matches = []
    for idx, score in zip(I[0], D[0]):
        if idx == -1:
            continue
        term = ontology_id_map.get(idx, {})
        matches.append({
            "uri": term.get("uri"),
            "label": term.get("label"),
            "score": float(score),
            "description": term.get("description")
        })

    logging.info(f"Found {len(matches)} match(es) for input text.")
    return matches
