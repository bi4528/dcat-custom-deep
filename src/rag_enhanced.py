import logging
import sys

# -------------------- Logging --------------------
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout,
    force=True
)

logging.info("Welcome to RAG...")

# -------------------- Import --------------------

import os
import json
import pandas as pd
import faiss
import numpy as np
import chardet
import gc
import re
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

# -------------------- Token Load --------------------
try:
    with open("./src/.hf_token", "r") as f:
        hf_token = f.read().strip()
        logging.info("Hugging Face token is ready!")
except FileNotFoundError:
    logging.error("Token file './src/.hf_token' not found.")
    sys.exit(1)
except Exception as e:
    logging.error(f"Failed to read Hugging Face token: {e}")
    sys.exit(1)

# -------------------- Paths --------------------
FAISS_INDEX_PATH = "./data/models/faiss-index-general.index"
ID_TRACKER_JSON = "./data/models/id_tracker.json"
DATASET_FOR_TEST = "./data/datasets/opisi-test/"
OUTPUT_FILE = "./data/models/batch_rag_results_structured.json"

# -------------------- Load Models --------------------
logging.info(f"Loading FAISS index from: {FAISS_INDEX_PATH}")
index = faiss.read_index(FAISS_INDEX_PATH)

logging.info(f"Loading ID tracker from: {ID_TRACKER_JSON}")
with open(ID_TRACKER_JSON, "r") as f:
    id_tracker = json.load(f)

logging.info("Loading sentence embedding model...")
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

logging.info("Loading Gemma model for text generation (GPU)...")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it", token=hf_token)
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2b-it",
    device_map="auto",
    torch_dtype="auto",
    token=hf_token
)
llm = pipeline("text-generation", model=model, tokenizer=tokenizer)

# -------------------- Utils --------------------
def read_csv_with_encoding_detection(csv_path):
    with open(csv_path, 'rb') as f:
        result = chardet.detect(f.read(10000))
        encoding = result['encoding']
    logging.info(f"Detected encoding for {os.path.basename(csv_path)}: {encoding}")
    return pd.read_csv(csv_path, delimiter=";", encoding=encoding)

def extract_metadata(text):
    title = re.search(r"\*\*Title:\*\* (.*)", text)
    notes = re.search(r"\*\*Notes:\*\* (.*)", text)
    license = re.search(r"\*\*License:\*\* (.*)", text)
    return {
        "title": title.group(1).strip() if title else "",
        "notes": notes.group(1).strip() if notes else "",
        "license": license.group(1).strip() if license else "",
    }

# -------------------- Main Loop --------------------
batch_results = []

for file in os.listdir(DATASET_FOR_TEST):
    if not file.endswith(".csv"):
        continue

    dataset_id = file.replace(".csv", "")
    csv_path = os.path.join(DATASET_FOR_TEST, file)
    json_path = os.path.join(DATASET_FOR_TEST, f"{dataset_id}.json")

    if not os.path.exists(json_path):
        logging.warning(f"Metadata missing for {dataset_id}. Skipping.")
        continue

    logging.info(f"Processing dataset: {dataset_id}")
    df = read_csv_with_encoding_detection(csv_path)
    with open(json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    query_text = meta.get("title", "") + "\n" + meta.get("notes", "") + "\n" + df.head(5).to_string()
    query_embedding = embedding_model.encode(query_text).reshape(1, -1)

    distances, indices = index.search(query_embedding, k=3)

    context = ""
    for idx in indices[0]:
        if idx == -1:
            continue
        try:
            dataset_idx = id_tracker[indices[0].tolist().index(idx)]
        except IndexError:
            logging.warning(f"Invalid index mapping for FAISS index {idx}")
            continue

        json_context_path = f"./data/datasets/opisi-train/{dataset_idx}.json"
        if os.path.exists(json_context_path):
            with open(json_context_path, "r", encoding="utf-8") as f:
                meta_context = json.load(f)
            context += f"Title: {meta_context.get('title', '')}\nNotes: {meta_context.get('notes', '')}\n\n"

    prompt = (
        f"Based on the following examples from the FAISS index:\n{context}\n\n"
        f"Generate metadata (title, notes, license) for the following dataset:\n{query_text}"
    )

    result = llm(prompt, max_new_tokens=256)[0]["generated_text"]
    structured = extract_metadata(result)
    logging.info(f"Metadata generated for {dataset_id}. Memory cleared.")

    batch_results.append({
        "dataset_id": dataset_id,
        "generated_metadata": result,
        "generated_metadata_extracted": structured
    })
    gc.collect()

# Save results
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(batch_results, f, indent=4, ensure_ascii=False)

logging.info(f"Batch RAG generation completed successfully. Results saved to: {OUTPUT_FILE}")
