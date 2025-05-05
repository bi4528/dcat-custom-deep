import os
import json
import logging
import sys
import pandas as pd
import numpy as np
import faiss
import chardet
import gc
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import yaml

# -------------------- Load Config --------------------
CONFIG_PATH = "./config.yaml"
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

DATASET_FOR_TEST = config["paths"]["dataset_for_test"]
FAISS_INDEX_PATH = config["paths"]["faiss_index_metadata_path"]
ID_TRACKER_JSON = config["paths"]["id_metadata_tracker_json"]
OUTPUT_DIR = config["paths"]["metadata_dir"]
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"
GENERATION_MODEL_NAME = "google/gemma-2b-it"
K_NEIGHBORS = config["k_neighbours"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

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
    with open(csv_path, 'rb') as f:
        result = chardet.detect(f.read(10000))
        encoding = result['encoding']
        if encoding.lower() in ["ascii", "windows-1252", "iso-8859-1"]:
            encoding = "windows-1250"
    try:
        df = pd.read_csv(csv_path, delimiter=";", encoding=encoding)
        logging.info(f"Loaded {os.path.basename(csv_path)} with encoding {encoding}")
        return df
    except Exception as e:
        logging.error(f"Failed to read {csv_path} with detected encoding {encoding}: {e}")
        return None

def normalize_embeddings(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norms

def create_prompt(examples, meta, df):
    csv_columns = ", ".join(df.columns.tolist())
    csv_sample = df.head(5).to_string()

    prompt = (
        f"<start_of_turn>user\n"
        f"You are a metadata assistant for Slovenian open data datasets.\n\n"
        f"Your task:\n"
        f"- ALWAYS reply strictly in Slovenian.\n"
        f"- Generate metadata as JSON with keys: title, notes, license, tags (list), organization.\n\n"
    )

    if examples:
        examples_text = ""
        for ex in examples:
            examples_text += (
                f"Title: {ex.get('title', '')}\n"
                f"Notes: {ex.get('notes', '')}\n"
                f"Tags: {', '.join(ex.get('tags', []))}\n"
                f"Organization: {ex.get('organization', '')}\n\n"
            )
        prompt += f"Here are some examples:\n{examples_text}\n"

    prompt += (
        f"Now generate metadata for the following dataset:\n\n"
        f"Title: {meta.get('title', '')}\n"
        f"Notes: {meta.get('notes', '')}\n"
        f"CSV Columns: {csv_columns}\n"
        f"CSV Sample:\n{csv_sample}\n\n"
        f"Return JSON only.\n"
        f"<end_of_turn>\n"
        f"<start_of_turn>model"
    )

    return prompt

def extract_metadata(text):
    try:
        json_start = text.find('{')
        json_end = text.rfind('}') + 1
        json_text = text[json_start:json_end]
        data = json.loads(json_text)
        return {
            "title": data.get("title", ""),
            "notes": data.get("notes", ""),
            "license": data.get("license", ""),
            "tags": data.get("tags", []),
            "organization": data.get("organization", "")
        }
    except Exception as e:
        logging.warning(f"Failed to parse metadata JSON: {e}")
        return {}

# -------------------- Main --------------------
def main():
    logging.info("Loading FAISS and ID tracker...")
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(ID_TRACKER_JSON, "r", encoding="utf-8") as f:
        id_tracker = json.load(f)

    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    try:
        with open("./src/.hf_token", "r") as f:
            hf_token = f.read().strip()
    except Exception as e:
        logging.error(f"Failed to load Hugging Face token: {e}")
        sys.exit(1)

    tokenizer = AutoTokenizer.from_pretrained(GENERATION_MODEL_NAME, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        GENERATION_MODEL_NAME, device_map="auto", torch_dtype="auto", token=hf_token
    )
    llm = pipeline("text-generation", model=model, tokenizer=tokenizer)

    for file in os.listdir(DATASET_FOR_TEST):
        if not file.endswith(".csv"):
            continue

        dataset_id = file.replace(".csv", "")
        csv_path = os.path.join(DATASET_FOR_TEST, file)
        json_path = os.path.join(DATASET_FOR_TEST, f"{dataset_id}.json")

        if not os.path.exists(json_path):
            logging.warning(f"Metadata JSON missing for {dataset_id}. Skipping.")
            continue

        df = read_csv_with_encoding_detection(csv_path)
        if df is None:
            continue

        with open(json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        combined_text = (
            meta.get("title", "") + "\n" +
            meta.get("notes", "") + "\n" +
            df.head(5).to_string()
        )

        query_embedding = embedding_model.encode(combined_text).reshape(1, -1)
        query_embedding = normalize_embeddings(query_embedding)

        distances, indices = index.search(query_embedding, K_NEIGHBORS)

        examples = []
        for idx in indices[0]:
            if idx == -1:
                continue
            try:
                dataset_idx = id_tracker[indices[0].tolist().index(idx)]
            except IndexError:
                continue
            example_json = os.path.join("./data/datasets/opsi-train", f"{dataset_idx}.json")
            if os.path.exists(example_json):
                with open(example_json, "r", encoding="utf-8") as ef:
                    ex_meta = json.load(ef)
                examples.append(ex_meta)

        prompt = create_prompt(examples, meta, df)
        result = llm(prompt, max_new_tokens=128)[0]["generated_text"]

        metadata = extract_metadata(result)

        with open(os.path.join(OUTPUT_DIR, f"{dataset_id}.metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)

        logging.info(f"Generated metadata saved for {dataset_id}.")
        gc.collect()

if __name__ == "__main__":
    logging.info("Welcome to metadata RAG generation...")
    main()
