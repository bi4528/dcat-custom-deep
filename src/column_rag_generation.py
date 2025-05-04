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
import re

# -------------------- Logging Setup --------------------
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout,
    force=True
)

# -------------------- Config --------------------
DATASET_FOR_TEST = "./data/datasets/opisi-test"
FAISS_INDEX_PATH = "./data/models/faiss-index-columns.index"
ID_TRACKER_JSON = "./data/models/id_tracker_columns.json"
OUTPUT_DIR = "./data/models/column_metadata"
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"
GENERATION_MODEL_NAME = "google/gemma-2b-it"
K_NEIGHBORS = 3
BATCH_SIZE = 16

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------- Utils --------------------
def read_csv_with_encoding_detection(csv_path):
    with open(csv_path, 'rb') as f:
        result = chardet.detect(f.read(10000))
        encoding = result['encoding']
    return pd.read_csv(csv_path, delimiter=";", encoding=encoding)

def normalize_embeddings(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norms

def infer_datatype(values):
    try:
        for v in values:
            float(v)
        return "number"
    except:
        if any(re.search(r"\d{4}-\d{2}-\d{2}", v) for v in values):
            return "date"
        return "string"

def create_prompt(examples, current_col, sample_values):
    sample_values_text = "\n".join(sample.strip().replace("\n", " ") for sample in sample_values)

    if examples:
        examples_text = ""
        for ex in examples:
            examples_text += f"- Column: {ex['column']}\n- Sample values:\n{ex['samples']}\n- Description: {ex['description']}\n\n"

        prompt = (
            f"<start_of_turn>user\n"
            f"You are a semantic description assistant for data columns.\n\n"
            f"Answer strictly in Slovenian language.\n\n"
            f"Given the following examples of columns:\n"
            f"{examples_text}\n"
            f"Now, describe the meaning of this new column:\n"
            f"- Column: {current_col}\n"
            f"- Sample values:\n{sample_values_text}\n\n"
            f"Answer with only a clear description sentence in Slovenian.\n"
            f"<end_of_turn>\n"
            f"<start_of_turn>model"
        )
    else:
        prompt = (
            f"<start_of_turn>user\n"
            f"You are a semantic description assistant for data columns.\n\n"
            f"Answer strictly in Slovenian language.\n\n"
            f"You have no examples to use.\n\n"
            f"Based only on the column name and sample values below, guess the meaning of the column.\n\n"
            f"- Column: {current_col}\n"
            f"- Sample values:\n{sample_values_text}\n\n"
            f"Answer with only a clear description sentence in Slovenian.\n"
            f"<end_of_turn>\n"
            f"<start_of_turn>model"
        )

    return prompt

# -------------------- Main --------------------
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

    try:
        df = read_csv_with_encoding_detection(csv_path)
    except Exception as e:
        logging.warning(f"Failed to read CSV {dataset_id}: {e}")
        continue

    columns_metadata = []

    for col in df.columns:
        sample_values = df[col].dropna().astype(str).head(5).tolist()
        if not sample_values:
            continue

        query_text = f"Column: {col}\nSample values:\n" + "\n".join(sample_values)
        query_embedding = embedding_model.encode(query_text).reshape(1, -1)
        query_embedding = normalize_embeddings(query_embedding)

        distances, indices = index.search(query_embedding, K_NEIGHBORS)

        examples = []
        for idx in indices[0]:
            if idx == -1:
                continue
            try:
                dataset_col = id_tracker[indices[0].tolist().index(idx)]
            except IndexError:
                continue
            dataset_name, column_name = dataset_col.split("::")
            examples.append({
                "column": column_name,
                "samples": "",  # No actual examples are loaded
                "description": ""  # No description available
            })

        prompt = create_prompt(examples, col, sample_values)
        result = llm(prompt, max_new_tokens=128)[0]["generated_text"]

        datatype = infer_datatype(sample_values)

        columns_metadata.append({
            "name": col,
            "titles": col,
            "description": result.strip(),
            "datatype": datatype
        })

        gc.collect()

    if columns_metadata:
        output = {
            "@context": "http://www.w3.org/ns/csvw",
            "tableSchema": {"columns": columns_metadata}
        }
        with open(os.path.join(OUTPUT_DIR, f"{dataset_id}.columns.json"), "w", encoding="utf-8") as f:
            json.dump(output, f, indent=4, ensure_ascii=False)
        logging.info(f"Saved CSVW metadata for {dataset_id}.")
