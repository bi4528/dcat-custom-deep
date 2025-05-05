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
import yaml

# -------------------- Load Config --------------------
CONFIG_PATH = "./config.yaml"
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

DATASET_FOR_TEST = config["paths"]["dataset_for_test"]
FAISS_INDEX_PATH = config["paths"]["faiss_index_column_path"]
ID_TRACKER_JSON = config["paths"]["id_column_tracker_json"]
OUTPUT_DIR = config["paths"]["columns_dir"]
EMBEDDING_MODEL_NAME = config["models"]["embedding_model"]
GENERATION_MODEL_NAME = config["models"]["generation_model"]
K_NEIGHBORS = config["k_neighbours"]
BATCH_SIZE = config["batch_size"]

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
        if encoding and encoding.lower() in ["ascii", "windows-1252", "iso-8859-1"]:
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

    prompt = (
        f"<start_of_turn>user\n"
        f"You are a semantic description assistant for data columns.\n\n"
        f"Always answer strictly in Slovenian language.\n"
        f"Provide only ONE clear Slovenian sentence as the description.\n\n"
    )

    if examples:
        examples_text = ""
        for ex in examples:
            examples_text += f"- Column: {ex['column']}\n- Sample values:\n{ex['samples']}\n- Description: {ex['description']}\n\n"

        prompt += (
            f"Given the following examples of columns:\n"
            f"{examples_text}\n"
        )

    prompt += (
        f"Now, describe the meaning of this new column:\n"
        f"- Column: {current_col}\n"
        f"- Sample values:\n{sample_values_text}\n\n"
        f"Answer with only one simple Slovenian sentence.\n"
        f"<end_of_turn>\n"
        f"<start_of_turn>model"
    )

    return prompt

def clean_llm_output(full_text):
    cleaned = re.split(r"<start_of_turn>model", full_text, maxsplit=1)
    text = cleaned[1] if len(cleaned) > 1 else full_text
    text = re.sub(r"^[^\wčšžČŠŽ]*Sure.*?:", "", text, flags=re.IGNORECASE).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    for sentence in reversed(sentences):
        if sentence.strip():
            return sentence.strip()
    return text.strip()

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

        df = read_csv_with_encoding_detection(csv_path)
        if df is None:
            continue

        columns_metadata = []
        prompts = []
        col_info_list = []

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
                    dataset_col = id_tracker[idx]  # Direct map using FAISS idx
                    dataset_name, column_name = dataset_col.split("::")
                    examples.append({
                        "column": column_name,
                        "samples": "",
                        "description": ""
                    })
                except Exception as e:
                    logging.warning(f"Index mapping failed for idx {idx}: {e}")

            prompt = create_prompt(examples, col, sample_values)
            prompts.append(prompt)
            col_info_list.append({
                "name": col,
                "sample_values": sample_values
            })

        if not prompts:
            continue

        logging.info(f"Running LLM in batch for {len(prompts)} columns...")
        results = llm(prompts, max_new_tokens=30)

        for i, output in enumerate(results):
            gen_text = output[0]["generated_text"]
            cleaned_result = clean_llm_output(gen_text)
            datatype = infer_datatype(col_info_list[i]["sample_values"])

            columns_metadata.append({
                "name": col_info_list[i]["name"],
                "titles": col_info_list[i]["name"],
                "description": cleaned_result.strip(),
                "datatype": datatype
            })

        if columns_metadata:
            output = {
                "@context": "http://www.w3.org/ns/csvw",
                "tableSchema": {"columns": columns_metadata}
            }
            with open(os.path.join(OUTPUT_DIR, f"{dataset_id}.columns.json"), "w", encoding="utf-8") as f:
                json.dump(output, f, indent=4, ensure_ascii=False)
            logging.info(f"Saved CSVW metadata for {dataset_id}.")

        gc.collect()

if __name__ == "__main__":
    main()
