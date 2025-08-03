import os
import json
import logging
import sys
import pandas as pd
import numpy as np
import chardet
import yaml
import re
from openai import AzureOpenAI
from datetime import datetime

# -------------------- Load Config --------------------
CONFIG_PATH = "./config.yaml"
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

AZURE_CONFIG_PATH = "./azure_config.yaml"
with open(AZURE_CONFIG_PATH, "r", encoding="utf-8") as f:
    azure_config = yaml.safe_load(f)

DATASET_FOR_TEST = config["paths"]["dataset_for_test"]
OUTPUT_DIR = config["paths"]["columns_dir"]
AZURE_OPENAI_DEPLOYMENT = azure_config["openai"]["deployment"]
AZURE_OPENAI_API_KEY = azure_config["openai"]["api_key"]
AZURE_OPENAI_ENDPOINT = azure_config["openai"]["endpoint"]
AZURE_OPENAI_API_VERSION = azure_config["openai"]["api_version"]

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

# -------------------- Azure OpenAI Setup --------------------
client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
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

def infer_datatype(values):
    try:
        for v in values:
            float(v)
        return "integer" if all(v.strip().isdigit() for v in values) else "number"
    except:
        if any(re.search(r"\\d{4}-\\d{2}-\\d{2}", v) for v in values):
            return "date"
        return "string"

def create_prompt(table_name, all_cols, target_col, sample_values):
    context = f"Ime tabele: {table_name}\n"
    context += "\nPodatki iz tabele:\n"
    for col in all_cols:
        if col == target_col:
            continue
        context += f"- {col}: primeri: \n  - " + ", ".join(all_cols[col][:2]) + "\n"

    col_values = "\n  - ".join(sample_values)
    instruction = (
        "Na podlagi podanih vrednosti in konteksta iz drugih stolpcev, opiši pomen stolpca '"
        f"{target_col}' v eni kratki povedi. Odgovori izključno v slovenščini brez dodatnega teksta."
    )
    return f"{context}\n\nStolpec: {target_col}\nPrimeri:\n  - {col_values}\n\n{instruction}"

# -------------------- Main --------------------
def main():
    for file in os.listdir(DATASET_FOR_TEST):
        if not file.endswith(".csv"):
            continue

        dataset_id = file.replace(".csv", "")
        csv_path = os.path.join(DATASET_FOR_TEST, file)
        df = read_csv_with_encoding_detection(csv_path)
        if df is None:
            continue

        all_values = {col: df[col].dropna().astype(str).tolist()[:5] for col in df.columns}

        columns_metadata = []

        for col in df.columns:
            values = all_values.get(col, [])
            if not values:
                continue

            prompt = create_prompt(dataset_id, all_values, col, values)

            try:
                response = client.chat.completions.create(
                    model=AZURE_OPENAI_DEPLOYMENT,
                    messages=[
                        {"role": "system", "content": "Si pomočnik za opisovanje podatkovnih stolpcev."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2,
                    max_tokens=50,
                )
                description = response.choices[0].message.content.strip()
            except Exception as e:
                logging.error(f"OpenAI call failed for column {col}: {e}")
                description = ""

            inferred_type = infer_datatype(values)

            columns_metadata.append({
                "name": col,
                "title": description,
                "datatype": inferred_type
            })

        if columns_metadata:
            output = {
                "@context": "http://www.w3.org/ns/csvw",
                "tableSchema": {"columns": columns_metadata}
            }
            with open(os.path.join(OUTPUT_DIR, f"{dataset_id}.columns.json"), "w", encoding="utf-8") as f:
                json.dump(output, f, indent=4, ensure_ascii=False)
            logging.info(f"Saved CSVW metadata for {dataset_id}.")

if __name__ == "__main__":
    main()
