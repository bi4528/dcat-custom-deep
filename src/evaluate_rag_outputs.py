import os
import json
import logging
import sys
from bert_score import score
import pandas as pd
import yaml

# -------------------- Load Config --------------------
CONFIG_PATH = "./config.yaml"
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

PIPELINE = config.get("pipeline", "both")
DATASET_FOR_TEST = config["paths"]["dataset_for_test"]
METADATA_DIR = config["paths"]["metadata_dir"]
COLUMNS_DIR = config["paths"]["columns_dir"]
REPORTS_DIR = config["paths"]["reports_dir"]

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

os.makedirs(REPORTS_DIR, exist_ok=True)

# -------------------- Utils --------------------
def evaluate_bert_score(candidates, references):
    candidates = [str(c) if not isinstance(c, str) else c for c in candidates]
    references = [str(r) if not isinstance(r, str) else r for r in references]
    P, R, F1 = score(candidates, references, lang="sl", verbose=True)
    return {
        "precision": P.mean().item(),
        "recall": R.mean().item(),
        "f1": F1.mean().item()
    }

# -------------------- Metadata Evaluation --------------------
def evaluate_metadata():
    results = {}

    for file in os.listdir(METADATA_DIR):
        if not file.endswith(".json"):
            continue

        dataset_id = file.replace(".metadata.json", "")
        gen_path = os.path.join(METADATA_DIR, file)
        gold_path = os.path.join(DATASET_FOR_TEST, f"{dataset_id}.json")

        if not os.path.exists(gold_path):
            logging.warning(f"Gold metadata missing for {dataset_id}. Skipping.")
            continue

        with open(gen_path, "r", encoding="utf-8") as f:
            generated = json.load(f)

        with open(gold_path, "r", encoding="utf-8") as f:
            gold = json.load(f)

        metrics = {}
        for field in ["title", "notes", "organization"]:
            gen_text = generated.get(field, "")
            gold_text = gold.get(field, "")
            metrics[field] = evaluate_bert_score([gen_text], [gold_text])

        # For tags (ensure string)
        gen_tags_list = generated.get("tags", [])
        gold_tags_list = gold.get("tags", [])

        if gen_tags_list and isinstance(gen_tags_list[0], dict):
            gen_tags = ", ".join(tag.get("name", "") for tag in gen_tags_list)
        else:
            gen_tags = ", ".join(gen_tags_list)

        if gold_tags_list and isinstance(gold_tags_list[0], dict):
            gold_tags = ", ".join(tag.get("name", "") for tag in gold_tags_list)
        else:
            gold_tags = ", ".join(gold_tags_list)

        metrics["tags"] = evaluate_bert_score([gen_tags], [gold_tags])

        results[dataset_id] = metrics
        logging.info(f"Evaluated metadata for {dataset_id}.")

    with open(os.path.join(REPORTS_DIR, "metadata_evaluation.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    logging.info("Metadata evaluation completed and saved.")

# -------------------- Column Evaluation --------------------
def evaluate_columns():
    results = {}

    for file in os.listdir(COLUMNS_DIR):
        if not file.endswith(".columns.json"):
            continue

        dataset_id = file.replace(".columns.json", "")
        csv_path = os.path.join(DATASET_FOR_TEST, f"{dataset_id}.csv")

        if not os.path.exists(csv_path):
            logging.warning(f"CSV missing for {dataset_id}. Skipping.")
            continue

        with open(file=os.path.join(COLUMNS_DIR, file), encoding="utf-8") as f:
            data = json.load(f)

        columns = data.get("tableSchema", {}).get("columns", [])

        try:
            df = pd.read_csv(csv_path, delimiter=";", encoding="utf-8", on_bad_lines="skip")
        except Exception as e:
            logging.warning(f"Failed to load CSV {dataset_id}: {e}")
            continue

        col_results = {}
        for col_meta in columns:
            col_name = col_meta.get("name", "")
            description = col_meta.get("description", "")

            if col_name not in df.columns:
                logging.warning(f"Column {col_name} not found in CSV {dataset_id}. Skipping.")
                continue

            sample_values = df[col_name].dropna().astype(str).head(5).tolist()
            context = col_name + "; " + ", ".join(sample_values)

            score_result = evaluate_bert_score([description], [context])
            col_results[col_name] = score_result

        results[dataset_id] = col_results
        logging.info(f"Evaluated columns for {dataset_id}.")

    with open(os.path.join(REPORTS_DIR, "columns_evaluation.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    logging.info("Column evaluation completed and saved.")

# -------------------- Main --------------------
if __name__ == "__main__":
    if PIPELINE in ("metadata", "both"):
        logging.info("Starting evaluation for Pipeline 1: Metadata.")
        evaluate_metadata()

    if PIPELINE in ("columns", "both"):
        logging.info("Starting evaluation for Pipeline 2: Columns.")
        evaluate_columns()

    logging.info("Evaluation process completed.")
