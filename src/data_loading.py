import requests
import os
import random
import json
import urllib.parse
import yaml
import logging
import sys

# -------------------- Load Config --------------------
CONFIG_PATH = "./config.yaml"
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

OUTPUT_DIR_TRAIN = config["paths"]["dataset_for_train"]
OUTPUT_DIR_TEST = config["paths"]["dataset_for_test"]
ERROR_LOG = config["data_load"]["error_log"]

TARGET_COUNT = config["data_load"]["target_count"]
API_LIST = config["data_load"]["opsi_api_list"]
API_SHOW = config["data_load"]["opsi_api_show"]

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

# Ensure directories exist
os.makedirs(OUTPUT_DIR_TRAIN, exist_ok=True)
os.makedirs(OUTPUT_DIR_TEST, exist_ok=True)

# -------------------- Data Download --------------------
logging.info("Fetching dataset IDs from OPSI...")
response = requests.get(API_LIST)
all_ids = response.json().get("result", [])
random.shuffle(all_ids)

valid_datasets = []

def has_csv(dataset_id):
    r = requests.get(API_SHOW + dataset_id)
    if r.status_code != 200:
        return None

    data = r.json()["result"]
    for res in data.get("resources", []):
        if res.get("format", "").lower() == "csv":
            return {
                "id": dataset_id,
                "csv_url": res.get("url"),
                "meta": data
            }
    return None

logging.info("Searching for datasets with CSV resources...")
for dataset_id in all_ids:
    dataset = has_csv(dataset_id)
    if dataset:
        valid_datasets.append(dataset)
        logging.info(f"Found CSV for dataset: {dataset_id}")
    if len(valid_datasets) >= TARGET_COUNT:
        break

logging.info(f"Retrieved {len(valid_datasets)} CSV datasets.")

random.shuffle(valid_datasets)
split_index = int(len(valid_datasets) * 0.8)
train = valid_datasets[:split_index]
test = valid_datasets[split_index:]

def save_dataset(entry, target_dir):
    csv_path = os.path.join(target_dir, f"{entry['id']}.csv")
    json_path = os.path.join(target_dir, f"{entry['id']}.json")

    if os.path.exists(csv_path) and os.path.exists(json_path):
        logging.info(f"Dataset with id={entry['id']} already exists.")
        return

    safe_csv_url = urllib.parse.quote(entry["csv_url"], safe=':/')

    try:
        csv_resp = requests.get(safe_csv_url)
        csv_resp.raise_for_status()
    except Exception as e:
        logging.error(f"Failed to download CSV: {safe_csv_url} | Error: {e}")
        with open(ERROR_LOG, "a") as f:
            f.write(f"{entry['id']}: {safe_csv_url} - {e}\n")
        return

    try:
        import chardet
        detected = chardet.detect(csv_resp.content)
        encoding = detected['encoding']

        if encoding and encoding.lower() in ["ascii", "windows-1252", "iso-8859-1"]:
            encoding = "windows-1250"

        text = csv_resp.content.decode(encoding or 'utf-8')
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(text)
        logging.info(f"CSV saved (UTF-8 re-encoded): {csv_path}")
    except Exception as e:
        logging.warning(f"Decoding failed ({e}), saving raw bytes...")
        with open(csv_path, "wb") as f:
            f.write(csv_resp.content)

    meta = entry["meta"]
    meta_data = {
        "title": meta.get("title", ""),
        "notes": meta.get("notes", ""),
        "license": meta.get("license_title", ""),
        "tags": [t["name"] for t in meta.get("tags", [])],
        "organization": meta.get("organization", {}).get("title", "")
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta_data, f, indent=4, ensure_ascii=False)
    logging.info(f"Metadata saved for dataset id: {entry['id']}")

logging.info("Saving train datasets...")
for entry in train:
    save_dataset(entry, OUTPUT_DIR_TRAIN)

logging.info("Saving test datasets...")
for entry in test:
    save_dataset(entry, OUTPUT_DIR_TEST)

logging.info("Data loading completed.")
