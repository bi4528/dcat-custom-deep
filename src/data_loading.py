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

successful_datasets = []

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

def save_dataset(entry, target_dir):
    csv_path = os.path.join(target_dir, f"{entry['id']}.csv")
    json_path = os.path.join(target_dir, f"{entry['id']}.json")

    if os.path.exists(csv_path) and os.path.exists(json_path):
        logging.info(f"Dataset with id={entry['id']} already exists.")
        return True  # Already saved previously

    safe_csv_url = urllib.parse.quote(entry["csv_url"], safe=':/')

    try:
        csv_resp = requests.get(safe_csv_url)
        csv_resp.raise_for_status()
    except Exception as e:
        logging.error(f"Failed to download CSV: {safe_csv_url} | Error: {e}")
        with open(ERROR_LOG, "a") as f:
            f.write(f"{entry['id']}: {safe_csv_url} - {e}\n")
        return False

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
    return True

# -------------------- Main Loop --------------------
logging.info("Searching and saving datasets until we have 100 successfully saved...")

idx_pointer = 0
while len(successful_datasets) < TARGET_COUNT and idx_pointer < len(all_ids):
    dataset_id = all_ids[idx_pointer]
    idx_pointer += 1

    dataset = has_csv(dataset_id)
    if dataset:
        # First save temporarily in train dir (we'll split later)
        success = save_dataset(dataset, OUTPUT_DIR_TRAIN)
        if success:
            successful_datasets.append(dataset["id"])
            logging.info(f"SUCCESS: {dataset['id']} saved ({len(successful_datasets)}/{TARGET_COUNT})")
    else:
        logging.info(f"No CSV found for {dataset_id}")

if len(successful_datasets) < TARGET_COUNT:
    logging.warning(f"Reached end of OPSI datasets, only {len(successful_datasets)} were saved.")
else:
    logging.info(f"Completed: 100 datasets saved.")

# -------------------- Split Train/Test --------------------
logging.info("Splitting into 80% train and 20% test...")
random.shuffle(successful_datasets)
split_index = int(TARGET_COUNT * 0.8)
train_ids = successful_datasets[:split_index]
test_ids = successful_datasets[split_index:]

for dataset_id in test_ids:
    # Move files to TEST folder
    csv_src = os.path.join(OUTPUT_DIR_TRAIN, f"{dataset_id}.csv")
    json_src = os.path.join(OUTPUT_DIR_TRAIN, f"{dataset_id}.json")
    csv_dst = os.path.join(OUTPUT_DIR_TEST, f"{dataset_id}.csv")
    json_dst = os.path.join(OUTPUT_DIR_TEST, f"{dataset_id}.json")

    try:
        os.rename(csv_src, csv_dst)
        os.rename(json_src, json_dst)
        logging.info(f"Moved {dataset_id} to test set.")
    except Exception as e:
        logging.error(f"Failed to move {dataset_id}: {e}")

logging.info("Data loading and splitting completed.")
