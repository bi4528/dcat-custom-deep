import os
import json
import yaml
import requests
import urllib.parse
import logging
import chardet
import zipfile

# -------------------- Config --------------------
CONFIG_PATH = "./config.yaml"
DATASET_ID = "mnzpprometne-nesrece-mesecno-od-leta-2019-dalje"
ZIP_URL = "https://www.policija.si/baza/pn2024.zip"
CSV_FILENAME_INSIDE_ZIP = "pn2024.csv"
API_SHOW = "https://podatki.gov.si/api/3/action/package_show?id="

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

OUTPUT_DIR = config["paths"]["dataset_opsi"]
ERROR_LOG = config["data_load"]["error_log"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------- Logging --------------------
log_level = getattr(logging, config.get("logging", {}).get("level", "INFO").upper(), logging.INFO)
logging.basicConfig(level=log_level, format='[%(asctime)s] %(levelname)s: %(message)s')

def download_and_save_dataset(dataset_id):
    zip_path = os.path.join(OUTPUT_DIR, f"{dataset_id}.zip")
    csv_path = os.path.join(OUTPUT_DIR, f"{dataset_id}.csv")
    json_path = os.path.join(OUTPUT_DIR, f"{dataset_id}.json")

    # --- Download ZIP ---
    try:
        logging.info(f"Downloading ZIP from {ZIP_URL}")
        r = requests.get(ZIP_URL)
        r.raise_for_status()
        with open(zip_path, "wb") as f:
            f.write(r.content)
        logging.info(f"Saved ZIP: {zip_path}")
    except Exception as e:
        logging.error(f"Failed to download ZIP: {e}")
        with open(ERROR_LOG, "a") as f:
            f.write(f"{dataset_id}: {ZIP_URL} - {e}\n")
        return

    # --- Extract CSV ---
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            if CSV_FILENAME_INSIDE_ZIP not in zip_ref.namelist():
                raise Exception(f"{CSV_FILENAME_INSIDE_ZIP} not found in ZIP")
            raw_bytes = zip_ref.read(CSV_FILENAME_INSIDE_ZIP)

        detected = chardet.detect(raw_bytes)
        encoding = detected['encoding'] or 'utf-8'
        if encoding.lower() in ["ascii", "windows-1252", "iso-8859-1"]:
            encoding = "windows-1250"

        text = raw_bytes.decode(encoding)
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(text)
        logging.info(f"Extracted and saved CSV: {csv_path}")

    except Exception as e:
        logging.error(f"Failed to extract CSV from ZIP: {e}")
        with open(ERROR_LOG, "a") as f:
            f.write(f"{dataset_id}: ZIP extract - {e}\n")
        return
    finally:
        if os.path.exists(zip_path):
            os.remove(zip_path)
            logging.info(f"Deleted ZIP file")

    # --- Fetch Metadata from CKAN API ---
    try:
        response = requests.get(API_SHOW + dataset_id)
        response.raise_for_status()
        meta = response.json()["result"]
        meta_data = {
            "title": meta.get("title", ""),
            "notes": meta.get("notes", ""),
            "license": meta.get("license_title", ""),
            "tags": [t["name"] for t in meta.get("tags", [])],
            "organization": meta.get("organization", {}).get("title", "")
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, indent=4, ensure_ascii=False)
        logging.info(f"Saved metadata: {json_path}")
    except Exception as e:
        logging.error(f"Failed to fetch metadata: {e}")
        with open(ERROR_LOG, "a") as f:
            f.write(f"{dataset_id}: metadata fetch - {e}\n")

# -------------------- Run --------------------
if __name__ == "__main__":
    download_and_save_dataset(DATASET_ID)
