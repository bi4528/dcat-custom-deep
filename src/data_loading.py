import requests
import os
import random
import json
import urllib.parse

# Pridobivam 3 CSV podatkovnih množic
TARGET_COUNT = 3
API_LIST = "https://podatki.gov.si/api/3/action/package_list"
API_SHOW = "https://podatki.gov.si/api/3/action/package_show?id="
OUTPUT_DIR = "./data/datasets"
ERROR_LOG = "./data/error_log.txt"

os.makedirs(f"{OUTPUT_DIR}/opsi-train", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/opsi-test", exist_ok=True)

print("Pridobivam seznam ID-jeva za datasete...")
response = requests.get(API_LIST)
all_ids = response.json()["result"]
random.shuffle(all_ids)

# Seznam validnih datasetova
valid_datasets = []

# Funkcija za preverjanje, ali dataset ima CSV
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

print("Iščem datasete s CSV datotekami...")
for dataset_id in all_ids:
    dataset = has_csv(dataset_id)
    if dataset:
        valid_datasets.append(dataset)
        print(f"Našel sem CSV za dataset: {dataset_id}")
    if len(valid_datasets) >= TARGET_COUNT:
        break

print(f"\n Pridobil sem {len(valid_datasets)} CSV datasetov!")

# Razdelim množico za treniranje in testiranje
random.shuffle(valid_datasets)
split_index = int(len(valid_datasets) * 0.8)
train = valid_datasets[:split_index]
test = valid_datasets[split_index:]

def save_dataset(entry, target_dir):
    csv_path = f"{target_dir}/{entry['id']}.csv"
    json_path = f"{target_dir}/{entry['id']}.json"

    if os.path.exists(csv_path) and os.path.exists(json_path):
        print(f"Dataset z id={entry['id']} že obstaja.")
        return

    safe_csv_url = urllib.parse.quote(entry["csv_url"], safe=':/')

    try:
        csv_resp = requests.get(safe_csv_url)
        csv_resp.raise_for_status()
    except Exception as e:
        print(f"Napaka pri prevzemu CSV: {safe_csv_url}\nPodrobnosti: {e}")
        with open(ERROR_LOG, "a") as f:
            f.write(f"{entry['id']}: {safe_csv_url} - {e}\n")
        return

    try:
        import chardet
        detected = chardet.detect(csv_resp.content)
        encoding = detected['encoding']

        if encoding.lower() in ["ascii", "windows-1252", "iso-8859-1"]:
            encoding = "windows-1250"

        text = csv_resp.content.decode(encoding)
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"CSV shranjen (UTF-8 re-encoding): {csv_path}")
    except Exception as e:
        print(f"Napaka pri dekodiranju CSV: {e}, shranjujem surove bajte...")
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
    print(f"Metapodatki shranjeni za dataset id: {entry['id']}")

print("\nShrani train dataset:")
for entry in train:
    save_dataset(entry, f"{OUTPUT_DIR}/opsi-train")

print("\nShrani test dataset:")
for entry in test:
    save_dataset(entry, f"{OUTPUT_DIR}/opsi-test")