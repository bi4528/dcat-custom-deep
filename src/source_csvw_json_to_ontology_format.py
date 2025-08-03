import json
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Load source file
source_path = "./result/generated/pn2024-100.columns.json"
output_path = "./notebook/source/pn2024_ontology_source.json"

try:
    with open(source_path, encoding="utf-8") as f:
        data = json.load(f)
    logging.info("Source JSON loaded successfully.")
except Exception as e:
    logging.error(f"Error reading source file: {e}")
    raise

columns = []
for col in data.get("tableSchema", {}).get("columns", []):
    name = col.get("name", "").strip()
    title = col.get("title", "").strip()

    if not name:
        continue

    uri = f"urn:pn2024-column:{name}"
    label = name
    definition = title
    synonyms = []

    text_parts = [
        f"Concept: {label}.",
        f"Defined as: {definition}" if definition else None
    ]
    text_for_embedding = " ".join(filter(None, text_parts))

    columns.append({
        "uri": uri,
        "label": label,
        "definition": definition,
        "synonyms": synonyms,
        "superclasses": [],
        "text_for_embedding": text_for_embedding
    })

logging.info(f"Extracted {len(columns)} columns from source.")

# Save output file
if not os.path.exists(output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(columns, f, indent=2, ensure_ascii=False)
    logging.info(f"File '{output_path}' created.")
else:
    logging.info(f"File '{output_path}' already exists. Skipping write.")
