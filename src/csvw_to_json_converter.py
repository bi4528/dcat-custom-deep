from rdflib import Graph, Namespace, URIRef, Literal
import json
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Load TTL file
g = Graph()
try:
    g.parse("./data/preglednica_zahtevkov_zavarovanj_csvw.ttl", format="turtle")
    logging.info("TTL file successfully parsed.")
except Exception as e:
    logging.error(f"Failed to parse TTL file: {e}")
    raise

CSVW = Namespace("http://www.w3.org/ns/csvw#")
DCT = Namespace("http://purl.org/dc/terms/")
SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")
RDFS = Namespace("http://www.w3.org/2000/01/rdf-schema#")

columns = []

for col in g.subjects(predicate=CSVW.name):
    uri = str(col)
    label = None
    definition = None
    synonyms = []

    for _, _, val in g.triples((col, CSVW.title, None)):
        if isinstance(val, Literal) and val.language == 'sl':
            label = str(val)
            break

    for _, _, val in g.triples((col, DCT.description, None)):
        if isinstance(val, Literal) and val.language == 'sl':
            definition = str(val)
            break

    for _, _, val in g.triples((col, SKOS.altLabel, None)):
        if isinstance(val, Literal) and val.language == 'sl':
            synonyms.append(str(val))

    text_parts = [
        f"Concept: {label or uri}.",
        f"Also known as: {', '.join(synonyms)}." if synonyms else None,
        f"Defined as: {definition}" if definition else None
    ]
    text_for_embedding = " ".join(filter(None, text_parts))

    columns.append({
        "uri": uri,
        "label": label or "",
        "definition": definition or "",
        "synonyms": synonyms,
        "superclasses": [],
        "text_for_embedding": text_for_embedding
    })

logging.info(f"Extracted {len(columns)} columns from TTL graph.")

# Save as JSON if it doesn't exist
output_path = "./notebook/target/preglednica_zahtevkov_zavarovanj_target.json"
if not os.path.exists(output_path):
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(columns, f, indent=2, ensure_ascii=False)
        logging.info(f"File '{output_path}' created.")
    except Exception as e:
        logging.error(f"Failed to write JSON file: {e}")
else:
    logging.info(f"File '{output_path}' already exists. Skipping write.")

