import json
import logging
from rdflib import Graph, Namespace, URIRef, Literal, BNode
from rdflib.namespace import RDF

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# File paths
SOURCE_JSON = "./result/generated/pn2024-100.columns.json"
MATCH_FILE = "./notebook/result/opsi_match.json"
TARGET_TTL = "./data/preglednica_zahtevkov_zavarovanj_csvw.ttl"
OUTPUT_TTL = "./notebook/result/pn2024-100_csvw.ttl"

# Namespaces
CSVW = Namespace("http://www.w3.org/ns/csvw#")
DCT = Namespace("http://purl.org/dc/terms/")
XSD = Namespace("http://www.w3.org/2001/XMLSchema#")

# Load files
with open(SOURCE_JSON, encoding="utf-8") as f:
    source_data = json.load(f)

with open(MATCH_FILE, encoding="utf-8") as f:
    match_data = json.load(f)

# Build mapping: source_name -> matched_target_title
name_to_targettitle = {}
for m in match_data:
    if "top_match" in m and m["top_match"]:
        source_uri = m["source_uri"]
        target_label = m["top_match"].get("label")
        name = source_uri.split(":")[-1]
        name_to_targettitle[name] = target_label
        logging.info(f"Mapping source name '{name}' to target title '{target_label}'")

# Parse target ontology
target_graph = Graph()
target_graph.parse(TARGET_TTL, format="turtle")

# Build lookup: csvw:title (sl) -> {predicate: object}
target_column_details = {}
for col in target_graph.subjects(RDF.type, CSVW.Column):
    target_title = None
    for _, _, val in target_graph.triples((col, CSVW.title, None)):
        if isinstance(val, Literal) and val.language == "sl":
            target_title = str(val)
            break
    if target_title:
        pred_map = {}
        for p, o in target_graph.predicate_objects(subject=col):
            pred_map[p] = o
        target_column_details[target_title] = pred_map
        logging.info(f"Found target column title '{target_title}' with predicates: {[str(p) for p in pred_map]}")

# Create new graph
output_graph = Graph()
output_graph.bind("csvw", CSVW)
output_graph.bind("dct", DCT)
output_graph.bind("xsd", XSD)

# Create table and schema
table = URIRef("http://example.org/pn2024-100-table")
schema = BNode()
output_graph.add((table, RDF.type, CSVW.Table))
output_graph.add((table, CSVW.tableSchema, schema))

# Build columns
column_nodes = []
for col in source_data["tableSchema"]["columns"]:
    col_uri = URIRef(f"http://example.org/column/{col['name']}")
    output_graph.add((col_uri, RDF.type, CSVW.Column))
    output_graph.add((col_uri, CSVW.name, Literal(col["name"])))
    output_graph.add((col_uri, CSVW.title, Literal(col["title"], lang="sl")))
    output_graph.add((col_uri, CSVW.datatype, URIRef(XSD[col["datatype"]])))

    matched_title = name_to_targettitle.get(col['name'])
    if matched_title:
        if matched_title in target_column_details:
            pred_map = target_column_details[matched_title]
            for p, o in pred_map.items():
                if p not in [CSVW.name, CSVW.title, DCT.description]:
                    output_graph.add((col_uri, p, o))
                    logging.info(f"Added attribute to column '{col['name']}': {p} -> {o}")
        else:
            logging.warning(f"Matched title '{matched_title}' not found in target_column_details")
    else:
        logging.info(f"No match found for column '{col['name']}'")

    column_nodes.append(col_uri)

# Link columns to schema
for col_node in column_nodes:
    output_graph.add((schema, CSVW.column, col_node))

# Save output
output_graph.serialize(destination=OUTPUT_TTL, format="turtle")
logging.info(f"Generated: {OUTPUT_TTL}")