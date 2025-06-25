# Re-running after kernel reset
from rdflib import Graph, URIRef, RDFS, RDF, OWL, Literal
import os
import json

# Example paths - replace with your actual file paths
MOUSE_OWL_PATH = "/mnt/data/mouse.owl"
ALIGNMENT_RDF_PATH = "/mnt/data/reference-alignment.rdf"
HUMAN_NAMESPACE = "http://human.owl#"
MOUSE_NAMESPACE = "http://mouse.owl#"

# Load the mouse ontology
mouse_graph = Graph()
mouse_graph.parse(MOUSE_OWL_PATH)

# Load the alignment RDF to get gold mappings
alignment_graph = Graph()
alignment_graph.parse(ALIGNMENT_RDF_PATH)

# Extract gold alignments: source URI -> target URI
gold_mappings = {}
for s, p, o in alignment_graph.triples((None, URIRef("http://knowledgeweb.semanticweb.org/heterogeneity/alignment#map"), None)):
    cell = o
    entity1 = alignment_graph.value(cell, URIRef("http://knowledgeweb.semanticweb.org/heterogeneity/alignment#entity1"))
    entity2 = alignment_graph.value(cell, URIRef("http://knowledgeweb.semanticweb.org/heterogeneity/alignment#entity2"))
    if entity1 and entity2:
        mouse_uri = alignment_graph.value(entity1, URIRef("rdf:resource")) or entity1
        human_uri = alignment_graph.value(entity2, URIRef("rdf:resource")) or entity2
        if isinstance(mouse_uri, URIRef) and isinstance(human_uri, URIRef):
            gold_mappings[str(mouse_uri)] = str(human_uri)

# Extract all mouse classes and their labels/comments
testset = []
for s in mouse_graph.subjects(RDF.type, OWL.Class):
    label = mouse_graph.value(s, RDFS.label)
    comment = mouse_graph.value(s, RDFS.comment)
    if label:
        entry = {
            "uri": str(s),
            "label": str(label),
            "description": str(comment) if isinstance(comment, Literal) else "",
            "gold_uri": gold_mappings.get(str(s), "")
        }
        if entry["gold_uri"]:
            testset.append(entry)

# Save to JSON
output_path = "/mnt/data/mouse_testset.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(testset, f, indent=2, ensure_ascii=False)

output_path, len(testset)
