# %% [markdown]
# # Mouse Human Ontology Matching Pipeline

# %% [markdown]
# ## Setup & Config

# %% [markdown]
# ### Import required packages

# %%
from rdflib import Graph, URIRef, RDFS, RDF, OWL, Literal, Namespace
import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
# import re
import yaml
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
from rapidfuzz import fuzz

# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
# from tqdm import tqdm

# %% [markdown]
# #### Load paths and constans from configuration

# %%
CONFIG_PATH = "./config.yaml"
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# %%
ALIGNMENT_RDF_PATH = config["paths"]["mouse_human_alignment"]

HUMAN_OWL_PATH = config["paths"]["human_ontology"]
HUMAN_INDEX_PATH = config["paths"]["human_index"]
HUMAN_ID_TRACKER_PATH = config["paths"]["human_id_tracker"]
HUMAN_TERMS_JSON_PATH = config["paths"]["human_terms_json"]

MOUSE_OWL_PATH = config["paths"]["mouse_ontology"]
MOUSE_INDEX_PATH = config["paths"]["mouse_index"]
MOUSE_ID_TRACKER_PATH = config["paths"]["mouse_id_tracker"]
MOUSE_TERMS_JSON_PATH = config["paths"]["mouse_terms_json"]

HUMAN_NAMESPACE = "http://human.owl#"
MOUSE_NAMESPACE = "http://mouse.owl#"

EMBEDDING_MODEL_NAME = config["models"]["embedding_model"]
OBO = Namespace("http://www.geneontology.org/formats/oboInOwl#")

TESTSET_PATH = config["paths"]["mouse_testset"]

# %%
with open("./src/.hf_token", "r") as f:
            hf_token = f.read().strip()

# %% [markdown]
# #### Utils functions for ontologies

# %%
def load_graph(path):
    graph = Graph()
    graph.parse(path)
    return graph

# %%
def get_label(graph, uri):
    label = graph.value(uri, RDFS.label)
    return str(label) if isinstance(label, Literal) else None

# %%
def extract_related_uris(graph, subject, predicate):
    """Dereferences URIs linked by the predicate and returns their rdfs:label."""
    values = []
    for obj in graph.objects(subject, predicate):
        label = get_label(graph, obj)
        if label:
            values.append(label)
    return values

# %%
def extract_superclass_labels(graph, subject):
    """Get human-readable labels of direct superclasses."""
    super_labels = []
    for superclass in graph.objects(subject, RDFS.subClassOf):
        if isinstance(superclass, URIRef):
            label = get_label(graph, superclass)
            if label:
                super_labels.append(label)
        elif (superclass, RDF.type, OWL.Restriction) in graph:
            filler = graph.value(superclass, OWL.someValuesFrom)
            if isinstance(filler, URIRef):
                super_labels.append(str(filler).split("#")[-1])
    return super_labels

# %% [markdown]
# #### Utils functions for generating terms JSON

# %%
def build_text_for_embedding(label, definition=None, synonyms=None, superclasses=None):
    parts = [f"Concept: {label}"]

    if synonyms:
        parts.append(f"Also known as: {', '.join(synonyms)}")

    if superclasses:
        parts.append(f"Part of: {', '.join(superclasses)}")

    if definition:
        parts.append(f"Defined as: {definition}")

    return ". ".join(parts)

# %%
def extract_enriched_terms(graph):
    terms = []
    for s in graph.subjects(RDF.type, OWL.Class):
        label = get_label(graph, s)
        if not label:
            continue

        definition = extract_related_uris(graph, s, OBO.hasDefinition)
        synonyms = extract_related_uris(graph, s, OBO.hasRelatedSynonym)
        superclasses = extract_superclass_labels(graph, s)

        enriched_text = build_text_for_embedding(
            label=label,
            definition=definition[0] if definition else None,
            synonyms=synonyms,
            superclasses=superclasses
        )

        terms.append({
            "uri": str(s),
            "label": label,
            "definition": definition[0] if definition else "",
            "synonyms": synonyms,
            "superclasses": superclasses,
            "text_for_embedding": enriched_text
        })

    return terms

# %%
def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

# %%
def ontology_indexing(ontology_json, faiss_index_path, id_tracker_json):

    with open(ontology_json, "r", encoding="utf-8") as f:
            ontology_terms = json.load(f)

    texts = []
    ids = []
    valid_terms = []

    for i, term in enumerate(ontology_terms):
        text = term.get("text_for_embedding")
        if not text:
            raise ValueError("Missing 'text_for_embedding'")
        texts.append(text)
        ids.append(abs(hash(term["uri"])) % (10**12))
        valid_terms.append(term)

    # Embedding
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embeddings = model.encode(texts, batch_size=16, show_progress_bar=True)
    embeddings = normalize_embeddings(np.array(embeddings))

    # FAISS indexing
    dimension = embeddings.shape[1]
    base_index = faiss.IndexFlatIP(dimension)
    index = faiss.IndexIDMap(base_index)
    index.add_with_ids(embeddings, np.array(ids))
    os.makedirs(os.path.dirname(faiss_index_path), exist_ok=True)
    faiss.write_index(index, faiss_index_path)

    # Save ID tracker
    id_map = {str(id_): term for id_, term in zip(ids, valid_terms)}
    with open(id_tracker_json, "w", encoding="utf-8") as f:
        json.dump(id_map, f, indent=4, ensure_ascii=False)

# %% [markdown]
# #### Utils for matching

# %%
def rerank_by_label_similarity(source_label: str, candidates: List[Dict], weight_faiss: float = 0.7, weight_label: float = 0.3) -> List[Dict]:
    """Combines semantic score and lexical similarity to rerank matches."""
    reranked = []
    for match in candidates:
        label_sim = fuzz.ratio(source_label, match["label"]) / 100
        combined_score = weight_faiss * match["score"] + weight_label * label_sim
        reranked.append({**match, "combined_score": combined_score})
    return sorted(reranked, key=lambda x: x["combined_score"], reverse=True)

# %%
def build_index_lookup(index_path: str, id_tracker_path: str):
    index = faiss.read_index(index_path)
    with open(id_tracker_path, "r", encoding="utf-8") as f:
        id_map = {int(k): v for k, v in json.load(f).items()}
    return index, id_map

# %%
def faiss_batch_search(embeddings: np.ndarray, index, top_k: int):
    return index.search(embeddings, top_k)

# %%
def precompute_reverse_matches(
    target_terms: List[Dict],
    reverse_index,
    reverse_id_map: Dict[int, Dict],
    model,
    top_k: int = 1
) -> Dict[str, str]:
    """
    Computes best reverse matches (target → source).
    Returns dict: target_uri → best source_uri.
    """
    reverse_lookup = {}

    valid_terms = [t for t in target_terms if t.get("text_for_embedding")]
    texts = [t["text_for_embedding"] for t in valid_terms]
    uris = [t["uri"] for t in valid_terms]

    # Batch encoding
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)
    embeddings = normalize_embeddings(np.array(embeddings).astype(np.float32))

    # Batch FAISS search
    D, I = reverse_index.search(embeddings, top_k)

    for i, (indices, scores) in enumerate(zip(I, D)):
        best_idx = indices[0]
        if best_idx == -1:
            continue
        match = reverse_id_map.get(best_idx)
        if match:
            reverse_lookup[uris[i]] = match["uri"]

    return reverse_lookup


# %%
def precompute_reverse_matches_topk(
    target_terms: list,
    reverse_index,
    reverse_id_map: dict,
    model,
    top_k: int = 5
) -> dict:
    """
    Computes top-k reverse matches (target → source).
    Returns dict: target_uri → list of source_uris.
    """
    reverse_lookup_k = {}

    valid_terms = [t for t in target_terms if t.get("text_for_embedding")]
    texts = [t["text_for_embedding"] for t in valid_terms]
    uris = [t["uri"] for t in valid_terms]

    # Batch encoding
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Batch FAISS search
    D, I = reverse_index.search(np.array(embeddings).astype(np.float32), top_k)

    for i, indices in enumerate(I):
        matches = [reverse_id_map[idx]["uri"] for idx in indices if idx != -1 and reverse_id_map.get(idx)]
        reverse_lookup_k[uris[i]] = matches

    return reverse_lookup_k

# %%
def embed_terms(terms: List[Dict], model) -> np.ndarray:
    texts = [t["text_for_embedding"] for t in terms if t.get("text_for_embedding")]
    return normalize_embeddings(model.encode(texts, batch_size=32, show_progress_bar=True))


# %%
def map_faiss_results(source_terms, D, I, target_id_map):
    matches = []
    for i, (distances, indices) in enumerate(zip(D, I)):
        src = source_terms[i]
        results = []
        for idx, score in zip(indices, distances):
            if idx == -1:
                continue
            t = target_id_map.get(idx)
            if not t:
                continue
            results.append({
                "uri": t["uri"],
                "label": t["label"],
                "score": float(score)
            })
        matches.append({
            "source_uri": src["uri"],
            "source_label": src["label"],
            "top_k_matches": results
        })
    return matches


# %%
# model_name = "google/gemma-2b-it"  # or gemma-7b-it

# tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     device_map="auto",
#     torch_dtype="auto",  # or torch.float16
#     token=hf_token
# )

# llm = pipeline("text-generation", model=model, tokenizer=tokenizer)

# %%
# def rerank_with_gemma_batched(source_batch, llm, max_tokens=512):
#     """
#     Batched reranking using Gemma.

#     Args:
#         source_batch: List of (source_label, candidate_matches), where:
#                       - source_label is str
#                       - candidate_matches is List[Dict] with 'label', 'uri', 'comment'
#         llm: Hugging Face pipeline (text-generation)
#         max_tokens: Max tokens to generate

#     Returns:
#         List of best candidate dicts (one per input), or None if no match
#     """
#     prompts = []

#     for source_label, candidates in source_batch:
#         prompt = f"""You are an expert in biomedical ontologies.
# Given the source concept: "{source_label}", choose the best matching target concept from the list below.
# Respond with only the label of the best match.

# Target candidates:
# """
#         for c in candidates:
#             label = c["label"]
#             comment = c.get("comment", "")
#             prompt += f"- {label}: {comment}\n"

#         prompt += "\nBest match:"
#         prompts.append(prompt)

#     responses = llm(prompts, max_new_tokens=16, do_sample=False)

#     best_candidates = []
#     for (source_label, candidates), resp in zip(source_batch, responses):
#         try:
#             generated = resp["generated_text"]
#             selected_label = generated.split("Best match:")[-1].strip().split("\n")[0]

#             best = next((c for c in candidates if selected_label.lower() in c["label"].lower()), None)
#             best_candidates.append(best)
#         except Exception as e:
#             best_candidates.append(None)

#     return best_candidates

# %%
# def batch(iterable, size):
#     for i in range(0, len(iterable), size):
#         yield iterable[i:i+size]

# %%
def apply_label_reranking(matches: List[Dict]):
    reranked = []
    for match in matches:
        ranked = rerank_by_label_similarity(match["source_label"], match["top_k_matches"])
        match["top_k_matches"] = ranked
        match["top_match"] = ranked[0] if ranked else None
        reranked.append(match)
    return reranked


# %% [markdown]
# #### Loading RDF into graph

# %%
mouse_graph = load_graph(MOUSE_OWL_PATH)
human_graph = load_graph(HUMAN_OWL_PATH)
alignment_graph = load_graph(ALIGNMENT_RDF_PATH)

# %% [markdown]
# #### Loading embedding model

# %%
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# %% [markdown]
# #### Generate enriched JSON of ontology terms

# %% [markdown]
# Generate enriched terms for human ontology

# %%
human_terms = extract_enriched_terms(human_graph)

with open(HUMAN_TERMS_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(human_terms, f, indent=2, ensure_ascii=False)

# %%
len(human_terms)

# %% [markdown]
# Generate enriched terms for mouse ontology

# %%
mouse_terms = extract_enriched_terms(mouse_graph)

with open(MOUSE_TERMS_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(mouse_terms, f, indent=2, ensure_ascii=False)

# %%
len(mouse_terms)

# %% [markdown]
# #### Ontology indexing

# %% [markdown]
# Create and populate FAISS with human entities

# %%
ontology_indexing(ontology_json=HUMAN_TERMS_JSON_PATH, faiss_index_path=HUMAN_INDEX_PATH, id_tracker_json=HUMAN_ID_TRACKER_PATH)

# %% [markdown]
# Create and populate FAISS with mouse entities

# %%
ontology_indexing(ontology_json=MOUSE_TERMS_JSON_PATH, faiss_index_path=MOUSE_INDEX_PATH, id_tracker_json=MOUSE_ID_TRACKER_PATH)

# %% [markdown]
# #### Execute matching on mouse human pair of ontologies

# %%
human_index = faiss.read_index(HUMAN_INDEX_PATH)
with open(HUMAN_ID_TRACKER_PATH, "r", encoding="utf-8") as f:
    human_id_map = {int(k): v for k, v in json.load(f).items()}

# %%
mouse_index = faiss.read_index(MOUSE_INDEX_PATH)

# %%
embs = embed_terms(mouse_terms, embedding_model)

# %%
embs_h = embed_terms(human_terms, embedding_model)

# %%
len(human_id_map)

# %%
# Step-by-step matching
D, I = faiss_batch_search(embs, human_index, top_k=5)
matches = map_faiss_results(mouse_terms, D, I, human_id_map)

output_path = "./data/datasets/anatomy-dataset/mouse_to_human_matches_unranked.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(matches, f, indent=2, ensure_ascii=False)

print(f"len match: {len(matches)}")

# Optional refinements
matches = apply_label_reranking(matches)

output_path = "./data/datasets/anatomy-dataset/mouse_to_human_matches_ranked.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(matches, f, indent=2, ensure_ascii=False)

print(f"len match: {len(matches)}")


# %%
mouse_index = faiss.read_index(MOUSE_INDEX_PATH)
with open(MOUSE_ID_TRACKER_PATH, "r", encoding="utf-8") as f:
    mouse_id_map = {int(k): v for k, v in json.load(f).items()}

# %%
reverse_lookup = precompute_reverse_matches(
    target_terms=human_terms,
    reverse_index=mouse_index,
    reverse_id_map=mouse_id_map,
    model=embedding_model,
    top_k=5
)

# %%
reverse_lookup_topk = precompute_reverse_matches_topk(
    target_terms=human_terms,
    reverse_index=mouse_index,
    reverse_id_map=mouse_id_map,
    model=embedding_model,
    top_k=3
)

# %%
len(reverse_lookup)

# %%
output_path = "./data/datasets/anatomy-dataset/reverse_lookup.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(reverse_lookup, f, indent=2, ensure_ascii=False)

# %%
def apply_hcb(matches, reverse_lookup, fallback_threshold=0.95):
    filtered = []
    skipped_top_match = 0
    no_predicited_uri = 0
    for m in matches:
        source_uri = m["source_uri"]
        top_match = m.get("top_match")

        if not top_match:
            skipped_top_match += 1
            continue

        predicted_uri = top_match["uri"]
        confidence = top_match.get("combined_score", top_match.get("score", 0))

        if reverse_lookup.get(predicted_uri) == source_uri:
            filtered.append(m)  # standard HCB
        elif confidence >= fallback_threshold:
            filtered.append(m)  # allow fallback based on confidence
        else: 
            no_predicited_uri += 1

    print(f"skipped_top_match {skipped_top_match}")
    print(f"no_predicited_uri {no_predicited_uri}")

    return filtered


# %%
def apply_hcb_with_topk(matches, reverse_lookup_top1, reverse_lookup_topk):
    """
    Filters matches using bidirectional match (HCB), extended to top-k reverse lookup and confidence fallback.
    
    Args:
        matches: list of match dicts (with top_match + score)
        reverse_lookup_top1: dict[target_uri → best source_uri]
        reverse_lookup_topk: dict[target_uri → list of top-k source_uris]
        fallback_threshold: minimum score to allow fallback if HCB fails

    Returns:
        List of filtered matches
    """
    filtered = []
    stats = {
        "strict_hcb": 0,
        "semi_hcb": 0,
        "skipped": 0
    }

    for m in matches:
        source_uri = m["source_uri"]
        top_match = m.get("top_match")

        if not top_match:
            stats["skipped"] += 1
            continue

        predicted_uri = top_match["uri"]

        if reverse_lookup_top1.get(predicted_uri) == source_uri:
            filtered.append(m)
            stats["strict_hcb"] += 1
        elif source_uri in reverse_lookup_topk.get(predicted_uri, []):
            filtered.append(m)
            stats["semi_hcb"] += 1
        else:
            stats["skipped"] += 1

    print("HCB Filtering Summary:")
    for key, count in stats.items():
        print(f"  {key}: {count}")

    return filtered


# %%
matches = apply_hcb_with_topk(matches, reverse_lookup, reverse_lookup_topk)

len(matches)

# %%
output_path = "./data/datasets/anatomy-dataset/mouse_to_human_matches_3.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(matches, f, indent=2, ensure_ascii=False)

print(f"Matching complete. Results saved to {output_path}")

# %% [markdown]
# #### Prepare gold mappings

# %%
ALIGN = Namespace("http://knowledgeweb.semanticweb.org/heterogeneity/alignment")

gold_mappings = {}
for cell in alignment_graph.subjects(RDF.type, ALIGN.Cell):
    mouse_uri = alignment_graph.value(cell, ALIGN.entity1)
    human_uri = alignment_graph.value(cell, ALIGN.entity2)
    if isinstance(mouse_uri, URIRef) and isinstance(human_uri, URIRef):
        gold_mappings[str(mouse_uri)] = str(human_uri)

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

with open(TESTSET_PATH, "w", encoding="utf-8") as f:
    json.dump(testset, f, indent=2, ensure_ascii=False)

TESTSET_PATH, len(testset)

# %% [markdown]
# #### Evaluation

# %%
import json

THRESHOLD = 0.8

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def evaluate_predictions(matches, gold_lookup, threshold=0.8):
    """
    Evaluates predictions in three ways:
    1. OAEI Standard (default): Any confident prediction not matching gold = FP
    2. Relaxed: Skip evaluation if gold is missing
    3. Strict: Also penalize confident predictions for entries missing in gold
    """
    TP_oaei = FP_oaei = FN_oaei = 0
    TP_relaxed = FP_relaxed = FN_relaxed = 0
    total_predicted = 0

    for m in matches:
        mouse_uri = m["source_uri"]
        mouse_label = m["source_label"]
        top_k = m.get("top_k_matches", [])
        total_predicted += 1

        # reranked = rerank_by_label_similarity(mouse_label, top_k, 0.8, 0.2)
        reranked = top_k
        if not reranked or not reranked[0]:
            continue

        top_match = reranked[0]
        if top_match["combined_score"] < threshold:
            continue  # Not confident enough → not considered a prediction

        predicted_uri = top_match["uri"]
        gold_uri = gold_lookup.get(mouse_uri)

        # OAEI standard: penalize all confident predictions not matching gold
        if gold_uri is None:
            FP_oaei += 1
        elif predicted_uri == gold_uri:
            TP_oaei += 1
        else:
            FP_oaei += 1

        # Relaxed evaluation (only where gold exists)
        if gold_uri:
            if predicted_uri == gold_uri:
                TP_relaxed += 1
            else:
                FP_relaxed += 1
        # Relaxed FN still computed
        elif gold_uri and top_match["combined_score"] < threshold:
            FN_relaxed += 1

    # Compute total gold for recall
    total_gold = len(gold_lookup)
    FN_oaei = total_gold - TP_oaei

    # Metrics
    def metrics(tp, fp, fn):
        prec = tp / (tp + fp) if tp + fp > 0 else 0
        rec = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0
        return prec, rec, f1

    oaei = metrics(TP_oaei, FP_oaei, FN_oaei)
    relaxed = metrics(TP_relaxed, FP_relaxed, FN_relaxed)

    return {
        "OAEI": {"TP": TP_oaei, "FP": FP_oaei, "FN": FN_oaei, "metrics": oaei},
        "Relaxed": {"TP": TP_relaxed, "FP": FP_relaxed, "FN": FN_relaxed, "metrics": relaxed},
        "Counts": {"Total predicted": total_predicted, "Total gold": total_gold}
    }

def print_report(result):
    print("OAEI Standard Evaluation")
    p, r, f = result["OAEI"]["metrics"]
    print(f"Precision: {p:.2%}")
    print(f"Recall:    {r:.2%}")
    print(f"F1 Score:  {f:.2%}")
    print(f"TP: {result['OAEI']['TP']}  FP: {result['OAEI']['FP']}  FN: {result['OAEI']['FN']}")
    print("")

    print("Relaxed Evaluation (Only matched entries)")
    p, r, f = result["Relaxed"]["metrics"]
    print(f"Precision: {p:.2%}")
    print(f"Recall:    {r:.2%}")
    print(f"F1 Score:  {f:.2%}")
    print(f"TP: {result['Relaxed']['TP']}  FP: {result['Relaxed']['FP']}  FN: {result['Relaxed']['FN']}")
    print("")

    print("Counts")
    print(f"Total predictions attempted: {result['Counts']['Total predicted']}")
    print(f"Total in gold reference:     {result['Counts']['Total gold']}")


matches = load_json("./data/datasets/anatomy-dataset/mouse_to_human_matches_3.json")
testset = load_json("./data/datasets/anatomy-dataset/mouse_testset.json")
gold_lookup = {entry["uri"]: entry["gold_uri"] for entry in testset}

evaluation_result = evaluate_predictions(matches, gold_lookup, threshold=THRESHOLD)
print_report(evaluation_result)




