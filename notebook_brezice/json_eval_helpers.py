
import json
from typing import Dict, List, Set, Optional

def load_gold_json(path: str) -> Dict[str, Set[str]]:
    """
    Load gold standard JSON and return: source_column -> set(valid target labels)
    Expected structure (list of dicts with keys: source_column, target_columns).
    """
    data = json.load(open(path, "r", encoding="utf-8"))
    gold = {}
    for item in data:
        src = (item.get("source_column") or "").strip()
        tgts = set()
        for t in item.get("target_columns", []):
            if t:
                tgts.add(str(t).strip())
        if src:
            gold[src] = tgts
    return gold

def load_predictions_json(
    path: str,
    source_label_field: str = "source_label",
    source_fallback_field: Optional[str] = "source_uri",
    candidates_field: str = "top_k_matches",
    candidate_label_field: str = "label",
    candidate_uri_field: Optional[str] = "uri",
    score_field_options: Optional[list] = None,
    top_k: int = 5,
) -> Dict[str, List[str]]:
    """
    Load predictions JSON like 'opsi_match.json' and produce:
    source_label -> [candidate_label_1, candidate_label_2, ..., candidate_label_k]
    If candidate_label_field missing, we can fallback to candidate_uri_field.
    Score is not required here, as ordering is assumed to be already top-k;
    otherwise, you can provide score_field_options in priority order to sort.
    """
    if score_field_options is None:
        score_field_options = ["combined_score", "score"]
    data = json.load(open(path, "r", encoding="utf-8"))
    topk_map: Dict[str, List[str]] = {}

    for item in data:
        src = item.get(source_label_field)
        if (not src) and source_fallback_field:
            src = item.get(source_fallback_field)
        src = (src or "").strip()
        if not src:
            continue

        cands = item.get(candidates_field, []) or []

        # sort by best available score if provided
        def best_score(c):
            for f in score_field_options:
                if f in c and c[f] is not None:
                    try:
                        return float(c[f])
                    except Exception:
                        pass
            return None

        # If scores exist, sort desc by score; else keep as is
        if any(best_score(c) is not None for c in cands):
            cands = sorted(cands, key=lambda c: (best_score(c) or float("-inf")), reverse=True)

        labels: List[str] = []
        for c in cands[:top_k]:
            lab = c.get(candidate_label_field) if candidate_label_field in c else None
            if not lab and candidate_uri_field:
                lab = c.get(candidate_uri_field)
            labels.append((lab or "").strip())

        topk_map[src] = labels

    return topk_map

def normalize(s: str) -> str:
    return (s or "").strip().lower()

def compute_metrics_from_maps(topk_map: Dict[str, List[str]], gold_map: Dict[str, Set[str]], k: int = 5) -> dict:
    """
    Compute Acc@1, P@5, and MRR using label equality (case-insensitive, trimmed).
    """
    n = 0
    acc1 = 0
    p_at_k_sum = 0.0
    mrr_sum = 0.0
    details = []

    for src, gold_set in gold_map.items():
        n += 1
        preds = [normalize(x) for x in topk_map.get(src, [])]
        gold_norm = set(normalize(x) for x in gold_set if x)
        hit = False
        rank_hit = None
        hits_in_k = 0

        for i, p in enumerate(preds, start=1):
            if p in gold_norm:
                hit = True
                if rank_hit is None:
                    rank_hit = i
                hits_in_k += 1

        if preds:
            if rank_hit == 1:
                acc1 += 1
            p_at_k_sum += (hits_in_k / min(len(preds), k))
        if rank_hit is not None:
            mrr_sum += 1.0 / rank_hit

        details.append({
            "source_column": src,
            "gold": sorted(list(gold_set)),
            "predictions": topk_map.get(src, []),
            "hit": hit,
            "rank_hit": rank_hit,
            "hits_in_k": hits_in_k
        })

    return {
        "count": n,
        "acc_at_1": acc1 / n if n else 0.0,
        "p_at_5": p_at_k_sum / n if n else 0.0,
        "mrr": mrr_sum / n if n else 0.0,
        "details": details
    }
