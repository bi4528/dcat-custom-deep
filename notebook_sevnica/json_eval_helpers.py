
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

def load_predictions_with_scores(
    path: str,
    source_label_field: str = "source_label",
    source_fallback_field: Optional[str] = "source_uri",
    candidates_field: str = "top_k_matches",
    candidate_label_field: str = "label",
    candidate_uri_field: Optional[str] = "uri",
    score_field_options: Optional[list] = None,
    top_k: int = 5,
):
    """
    Vrne: src -> [(label, score_or_None), ...] (dolžine do top_k), urejeno po najboljšem znanem score (desc).
    """
    if score_field_options is None:
        score_field_options = ["combined_score", "score"]

    data = json.load(open(path, "r", encoding="utf-8"))
    result: Dict[str, List[tuple]] = {}

    def best_score(c):
        for f in score_field_options:
            if f in c and c[f] is not None:
                try:
                    return float(c[f])
                except Exception:
                    pass
        return None

    for item in data:
        src = item.get(source_label_field) or (source_fallback_field and item.get(source_fallback_field)) or ""
        src = src.strip()
        if not src:
            continue

        cands = item.get(candidates_field, []) or []
        if any(best_score(c) is not None for c in cands):
            cands = sorted(cands, key=lambda c: (best_score(c) or float("-inf")), reverse=True)

        pairs: List[tuple] = []
        for c in cands[:top_k]:
            lab = c.get(candidate_label_field) if candidate_label_field in c else None
            if not lab and candidate_uri_field:
                lab = c.get(candidate_uri_field)
            pairs.append(((lab or "").strip(), best_score(c)))
        result[src] = pairs
    return result

def decide_label_with_threshold(pairs: List[tuple], threshold: float) -> Optional[str]:
    """
    pairs: [(label, score_or_None), ...] urejeno po desc score.
    Vrne prvi label s score >= threshold. Če ga ni, vrne None (NO_MATCH).
    Če score-ov ni, vrne None (raje zavrni kot napačno poveži).
    """
    for lab, sc in pairs:
        if sc is not None and sc >= threshold and lab:
            return lab.strip()
    return None  # NO_MATCH


# Na vrhu datoteke (če še nimaš):
from typing import Optional, List, Tuple, Dict, Set

def compute_open_world_metrics(
    gold_map: Dict[str, Set[str]],
    pred_pairs_map: Dict[str, List[Tuple[str, Optional[float]]]],
    threshold: float = 0.5,
    k: int = 5,
    exclude_no_match_from_mrr: bool = True,
):
    """
    Open-world odločitvene metrike (TP/FP/FN/TN + Precision/Recall/F1/Accuracy) in
    rangirne metrike med pozitivnimi (Acc@1, P@k, MRR).

    gold_map: src -> set(pravilnih target labelov); prazen set pomeni NO_MATCH (negativen primer)
    pred_pairs_map: src -> [(label, score_or_None), ...] urejeno po boljšem (desc), do k elementov
    threshold: če prvi kandidat s score >= threshold obstaja, napovemo 'poveži', sicer NO_MATCH
    k: cut-off za P@k in omejitev števila obravnavanih kandidatov
    exclude_no_match_from_mrr: če True, primeri brez zadetka ne znižujejo MRR (dodamo 0 le, če je False)
    """
    # Števci odločitev
    TP = FP = FN = TN = 0

    # Za rangiranje med pozitivnimi
    n_rel = 0
    acc1 = 0
    p_at_k_sum = 0.0
    mrr_sum = 0.0

    details: List[Dict] = []

    def norm(s: Optional[str]) -> str:
        return (s or "").strip().lower()

    # Fallback, če helper ni definiran
    _decide = None
    if "decide_label_with_threshold" in globals():
        _decide = globals()["decide_label_with_threshold"]
    else:
        def _decide(pairs: List[Tuple[str, Optional[float]]], t: float) -> Optional[str]:
            for lab, sc in pairs:
                if lab and sc is not None and sc >= t:
                    return lab.strip()
            return None

    for src, gold_set in gold_map.items():
        gold_norm = set(norm(x) for x in gold_set if x)
        is_no_match_gold = (len(gold_norm) == 0)

        pairs = pred_pairs_map.get(src, []) or []
        chosen = _decide(pairs, threshold)  # None => NO_MATCH

        # Odločitev (TP/FP/FN/TN)
        if is_no_match_gold:
            if chosen is None:
                TN += 1
                outcome = "TN"
            else:
                FP += 1
                outcome = "FP"
        else:
            if chosen is None:
                FN += 1
                outcome = "FN"
            else:
                if norm(chosen) in gold_norm:
                    TP += 1
                    outcome = "TP"
                else:
                    FP += 1
                    outcome = "FP"

        # Detajli (za izvoz/analizo)
        details.append({
            "source": src,
            "gold": sorted(list(gold_set)),
            "chosen": chosen,
            "outcome": outcome,
            "candidates": [lab for lab, _ in pairs],
            "scores": [sc for _, sc in pairs],
        })

        # Rangirne metrike računamo samo za pozitivne primere v zl. standardu
        if not is_no_match_gold:
            n_rel += 1
            preds_labels = [norm(lab) for lab, _ in pairs]

            # Acc@1
            if preds_labels and preds_labels[0] in gold_norm:
                acc1 += 1

            # P@k
            if preds_labels:
                denom = min(len(preds_labels), k)
                hits_in_k = sum(1 for p in preds_labels[:k] if p in gold_norm)
                p_at_k_sum += hits_in_k / denom

            # MRR
            rank_hit = None
            for i, p in enumerate(preds_labels, start=1):
                if p in gold_norm:
                    rank_hit = i
                    break
            if rank_hit is not None:
                mrr_sum += 1.0 / rank_hit
            elif not exclude_no_match_from_mrr:
                mrr_sum += 0.0  # eksplicitno, za jasnost

    # Odločitvene metrike
    precision = TP / (TP + FP) if (TP + FP) else 0.0
    recall    = TP / (TP + FN) if (TP + FN) else 0.0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    accuracy  = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) else 0.0

    # Povzetek rangirnih metrik
    ranking = {
        "count_positives": n_rel,
        "acc_at_1": (acc1 / n_rel) if n_rel else 0.0,
        "p_at_%d" % k: (p_at_k_sum / n_rel) if n_rel else 0.0,
        "mrr": (mrr_sum / n_rel) if n_rel else 0.0,
    }

    return {
        "decision_metrics": {
            "threshold": threshold,
            "TP": TP, "FP": FP, "FN": FN, "TN": TN,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
        },
        "ranking_metrics_among_positives": ranking,
        "details": details,
    }
