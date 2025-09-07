
import os, re, json, pandas as pd
from pathlib import Path

def norm_uri(label: str, prefix: str="col") -> str:
    s = str(label).strip().lower()
    s = re.sub(r"[^\w]+", "_", s, flags=re.UNICODE)
    s = re.sub(r"_+", "_", s).strip("_")
    return f"{prefix}:{s}"

def detect_kind(series: pd.Series):
    sample = series.dropna().astype(str).head(40).tolist()
    if not sample:
        return "text"
    truthy = {"da","ne","true","false","yes","no","y","n","1","0"}
    if sum(1 for v in sample if v.strip().lower() in truthy) / len(sample) > 0.7:
        return "boolean"
    nums = 0; ints = 0
    for v in sample:
        v2 = v.replace(",", ".")
        try:
            f = float(v2); nums += 1
            if "." not in v2: ints += 1
        except Exception:
            pass
    if nums/len(sample) > 0.8:
        return "integer" if ints/max(nums,1) > 0.9 else "decimal"
    if sum(1 for v in sample if re.search(r"\b-?\d{1,3}\.\d{3,}", v)) / len(sample) > 0.5:
        return "gps"
    if sum(1 for v in sample if re.search(r"\d{4}-\d{2}-\d{2}|\d{1,2}\.\s?\d{1,2}\.\s?\d{2,4}", v)) / len(sample) > 0.4:
        return "date"
    return "text"

def build_definition(label: str, kind: str, examples: list, context_cols: list) -> str:
    ex = "; ".join(str(x) for x in examples[:2]) if examples else ""
    cx = ", ".join([c for c in context_cols if str(c) != str(label)][:3])
    if kind == "gps":
        core = f"geografski podatek o legi parkirišča"
    elif kind == "integer":
        core = f"številski podatek o količini ali kapaciteti parkirišča"
    elif kind == "decimal":
        core = f"numerični podatek, povezan s parkirišči"
    elif kind == "boolean":
        core = f"dvovrednostni (DA/NE) podatek o lastnosti parkirišča"
    elif kind == "date":
        core = f"datumovni podatek, ki označuje časovno lastnost zapisa"
    else:
        core = f"opisni podatek o parkirišču v kontekstu stolpcev {cx}" if cx else "opisni podatek o parkirišču"
    if ex:
        return f"Stolpec '{label}' predstavlja {core} (npr. {ex})."
    else:
        return f"Stolpec '{label}' predstavlja {core}."

def build_text_for_embedding(label: str, definition: str, examples: list) -> str:
    ex_str = ", ".join(str(x) for x in examples[:3]) if examples else ""
    parts = [f"Concept: {label}."]
    if ex_str:
        parts.append(f"Examples: {ex_str}.")
    parts.append(f"Defined as: {definition}")
    return " ".join(parts)

def dataframe_to_target(df: pd.DataFrame, uri_prefix="col"):
    cols = list(df.columns)
    target = []
    for c in cols:
        series = df[c]
        ex_vals = pd.Series(series.dropna().astype(str).unique()).head(5).tolist()
        kind = detect_kind(series)
        definition = build_definition(str(c), kind, ex_vals, cols)
        target.append({
            "uri": norm_uri(c, prefix=uri_prefix),
            "label": str(c),
            "definition": definition,
            "synonyms": [],
            "superclasses": [],
            "text_for_embedding": build_text_for_embedding(str(c), definition, ex_vals),
        })
    return target

def generate_target_from_file(input_path: str, output_path: str, sheet: str=None, uri_prefix="col"):
    ext = os.path.splitext(input_path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        xls = pd.ExcelFile(input_path)
        sheet_name = sheet or xls.sheet_names[0]
        df = pd.read_excel(input_path, sheet_name=sheet_name)
    else:
        df = pd.read_csv(input_path)
    target = dataframe_to_target(df, uri_prefix=uri_prefix)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(target, f, ensure_ascii=False, indent=2)
    return output_path

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Generate target ontology JSON with Examples and 'Defined as' order for text_for_embedding.")
    ap.add_argument("input_path", help="Path to CSV/XLSX file")
    ap.add_argument("output_path", help="Where to save JSON")
    ap.add_argument("--sheet", help="Excel sheet name", default=None)
    ap.add_argument("--uri-prefix", help="URI prefix (e.g., 'sev')", default="col")
    args = ap.parse_args()
    out = generate_target_from_file(args.input_path, args.output_path, sheet=args.sheet, uri_prefix=args.uri_prefix)
    print(out)
