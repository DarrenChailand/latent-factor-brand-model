#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build Brand × Attribute Matrix from:
  1) data/processed/responses/Appendix_A_responses.jsonl ("response" field from each line)
  2) A small, hard-coded example list (same as in the prompt)

Outputs:
  data/processed/brand_attribute_matrix/examples_matrix.csv
  data/processed/brand_attribute_matrix/examples_heatmap.png
  data/processed/brand_attribute_matrix/jsonl_matrix.csv
  data/processed/brand_attribute_matrix/jsonl_heatmap.png

Usage:
  python build_brand_attribute_matrix.py \
    --input data/processed/responses/Appendix_A_responses.jsonl \
    --outdir data/processed/brand_attribute_matrix \
    [--prefix-adjectives]

Requirements (tested versions):
  pip install spacy==3.7.4 pandas==2.2.2 matplotlib==3.9.2
  python -m spacy download en_core_web_sm
"""

import argparse
import json
import os
from collections import defaultdict, Counter
from datetime import datetime  # <-- added

import spacy
from spacy.pipeline import EntityRuler
import pandas as pd
import matplotlib.pyplot as plt

# ================== Config ==================
BRANDS = ["Google", "Apple", "Samsung", "Microsoft", "Amazon", "Meta", "Nvidia", "Tesla"]

# ================== Small logger ==================
def _log_step(func_name: str, idx: int):
    print(f"[{datetime.now().isoformat()}] {func_name}: index={idx}")

# ================== NLP ==================
def build_nlp(brands=BRANDS):
    nlp = spacy.load("en_core_web_sm")
    ruler = nlp.add_pipe("entity_ruler", before="ner")
    ruler.add_patterns([{"label": "BRAND", "pattern": b} for b in brands])
    return nlp

# ================== Helpers ==================
def brand_spans(sent):
    return [ent for ent in sent.ents if ent.label_ == "BRAND"]

def token_is_brand(token, brand_ents):
    return any(ent.start <= token.i < ent.end for ent in brand_ents)

def last_nonbrand_noun_before(tokens, idx, brand_ents):
    for t in reversed(tokens[:idx]):
        if t.is_alpha and t.pos_ in ("NOUN","PROPN") and not token_is_brand(t, brand_ents):
            return t
    return None

def noun_phrase_from_head(head):
    """
    Return ONLY noun–noun compounds + head (no adjectives, no adverbs).
    Examples:
      'image sensor' -> 'image sensor' (kept: compound nouns)
      'well-designed camera' -> 'camera' (dropped 'well-designed')
    """
    left = [t for t in head.lefts
            if t.dep_ == "compound" and t.is_alpha and t.pos_ in ("NOUN","PROPN")]
    right = [t for t in head.rights
             if t.dep_ == "compound" and t.is_alpha and t.pos_ in ("NOUN","PROPN")]
    toks = sorted(left + [head] + right, key=lambda x: x.i)
    return " ".join(t.lemma_.lower() for t in toks)

def add_feature(counter_map, brands, head_token):
    phrase = noun_phrase_from_head(head_token)
    if phrase:
        for b in brands:
            counter_map[b][phrase] += 1

# ================== Core logic ==================
def handle_comparatives_than_only(sent, brand_features, brand_ents):
    """
    If sentence matches '... <feature NOUN> ... than <BRAND> ...',
    add that feature to BOTH: every brand before 'than' and the next brand after 'than'.
    """
    tokens = list(sent)
    for i, tok in enumerate(tokens):
        if tok.lemma_.lower() != "than":
            continue
        right_brand_ents = [ent for ent in brand_ents if ent.start >= tok.i]
        left_brand_ents  = [ent for ent in brand_ents if ent.end <= tok.i]
        if not right_brand_ents or not left_brand_ents:
            continue
        right_brand = min(right_brand_ents, key=lambda e: e.start).text
        left_brands = list({ent.text for ent in left_brand_ents})
        feat_head = last_nonbrand_noun_before(tokens, i, brand_ents)
        if feat_head is not None:
            add_feature(brand_features, left_brands + [right_brand], feat_head)

def process(texts, nlp):
    """
    Returns:
      per_brand_features: {brand -> Counter(noun/proper-noun features)}
      per_brand_descriptors: {brand -> Counter(adjectives)}
      overall_features: Counter of all features (nouns/proper-nouns)
    """
    per_brand_features = defaultdict(Counter)
    per_brand_descriptors = defaultdict(Counter)
    overall_features = Counter()

    for i, text in enumerate(texts):
        _log_step('process', i)  # <-- added
        doc = nlp(text)
        for sent in doc.sents:
            b_ents = brand_spans(sent)
            if not b_ents:
                continue
            sent_brands = [ent.text for ent in b_ents]

            # (A) comparative rule (pure 'than')
            handle_comparatives_than_only(sent, per_brand_features, b_ents)

            # (B) count-everything per sentence (skip brand tokens as features)
            for tok in sent:
                if tok.is_stop or not tok.is_alpha:
                    continue
                if token_is_brand(tok, b_ents):
                    continue
                if tok.pos_ in ("NOUN","PROPN"):
                    feat = tok.lemma_.lower()
                    overall_features[feat] += 1
                    for b in sent_brands:
                        per_brand_features[b][feat] += 1
                elif tok.pos_ == "ADJ":
                    adj = tok.lemma_.lower()
                    for b in sent_brands:
                        per_brand_descriptors[b][adj] += 1

    return per_brand_features, per_brand_descriptors, overall_features

# ================== Brand × Attribute matrix ==================
def build_attribute_counts(per_brand_features, per_brand_descriptors, prefix_adjectives=False):
    brand_attrs = {}
    for brand in sorted(set(per_brand_features) | set(per_brand_descriptors)):
        feats = per_brand_features.get(brand, Counter())
        descs = per_brand_descriptors.get(brand, Counter())
        merged = Counter(feats)
        if prefix_adjectives:
            merged.update({f"adj:{k}": v for k, v in descs.items()})
        else:
            merged.update(descs)  # collisions allowed: noun/adjective same surface form
        brand_attrs[brand] = merged
    return brand_attrs

def build_attribute_matrix(brand_attribute_counts):
    all_attrs = sorted({a for _, c in brand_attribute_counts.items() for a in c})
    brands = sorted(brand_attribute_counts.keys())
    data = [[brand_attribute_counts[b].get(a, 0) for a in all_attrs] for b in brands]
    return pd.DataFrame(data, index=brands, columns=all_attrs)

def save_outputs(df, outdir, stem):
    os.makedirs(outdir, exist_ok=True)
    csv_path = os.path.join(outdir, f"{stem}_matrix.csv")
    png_path = os.path.join(outdir, f"{stem}_heatmap.png")
    df.to_csv(csv_path, index=True)

    # Heatmap
    plt.figure()
    plt.imshow(df.values, aspect="auto")
    plt.xticks(range(len(df.columns)), df.columns, rotation=90)
    plt.yticks(range(len(df.index)), df.index)
    plt.title(f"Brand × Attribute Counts ({stem})")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(png_path, dpi=180)
    plt.close()

    print(f"Saved: {csv_path}")
    print(f"Saved: {png_path}")

# ================== I/O helpers ==================
def read_jsonl_responses(path):
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            _log_step('read_jsonl_responses', i)  # <-- added
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "response" in obj and isinstance(obj["response"], str):
                texts.append(obj["response"])
    return texts

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str,
                        default="data/processed/responses/Appendix_A_responses.jsonl",
                        help="Path to JSONL with lines containing a 'response' field.")
    parser.add_argument("--outdir", type=str,
                        default="data/processed/brand_attribute_matrix",
                        help="Directory to save CSVs and heatmaps.")
    parser.add_argument("--prefix-adjectives", action="store_true",
                        help="If set, adjectives are prefixed as 'adj:...' columns.")
    args = parser.parse_args()

    nlp = build_nlp()

    # ---------- JSONL responses ----------
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input JSONL not found: {args.input}")

    jsonl_texts = read_jsonl_responses(args.input)
    if len(jsonl_texts) == 0:
        print("Warning: no 'response' strings found in the JSONL.")
    bfeat_j, bdesc_j, _ = process(jsonl_texts, nlp)
    counts_j = build_attribute_counts(bfeat_j, bdesc_j, prefix_adjectives=args.prefix_adjectives)
    df_j = build_attribute_matrix(counts_j)
    save_outputs(df_j, args.outdir, "raw")

    print("\nDone.")

if __name__ == "__main__":
    main()
