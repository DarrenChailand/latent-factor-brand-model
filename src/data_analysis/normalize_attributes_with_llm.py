#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Normalize / group attribute columns in a Brand × Attribute matrix using a local LLM (Ollama).

Inputs:
  - filtered.csv (brands × attributes)

Outputs (written to outdir):
  - attribute_groups.json
  - attribute_mapping.json
  - filtered_normalized.csv
"""

import json
import os
import time
import textwrap
from collections import defaultdict

import ollama
import pandas as pd


# =========================
# 1) LLM client + ask()
# =========================

MODEL = "gemma3:4b"


def init_client() -> ollama.Client:
    """Initialize the Ollama client and optionally warm up the model."""
    client = ollama.Client()

    _ = client.generate(
        model=MODEL,
        prompt="OK",
        options={
            "num_ctx": 512,
            "num_predict": 16,
            "num_thread": os.cpu_count(),
        },
        keep_alive="10m",
    )
    return client


CLIENT = init_client()


def ask(prompt: str) -> str:
    """Send a prompt to the Ollama model and return its text response."""
    t0 = time.time()
    r = CLIENT.generate(
        model=MODEL,
        prompt=prompt.strip(),
        options={
            "num_ctx": 1024,
            "num_predict": -1,
            "num_thread": os.cpu_count(),
        },
        keep_alive="10m",
        stream=False,
    )
    dt = time.time() - t0
    print(f"[LLM] {len(r.response)} chars in {dt:.2f}s")
    return r.response


# =========================
# 2) Prompt construction
# =========================

def build_grouping_prompt(attributes, brand_names):
    """
    Build a prompt asking the LLM to group attributes and normalize phrasing,
    using the actual matrix brands as context.

    Note: brands are not always technology brands; treat them as general product/service brands.
    """
    if len(brand_names) <= 15:
        brands_context = ", ".join(brand_names)
    else:
        sample = ", ".join(list(brand_names)[:10])
        brands_context = f"{sample}, ... (total {len(brand_names)} brands)"

    attr_block = "\n".join(f"- {a}" for a in attributes)

    prompt = f"""
    You are helping to clean up column names in a Brand × Attribute matrix.

    Matrix structure:
    - Each ROW is a **brand** (not necessarily a technology brand). Examples from this matrix include:
      {brands_context}
    - Each COLUMN is a candidate **attribute** that describes how people talk about that brand:
      product features, service quality, experience, reputation, etc.
    - Each CELL (brand, attribute) is a count of how often that attribute is mentioned in
      descriptions of that brand.

    Your task:
    - Group **synonyms, near-synonyms, spelling variants, and simple phrasing variations**
      that represent essentially the same brand attribute.
    - For each group, choose a **short, clear, canonical attribute name** that would work
      as a column label in a brand-perception matrix.
    - Think in terms of what a consumer would perceive as the *same underlying idea*.

    Important:
    - Brands are general product/service brands (they may be tech, fashion, food, finance, etc.).
    - Attributes can describe:
        • physical product features (e.g., camera, battery capacity, fabric quality)
        • performance (e.g., speed, reliability, durability)
        • usability / experience (e.g., ease of use, user interface)
        • trust / perception (e.g., privacy, security, prestige)
    - You should still recognize common tech-related terms (e.g. "OLED", "GPU", "battery life"),
      but do not assume all brands are tech-only.

    Grouping guidelines:
    - Group together:
        • singular vs plural (e.g. "display" vs "displays")
        • minor wording differences (e.g. "screen" vs "display")
        • obvious duplicates ("touchscreen" vs "touch screens")
    - Keep separate when they are meaningfully different to brand perception, for example:
        • "battery capacity" vs "battery life" → different ideas (size vs duration)
        • "GPU" vs "GPU performance" → component vs performance judgement
        • "privacy" vs "security" → related but distinct concerns

    Constraints:
    - Every input attribute must appear in **exactly one** group.
    - Canonical names should be:
        • concise (1–3 words),
        • easy to read,
        • no weird punctuation.

    Output format:
    - Return **only valid JSON**, no explanations or extra text.
    - Use this exact structure:

    {{
      "groups": [
        {{
          "canonical": "battery life",
          "members": ["battery", "battery life", "iphone battery"]
        }},
        {{
          "canonical": "display",
          "members": ["display", "displays", "screen", "quality display"]
        }}
      ]
    }}

    Now here are the attributes you must group and normalize:

    {attr_block}
    """
    return textwrap.dedent(prompt).strip()


# =========================
# 3) Grouping + mapping
# =========================

def safe_parse_json(raw: str) -> dict:
    """
    Extract JSON from an LLM response that might be:
    - empty
    - wrapped in ```json ... ```
    - have extra text before/after JSON
    """
    if raw is None:
        raise ValueError("LLM returned None instead of a string.")

    txt = raw.strip()
    if not txt:
        raise ValueError(
            "LLM returned an empty response. "
            "Check that MODEL is correct and that client.generate(...) is working."
        )

    if txt.startswith("```"):
        lines = txt.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        txt = "\n".join(lines).strip()

    first_brace = txt.find("{")
    last_brace = txt.rfind("}")

    if first_brace == -1 or last_brace == -1 or last_brace < first_brace:
        print("=== RAW LLM RESPONSE (no JSON braces found) ===")
        print(raw)
        raise ValueError("Could not find JSON object in LLM response.")

    candidate = txt[first_brace:last_brace + 1]

    try:
        return json.loads(candidate)
    except json.JSONDecodeError as e:
        print("=== RAW LLM RESPONSE THAT FAILED JSON PARSE ===")
        print(raw)
        print("=== JSON CANDIDATE SUBSTRING ===")
        print(candidate)
        raise e


def group_attributes_with_llm(attributes, brand_names, chunk_size=60):
    """
    Use the LLM to group/normalize attribute names.

    Returns:
      all_groups: list of {"canonical": str, "members": [str, ...]}
    """
    attributes = [a.strip() for a in attributes if a.strip()]
    all_groups = []

    for start in range(0, len(attributes), chunk_size):
        chunk = attributes[start:start + chunk_size]
        print(f"[LLM] Grouping attributes {start}–{start + len(chunk) - 1} / {len(attributes)}")
        prompt = build_grouping_prompt(chunk, brand_names)
        raw = ask(prompt)
        data = safe_parse_json(raw)
        groups = data.get("groups", [])
        all_groups.extend(groups)

    if len(attributes) > chunk_size:
        canonical_names = sorted({g["canonical"] for g in all_groups})
        print(f"[LLM] Second-pass merge over {len(canonical_names)} canonical names")
        prompt = build_grouping_prompt(canonical_names, brand_names)
        raw = ask(prompt)
        merged_data = safe_parse_json(raw)
        merged_groups = merged_data.get("groups", [])

        canonical_to_final = {}
        for g in merged_groups:
            final_canon = g["canonical"]
            for member in g.get("members", []):
                canonical_to_final[member] = final_canon

        new_all_groups = []
        for g in all_groups:
            old_canon = g["canonical"]
            final_canon = canonical_to_final.get(old_canon, old_canon)
            new_all_groups.append(
                {"canonical": final_canon, "members": g.get("members", [])}
            )
        all_groups = new_all_groups

    return all_groups


def build_attr_to_canonical(groups):
    """Convert list of groups into a flat mapping: original_attr -> canonical_attr."""
    mapping = {}
    for g in groups:
        canon = g["canonical"]
        for m in g.get("members", []):
            mapping[m] = canon
    return mapping


def apply_attribute_mapping(df, attr_to_canonical):
    """
    Given df and mapping original -> canonical attribute, return a new df where:
      - columns are canonical attribute names
      - synonym columns are summed
    """
    complete_mapping = {col: attr_to_canonical.get(col, col) for col in df.columns}

    grouped_cols = defaultdict(list)
    for col, canon in complete_mapping.items():
        grouped_cols[canon].append(col)

    new_cols = {}
    for canon, cols in grouped_cols.items():
        if len(cols) == 1:
            new_cols[canon] = df[cols[0]]
        else:
            new_cols[canon] = df[cols].sum(axis=1)

    new_df = pd.DataFrame(new_cols, index=df.index)
    new_df = new_df.reindex(sorted(new_df.columns), axis=1)
    return new_df


# =========================
# Notebook-friendly entry point
# =========================

def run_normalize_attributes(
    input_csv: str,
    outdir: str = "data/processed/brand_attribute_matrix",
    chunk_size: int = 60,
) -> str:
    """
    Notebook-friendly wrapper.

    Writes:
      - attribute_groups.json
      - attribute_mapping.json
      - filtered_normalized.csv

    Returns normalized CSV path.
    """
    groups_path = os.path.join(outdir, "attribute_groups.json")
    mapping_path = os.path.join(outdir, "attribute_mapping.json")
    normalized_path = os.path.join(outdir, "filtered_normalized.csv")

    os.makedirs(outdir, exist_ok=True)

    print(f"Loading matrix from: {input_csv}")
    df = pd.read_csv(input_csv, index_col=0)

    brand_names = list(df.index)
    attributes = list(df.columns)

    print(f"Loaded {len(brand_names)} brands and {len(attributes)} attributes.")
    print("Example brands:", brand_names[:5])

    groups = group_attributes_with_llm(attributes, brand_names, chunk_size=chunk_size)

    with open(groups_path, "w", encoding="utf-8") as f:
        json.dump(groups, f, indent=2, ensure_ascii=False)
    print(f"Saved attribute groups to: {groups_path}")

    attr_to_canonical = build_attr_to_canonical(groups)
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(attr_to_canonical, f, indent=2, ensure_ascii=False)
    print(f"Saved attribute mapping to: {mapping_path}")

    normalized_df = apply_attribute_mapping(df, attr_to_canonical)
    normalized_df.to_csv(normalized_path)
    print(f"Saved normalized brand × attribute matrix to: {normalized_path}")

    print("\n=== Preview of normalized columns ===")
    print(list(normalized_df.columns)[:20])

    return normalized_path
