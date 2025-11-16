#!/usr/bin/env python
"""
Normalize / group attribute columns in a Brand × Attribute matrix using a local LLM (Ollama).

Input:
    data/processed/brand_attribute_matrix/filtered.csv

Output:
    data/processed/brand_attribute_matrix/attribute_groups.json
    data/processed/brand_attribute_matrix/attribute_mapping.json
    data/processed/brand_attribute_matrix/filtered_normalized.csv
"""

import os
import time
import json
import textwrap
from collections import defaultdict

import pandas as pd
import ollama


# =========================
# 1. LLM client + ask()
# =========================

MODEL = "gemma3:4b"


def init_client() -> ollama.Client:
    """
    Initialize the Ollama client and optionally warm up the model.
    """
    client = ollama.Client()

    # Warm up so the model is loaded into RAM (optional but nice)
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
    """
    Send a prompt to the Ollama model and return its text response.
    """
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
# 2. Prompt construction
# =========================

def build_grouping_prompt(attributes, brand_names):
    """
    Build a prompt asking the LLM to group attributes and normalize phrasing,
    using the actual matrix brands as context.

    Note: brands are *not always* technology brands; treat them as general
    product / service brands (could be tech, fashion, food, etc.).
    """
    # Compress brand list into a readable context string
    if len(brand_names) <= 15:
        brands_context = ", ".join(brand_names)
    else:
        # Show a sample + count
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
# 3. Grouping + mapping
# =========================

def safe_parse_json(raw: str) -> dict:
    """
    Try to robustly extract JSON from an LLM response that might be:
    - empty
    - wrapped in ```json ... ```
    - have extra text before/after the JSON.

    Raises a clear error if nothing JSON-like is found.
    """
    if raw is None:
        raise ValueError("LLM returned None instead of a string.")

    txt = raw.strip()

    if not txt:
        # This is exactly the error you're hitting: empty string → JSONDecodeError.
        raise ValueError(
            "LLM returned an empty response. "
            "Check that MODEL is correct and that client.generate(...) is working."
        )

    # If the response is wrapped in ```...``` fences, strip them.
    if txt.startswith("```"):
        # Strip leading and trailing fences safely
        # (anything like ```json ... ``` or ``` ...)
        lines = txt.splitlines()
        # Drop first and last line if they look like fences
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        txt = "\n".join(lines).strip()

    # At this point txt may still have extra explanation text.
    # Try to locate the first '{' and the last '}' and parse that slice.
    first_brace = txt.find("{")
    last_brace = txt.rfind("}")

    if first_brace == -1 or last_brace == -1 or last_brace < first_brace:
        # No JSON-looking content found
        print("=== RAW LLM RESPONSE (no JSON braces found) ===")
        print(raw)
        raise ValueError(
            "Could not find JSON object in LLM response. "
            "Check the prompt or log the raw response above."
        )

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

    - If there are many attributes, we process them in chunks to avoid hitting
      context limits.
    - Optionally, a 2nd pass merges canonical names across chunks.

    Returns:
        all_groups: list of { "canonical": str, "members": [str, ...] }
    """
    attributes = [a.strip() for a in attributes if a.strip()]
    all_groups = []

    # First pass: group within each chunk
    for start in range(0, len(attributes), chunk_size):
        chunk = attributes[start:start + chunk_size]
        print(f"[LLM] Grouping attributes {start}–{start + len(chunk) - 1} / {len(attributes)}")
        prompt = build_grouping_prompt(chunk, brand_names)
        raw = ask(prompt)
        data = safe_parse_json(raw)
        groups = data.get("groups", [])
        all_groups.extend(groups)

    # Optional second pass: unify canonical names if multiple chunks
    if len(attributes) > chunk_size:
        canonical_names = sorted({g["canonical"] for g in all_groups})
        print(f"[LLM] Second-pass merge over {len(canonical_names)} canonical names")
        prompt = build_grouping_prompt(canonical_names, brand_names)
        raw = ask(prompt)
        merged_data = safe_parse_json(raw)
        merged_groups = merged_data.get("groups", [])

        # Map old canonical name -> final canonical
        canonical_to_final = {}
        for g in merged_groups:
            final_canon = g["canonical"]
            for member in g.get("members", []):
                canonical_to_final[member] = final_canon

        # Rebuild all_groups with final canonicals
        new_all_groups = []
        for g in all_groups:
            old_canon = g["canonical"]
            final_canon = canonical_to_final.get(old_canon, old_canon)
            new_all_groups.append(
                {
                    "canonical": final_canon,
                    "members": g.get("members", []),
                }
            )
        all_groups = new_all_groups

    return all_groups


def build_attr_to_canonical(groups):
    """
    Convert list of groups into a flat mapping: original_attr -> canonical_attr.
    """
    mapping = {}
    for g in groups:
        canon = g["canonical"]
        for m in g.get("members", []):
            mapping[m] = canon
    return mapping


def apply_attribute_mapping(df, attr_to_canonical):
    """
    Given the original df and a mapping original -> canonical attribute,
    return a new df where:
      - columns are canonical attribute names
      - counts of synonym columns are summed.
    """
    # Ensure every column has a mapping (fallback: itself)
    complete_mapping = {
        col: attr_to_canonical.get(col, col)
        for col in df.columns
    }

    groups = defaultdict(list)
    for col, canon in complete_mapping.items():
        groups[canon].append(col)

    new_cols = {}
    for canon, cols in groups.items():
        if len(cols) == 1:
            new_cols[canon] = df[cols[0]]
        else:
            new_cols[canon] = df[cols].sum(axis=1)

    new_df = pd.DataFrame(new_cols, index=df.index)
    # Optional: sort columns alphabetically
    new_df = new_df.reindex(sorted(new_df.columns), axis=1)
    return new_df


# =========================
# 4. Main script
# =========================

def main():
    # Paths
    csv_path = "data/processed/brand_attribute_matrix/filtered.csv"
    groups_path = "data/processed/brand_attribute_matrix/attribute_groups.json"
    mapping_path = "data/processed/brand_attribute_matrix/attribute_mapping.json"
    normalized_path = "data/processed/brand_attribute_matrix/filtered_normalized.csv"

    # Ensure output directory exists
    os.makedirs(os.path.dirname(groups_path), exist_ok=True)

    # Load brand × attribute matrix
    print(f"Loading matrix from: {csv_path}")
    df = pd.read_csv(csv_path, index_col=0)

    brand_names = list(df.index)
    attributes = list(df.columns)

    print(f"Loaded {len(brand_names)} brands and {len(attributes)} attributes.")
    print("Example brands:", brand_names[:5])

    # 1) LLM grouping
    groups = group_attributes_with_llm(attributes, brand_names)

    # 2) Save raw groups
    with open(groups_path, "w", encoding="utf-8") as f:
        json.dump(groups, f, indent=2, ensure_ascii=False)
    print(f"Saved attribute groups to: {groups_path}")

    # 3) Build and save flat mapping
    attr_to_canonical = build_attr_to_canonical(groups)
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(attr_to_canonical, f, indent=2, ensure_ascii=False)
    print(f"Saved attribute mapping to: {mapping_path}")

    # 4) Apply mapping to matrix
    normalized_df = apply_attribute_mapping(df, attr_to_canonical)
    normalized_df.to_csv(normalized_path)
    print(f"Saved normalized brand × attribute matrix to: {normalized_path}")

    # Show a small preview
    print("\n=== Preview of normalized columns ===")
    print(list(normalized_df.columns)[:20])


if __name__ == "__main__":
    main()
