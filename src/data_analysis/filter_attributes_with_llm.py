#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Filter Brand × Attribute matrix columns using an Ollama LLM.

Usage (example):
  python filter_attributes_with_llm.py \
      --input data/processed/brand_attribute_matrix/raw.csv \
      --output data/processed/brand_attribute_matrix/filtered.csv \
      --decisions-log data/processed/brand_attribute_matrix/attribute_decisions.csv
"""

import argparse
import json
import os
import time
from typing import Literal, Dict, List, Optional

from pathlib import Path

import pandas as pd
import ollama

# ================== Ollama setup (same style as your old script) ==================

client = ollama.Client()
MODEL = "gemma3:4b"

# Warm up so the model is loaded into RAM (optional but nice)
_ = client.generate(
    model=MODEL,
    prompt="OK",
    options={"num_ctx": 512, "num_predict": 16, "num_thread": os.cpu_count()},
    keep_alive="10m",
)


def ask(prompt: str) -> str:
    """
    Send a prompt to the Ollama model and return its text response.
    """
    t0 = time.time()
    r = client.generate(
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
    print(f"[{len(r.response)} chars in {dt:.2f}s]")
    return r.response


# ================== Prompt template ==================
# Note: {brands_context} and {attribute} will be filled in at runtime.  # <<< CHANGED

PROMPT_TEMPLATE = """You are cleaning up columns in a Brand × Attribute matrix.

Context about the matrix:
- Each ROW is a technology brand (e.g. Apple, Google, Samsung, Nvidia, Tesla, Xiaomi, etc.).
- Each COLUMN is a candidate attribute (a word or short phrase).
- Each CELL (brand, attribute) is a count of how many times that attribute was mentioned
  in natural-language descriptions of that brand (from LLM-generated product text).

The brands currently present as rows in this matrix include:
{brands_context}

Your task:
For a given candidate attribute (a column name), decide whether it is a **meaningful
product/brand attribute** that should remain as a column in the matrix.

Be VERY STRICT. If you are not clearly convinced that it is a product/brand attribute,
you MUST answer DROP.

----------------------------
KEEP if the candidate is:
----------------------------
  - A product feature or component
      e.g. "camera", "battery", "battery life", "screen", "display", "chip", "CPU",
           "GPU", "AI assistant", "image sensor", "performance", "telephoto",
           "thermal", "touch id", "user interface"
  - A quality / perception adjective or phrase about the product or user experience
      e.g. "reliable", "innovative", "expensive", "cheap", "fast", "smooth",
           "user-friendly", "secure", "private", "unreliable", "unresponsive",
           "understated", "underwhelming", "vibrant", "vivid"
  - A usage or experience concept clearly related to products or services
      e.g. "ecosystem", "integration", "usability", "privacy", "security",
           "compatibility", "stability", "latency", "durability",
           "user experience", "update policy", "wearables"

----------------------------
DROP in ALL these cases:
----------------------------
  - Company names, stock tickers, or corporate groups
      e.g. "AAPL", "Alphabet", "Meta", "TSMC", "Verizon", "Trustpilot", "YouTube"
  - Regions / countries / continents / demonyms / days / time words
      e.g. "america", "africa", "asia", "uk", "tuesday", "today", "year", "week"
  - Generic people / organization / process / meta words
      e.g. "team", "teacher", "user", "vendor", "university", "tutorial",
           "trade", "traffic", "topic", "test", "testing", "update", "updates"
  - Generic verbs or action words that do not describe a specific product feature
      e.g. "taking", "talk", "think", "work", "writing", "translate", "use",
           "uses", "using", "wait", "watching"
  - Very generic abstract nouns with no clear product-specific meaning
      e.g. "ability", "addition", "amount", "age", "activity", "thing", "time",
           "world", "word", "title", "trend", "trends"
  - Words mostly about organization, finance, or marketing
      e.g. "acquisition", "advertising", "adwords", "adsense", "trade", "trading"
  - Product line codes / vague series names / short codes
      e.g. "a series", "xl", "uhd", "ufs", "tflop", "tb", "tsmc", "xe"
  - Extremely vague adjectives without clear relation to product qualities
      e.g. "true", "vast", "typical", "ultimate"
  - Anything that looks like a function word, filler, grammar artifact
      e.g. "up", "well", "w", "z", "x"

Borderline case rule:
- If you are unsure, you MUST choose "DROP".
- Only choose "KEEP" when the word clearly describes how users perceive or use the product,
  or a concrete component/feature (e.g. "touchscreen", "wearables", "wallet", "zoom").

Now classify this attribute candidate:

  attribute: "{attribute}"

Return your answer in **exactly** this JSON format with no extra text:

{{
  "attribute": "{attribute}",
  "decision": "KEEP" or "DROP"
}}
"""



def build_prompt(attribute: str, brands_context: str) -> str:  # <<< CHANGED
    """
    Replace placeholders in the prompt template with:
      - the actual attribute string
      - a short description/list of brands present in the matrix
    """
    return PROMPT_TEMPLATE.format(attribute=attribute, brands_context=brands_context)


# ================== Core LLM-based classification ==================

def classify_attribute_with_llm(attribute: str, brands_context: str) -> Literal["KEEP", "DROP"]:  # <<< CHANGED
    """
    Call the Ollama model with a prompt for this attribute and parse its JSON answer.

    If parsing fails or decision is invalid, default to "DROP"
    (safer than accidentally keeping junk).
    """
    prompt = build_prompt(attribute, brands_context)
    raw = ask(prompt)

    # Try to find JSON in the response.
    # Model *should* output only JSON, but we defend against noise.
    text = raw.strip()

    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        json_str = text[start:end]
        data = json.loads(json_str)
        decision = str(data.get("decision", "")).strip().upper()
        if decision not in ("KEEP", "DROP"):
            print(f"  ⚠️ Invalid decision for '{attribute}': {decision!r}, defaulting to DROP")
            return "DROP"
        print(f"  → {attribute!r}: {decision}")
        return decision  # type: ignore[return-value]
    except Exception as e:
        print(f"  ⚠️ Failed to parse JSON for '{attribute}': {e}")
        print("  Raw response was:")
        print(raw)
        return "DROP"


# ================== Matrix filtering ==================

def filter_matrix(
    input_csv: str,
    output_csv: str,
    decisions_csv: Optional[str] = None,
) -> None:
    """
    Load the brand × attribute matrix, ask the LLM for each attribute
    whether to KEEP or DROP, and write the filtered matrix.
    """
    print(f"Loading matrix from: {input_csv}")
    df = pd.read_csv(input_csv, index_col=0)

    # Build a short context string listing the brands (row index).  # <<< CHANGED
    brands = [str(b) for b in df.index.tolist()]
    # If there are many brands, truncate the list in the prompt.
    if len(brands) > 20:
        brands_context = ", ".join(brands[:20]) + ", ..."
    else:
        brands_context = ", ".join(brands)

    kept_columns: List[str] = []
    decisions_log: List[Dict[str, str]] = []

    total_attrs = len(df.columns)
    print(f"Total attributes (columns): {total_attrs}")
    print(f"Brands in matrix: {brands_context}")

    for i, attr in enumerate(df.columns):
        print(f"\n[{i+1}/{total_attrs}] Classifying attribute: {attr!r}")
        decision = classify_attribute_with_llm(attr, brands_context)  # <<< CHANGED
        decisions_log.append({"attribute": attr, "decision": decision})
        if decision == "KEEP":
            kept_columns.append(attr)

    print(f"\nKeeping {len(kept_columns)} / {total_attrs} attributes")

    filtered_df = df[kept_columns]
    Path(os.path.dirname(output_csv) or ".").mkdir(parents=True, exist_ok=True)
    filtered_df.to_csv(output_csv)
    print(f"Filtered matrix written to: {output_csv}")

    if decisions_csv is not None:
        Path(os.path.dirname(decisions_csv) or ".").mkdir(parents=True, exist_ok=True)
        pd.DataFrame(decisions_log).to_csv(decisions_csv, index=False)
        print(f"Decisions log written to:   {decisions_csv}")


# ================== CLI ==================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter Brand × Attribute matrix using an Ollama LLM to decide KEEP/DROP per attribute."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/processed/brand_attribute_matrix/raw.csv",  # <<< CHANGED
        help="Path to input Brand × Attribute matrix CSV.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/brand_attribute_matrix/filtered.csv",
        help="Path to output filtered matrix CSV.",
    )
    parser.add_argument(
        "--decisions-log",
        type=str,
        default="data/processed/brand_attribute_matrix/decisions-log.csv",
        help="Optional path to save a CSV of {attribute, decision}.",
    )

    args = parser.parse_args()

    filter_matrix(
        input_csv=args.input,
        output_csv=args.output,
        decisions_csv=args.decisions_log,
    )


if __name__ == "__main__":
    main()
