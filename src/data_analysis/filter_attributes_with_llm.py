#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filter Brand × Attribute matrix columns using either:
- OpenAI (ChatGPT), or
- a local Ollama model.

Inputs:
  - CSV matrix (rows=brands, cols=attributes, values=counts)

Outputs:
  - filtered CSV matrix (subset of columns)
  - optional decisions log CSV (attribute, decision)
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Literal, Optional

import ollama
import pandas as pd
from openai import OpenAI


# ================== Backend config ==================

USE_OPENAI = True
OPENAI_MODEL = "gpt-5.1"

MODEL = "gemma3:4b"
client = ollama.Client()

openai_client = None  # initialized lazily


# Warm up Ollama (optional; matches your current behavior)
_ = client.generate(
    model=MODEL,
    prompt="OK",
    options={"num_ctx": 512, "num_predict": 16, "num_thread": os.cpu_count()},
    keep_alive="10m",
)


# ================== LLM calls ==================

def ask(prompt: str) -> str:
    """Send a prompt to Ollama and return the text response."""
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
    print(f"[Ollama {len(r.response)} chars in {dt:.2f}s]")
    return r.response


def init_openai_if_needed():
    """Initialize OpenAI client (interactive key prompt)."""
    global openai_client

    if not USE_OPENAI:
        return
    if openai_client is not None:
        return

    print("\n=== OpenAI backend is enabled ===")
    print("Please enter your OpenAI API key (starting with sk-... or sk-proj-...).")
    api_key = input("OpenAI API Key: ").strip()

    os.environ["OPENAI_API_KEY"] = api_key
    openai_client = OpenAI(api_key=api_key)
    print("OpenAI client initialized.\n")


def ask_openai(prompt: str) -> str:
    """Send a prompt to OpenAI ChatGPT model and return the text response."""
    init_openai_if_needed()

    t0 = time.time()
    resp = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt.strip()}],
        temperature=0.0,
    )
    text = resp.choices[0].message.content or ""
    dt = time.time() - t0
    print(f"[OpenAI {OPENAI_MODEL} {len(text)} chars in {dt:.2f}s]")
    return text


# ================== Prompt template ==================

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


def build_prompt(attribute: str, brands_context: str) -> str:
    return PROMPT_TEMPLATE.format(attribute=attribute, brands_context=brands_context)


# ================== Classification ==================

def classify_attribute_with_llm(attribute: str, brands_context: str) -> Literal["KEEP", "DROP"]:
    """Call Ollama or OpenAI depending on USE_OPENAI."""
    prompt = build_prompt(attribute, brands_context)

    raw = ask_openai(prompt) if USE_OPENAI else ask(prompt)
    text = raw.strip()

    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        data = json.loads(text[start:end])
        decision = str(data.get("decision", "")).strip().upper()

        if decision not in ("KEEP", "DROP"):
            print(f"  ⚠️ Invalid decision for '{attribute}': {decision!r}, defaulting to DROP")
            return "DROP"

        print(f"  → {attribute!r}: {decision}")
        return decision

    except Exception as e:
        print(f"  ⚠️ Failed to parse JSON for '{attribute}': {e}")
        print("  Raw response was:")
        print(raw)
        return "DROP"


# ================== Matrix filtering ==================

def filter_matrix(input_csv: str, output_csv: str, decisions_csv: Optional[str] = None) -> None:
    """Load matrix, classify each attribute, write filtered output."""
    print(f"Loading matrix from: {input_csv}")
    df = pd.read_csv(input_csv, index_col=0)

    brands = [str(b) for b in df.index.tolist()]
    brands_context = ", ".join(brands[:20]) + (", ..." if len(brands) > 20 else "")

    kept_columns: List[str] = []
    decisions_log: List[Dict[str, str]] = []

    total_attrs = len(df.columns)
    print(f"Total attributes (columns): {total_attrs}")
    print(f"Brands in matrix: {brands_context}")
    print(f"Backend: {'OpenAI ' + OPENAI_MODEL if USE_OPENAI else 'Ollama ' + MODEL}")

    for i, attr in enumerate(df.columns):
        print(f"\n[{i+1}/{total_attrs}] Classifying attribute: {attr!r}")
        decision = classify_attribute_with_llm(attr, brands_context)
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
        print(f"Decisions log written to: {decisions_csv}")


def run_filter_attributes(
    input_csv: str,
    output_csv: str = "data/processed/brand_attribute_matrix/filtered.csv",
    decisions_log_csv: str = "data/processed/brand_attribute_matrix/decisions-log.csv",
) -> str:
    """
    Notebook-friendly wrapper.

    Filters columns in the Brand × Attribute matrix using the existing backend settings.
    Returns output_csv.
    """
    filter_matrix(
        input_csv=input_csv,
        output_csv=output_csv,
        decisions_csv=decisions_log_csv,
    )
    return output_csv
