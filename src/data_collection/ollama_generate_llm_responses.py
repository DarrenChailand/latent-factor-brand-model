import ollama, os, time
import json
import time
from pathlib import Path
from typing import List

client = ollama.Client()
MODEL = "llama3.2:1b"

# Warm up so the model is loaded into RAM (improves first real query latency)
_ = client.generate(
    model=MODEL,
    prompt="OK",
    options={"num_ctx": 512, "num_predict": 16, "num_thread": os.cpu_count()},
    keep_alive="10m",
)

def ask(prompt: str) -> str:
    t0 = time.time()
    r = client.generate(
        model=MODEL,
        prompt=prompt.strip(),
        options={
            "num_ctx": 1024,          # try 512 or 1024; smaller = faster on 8 GB
            "num_predict": -1,      # cap output length
            "num_thread": os.cpu_count(),
        },
        keep_alive="10m",            # keep the model in RAM between calls
        stream=False                 # set True if you want tokens ASAP
    )
    dt = time.time() - t0
    print(f"[{len(r.response)} chars in {dt:.2f}s]")
    return r.response

def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)

def incremental_save_jsonl(out_path: str, item: dict):
    """
    Save one record per line (JSONL style) — append-safe and crash-resistant.
    Later, you can convert it to a proper .json array easily.
    """
    with open(out_path, "a", encoding="utf-8") as f:
        json.dump(item, f, ensure_ascii=False)
        f.write("\n")

def process_file(prompts_path: str, out_path: str):
    ensure_dir(os.path.dirname(out_path))
    with open(prompts_path, "r", encoding="utf-8") as f:
        prompts = json.load(f)

    total = len(prompts)
    print(f"Processing {total} prompts from {prompts_path}")

    # Resume support: skip already processed lines
    processed = 0
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            processed = sum(1 for _ in f)
        print(f"Resuming from index {processed}")

    for i, prompt in enumerate(prompts):
        if i < processed:
            continue
        try:
            t0 = time.time()
            resp = ask(prompt)
            dt = time.time() - t0
            record = {
                "index": i,
                "prompt": prompt,
                "response": resp,
                "elapsed_seconds": round(dt, 3),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
            }
            incremental_save_jsonl(out_path, record)
            print(f"✓ [{i+1}/{total}] saved ({len(resp)} chars, {dt:.2f}s)")
        except Exception as e:
            print(f"⚠️ Error at index {i}: {e}")
            time.sleep(3)

    print(f"✅ Finished {prompts_path}")

if __name__ == "__main__":
    process_file(
        "data/processed/prompts/Appendix_A_generated_prompts.json",
        "data/processed/responses/Appendix_A_responses.jsonl"
    )
    process_file(
        "data/processed/prompts/Appendix_B_generated_prompts.json",
        "data/processed/responses/Appendix_B_responses.jsonl"
    )