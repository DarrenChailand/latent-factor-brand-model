import json
import re
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

FILE = "data/processed/responses/Appendix_A_responses.jsonl"

companies = ["apple", "samsung", "google"]

def normalize(text):
    return re.sub(r"[^a-zA-Z0-9 ]", " ", text.lower())

# Load all responses
responses = []
with open(FILE, "r") as f:
    for line in f:
        obj = json.loads(line)
        responses.append(obj["response"])

# Split into sentences/statements
statements = []
for r in responses:
    parts = re.split(r"[.!?]\s+", r)
    statements.extend([p.strip() for p in parts if p.strip()])

def company_hits(text):
    text = normalize(text)
    return {c: (c in text) for c in companies}

# Count stats
counts = {
    "total_statements": len(statements),
    "contains_3": 0,
    "contains_2": 0,
    "contains_1": 0,
    "contains_0": 0,
    "only_apple": 0,
    "only_samsung": 0,
    "only_google": 0,
}

for stmt in statements:
    hits = company_hits(stmt)
    num = sum(hits.values())

    if num == 3: counts["contains_3"] += 1
    elif num == 2: counts["contains_2"] += 1
    elif num == 1: counts["contains_1"] += 1
    else: counts["contains_0"] += 1

    if hits["apple"] and not hits["samsung"] and not hits["google"]:
        counts["only_apple"] += 1
    if hits["samsung"] and not hits["apple"] and not hits["google"]:
        counts["only_samsung"] += 1
    if hits["google"] and not hits["apple"] and not hits["samsung"]:
        counts["only_google"] += 1

# Display results
print("=== STATISTICS ===")
for k,v in counts.items():
    print(f"{k}: {v}")

# Bar chart visualization
labels = [
    "Only Apple", "Only Samsung", "Only Google",
    "2 of 3", "All 3", "No Company"
]
values = [
    counts["only_apple"],
    counts["only_samsung"],
    counts["only_google"],
    counts["contains_2"],
    counts["contains_3"],
    counts["contains_0"],
]

import json

# Dict to store categorized statements
statement_buckets = {
    "only_apple": [],
    "only_samsung": [],
    "only_google": [],
    "contains_3": [],
    "contains_2": [],
    "contains_1": [],
    "contains_0": []
}

for stmt in statements:
    hits = company_hits(stmt)
    num = sum(hits.values())

    # classify
    if hits["apple"] and not hits["samsung"] and not hits["google"]:
        statement_buckets["only_apple"].append(stmt)
    elif hits["samsung"] and not hits["apple"] and not hits["google"]:
        statement_buckets["only_samsung"].append(stmt)
    elif hits["google"] and not hits["apple"] and not hits["samsung"]:
        statement_buckets["only_google"].append(stmt)
    elif num == 3:
        statement_buckets["contains_3"].append(stmt)
    elif num == 2:
        statement_buckets["contains_2"].append(stmt)
    elif num == 1:
        statement_buckets["contains_1"].append(stmt)
    else:
        statement_buckets["contains_0"].append(stmt)

# save file
with open("src/data_analysis/appendix_A_statements_by_group.json", "w") as f:
    json.dump(statement_buckets, f, indent=4)

print("✅ Saved src/data_analysis/appendix_A_statements_by_group.json")


plt.bar(labels, values)
plt.xticks(rotation=45)
plt.title("Company Mentions per Statement")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()
