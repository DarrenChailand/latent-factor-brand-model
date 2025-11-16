import json
import re
import itertools
import os

def replace_placeholders(text, mapping):
    return re.sub(r"\[(.*?)\]", lambda m: mapping.get(m.group(1), m.group(0)), text)

# --- Load configs
with open("data/raw/demo_brand_prompt_config.json", "r") as f:
    config = json.load(f)

with open("data/raw/brand_prompt_templates.json", "r") as f:
    template = json.load(f)

template_appendix_b = template["latent_need_templates"]
product_categories = config["product_categories"]
demographic_targets = [""] + config["demographic_targets"]
usage_contexts = [""] + config["usage_contexts"]
temporal_framing = [""] + config["temporal_framing"]

# --- Store all generated prompts
generated_prompts = []

for i1 in template_appendix_b:
    template_question = template_appendix_b[i1]
    for i2 in product_categories:
        for i3 in demographic_targets:
            for i4 in usage_contexts:
                for i5 in temporal_framing:
                    replacement = {}
                    placeholders = re.findall(r"\[(.*?)\]", template_question)
                    for x in placeholders:
                        if x.startswith("Product Category"):
                            replacement[x] = i2
                        elif x.startswith("Demographic Target"):
                            replacement[x] = i3
                        elif x.startswith("Usage Context"):
                            replacement[x] = i4
                        elif x.startswith("Temporal Framing"):
                            replacement[x] = i5

                    base_template = replace_placeholders(template_question, replacement)
                    generated_prompts.append(base_template)

# --- Save results to JSON and TXT
output_dir = "data/processed/prompts"
os.makedirs(output_dir, exist_ok=True)

json_path = os.path.join(output_dir, "Appendix_B_generated_prompts.json")

# Save JSON
with open(json_path, "w") as f:
    json.dump(generated_prompts, f, indent=2, ensure_ascii=False)


print(f"✅ Saved {len(generated_prompts)} prompts to:")
print(f"   • {json_path}")

   