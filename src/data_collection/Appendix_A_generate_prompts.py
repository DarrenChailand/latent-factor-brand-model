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

template_appendix_a = template["brand_perception_templates"]
brands = config["brands"]
industries = config["industries"]
factors = config["comparison_factors"]
personas = [""] + config["personas"]
temporal_modifiers = [""] + config["temporal_modifiers"]

# --- Store all generated prompts
generated_prompts = []

for i1 in template_appendix_a:
    template_question = template_appendix_a[i1]
    placeholders = re.findall(r"\[(.*?)\]", template_question)

    total_n_per_template = {"Brands": 0, "factors": 0, "Industries": 0}
    for x in placeholders:
        if x.startswith("Brand"):
            total_n_per_template["Brands"] += 1
        elif x.startswith("factor"):
            total_n_per_template["factors"] += 1
        elif x.startswith("Industry"):
            total_n_per_template["Industries"] += 1

    # Create permutations for each placeholder type
    brands_permutation = list(itertools.permutations(brands, total_n_per_template["Brands"])) or [()]
    factors_permutation = list(itertools.permutations(factors, total_n_per_template["factors"])) or [()]
    industries_permutation = list(itertools.permutations(industries, total_n_per_template["Industries"])) or [()]

    # Fill templates
    for i4 in brands_permutation:
        for i5 in factors_permutation:
            for i6 in industries_permutation:
                i4_c = list(i4)
                i5_c = list(i5)
                i6_c = list(i6)
                replacement = {}
                for x in placeholders:
                    if x.startswith("Brand"):
                        replacement[x] = i4_c.pop(0)
                    elif x.startswith("factor"):
                        replacement[x] = i5_c.pop(0)
                    elif x.startswith("Industry"):
                        replacement[x] = i6_c.pop(0)

                base_template = replace_placeholders(template_question, replacement)

                for i2 in temporal_modifiers:
                    for i3 in personas:
                        filled_template = base_template
                        if i2:
                            filled_template = i2 + ", " + filled_template
                        if i3:
                            filled_template = i3 + ", " + filled_template
                        generated_prompts.append(filled_template)

# --- Save results to JSON and TXT
output_dir = "data/processed/prompts"
os.makedirs(output_dir, exist_ok=True)

json_path = os.path.join(output_dir, "Appendix_A_generated_prompts.json")

# Save JSON
with open(json_path, "w") as f:
    json.dump(generated_prompts, f, indent=2, ensure_ascii=False)


print(f"✅ Saved {len(generated_prompts)} prompts to:")
print(f"   • {json_path}")
