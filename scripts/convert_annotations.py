import json

with open("data/raw/annotated.json", "r", encoding="utf-8") as f:
    raw = json.load(f)

dataset = []

for item in raw:
    text = item["data"]["text"]
    entities = []
    domain_labels = []

    if item["annotations"]:
        for result in item["annotations"][0]["result"]:
            if result["type"] == "labels":
                v = result["value"]
                entities.append({
                    "start": v["start"],
                    "end": v["end"],
                    "label": "B-TECH",
                    "value": v["text"]
                })
            elif result["type"] == "choices":
                domain_labels = result["value"]["choices"]

    dataset.append({
        "text": text,
        "language": "fr" if any(c in text for c in ["é","è","à","ê","ù","ç"]) else "en",
        "entities": entities,
        "domain_labels": domain_labels
    })

with open("data/raw/dataset.jsonl", "w", encoding="utf-8") as out:
    for entry in dataset:
        out.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"Converted {len(dataset)} entries")