import csv
import json 

sentences = []
with open("data/raw/descriptions.csv", "r", encoding="utf-8") as f:
    reader = csv.reader(f,quotechar='"', delimiter=',')
    next(reader)
    for row in reader:
        if(len(row)>=2):
            title = row[0].strip()
            description = row[1].strip()
            text = f"{title}. {description}"
            sentences.append({"text": text})

with open("data/raw/raw_sentences.json", "w", encoding="utf-8") as out:
        json.dump(sentences,out,ensure_ascii=False, indent=2)

print(f"Exported {len(sentences)} sentences")