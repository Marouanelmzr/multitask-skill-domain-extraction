from transformers import AutoTokenizer
from pathlib import Path
from typing import List,Dict,Any
import json

#Config
BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_DIR = BASE_DIR / "data" / "cleaned" / "splits"
OUTPUT_DIR = BASE_DIR / "data" / "cleaned" / "processed"

TRAIN_FILE = INPUT_DIR / "train.jsonl"
VAL_FILE = INPUT_DIR / "val.jsonl"
TEST_FILE = INPUT_DIR / "test.jsonl"

TRAIN_OUT = OUTPUT_DIR / "train_processed.jsonl"
VAL_OUT = OUTPUT_DIR / "val_processed.jsonl"
TEST_OUT = OUTPUT_DIR / "test_processed.jsonl"

MODEL_NAME = "xlm-roberta-base"  #"microsoft/mdeberta-v3-base"
TAG2ID = {
    "O": 0,
    "B-TECH": 1,
    "I-TECH": 2
}
ID2TAG = {v: k for k, v in TAG2ID.items()}
DOMAIN2ID = {
    "Web Frontend":0,
    "Web Backend":1,
    "Mobile Development":2,
    "DevOps and Cloud Infrastructure":3,
    "Data Engineering":4,
    "Machine Learning and AI":5,
    "Cybersecurity":6,
    "Embedded Systems and IoT":7,
    "High Performance and Quantum Computing":8,
    "Other":9,
}
MAX_LENGTH = 256

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def load_jsonl(path):
    with open(path,"r",encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]
    return data
def save_jsonl(data: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
def sort_entities(entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(entities, key=lambda e: (e["start"], e["end"]))

# encoding domain_labels into 0 or 1 values
def encode_domains(domain_labels,domain2id):
    encode = [0] * len(domain2id)
    for domain in domain_labels:
        id = domain2id[domain]
        encode[id]=1
    return encode

# align_ner_labels
def align_ner_labels(text, entities,tokenizer,tag2id,max_length):
    encoding = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
        add_special_tokens=True,
    )
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]
    offset_mapping = encoding["offset_mapping"]

    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    ner_labels = [tag2id["O"]]*len(input_ids)

    for i, (start,end) in enumerate(offset_mapping):
        if start == end: # because start and end are given (0,0) by the tokenizer
            ner_labels[i] = -100
    
    sorted_ents = sort_entities(entities)

    for ent in sorted_ents:
        ent_start = ent["start"]
        ent_end = ent["end"]
        entity_token_indices = []

        for i, (tok_start,tok_end) in enumerate(offset_mapping):
            if tok_start == tok_end:
                continue
            overlaps = tok_start < ent_end and tok_end > ent_start
            if overlaps:
                entity_token_indices.append(i)
        if not entity_token_indices:
            continue

        first_idx = entity_token_indices[0]
        ner_labels[first_idx] = tag2id["B-TECH"]
        for idx in entity_token_indices[1:]:
            ner_labels[idx] = tag2id["I-TECH"]

    truncated = False
    if len(text) > 0:
        covered_positions = [end for start, end in offset_mapping if start != end]
        max_covered = max(covered_positions) if covered_positions else 0
        if max_covered < len(text):
            truncated = True

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "ner_labels": ner_labels,
        "tokens": tokens,
        "offset_mapping": offset_mapping,
        "truncated": truncated,
    }



#sample preprocessing
def preprocess_sample(sample,tokenizer,max_length):
    text = sample["text"]
    language = sample.get("language")
    entities = sample["entities"]
    domain_labels = sample["domain_labels"]

    aligned = align_ner_labels(
        text=text,
        entities=entities,
        tokenizer=tokenizer,
        tag2id=TAG2ID,
        max_length=max_length,
    )

    domain_vector = encode_domains(domain_labels, DOMAIN2ID)

    processed = {
        "text": text,
        "language": language,
        "tokens": aligned["tokens"],
        "input_ids": aligned["input_ids"],
        "attention_mask": aligned["attention_mask"],
        "ner_labels": aligned["ner_labels"],
        "domain_labels": domain_vector,
        "offset_mapping": aligned["offset_mapping"],
        "truncated": aligned["truncated"],
        # keep raw annotations too, useful for debugging
        "entities": entities,
        "original_domain_labels": domain_labels,
    }

    return processed

# split preprocessing
def preprocess_split(input_path,output_path,tokenizer,max_length):
    print(f"\nProcessing: {input_path}")
    data = load_jsonl(input_path)
    processed_data = []

    truncated_count = 0

    for idx, sample in enumerate(data):
        processed = preprocess_sample(sample, tokenizer, max_length=max_length)
        processed_data.append(processed)

        if processed["truncated"]:
            truncated_count += 1

        if idx < 3:
            print(f"\n--- Debug sample {idx} ---")
            print("Text:", processed["text"])
            print("Tokens:", processed["tokens"])
            print("NER labels:", processed["ner_labels"])
            print("Domains:", processed["original_domain_labels"])
            print("Domain vector:", processed["domain_labels"])
            print("Truncated:", processed["truncated"])

    save_jsonl(processed_data, output_path)

    print(f"Saved {len(processed_data)} samples to {output_path}")
    print(f"Truncated samples: {truncated_count}")

def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save mappings for reproducibility
    mappings = {
        "model_name": MODEL_NAME,
        "max_length": MAX_LENGTH,
        "tag2id": TAG2ID,
        "id2tag": ID2TAG,
        "domain2id": DOMAIN2ID,
    }

    with (OUTPUT_DIR / "mappings.json").open("w", encoding="utf-8") as f:
        json.dump(mappings, f, ensure_ascii=False, indent=2)

    preprocess_split(TRAIN_FILE, TRAIN_OUT, tokenizer, MAX_LENGTH)
    preprocess_split(VAL_FILE, VAL_OUT, tokenizer, MAX_LENGTH)
    preprocess_split(TEST_FILE, TEST_OUT, tokenizer, MAX_LENGTH)

    print("\nDone.")
    print(f"Mappings saved to: {OUTPUT_DIR / 'mappings.json'}")


if __name__ == "__main__":
    main()