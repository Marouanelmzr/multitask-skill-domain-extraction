import json
import random
from pathlib import Path


# CONFIG
INPUT_FILE = "data/cleaned/dataset_corrected.jsonl"
OUTPUT_DIR = "data/cleaned/splits"

TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
TEST_RATIO  = 0.10

SEED = 42



# LOAD DATA
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f if line.strip()]

print(f"Loaded {len(data)} samples")

# SHUFFLE
random.seed(SEED)
random.shuffle(data)

# COMPUTE INDICES
n = len(data)

train_end = int(n * TRAIN_RATIO)
val_end   = train_end + int(n * VAL_RATIO)

train_data = data[:train_end]
val_data   = data[train_end:val_end]
test_data  = data[val_end:]

# CREATE OUTPUT FOLDER
Path(OUTPUT_DIR).mkdir(exist_ok=True)

# SAVE FUNCTION
def save_jsonl(path, samples):
    with open(path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

# SAVE FILES
save_jsonl(f"{OUTPUT_DIR}/train.jsonl", train_data)
save_jsonl(f"{OUTPUT_DIR}/val.jsonl", val_data)
save_jsonl(f"{OUTPUT_DIR}/test.jsonl", test_data)

# REPORT
print("\nSplit completed:")
print(f"Train: {len(train_data)}")
print(f"Val:   {len(val_data)}")
print(f"Test:  {len(test_data)}")