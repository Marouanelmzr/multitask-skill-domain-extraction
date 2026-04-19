import json
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split

#Config
INPUT_FILE = "data/cleaned/dataset_corrected.jsonl"
OUTPUT_DIR = "data/cleaned/splits"

TEST_RATIO = 0.10
VAL_RATIO = 0.10
SEED = 42

# LOAD JSONL
data = []

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            data.append(json.loads(line))

print(f"Loaded {len(data)} samples")

# PRIMARY LABEL FOR STRATIFICATION
def get_primary_label(sample):
    domains = sample.get("domain_labels", [])
    if not domains:
        return "NO_LABEL"
    return domains[0]

labels = [get_primary_label(sample) for sample in data]


# CHECK LABEL COUNTS
label_counts = Counter(labels)

print("\nLabel counts before split:")
for label, count in label_counts.items():
    print(f"{label}: {count}")

rare_labels = [label for label, count in label_counts.items() if count < 2]
if rare_labels:
    raise ValueError(
        f"These labels have fewer than 2 samples and cannot be stratified safely: {rare_labels}"
    )

# FIRST SPLIT: TRAIN+VAL / TEST
train_val, test = train_test_split(
    data,
    test_size=TEST_RATIO,
    random_state=SEED,
    stratify=labels
)

# SECOND SPLIT: TRAIN / VAL
train_val_labels = [get_primary_label(sample) for sample in train_val]

val_relative_ratio = VAL_RATIO / (1 - TEST_RATIO)

train, val = train_test_split(
    train_val,
    test_size=val_relative_ratio,
    random_state=SEED,
    stratify=train_val_labels
)

# SAVE JSONL
Path(OUTPUT_DIR).mkdir(exist_ok=True)

def save_jsonl(path, samples):
    with open(path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

save_jsonl(f"{OUTPUT_DIR}/train.jsonl", train)
save_jsonl(f"{OUTPUT_DIR}/val.jsonl", val)
save_jsonl(f"{OUTPUT_DIR}/test.jsonl", test)


# REPORT
def show_distribution(name, samples):
    counts = Counter(get_primary_label(sample) for sample in samples)
    print(f"\n{name} distribution:")
    for label, count in counts.items():
        print(f"{label}: {count}")

print("\nSplit completed:")
print(f"Train: {len(train)}")
print(f"Val:   {len(val)}")
print(f"Test:  {len(test)}")

show_distribution("TRAIN", train)
show_distribution("VAL", val)
show_distribution("TEST", test)