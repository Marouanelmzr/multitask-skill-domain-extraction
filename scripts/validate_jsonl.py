import json
from pathlib import Path


DATASET_PATH = "data/raw/dataset_corrected.jsonl"

ALLOWED_DOMAIN_LABELS = {
    "Web Frontend",
    "Web Backend",
    "Mobile Development",
    "DevOps and Cloud Infrastructure",
    "Data Engineering",
    "Machine Learning and AI",
    "Cybersecurity",
    "Embedded Systems and IoT",
    "High Performance and Quantum Computing",
}

ALLOW_OVERLAPS = False


# validation helpers:
def entities_overlap(ent1, ent2):
    return ent1["start"] < ent2["end"] and ent2["start"] < ent1["end"]


def print_error(line_num, sample_text, message, entity=None, extracted=None):
    print("\n" + "=" * 80)
    print(f"Line number: {line_num}")
    print(f"Error: {message}")
    print(f"Bad sample text: {repr(sample_text)}")
    if entity is not None:
        print(f"Offending entity: {entity}")
    if extracted is not None:
        print(f"Extracted substring from text: {repr(extracted)}")
    print("=" * 80)


def validate_sample(sample, line_num):
    is_valid = True

    # 1. Validate text existance
    text = sample.get("text")
    if not isinstance(text, str) or not text.strip():
        print_error(
            line_num=line_num,
            sample_text=text,
            message="text is missing, not a string, or empty"
        )
        return False


    # 2. Validate entities existence/type
    entities = sample.get("entities", [])
    if not isinstance(entities, list):
        print_error(
            line_num=line_num,
            sample_text=text,
            message="'entities' must be a list"
        )
        return False


    # 3. Validate domain labels
    domain_labels = sample.get("domain_labels", [])
    if not isinstance(domain_labels, list):
        print_error(
            line_num=line_num,
            sample_text=text,
            message="'domain_labels' must be a list"
        )
        is_valid = False
    else:
        for domain in domain_labels:
            if domain not in ALLOWED_DOMAIN_LABELS:
                print_error(
                    line_num=line_num,
                    sample_text=text,
                    message=f"invalid domain label: {domain}"
                )
                is_valid = False


    # 4. Validate each entity
    seen_spans = set()

    for entity in entities:
        if not isinstance(entity, dict):
            print_error(
                line_num=line_num,
                sample_text=text,
                message="entity is not a dictionary",
                entity=entity
            )
            is_valid = False
            continue

        start = entity.get("start")
        end = entity.get("end")
        value = entity.get("value")
        label = entity.get("label")

        # Check required fields
        if start is None or end is None or value is None or label is None:
            print_error(
                line_num=line_num,
                sample_text=text,
                message="entity is missing one of: start, end, value, label",
                entity=entity
            )
            is_valid = False
            continue

        # Check types
        if not isinstance(start, int) or not isinstance(end, int):
            print_error(
                line_num=line_num,
                sample_text=text,
                message="entity start/end must be integers",
                entity=entity
            )
            is_valid = False
            continue

        if not isinstance(value, str):
            print_error(
                line_num=line_num,
                sample_text=text,
                message="entity value must be a string",
                entity=entity
            )
            is_valid = False
            continue

        # Check start < end
        if start >= end:
            extracted = text[start:end] if 0 <= start <= len(text) and 0 <= end <= len(text) else None
            print_error(
                line_num=line_num,
                sample_text=text,
                message="entity has invalid span: start must be < end",
                entity=entity,
                extracted=extracted
            )
            is_valid = False
            continue

        # Check bounds
        if start < 0 or end > len(text):
            extracted = text[start:end] if 0 <= start < len(text) else None
            print_error(
                line_num=line_num,
                sample_text=text,
                message="entity span is out of text bounds",
                entity=entity,
                extracted=extracted
            )
            is_valid = False
            continue

        # Check exact substring match
        extracted = text[start:end]
        if extracted != value:
            print_error(
                line_num=line_num,
                sample_text=text,
                message="text[start:end] does not match entity value",
                entity=entity,
                extracted=extracted
            )
            is_valid = False

        # Check duplicate spans
        span = (start, end)
        if span in seen_spans:
            print_error(
                line_num=line_num,
                sample_text=text,
                message="duplicate entity span found in same sample",
                entity=entity,
                extracted=extracted
            )
            is_valid = False
        else:
            seen_spans.add(span)


    # 5. Check overlaps
    if not ALLOW_OVERLAPS:
        sorted_entities = sorted(
            [e for e in entities if isinstance(e, dict) and isinstance(e.get("start"), int) and isinstance(e.get("end"), int)],
            key=lambda x: (x["start"], x["end"]) # lambda function is a function used temporarly without the need of a name, the x is iterator over entities
        )

        for i in range(len(sorted_entities)):
            for j in range(i + 1, len(sorted_entities)):
                e1 = sorted_entities[i]
                e2 = sorted_entities[j]

                if entities_overlap(e1, e2):
                    print_error(
                        line_num=line_num,
                        sample_text=text,
                        message="overlapping entities found",
                        entity={"entity_1": e1, "entity_2": e2},
                        extracted={
                            "entity_1_text": text[e1["start"]:e1["end"]],
                            "entity_2_text": text[e2["start"]:e2["end"]]
                        }
                    )
                    is_valid = False

    return is_valid



# MAIN
def validate_jsonl_file(path):
    path = Path(path)

    if not path.exists():
        print(f"File not found: {path}")
        return

    total = 0
    valid = 0
    invalid = 0

    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                print(f"\nLine {line_num} is empty.")
                invalid += 1
                total += 1
                continue

            try:
                sample = json.loads(line)
            except json.JSONDecodeError as e:
                print("\n" + "=" * 80)
                print(f"Line number: {line_num}")
                print("Error: invalid JSON")
                print(f"Raw line: {line}")
                print(f"JSON decode error: {e}")
                print("=" * 80)
                invalid += 1
                total += 1
                continue

            total += 1
            if validate_sample(sample, line_num):
                valid += 1
            else:
                invalid += 1

    print("\n" + "#" * 80)
    print("VALIDATION SUMMARY")
    print(f"Total samples:   {total}")
    print(f"Valid samples:   {valid}")
    print(f"Invalid samples: {invalid}")
    print("#" * 80)


if __name__ == "__main__":
    validate_jsonl_file(DATASET_PATH)