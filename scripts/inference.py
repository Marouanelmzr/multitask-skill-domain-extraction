import torch
from scripts.model import MultitaskXLM
from transformers import AutoTokenizer

MODEL_NAME = "xlm-roberta-base"
WEIGHTS_PATH = "models/best_model.pt"

ID2DOMAIN = {
    0: "Web Frontend",
    1: "Web Backend",
    2: "Mobile Development",
    3: "DevOps and Cloud Infrastructure",
    4: "Data Engineering",
    5: "Machine Learning and AI",
    6: "Cybersecurity",
    7: "Embedded Systems and IoT",
    8: "High Performance and Quantum Computing",
    9: "Other",
}

def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MultitaskXLM(
        model_name=MODEL_NAME,
        num_domain_labels=10,
        num_ner_labels=3
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model.load_state_dict(
        torch.load(WEIGHTS_PATH, map_location=device)
    )

    model.to(device)
    model.eval()

    return model, tokenizer, device

def extract_technologies(text, offsets, ner_predictions):
    technologies = []
    current_start = None
    current_end = None

    for (start, end), label_id in zip(offsets, ner_predictions):
        # special tokens usually have offset (0, 0)
        if start == end:
            continue

        if label_id == 1:  # B-TECH
            if current_start is not None:
                technologies.append(text[current_start:current_end].strip())

            current_start = start
            current_end = end

        elif label_id == 2:  # I-TECH
            if current_start is not None:
                current_end = end

        else:  # O
            if current_start is not None:
                technologies.append(text[current_start:current_end].strip())
                current_start = None
                current_end = None

    if current_start is not None:
        technologies.append(text[current_start:current_end].strip())

    return list(dict.fromkeys(technologies))


def predict_text(text, model, tokenizer, device, domain_threshold=0.5):
    encoding = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256,
        return_offsets_mapping=True
    )

    offsets = encoding["offset_mapping"][0].tolist()

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        ner_logits = outputs["ner_logits"]
        domain_logits = outputs["domain_logits"]

        ner_predictions = torch.argmax(ner_logits, dim=-1)[0].cpu().tolist()
        domain_probs = torch.sigmoid(domain_logits)[0].cpu()

        technologies = extract_technologies(
            text=text,
            offsets=offsets,
            ner_predictions=ner_predictions
        )

        domains = [
            ID2DOMAIN[i]
            for i, prob in enumerate(domain_probs)
            if prob >= domain_threshold and ID2DOMAIN[i] != "Other"
        ]

    return {
        "text": text,
        "technologies": technologies,
        "domains": domains
    }