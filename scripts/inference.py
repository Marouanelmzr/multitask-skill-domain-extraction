import numpy as np
import onnxruntime as ort
from pathlib import Path
from tokenizers import Tokenizer
from transformers import AutoTokenizer
from scripts.normalise import TechNormaliser

ROOT = Path(__file__).parent.parent

MODEL_PATH     = ROOT / "models" / "model.onnx"
TOKENIZER_PATH = ROOT / "models" / "tokenizer"

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
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(
        str(MODEL_PATH),
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )

    # HF tokenizer needed for offset_mapping (tokenizers lib doesn't expose it as cleanly)
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_PATH))

    tech_normaliser = TechNormaliser(
        model_name="models/normaliser_onnx",
        semantic_threshold=0.6,
        registry_path="data/tech_registry.json"
    )

    return session, tokenizer, tech_normaliser


def extract_technologies(text, offsets, ner_predictions):
    technologies = []
    current_start = None
    current_end = None

    for (start, end), label_id in zip(offsets, ner_predictions):
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


def predict_text(text, session, tokenizer, tech_normaliser, domain_threshold=0.5):
    encoding = tokenizer(
        text,
        return_tensors="np",        # numpy directly, no torch needed
        truncation=True,
        padding=True,
        max_length=256,
        return_offsets_mapping=True,
    )

    offsets = encoding["offset_mapping"][0].tolist()

    domain_logits, ner_logits = session.run(None, {
        "input_ids":      encoding["input_ids"].astype(np.int64),
        "attention_mask": encoding["attention_mask"].astype(np.int64),
    })

    # ner_logits: (1, seq_len, 3)
    ner_predictions = np.argmax(ner_logits[0], axis=-1).tolist()

    # domain_logits: (1, 10) — sigmoid threshold
    domain_probs = 1 / (1 + np.exp(-domain_logits[0]))

    raw_technologies = extract_technologies(
        text=text,
        offsets=offsets,
        ner_predictions=ner_predictions,
    )

    normalized_technologies = tech_normaliser.normalise_batch(raw_technologies)

    domains = [
        ID2DOMAIN[i]
        for i, prob in enumerate(domain_probs)
        if prob >= domain_threshold and ID2DOMAIN[i] != "Other"
    ]

    return {
        "text": text,
        "technologies": normalized_technologies,
        "raw_technologies": raw_technologies,
        "domains": domains,
    }