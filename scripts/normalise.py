from pathlib import Path
import numpy as np
import json
import threading
import onnxruntime as ort
from transformers import AutoTokenizer

def cosine_similarity_single(query, corpus):
    query = np.asarray(query, dtype=np.float32)
    corpus = np.asarray(corpus, dtype=np.float32)

    query_norm = np.linalg.norm(query)
    corpus_norms = np.linalg.norm(corpus, axis=1)

    return (corpus @ query) / (corpus_norms * query_norm)


class ONNXEncoder:
    def __init__(self, model_dir: str):
        self.session = ort.InferenceSession(
            f"{model_dir}/model.onnx",
            providers=["CPUExecutionProvider"]
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

    def encode(self, text: str) -> np.ndarray:
        enc = self.tokenizer(
            text, return_tensors="np",
            padding=True, truncation=True, max_length=128
        )
        outputs = self.session.run(None, {
            "input_ids":      enc["input_ids"].astype(np.int64),
            "attention_mask": enc["attention_mask"].astype(np.int64),
        })
        # optimum exports sentence_embedding directly — no pooling needed
        return outputs[1][0]  # (384,)


class TechNormaliser:
    def __init__(
        self,
        model_name="models/normaliser_onnx",  # now points to local ONNX
        semantic_threshold=0.85,
        registry_path="data/tech_registry.json"
    ):
        self.model = ONNXEncoder(model_name)
        self.semantic_threshold = semantic_threshold
        self.registry_path = Path(registry_path)

        self.registry = {}
        self._lock = threading.Lock()

        if self.registry_path.exists():
            self._load_registry()

    def normalise(self, technology):
        with self._lock:
            result = self._normalise_unlocked(technology)
            self._save_registry()
        return result

    def normalise_batch(self, technologies):
        with self._lock:
            results = [self._normalise_unlocked(tech) for tech in technologies]
            self._save_registry()
        return list(dict.fromkeys(results))

    def _normalise_unlocked(self, technology):
        technology = technology.strip()
        if not technology:
            return technology
        if not self.registry:
            embedding = self.model.encode(technology)
            self._add_new_canonical(technology, embedding)
            return technology
        return self._semantic_similarity(technology)

    def _semantic_similarity(self, technology):
        embedding = self.model.encode(technology).flatten()
        corpus_embeddings = np.array([d["embedding"].flatten() for d in self.registry.values()])
        canonical_names = list(self.registry.keys())
        similarities = cosine_similarity_single( embedding, corpus_embeddings)
        max_idx = similarities.argmax()
        max_similarity = similarities[max_idx]
        canonical_tech = canonical_names[max_idx]

        if max_similarity >= self.semantic_threshold:
            if technology not in self.registry[canonical_tech]["mentions"]:
                self.registry[canonical_tech]["mentions"].append(technology)
            return canonical_tech

        self._add_new_canonical(technology, embedding)
        return technology

    def _add_new_canonical(self, technology, embedding):
        self.registry[technology] = {
            "mentions": [technology],
            "embedding": embedding.flatten()
        }

    def _save_registry(self):
        json_registry = {
            canonical: {
                "mentions": data["mentions"],
                "embedding": data["embedding"].tolist()
            }
            for canonical, data in self.registry.items()
        }
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(json_registry, f, ensure_ascii=False, indent=2)

    def _load_registry(self):
        with open(self.registry_path, "r", encoding="utf-8") as f:
            json_registry = json.load(f)
        self.registry = {
            canonical: {
                "mentions": data["mentions"],
                "embedding": np.array(data["embedding"], dtype=np.float32)
            }
            for canonical, data in json_registry.items()
        }