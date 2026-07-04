from pathlib import Path
from scripts.model import MultitaskXLM
import torch
import torch.nn as nn
from transformers import AutoTokenizer
import onnx
import onnxruntime as ort
import numpy as np

ROOT = Path(__file__).parent.parent  # ai/

class MultitaskXLMONNX(nn.Module):
    def __init__(self, base: MultitaskXLM):
        super().__init__()
        self.model = base

    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids, attention_mask)
        return out["domain_logits"], out["ner_logits"]


# 1. Load trained weights
print("Loading model weights...")
base_model = MultitaskXLM(
    model_name="xlm-roberta-base",
    num_domain_labels=10,
    num_ner_labels=3,
)
base_model.load_state_dict(torch.load(ROOT / "models" / "best_model.pt", map_location="cpu"))
base_model.eval()

export_model = MultitaskXLMONNX(base_model)
export_model.eval()

# 2. Dummy inputs for tracing
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
dummy = tokenizer(
    ["Bonjour le monde", "Hello world"],
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=128,
)
dummy_ids  = dummy["input_ids"]
dummy_mask = dummy["attention_mask"]

# 3. Sanity check before export
with torch.no_grad():
    d_logits, n_logits = export_model(dummy_ids, dummy_mask)
    print(f"Forward pass OK, domain: {d_logits.shape}, ner: {n_logits.shape}")

# 4. Export to ONNX
ONNX_PATH = ROOT / "models" / "model.onnx"
print("Exporting to ONNX...")
torch.onnx.export(
    export_model,
    (dummy_ids, dummy_mask),
    str(ONNX_PATH),
    input_names=["input_ids", "attention_mask"],
    output_names=["domain_logits", "ner_logits"],
    dynamic_axes={
        "input_ids":      {0: "batch", 1: "seq_len"},
        "attention_mask": {0: "batch", 1: "seq_len"},
        "domain_logits":  {0: "batch"},
        "ner_logits":     {0: "batch", 1: "seq_len"},
    },
    opset_version=17,
    do_constant_folding=True,
)
print(f"Exported to {ONNX_PATH}")

# 5. Verify the exported graph
print("Verifying ONNX graph...")
onnx_model = onnx.load(str(ONNX_PATH))
onnx.checker.check_model(onnx_model)
print("ONNX graph is valid")

# 6. Verify outputs match PyTorch exactly
print("Comparing PyTorch vs ONNX outputs...")
sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
onnx_domain, onnx_ner = sess.run(None, {
    "input_ids":      dummy_ids.numpy(),
    "attention_mask": dummy_mask.numpy(),
})

with torch.no_grad():
    pt_domain, pt_ner = export_model(dummy_ids, dummy_mask)

domain_diff = np.abs(pt_domain.numpy() - onnx_domain).max()
ner_diff    = np.abs(pt_ner.numpy()    - onnx_ner).max()
print(f"Max domain logits diff : {domain_diff:.2e}  (expect < 1e-5)")
print(f"Max NER logits diff    : {ner_diff:.2e}  (expect < 1e-5)")

print("\n Export complete, model.onnx is ready for production")