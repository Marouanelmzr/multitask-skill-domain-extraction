import torch
from scripts.model import MultitaskXLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    model = MultitaskXLM(
        num_domain_labels=10,
        num_ner_labels=3
    )

    model.load_state_dict(
        torch.load("weights/best_model.pt", map_location=device)
    )

    model.to(device)
    model.eval()

    return model