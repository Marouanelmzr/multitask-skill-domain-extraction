import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path
import json

#Config
BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_DIR = BASE_DIR / "data" / "cleaned" / "processed"


def load_jsonl(path):
    with open(path,"r",encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]
    return data


class PortfolioDataset(Dataset):
    def __init__(self,file_path):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")
        self.samples = load_jsonl(self.file_path)
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, index):
        sample = self.samples[index]
        item = {
            "input_ids": torch.tensor(sample["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(sample["attention_mask"], dtype=torch.long),
            "ner_labels": torch.tensor(sample["ner_labels"], dtype=torch.long),
            "domain_labels": torch.tensor(sample["domain_labels"], dtype=torch.float),
            "tokens": sample["tokens"],
            "entities": sample["entities"],
            "original_domain_labels": sample["original_domain_labels"],
            "text": sample["text"],
        }
        return item


def multitask_collate_fn(batch):
    input_ids = [item["input_ids"] for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]
    ner_labels = [item["ner_labels"] for item in batch]
    domain_labels = torch.stack([item["domain_labels"] for item in batch])

    tokens = [item["tokens"] for item in batch]
    entities = [item["entities"] for item in batch]
    original_domain_labels = [item["original_domain_labels"] for item in batch]
    texts = [item["text"] for item in batch]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    ner_labels = pad_sequence(ner_labels, batch_first=True, padding_value=-100)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "ner_labels": ner_labels,
        "domain_labels": domain_labels,
        "tokens": tokens,
        "entities": entities,
        "original_domain_labels": original_domain_labels,
        "text": texts,
    }
