import torch
from torch.utils.data import DataLoader
from scripts.dataset import PortfolioDataset,multitask_collate_fn

train_dataset = PortfolioDataset("../data/cleaned/processed/train_processed.jsonl")
train_dataloader = DataLoader(train_dataset, 
    batch_size=8, 
    shuffle=True, 
    collate_fn=multitask_collate_fn)

batch = next(iter(train_dataloader))
for k, v in batch.items():
    if hasattr(v, "shape"):
        print(k, v.shape)