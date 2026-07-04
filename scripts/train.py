import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm
import wandb
from scripts.dataset import PortfolioDataset, multitask_collate_fn
from scripts.model import MultitaskXLM
from scripts.eval import eval_step

PROJECT_ROOT = Path.cwd().parent
sys.path.append(str(PROJECT_ROOT))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

train_dataset = PortfolioDataset(
    str(PROJECT_ROOT / "data" / "cleaned" / "processed" / "train_processed.jsonl")
    )
validation_dataset = PortfolioDataset(
    str(PROJECT_ROOT / "data" / "cleaned" / "processed" / "val_processed.jsonl")
    )

train_dataloader = DataLoader(train_dataset,
    batch_size=8,
    shuffle=True,
    collate_fn=multitask_collate_fn,
    num_workers=2,
    pin_memory=True,)

validation_dataloader = DataLoader(validation_dataset,
    batch_size=8,
    shuffle=True,
    collate_fn=multitask_collate_fn,
    num_workers=2,
    pin_memory=True,)


model = MultitaskXLM(num_domain_labels=10, model_name="xlm-roberta-base", num_ner_labels=3)
model.to(device)

wandb.login(key=os.environ.get("WANDB_API_KEY"))

scaler = GradScaler('cuda')

def train_step(model, dataloader, optimizer, device, scheduler=None):
    model.train()
    train_loss = 0

    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        domain_labels = batch["domain_labels"].to(device)
        ner_labels = batch["ner_labels"].to(device)

        optimizer.zero_grad()

        with autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                domain_labels=domain_labels,
                ner_labels=ner_labels
            )
            loss = outputs["loss"]

        scaler.scale(loss).backward()
        # Unscale BEFORE clipping so clip sees real gradient magnitudes
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()
        if batch_idx % 10 == 0:
            print(f"Step {batch_idx} | Loss {loss.item():.4f}")

    return train_loss / len(dataloader)


wandb.init(
    project="portfolio-nlp",
    name="FINAL",
    config={
        "epochs": 10,
        "optimizer": "AdamW",
        "model": "xlm-roberta-base",
        "lr": "2e-5",
    }
)


def train(
    model,
    train_dataloader,
    val_dataloader,
    optimizer,
    device,
    epochs=5,
    scheduler=None
):
    model.to(device)

    history = {
        "train_loss": [],
        "val_loss": [],
        "domain_f1": [],
        "ner_f1": []
    }

    best_val_loss = float("inf")

    for epoch in range(epochs):

        print(f"\n===== Epoch {epoch+1}/{epochs} =====")

        # Train
        train_loss = train_step(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            device=device,
            scheduler= scheduler,
        )

        # Validation
        val_metrics = eval_step(
            model=model,
            dataloader=val_dataloader,
            device=device
        )

        # for cosine annealing we do per epoch step
        if scheduler:
          scheduler.step()

        val_loss = val_metrics["loss"]

        # Save history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["domain_f1"].append(val_metrics["domain_micro_f1"])
        history["ner_f1"].append(val_metrics["ner_micro_f1"])

        print(f"Train loss : {train_loss:.4f}")
        print(f"Val loss   : {val_loss:.4f}")
        print(f"Domain F1  : {val_metrics['domain_micro_f1']:.4f}")
        print(f"NER F1     : {val_metrics['ner_micro_f1']:.4f}")
        print(f"LR         : {optimizer.param_groups[0]['lr']}")

        #wandb logs
        lr = optimizer.param_groups[0]['lr']
        wandb.log({
            "epoch": epoch + 1,

            # losses
            "train/loss": train_loss,
            "val/loss": val_loss,

            # metrics
            "val/domain_f1": val_metrics["domain_micro_f1"],
            "val/ner_f1": val_metrics["ner_micro_f1"],

            # learning rate
            "lr": lr,
        })

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pt")
            print("Best model saved.")

    return history


optimizer = torch.optim.AdamW(model.parameters(), lr=4e-5)

EPOCHS = 15

scheduler = CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS,        # one full cosine cycle over all epochs
    eta_min=1e-8         # minimum LR at the bottom of the curve
)

losses = train(
    model=model,
    train_dataloader=train_dataloader,
    val_dataloader= validation_dataloader,
    optimizer=optimizer,
    device=device,
    epochs= EPOCHS,
    scheduler= scheduler,
)