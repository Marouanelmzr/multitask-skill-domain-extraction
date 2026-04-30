"""
Multi-Task NLP Model: Domain Classification + Technology NER
Shared XLM-RoBERTa encoder with task-specific heads.
Supports staged training (freeze → gradual unfreeze → full fine-tune).
"""

import torch
import torch.nn as nn
from transformers import XLMRobertaModel, XLMRobertaConfig
from torchcrf import CRF  # pip install pytorch-crf


# ─────────────────────────────────────────────
# 1. TASK HEADS
# ─────────────────────────────────────────────

class ClassificationHead(nn.Module):
    """
    Sentence-level classification over 9 domains.
    Uses the [CLS] token representation.
    """
    def __init__(self, hidden_size: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, cls_output: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cls_output: (batch_size, hidden_size) — [CLS] token from encoder
        Returns:
            logits: (batch_size, num_classes)
        """
        x = self.dropout(cls_output)
        return self.classifier(x)


class NERHead(nn.Module):
    """
    Token-level NER head with optional CRF layer.
    CRF is recommended for technology span detection (multi-token entities).

    BIO tag scheme example:
        0: O
        1: B-TECH
        2: I-TECH
    """
    def __init__(self, hidden_size: int, num_tags: int, dropout: float = 0.1, use_crf: bool = True):
        super().__init__()
        self.use_crf = use_crf
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, num_tags)

        if use_crf:
            self.crf = CRF(num_tags, batch_first=True)

    def forward(
        self,
        token_output: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor = None,
    ):
        """
        Args:
            token_output:   (batch_size, seq_len, hidden_size)
            attention_mask: (batch_size, seq_len) — 1 for real tokens, 0 for padding
            labels:         (batch_size, seq_len) — tag ids, or None at inference

        Returns:
            If labels provided  → (loss, predictions)
            If no labels        → (None, predictions)
        """
        emissions = self.linear(self.dropout(token_output))  # (B, L, num_tags)
        bool_mask = attention_mask.bool()

        if self.use_crf:
            if labels is not None:
                # CRF returns negative log-likelihood; negate for loss
                loss = -self.crf(emissions, labels, mask=bool_mask, reduction="mean")
                predictions = self.crf.decode(emissions, mask=bool_mask)
                return loss, predictions
            else:
                predictions = self.crf.decode(emissions, mask=bool_mask)
                return None, predictions
        else:
            if labels is not None:
                loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
                # Flatten for cross entropy
                active_logits = emissions[bool_mask]       # (num_active, num_tags)
                active_labels = labels[bool_mask]          # (num_active,)
                loss = loss_fn(active_logits, active_labels)
                predictions = emissions.argmax(dim=-1)
                return loss, predictions
            else:
                predictions = emissions.argmax(dim=-1)
                return None, predictions


# ─────────────────────────────────────────────
# 2. MULTI-TASK MODEL
# ─────────────────────────────────────────────

class MultiTaskNLPModel(nn.Module):
    """
    Shared XLM-RoBERTa encoder with:
      - ClassificationHead  → 9-domain classification ([CLS] token)
      - NERHead             → technology entity tagging (all tokens)

    Loss:
        total_loss = alpha * cls_loss + beta * ner_loss
    """

    def __init__(
        self,
        model_name: str = "xlm-roberta-base",
        num_classes: int = 9,
        num_ner_tags: int = 3,           # e.g. O, B-TECH, I-TECH
        clf_dropout: float = 0.1,
        ner_dropout: float = 0.1,
        use_crf: bool = True,
        alpha: float = 1.0,              # weight for classification loss
        beta: float = 1.0,              # weight for NER loss
    ):
        super().__init__()

        self.alpha = alpha
        self.beta = beta

        # ── Shared Encoder ──────────────────────────────────────────────
        self.encoder = XLMRobertaModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size  # 768 for base

        # ── Task Heads ──────────────────────────────────────────────────
        self.clf_head = ClassificationHead(hidden_size, num_classes, clf_dropout)
        self.ner_head = NERHead(hidden_size, num_ner_tags, ner_dropout, use_crf)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        clf_labels: torch.Tensor = None,   # (batch_size,)
        ner_labels: torch.Tensor = None,   # (batch_size, seq_len)
    ) -> dict:
        """
        Returns a dict with keys:
            total_loss      : combined loss (if labels provided)
            clf_loss        : classification loss
            ner_loss        : NER loss
            clf_logits      : (B, num_classes)
            ner_predictions : list of tag-id lists
        """
        # ── Encode ──────────────────────────────────────────────────────
        encoder_out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # [CLS] → classification
        cls_output   = encoder_out.last_hidden_state[:, 0, :]   # (B, H)
        # all tokens → NER
        token_output = encoder_out.last_hidden_state             # (B, L, H)

        # ── Classification Head ──────────────────────────────────────────
        clf_logits = self.clf_head(cls_output)

        clf_loss = None
        if clf_labels is not None:
            clf_loss = nn.CrossEntropyLoss()(clf_logits, clf_labels)

        # ── NER Head ────────────────────────────────────────────────────
        ner_loss, ner_predictions = self.ner_head(
            token_output, attention_mask, ner_labels
        )

        # ── Combined Loss ────────────────────────────────────────────────
        total_loss = None
        if clf_loss is not None and ner_loss is not None:
            total_loss = self.alpha * clf_loss + self.beta * ner_loss
        elif clf_loss is not None:
            total_loss = clf_loss
        elif ner_loss is not None:
            total_loss = ner_loss

        return {
            "total_loss":      total_loss,
            "clf_loss":        clf_loss,
            "ner_loss":        ner_loss,
            "clf_logits":      clf_logits,
            "ner_predictions": ner_predictions,
        }

    # ─────────────────────────────────────────────
    # 3. STAGED TRAINING HELPERS
    # ─────────────────────────────────────────────

    def freeze_encoder(self):
        """Phase 1: Freeze entire encoder, train heads only."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        print("✓ Encoder frozen — training heads only.")

    def unfreeze_top_layers(self, num_layers: int = 4):
        """
        Phase 2: Unfreeze the top N transformer layers of the encoder.
        XLM-RoBERTa-base has 12 layers (indexed 0–11).
        """
        # Always unfreeze the pooler and embeddings projection
        for param in self.encoder.pooler.parameters():
            param.requires_grad = True

        total_layers = len(self.encoder.encoder.layer)
        for i in range(total_layers - num_layers, total_layers):
            for param in self.encoder.encoder.layer[i].parameters():
                param.requires_grad = True

        unfrozen = [
            total_layers - num_layers + i for i in range(num_layers)
        ]
        print(f"✓ Unfrozen encoder layers: {unfrozen}")

    def unfreeze_encoder(self):
        """Phase 3: Unfreeze full encoder for complete fine-tuning."""
        for param in self.encoder.parameters():
            param.requires_grad = True
        print("✓ Full encoder unfrozen — end-to-end fine-tuning.")

    def get_parameter_groups(
        self,
        encoder_lr: float = 2e-5,
        head_lr: float = 1e-4,
        weight_decay: float = 0.01,
    ) -> list:
        """
        Discriminative learning rates:
          - Encoder layers get encoder_lr (lower)
          - Task heads get head_lr (higher)

        Pass the returned list directly to your optimizer:
            optimizer = AdamW(model.get_parameter_groups(...))
        """
        no_decay = ["bias", "LayerNorm.weight"]

        encoder_params_decay    = []
        encoder_params_no_decay = []
        head_params_decay       = []
        head_params_no_decay    = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            is_head = name.startswith("clf_head") or name.startswith("ner_head")
            no_wd   = any(nd in name for nd in no_decay)

            if is_head:
                (head_params_no_decay if no_wd else head_params_decay).append(param)
            else:
                (encoder_params_no_decay if no_wd else encoder_params_decay).append(param)

        return [
            {"params": encoder_params_decay,    "lr": encoder_lr, "weight_decay": weight_decay},
            {"params": encoder_params_no_decay, "lr": encoder_lr, "weight_decay": 0.0},
            {"params": head_params_decay,       "lr": head_lr,    "weight_decay": weight_decay},
            {"params": head_params_no_decay,    "lr": head_lr,    "weight_decay": 0.0},
        ]


# ─────────────────────────────────────────────
# 4. TRAINING LOOP SKETCH
# ─────────────────────────────────────────────

def run_epoch(model, dataloader, optimizer, scheduler=None, device="cuda"):
    """
    Single training epoch.

    Expected batch dict keys:
        input_ids      : (B, L)
        attention_mask : (B, L)
        clf_labels     : (B,)
        ner_labels     : (B, L)  — use -100 for special/padding tokens
    """
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        clf_labels     = batch["clf_labels"].to(device)
        ner_labels     = batch["ner_labels"].to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            clf_labels=clf_labels,
            ner_labels=ner_labels,
        )

        loss = outputs["total_loss"]
        loss.backward()

        # Gradient clipping — important for stability
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        if scheduler:
            scheduler.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


# ─────────────────────────────────────────────
# 5. FULL STAGED TRAINING ORCHESTRATOR
# ─────────────────────────────────────────────

def staged_training(model, train_loader, val_loader=None, device="cuda"):
    """
    Orchestrates the 3-phase staged training strategy.
    Adjust epoch counts and LRs to your dataset size.
    """
    from torch.optim import AdamW
    from transformers import get_linear_schedule_with_warmup

    model.to(device)

    def make_optimizer_and_scheduler(encoder_lr, head_lr, num_epochs):
        param_groups = model.get_parameter_groups(encoder_lr=encoder_lr, head_lr=head_lr)
        opt = AdamW(param_groups)
        total_steps = len(train_loader) * num_epochs
        warmup_steps = int(0.1 * total_steps)
        sch = get_linear_schedule_with_warmup(opt, warmup_steps, total_steps)
        return opt, sch

    # ── Phase 1: Warm-up heads ───────────────────────────────────────────
    print("\n━━━ Phase 1: Warm-up (heads only) ━━━")
    model.freeze_encoder()
    optimizer, scheduler = make_optimizer_and_scheduler(
        encoder_lr=0.0, head_lr=5e-3, num_epochs=3
    )
    for epoch in range(3):
        loss = run_epoch(model, train_loader, optimizer, scheduler, device)
        print(f"  Epoch {epoch+1}/3 — loss: {loss:.4f}")

    # ── Phase 2: Unfreeze top layers ─────────────────────────────────────
    print("\n━━━ Phase 2: Gradual unfreeze (top 4 layers) ━━━")
    model.unfreeze_top_layers(num_layers=4)
    optimizer, scheduler = make_optimizer_and_scheduler(
        encoder_lr=5e-5, head_lr=1e-4, num_epochs=4
    )
    for epoch in range(4):
        loss = run_epoch(model, train_loader, optimizer, scheduler, device)
        print(f"  Epoch {epoch+1}/4 — loss: {loss:.4f}")

    # ── Phase 3: Full fine-tuning ────────────────────────────────────────
    print("\n━━━ Phase 3: Full fine-tuning ━━━")
    model.unfreeze_encoder()
    optimizer, scheduler = make_optimizer_and_scheduler(
        encoder_lr=2e-5, head_lr=5e-5, num_epochs=6
    )
    for epoch in range(6):
        loss = run_epoch(model, train_loader, optimizer, scheduler, device)
        print(f"  Epoch {epoch+1}/6 — loss: {loss:.4f}")

    return model


# ─────────────────────────────────────────────
# QUICK SANITY CHECK
# ─────────────────────────────────────────────

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = MultiTaskNLPModel(
        model_name="xlm-roberta-base",
        num_classes=9,
        num_ner_tags=3,   # O, B-TECH, I-TECH
        use_crf=True,
        alpha=1.0,
        beta=1.0,
    ).to(device)

    # Dummy batch
    B, L = 4, 64
    dummy = {
        "input_ids":      torch.randint(0, 250002, (B, L)).to(device),
        "attention_mask": torch.ones(B, L, dtype=torch.long).to(device),
        "clf_labels":     torch.randint(0, 9, (B,)).to(device),
        "ner_labels":     torch.randint(0, 3, (B, L)).to(device),
    }

    outputs = model(**dummy)
    print("total_loss :", outputs["total_loss"].item())
    print("clf_logits :", outputs["clf_logits"].shape)   # (4, 9)
    print("ner preds  :", len(outputs["ner_predictions"]), "sequences")
