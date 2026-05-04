import torch
import torch.nn as nn
from transformers import AutoModel

class MultitaskXLM(nn.Module):
    def __init__(
        self,
        model_name="xlm-roberta-base",
        num_domain_labels=10,
        num_ner_labels=3,
        dropout=0.1
    ):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(dropout)
        self.domain_classifier = nn.Linear(hidden_size, num_domain_labels)
        self.ner_classifier = nn.Linear(hidden_size, num_ner_labels)

    def forward(self, input_ids, attention_mask, domain_labels=None, ner_labels=None):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        hidden_states = outputs.last_hidden_state

        cls_vector = hidden_states[:, 0, :]

        domain_logits = self.domain_classifier(self.dropout(cls_vector))
        ner_logits = self.ner_classifier(self.dropout(hidden_states))

        loss = None

        if domain_labels is not None and ner_labels is not None:
            domain_loss_fn = nn.BCEWithLogitsLoss()
            ner_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

            domain_loss = domain_loss_fn(domain_logits, domain_labels.float())

            ner_loss = ner_loss_fn(
                ner_logits.reshape(-1, ner_logits.shape[-1]),
                ner_labels.reshape(-1)
            )

            loss = domain_loss + ner_loss

        return {
            "domain_logits": domain_logits,
            "ner_logits": ner_logits,
            "loss": loss
        }