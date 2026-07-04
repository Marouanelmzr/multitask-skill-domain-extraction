
import torch
from sklearn.metrics import f1_score

def eval_step(model, dataloader, device):
    model.eval()

    total_loss = 0

    all_domain_preds = []
    all_domain_labels = []

    all_ner_preds = []
    all_ner_labels = []

    with torch.inference_mode():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            domain_labels = batch["domain_labels"].to(device)
            ner_labels = batch["ner_labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                domain_labels=domain_labels,
                ner_labels=ner_labels,
            )

            loss = outputs["loss"]
            total_loss += loss.item()

            # Domain predictions
            domain_logits = outputs["domain_logits"]
            domain_preds = (torch.sigmoid(domain_logits) > 0.5).int()

            all_domain_preds.append(domain_preds.cpu())
            all_domain_labels.append(domain_labels.cpu().int())

            # NER predictions
            ner_logits = outputs["ner_logits"]
            ner_preds = ner_logits.argmax(dim=-1)

            mask = ner_labels != -100

            all_ner_preds.append(ner_preds[mask].cpu())
            all_ner_labels.append(ner_labels[mask].cpu())

    avg_loss = total_loss / len(dataloader)

    all_domain_preds = torch.cat(all_domain_preds).numpy()
    all_domain_labels = torch.cat(all_domain_labels).numpy()

    all_ner_preds = torch.cat(all_ner_preds).numpy()
    all_ner_labels = torch.cat(all_ner_labels).numpy()

    domain_micro_f1 = f1_score(
        all_domain_labels,
        all_domain_preds,
        average="micro",
        zero_division=0
    )

    domain_macro_f1 = f1_score(
        all_domain_labels,
        all_domain_preds,
        average="macro",
        zero_division=0
    )

    ner_micro_f1 = f1_score(
        all_ner_labels,
        all_ner_preds,
        average="micro",
        zero_division=0
    )

    ner_macro_f1 = f1_score(
        all_ner_labels,
        all_ner_preds,
        average="macro",
        zero_division=0
    )

    return {
        "loss": avg_loss,
        "domain_micro_f1": domain_micro_f1,
        "domain_macro_f1": domain_macro_f1,
        "ner_micro_f1": ner_micro_f1,
        "ner_macro_f1": ner_macro_f1,
    }
