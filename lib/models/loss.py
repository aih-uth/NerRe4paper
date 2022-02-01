import torch
import torch.utils.data
        

def compute_ner_loss(model, ner_res, tag):
     return -model.module.crf(ner_res, tag, mask=(tag!=0), reduction="mean")


def compute_loss(rel_logits, rel_golds, device):
    rel_criterion = torch.nn.CrossEntropyLoss(reduction='none')
    rel_loss = torch.tensor(0., dtype=torch.float).to(device)
    for b, rel_logit in enumerate(rel_logits):
        batch_labels = rel_golds[b]
        batch_logits = rel_logit
        batch_loss = rel_criterion(batch_logits.to(device), batch_labels.unsqueeze(0).to(device))
        batch_loss_masked = torch.triu(batch_loss, diagonal=1)
        rel_loss += batch_loss_masked.sum()
    return rel_loss