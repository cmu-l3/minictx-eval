## Acknowledgments
# Code borrowed from the [LeanDojo ReProver repository](https://github.com/lean-dojo/ReProver).

import os
import json
import torch
from typing import Union, List

@torch.no_grad()
def encode(model, tokenizer, s: Union[str, List[str]]) -> torch.Tensor:
    """Encode texts into feature vectors."""
    if isinstance(s, str):
        s = [s]
        should_squeeze = True
    else:
        should_squeeze = False
    device = model.device
    tokenized_s = tokenizer(s, return_tensors="pt", padding=True).to(device)
    hidden_state = model(tokenized_s.input_ids).last_hidden_state
    lens = tokenized_s.attention_mask.sum(dim=1)
    features = (hidden_state * tokenized_s.attention_mask.unsqueeze(2)).sum(dim=1) / lens.unsqueeze(1)
    if should_squeeze:
      features = features.squeeze()
    return features

@torch.no_grad()
def retrieve(model, tokenizer, state: str, premises: List[str], k: int, batch_size = 32) -> List[str]:
    """Retrieve the top-k premises given a state."""
    state_emb = encode(model, tokenizer, state)
    all_premise_embs = []
    for i in range(0, len(premises), batch_size):
        batch_premises = premises[i:i + batch_size]
        batch_embs = encode(model, tokenizer, batch_premises)
        all_premise_embs.append(batch_embs)
    all_premise_embs = torch.cat(all_premise_embs, dim=0)
    scores = (state_emb @ all_premise_embs.T)
    topk = scores.topk(k).indices.tolist()
    return [premises[i] for i in topk]
