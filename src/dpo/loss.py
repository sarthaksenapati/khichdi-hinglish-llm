"""DPO loss, implemented from scratch (matches TRL's 'sigmoid' loss).

Two pieces:
  sequence_logps  — log π(y|x): causal-shifted, gathered, response-masked, summed.
  dpo_loss        — the Bradley-Terry / DPO objective over chosen vs rejected.

Verified numerically against TRL in scripts/verify_dpo_loss.py.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


def sequence_logps(logits, labels, loss_mask):
    """Sum of per-token log-probs over the RESPONSE tokens of each sequence.

    logits:    (B, T, V) raw logits from the model
    labels:    (B, T)    token ids (the full prompt+response sequence)
    loss_mask: (B, T)    1.0 on response tokens, 0.0 on prompt/pad tokens

    Position t's logits predict token t+1, so we shift before scoring.
    Returns: (B,) summed log-prob of each response.
    """
    logits = logits[:, :-1, :]            # drop last step (nothing to predict after it)
    labels = labels[:, 1:]                # drop first token (no logit predicts it)
    loss_mask = loss_mask[:, 1:]
    logp = F.log_softmax(logits, dim=-1)                                   # (B, T-1, V)
    token_logp = torch.gather(logp, 2, labels.unsqueeze(2)).squeeze(2)     # (B, T-1)
    return (token_logp * loss_mask).sum(dim=-1)                            # (B,)


def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta: float = 0.1):
    """Vanilla (sigmoid) DPO loss.

    All four args are (B,) sequence log-probs. Returns (loss (B,), chosen_reward (B,),
    rejected_reward (B,)). The rewards are the implicit DPO reward = beta * logratio,
    detached — used only for logging (reward margin / accuracy).
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps     # policy's preference
    ref_logratios = ref_chosen_logps - ref_rejected_logps          # reference's preference
    margin = pi_logratios - ref_logratios                          # improvement over reference
    loss = -F.logsigmoid(beta * margin)

    chosen_reward = beta * (policy_chosen_logps - ref_chosen_logps).detach()
    rejected_reward = beta * (policy_rejected_logps - ref_rejected_logps).detach()
    return loss, chosen_reward, rejected_reward
