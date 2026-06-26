"""Verify our from-scratch DPO loss (src/dpo/loss.py).

Checks, in order:
  1. sequence_logps matches a brute-force per-token loop.
  2. Analytic invariant: policy == reference  =>  loss == log 2, rewards == 0.
  3. Gradient direction: raising chosen logp lowers loss; raising rejected logp raises it.
  4. Numerical match against TRL's sigmoid DPO loss (real TRL if installed,
     else an inline copy of TRL's exact formula).

  python scripts/verify_dpo_loss.py
"""
from __future__ import annotations
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dpo.loss import sequence_logps, dpo_loss

torch.manual_seed(0)
BETA = 0.1


def check(name, ok):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    assert ok, name


# ---- 1. sequence_logps vs brute force ----
B, T, V = 4, 7, 50
logits = torch.randn(B, T, V)
labels = torch.randint(0, V, (B, T))
mask = torch.zeros(B, T)
mask[:, 3:] = 1.0                        # first 3 tokens are "prompt"

ours = sequence_logps(logits, labels, mask)
brute = torch.zeros(B)
for b in range(B):
    for t in range(T - 1):               # position t predicts token t+1
        lp = F.log_softmax(logits[b, t], dim=-1)[labels[b, t + 1]]
        brute[b] += mask[b, t + 1] * lp
check("sequence_logps matches brute-force loop", torch.allclose(ours, brute, atol=1e-5))


# ---- 2. analytic invariant: policy == reference ----
p = torch.randn(B)
loss, cr, rr = dpo_loss(p, p - 1.0, p, p - 1.0, beta=BETA)   # policy logps == ref logps
check("policy==ref => loss == log 2", torch.allclose(loss, torch.full((B,), math.log(2)), atol=1e-6))
check("policy==ref => chosen_reward == 0", torch.allclose(cr, torch.zeros(B), atol=1e-6))
check("policy==ref => rejected_reward == 0", torch.allclose(rr, torch.zeros(B), atol=1e-6))


# ---- 3. gradient direction ----
pc = torch.tensor([0.0], requires_grad=True)
pr = torch.tensor([0.0], requires_grad=True)
rc = torch.tensor([0.0])
rr_ = torch.tensor([0.0])
loss, _, _ = dpo_loss(pc, pr, rc, rr_, beta=BETA)
loss.sum().backward()
check("d loss / d chosen_logp < 0 (raising chosen lowers loss)", pc.grad.item() < 0)
check("d loss / d rejected_logp > 0 (raising rejected raises loss)", pr.grad.item() > 0)


# ---- 4. match TRL ----
pc = torch.randn(B); pr = torch.randn(B); rc = torch.randn(B); rr = torch.randn(B)
ours_loss, ours_cr, ours_rr = dpo_loss(pc, pr, rc, rr, beta=BETA)

used_real_trl = False
try:
    from trl import DPOTrainer

    class _Dummy:                         # minimal object to bind the method to
        beta = BETA
        loss_type = "sigmoid"
        label_smoothing = 0.0
        reference_free = False
        f_divergence_type = getattr(__import__("trl"), "FDivergenceType", None)
    d = _Dummy()
    out = DPOTrainer.dpo_loss(d, pc, pr, rc, rr)
    trl_loss = out[0] if isinstance(out, tuple) else out
    used_real_trl = True
except Exception as e:                    # version drift / not installed -> inline TRL formula
    print(f"  (real TRL unavailable: {type(e).__name__}; using inline TRL formula)")
    pi_logratios = pc - pr
    ref_logratios = rc - rr
    logits_ = pi_logratios - ref_logratios
    trl_loss = -F.logsigmoid(BETA * logits_)   # TRL sigmoid branch, label_smoothing=0

check(f"loss matches TRL ({'real' if used_real_trl else 'inline formula'})",
      torch.allclose(ours_loss, trl_loss, atol=1e-6))

print("\nAll DPO-loss checks passed.")
