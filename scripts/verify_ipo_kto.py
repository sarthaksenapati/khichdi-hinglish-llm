"""Verify the IPO and KTO losses (src/dpo/loss.py).

Checks each loss against its analytic invariants and gradient directions, plus a
numerical match to the reference formula each method uses.

  python scripts/verify_ipo_kto.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dpo.loss import ipo_loss, kto_loss

torch.manual_seed(0)
BETA = 0.1


def check(name, ok):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    assert ok, name


# ============================ IPO ============================
B = 6

# 1. init invariant: policy == reference -> margin 0 -> loss = (1/(2β))²
p = torch.randn(B)
loss, _, _ = ipo_loss(p, p - 1.0, p, p - 1.0, beta=BETA)
target0 = (1.0 / (2 * BETA)) ** 2
check("IPO: policy==ref => loss == (1/(2β))²", torch.allclose(loss, torch.full((B,), target0), atol=1e-5))

# 2. zero-loss optimum: margin exactly 1/(2β) -> loss 0
#    set policy chosen high enough that margin == 1/(2β)
pc = torch.zeros(B) + 1.0 / (2 * BETA)
pr = torch.zeros(B)
rc = torch.zeros(B); rr = torch.zeros(B)
loss, _, _ = ipo_loss(pc, pr, rc, rr, beta=BETA)
check("IPO: margin == 1/(2β) => loss == 0", torch.allclose(loss, torch.zeros(B), atol=1e-6))

# 3. gradient: below target, raising chosen lowers loss
pc = torch.tensor([0.0], requires_grad=True)
pr = torch.tensor([0.0], requires_grad=True)
loss, _, _ = ipo_loss(pc, pr, torch.tensor([0.0]), torch.tensor([0.0]), beta=BETA)
loss.sum().backward()
check("IPO: d loss / d chosen_logp < 0 when below target", pc.grad.item() < 0)

# 4. match reference IPO formula
pc, pr, rc, rr = [torch.randn(B) for _ in range(4)]
ours, _, _ = ipo_loss(pc, pr, rc, rr, beta=BETA)
ref = ((pc - pr) - (rc - rr) - 1.0 / (2 * BETA)) ** 2
check("IPO: matches reference formula", torch.allclose(ours, ref, atol=1e-6))


# ============================ KTO ============================
# 1. init invariant: policy == reference -> r=0, z0=0 -> each loss = 1 - σ(0) = 0.5
pd = torch.randn(B); pu = torch.randn(B)
des, undes, _, _, kl = kto_loss(pd, pd, pu, pu, beta=BETA)   # policy == ref on both groups
check("KTO: policy==ref => z0 == 0", torch.allclose(kl, torch.tensor(0.0), atol=1e-6))
check("KTO: policy==ref => desirable loss == 0.5", torch.allclose(des, torch.full((B,), 0.5), atol=1e-6))
check("KTO: policy==ref => undesirable loss == 0.5", torch.allclose(undes, torch.full((B,), 0.5), atol=1e-6))

# 2. gradient: raising a DESIRABLE example's logprob lowers its loss
pd = torch.tensor([0.0], requires_grad=True)
rd = torch.tensor([0.0])
pu2 = torch.tensor([0.0]); ru2 = torch.tensor([0.0])
des, undes, _, _, _ = kto_loss(pd, rd, pu2, ru2, beta=BETA, kl=torch.tensor(0.0))
des.sum().backward()
check("KTO: d desirable_loss / d logp < 0 (raise desirable reward)", pd.grad.item() < 0)

# 3. gradient: raising an UNDESIRABLE example's logprob raises its loss
pu = torch.tensor([0.0], requires_grad=True)
des, undes, _, _, _ = kto_loss(torch.tensor([0.0]), torch.tensor([0.0]), pu, torch.tensor([0.0]),
                               beta=BETA, kl=torch.tensor(0.0))
undes.sum().backward()
check("KTO: d undesirable_loss / d logp > 0 (suppress undesirable reward)", pu.grad.item() > 0)

# 4. match reference KTO formula (fixed kl)
pd, rd, pu_, ru = [torch.randn(B) for _ in range(4)]
kl = torch.tensor(0.3)
des, undes, _, _, _ = kto_loss(pd, rd, pu_, ru, beta=BETA, kl=kl)
ref_des = 1 - torch.sigmoid(BETA * ((pd - rd) - kl))
ref_undes = 1 - torch.sigmoid(BETA * (kl - (pu_ - ru)))
check("KTO: desirable matches reference formula", torch.allclose(des, ref_des, atol=1e-6))
check("KTO: undesirable matches reference formula", torch.allclose(undes, ref_undes, atol=1e-6))

print("\nAll IPO/KTO checks passed.")
