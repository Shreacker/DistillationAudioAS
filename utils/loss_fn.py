import numpy as np
import math
import torch
from torch import nn
from torch.nn import functional as F

def _importance_weight(
        teacher_logits: torch.Tensor,
        gamma: float=1.,
        eps: float=1e-8
):
    p = F.softmax(teacher_logits.detach(), dim=-1)
    entropy = -(p * torch.log(p.clamp_min(eps))).sum(dim=-1)
    confidence = 1.0 - entropy/math.log(p.size(-1))

    return confidence.clamp(0.0, 1.0) ** gamma

class ClassroomLoss(nn.Module):
    def __init__(
            self,
            weight: torch.Tensor | None = None,
            label_smoothing: float=0.0,
            eps: float=1e-8
    ):
        super().__init__()
        self.register_buffer('weight', weight)
        self.label_smoothing = label_smoothing
        self.eps = eps

    def forward(
            self,
            student_logits: torch.Tensor,
            target: torch.Tensor,
            teacher_logits: torch.Tensor,
            classroom_probs: torch.Tensor | None = None,
            peer_probs: torch.Tensor=None,
            T: float=1.0,
            alpha: float=0.0,
            beta: float=0.0,
            lambda_: float=0.0,
            gamma: float = 1.0,
    ) -> torch.Tensor:
        
        w = _importance_weight(teacher_logits, gamma)
        
        ce = F.cross_entropy(
            student_logits, target,
            weight=self.weight,
            reduction='none',
            label_smoothing=self.label_smoothing
        )
        loss = ((1.0 - w) * ce).mean()

        log_ps = F.log_softmax(student_logits / T, dim=-1)
        
        if alpha > 0:
            p_t = F.softmax(teacher_logits.detach() / T, dim=-1)
            rco = F.kl_div(log_ps, p_t, reduction='none').sum(dim=-1)
            loss = loss + (alpha * T**2 * (w * rco).mean())

        if beta > 0 and classroom_probs is not None:
            p_c = F.softmax(torch.log(classroom_probs.clamp_min(self.eps)) / T, dim=-1)
            kl = F.kl_div(log_ps, p_c, reduction='none').sum(dim=-1)
            loss = loss + (beta * T**2 * kl.mean())

        if lambda_ > 0 and peer_probs is not None:
            p_s = F.softmax(student_logits, dim=-1)
            cos = F.cosine_similarity(p_s.unsqueeze(0), peer_probs, dim=-1)
            loss = loss + lambda_*cos.mean()

        return loss

class ClassroomCache:
    def __init__(self, n_students: int, n_samples: int, n_classes: int=2):
        self.n = n_students
        self.probs = torch.zeros(n_students, n_samples, n_classes)
        self.filled = [False] * n_students

    @torch.no_grad()
    def update(self, s: int, idx: torch.Tensor, student_logits: torch.Tensor):
        self.probs[s, idx.cpu()] = F.softmax(student_logits.detach(), dim=-1).cpu()

    def mark_filled(self, s: int):
        self.filled[s] = True

    def get(self, s: int, idx: torch.Tensor, device, ens_weight=None):
        others = [j for j in range(self.n) if j != s and self.filled[j]]
        if not others:
            return None, None
        
        peers = self.probs[others][:, idx.cpu()].to(device)
        if ens_weight is None:
            w = torch.ones(len(others))
        else:
            w = torch.as_tensor([ens_weight[j] for j in others], dtype=torch.float32)
        
        w = (w / w.sum()).to(device)
        classroom = torch.tensordot(w, peers, dims=1)

        return classroom, peers
