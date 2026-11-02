"""Batched schedule-free AdamW: a vectorized port of schedulefree.AdamWScheduleFree (version 1.4.1).

Upstream: https://github.com/facebookresearch/schedule_free, schedulefree/adamw_schedulefree.py, release 1.4.1
(r = 0, weight_lr_power = 2, foreach path). The math is unchanged; only the per-group Python scalars lr and
weight_decay become per-member tensors of shape (K, 1, 1) over parameters stacked along dim 0.

As upstream, parameters hold y while training; eval() moves them to x and train() back to y; z and exp_avg_sq are
state; the Adam bias correction is applied in the denominator. The averaging weight ckp1 = lr_max^2 / sum(lr_max^2)
is the same for every member because lr cancels, so it is a single float computed from the warmup schedule.
Verified member-by-member against the upstream optimizer in tests/test_optim.py.
"""
import torch


class BatchedAdamWScheduleFree:
    def __init__(self, params: list[tuple[torch.Tensor, bool]], lr: torch.Tensor, weight_decay: torch.Tensor,
                 beta: float, beta2: float, eps: float, warmup_steps: int):
        """params: (stacked parameter, apply_weight_decay). lr, weight_decay: (K,) tensors on the param device."""
        self.params = [p for p, _ in params]
        self.decay_mask = [d for _, d in params]
        k = self.params[0].shape[0]
        if lr.shape != (k,) or weight_decay.shape != (k,):
            raise ValueError(f"lr/weight_decay must have shape ({k},), got {tuple(lr.shape)}, "
                             f"{tuple(weight_decay.shape)}")
        self.lr = lr.reshape(k, 1, 1).to(self.params[0])
        self.weight_decay = weight_decay.reshape(k, 1, 1).to(self.params[0])
        self.beta, self.beta2, self.eps, self.warmup_steps = beta, beta2, eps, warmup_steps
        self.k = 0
        self.sched_max = -1.0
        self.weight_sum = 0.0
        self.last_sched = 0.0
        self.last_ckp1 = 0.0
        self.train_mode = False
        with torch.no_grad():
            self.z = [p.detach().clone() for p in self.params]
            self.exp_avg_sq = [torch.zeros_like(p) for p in self.params]

    @torch.no_grad()
    def eval(self) -> None:
        if self.train_mode:
            torch._foreach_lerp_(self.params, self.z, 1 - 1 / self.beta)
            self.train_mode = False

    @torch.no_grad()
    def train(self) -> None:
        if not self.train_mode:
            torch._foreach_lerp_(self.params, self.z, 1 - self.beta)
            self.train_mode = True

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

    @torch.no_grad()
    def step(self) -> None:
        if not self.train_mode:
            raise RuntimeError("BatchedAdamWScheduleFree.step called outside train mode; call train() first")
        k, beta, beta2 = self.k, self.beta, self.beta2
        sched = (k + 1) / self.warmup_steps if k < self.warmup_steps else 1.0
        bias_correction2 = 1 - beta2 ** (k + 1)
        self.sched_max = max(sched, self.sched_max)
        weight = self.sched_max ** 2
        self.weight_sum += weight
        ckp1 = weight / self.weight_sum if self.weight_sum != 0 else 0.0
        self.last_sched, self.last_ckp1 = sched, ckp1

        y, z, v = self.params, self.z, self.exp_avg_sq
        grad = [p.grad for p in y]
        torch._foreach_mul_(v, beta2)
        torch._foreach_addcmul_(v, grad, grad, value=1 - beta2)
        denom = torch._foreach_div(v, bias_correction2)
        torch._foreach_sqrt_(denom)
        torch._foreach_add_(denom, self.eps)
        torch._foreach_div_(grad, denom)

        lr = self.lr * sched
        y_coef = lr * (beta * (1 - ckp1) - 1)
        for yi, gi, decay in zip(y, grad, self.decay_mask):
            if decay:
                gi.addcmul_(yi, self.weight_decay)
        torch._foreach_lerp_(y, z, ckp1)
        for yi, zi, gi in zip(y, z, grad):
            yi.addcmul_(gi, y_coef)
            zi.addcmul_(gi, lr, value=-1)
        self.k = k + 1

    def state_dict(self) -> dict:
        def snapshot(t: torch.Tensor) -> torch.Tensor:
            return t.detach().to("cpu", copy=True)  # .cpu() alone aliases CPU tensors

        return {"k": self.k, "sched_max": self.sched_max, "weight_sum": self.weight_sum,
                "train_mode": self.train_mode, "z": [snapshot(t) for t in self.z],
                "exp_avg_sq": [snapshot(t) for t in self.exp_avg_sq],
                "lr": snapshot(self.lr), "weight_decay": snapshot(self.weight_decay)}

    @torch.no_grad()
    def load_state_dict(self, state: dict) -> None:
        if not torch.equal(state["lr"], self.lr.cpu()) or not torch.equal(state["weight_decay"],
                                                                           self.weight_decay.cpu()):
            raise ValueError("load_state_dict: checkpoint lr/weight_decay differ from this optimizer's")
        self.k, self.sched_max, self.weight_sum = state["k"], state["sched_max"], state["weight_sum"]
        self.train_mode = state["train_mode"]
        for dst, src in zip(self.z, state["z"]):
            dst.copy_(src)
        for dst, src in zip(self.exp_avg_sq, state["exp_avg_sq"]):
            dst.copy_(src)
