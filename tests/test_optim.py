"""The batched port must reproduce schedulefree.AdamWScheduleFree member by member."""
import pytest
import schedulefree
import torch

from capacity_study.model import BatchedMLP
from capacity_study.optim import BatchedAdamWScheduleFree

BETA, BETA2, EPS = 0.9, 0.999, 1e-8


def _setup(dtype, k=4, depth=2, width=8, warmup=50):
    torch.manual_seed(0)
    lrs = torch.tensor([3e-3, 1e-2, 3e-4, 1e-3], dtype=dtype)[:k]
    wds = torch.tensor([0.0, 1e-2, 1e-1, 3e-3], dtype=dtype)[:k]
    batched = BatchedMLP(k, 19, depth, width, 0.01).to(dtype)
    batched.initialize([11, 22, 33, 44][:k])
    opt = BatchedAdamWScheduleFree(batched.ordered_parameters(), lrs, wds, BETA, BETA2, EPS, warmup)
    refs = []
    for m in range(k):
        single = BatchedMLP(1, 19, depth, width, 0.01).to(dtype)
        with torch.no_grad():
            for (ps, _), (pb, _) in zip(single.ordered_parameters(), batched.ordered_parameters()):
                ps.copy_(pb[m:m + 1])
        weights = [p for p, is_w in single.ordered_parameters() if is_w]
        biases = [p for p, is_w in single.ordered_parameters() if not is_w]
        ref_opt = schedulefree.AdamWScheduleFree(
            [{"params": weights, "weight_decay": float(wds[m])}, {"params": biases, "weight_decay": 0.0}],
            lr=float(lrs[m]), betas=(BETA, BETA2), eps=EPS, warmup_steps=warmup)
        refs.append((single, ref_opt))
    return batched, opt, refs


def _train(batched, opt, refs, steps, dtype, eval_every=25, seed=1):
    gen = torch.Generator().manual_seed(seed)
    opt.train()
    for _, ref_opt in refs:
        ref_opt.train()
    k = batched.num_members
    for step in range(steps):
        x = torch.randn(k, 16, 19, generator=gen, dtype=dtype)
        x[:, :, 15] += 18.0  # approximation-coefficient scale gap
        y = 0.01 * torch.randn(k, 16, generator=gen, dtype=dtype)
        opt.zero_grad()
        ((batched(x) - y) ** 2).mean(dim=1).sum().backward()
        opt.step()
        for m, (single, ref_opt) in enumerate(refs):
            ref_opt.zero_grad()
            ((single(x[m:m + 1]) - y[m:m + 1]) ** 2).mean().backward()
            ref_opt.step()
        if (step + 1) % eval_every == 0:
            opt.eval(); opt.train()
            for _, ref_opt in refs:
                ref_opt.eval(); ref_opt.train()


def _assert_match(batched, opt, refs, rtol, atol):
    for m, (single, ref_opt) in enumerate(refs):
        for i, ((pb, _), (ps, _)) in enumerate(zip(batched.ordered_parameters(), single.ordered_parameters())):
            torch.testing.assert_close(pb[m:m + 1], ps, rtol=rtol, atol=atol, msg=f"y member {m} tensor {i}")
            torch.testing.assert_close(opt.z[i][m:m + 1], ref_opt.state[ps]["z"], rtol=rtol, atol=atol)
            torch.testing.assert_close(opt.exp_avg_sq[i][m:m + 1], ref_opt.state[ps]["exp_avg_sq"],
                                       rtol=rtol, atol=atol)


def test_port_matches_upstream_float64():
    batched, opt, refs = _setup(torch.float64)
    _train(batched, opt, refs, steps=700, dtype=torch.float64)
    _assert_match(batched, opt, refs, rtol=1e-9, atol=1e-12)
    opt.eval()
    for _, ref_opt in refs:
        ref_opt.eval()
    _assert_match(batched, opt, refs, rtol=1e-9, atol=1e-12)  # x


def test_port_matches_upstream_float32():
    batched, opt, refs = _setup(torch.float32)
    _train(batched, opt, refs, steps=200, dtype=torch.float32)
    _assert_match(batched, opt, refs, rtol=1e-4, atol=1e-6)


def test_biases_never_decayed():
    k = 2
    model = BatchedMLP(k, 19, 1, 8, 0.01)
    model.initialize([1, 2])
    opt = BatchedAdamWScheduleFree(model.ordered_parameters(), torch.full((k,), 1e-3), torch.full((k,), 0.5),
                                   BETA, BETA2, EPS, 1)
    opt.train()
    w_before = [w.detach().clone() for w in model.weights]
    b_before = [b.detach().clone() for b in model.biases]
    for p, _ in model.ordered_parameters():
        p.grad = torch.zeros_like(p)
    opt.step()
    for b, b0 in zip(model.biases, b_before):
        assert torch.equal(b.detach(), b0)
    for w, w0 in zip(model.weights, w_before):
        torch.testing.assert_close(w.detach(), w0 * (1 - 1e-3 * 0.5))  # zero gradient: decay only


def test_checkpoint_roundtrip_is_exact():
    batched, opt, refs = _setup(torch.float32)
    _train(batched, opt, refs, steps=60, dtype=torch.float32)
    state = opt.state_dict()
    params = [p.detach().clone() for p, _ in batched.ordered_parameters()]
    _train(batched, opt, refs, steps=40, dtype=torch.float32, seed=2)
    expected = [p.detach().clone() for p, _ in batched.ordered_parameters()]

    batched2, opt2, refs2 = _setup(torch.float32)
    with torch.no_grad():
        for (p, _), saved in zip(batched2.ordered_parameters(), params):
            p.copy_(saved)
    opt2.load_state_dict(state)
    _train(batched2, opt2, refs2[:0], steps=40, dtype=torch.float32, seed=2)
    for (p, _), e in zip(batched2.ordered_parameters(), expected):
        assert torch.equal(p.detach(), e)


def test_nan_member_does_not_leak():
    batched, opt, _ = _setup(torch.float32)
    clean, clean_opt, _ = _setup(torch.float32)
    with torch.no_grad():
        batched.weights[0][1].fill_(float("nan"))
    _train(batched, opt, [], steps=30, dtype=torch.float32)
    _train(clean, clean_opt, [], steps=30, dtype=torch.float32)
    for (p, _), (q, _) in zip(batched.ordered_parameters(), clean.ordered_parameters()):
        for m in (0, 2, 3):
            assert torch.equal(p[m], q[m])
        assert not torch.isfinite(p[1]).all()


@pytest.mark.parametrize("bad", [torch.zeros(3), torch.zeros(4, 1)])
def test_rejects_wrong_hyperparameter_shape(bad):
    model = BatchedMLP(4, 19, 1, 8, 0.01)
    with pytest.raises(ValueError):
        BatchedAdamWScheduleFree(model.ordered_parameters(), bad, torch.zeros(4), BETA, BETA2, EPS, 1)
