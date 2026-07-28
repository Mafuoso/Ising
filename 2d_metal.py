import time
import numpy as np
import torch
from tqdm import tqdm

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
torch.manual_seed(995)
DTYPE = torch.float32

_MASK = {}


def checkerboard(L):
    if L not in _MASK:
        i, j = torch.meshgrid(torch.arange(L, device=device),
                              torch.arange(L, device=device), indexing="ij")
        even = ((i + j) % 2 == 0).view(1, 1, L, L)
        _MASK[L] = (even, ~even)
    return _MASK[L]


def neighbour_sum(s):
    return (torch.roll(s, 1, -1) + torch.roll(s, -1, -1)
            + torch.roll(s, 1, -2) + torch.roll(s, -1, -2))


def compute_energies(s, J=1.0):
    """Total energy of each slot. Shape (B, R)."""
    return -0.5 * J * (s * neighbour_sum(s)).sum(dim=(-2, -1))


def metropolis(s, beta4, buf, J=1.0):
    """One sweep. One random buffer serves both sublattices."""
    even, odd = checkerboard(s.shape[-1])
    buf.uniform_()
    for mask in (even, odd):
        dE = 2.0 * J * s * neighbour_sum(s)
        flip = mask & (buf < torch.exp(-dE * beta4))
        s = torch.where(flip, -s, s)
    return s


def swap_labels(E_slot, pos, beta_lad, pairs, n_acc):
    """Exchange temperature labels. The spins do not move.

    pos[b, t] gives the slot that holds temperature t.
    """
    B = E_slot.shape[0]
    for p, (t1, t2) in enumerate(pairs):
        s1 = pos[:, t1]
        s2 = pos[:, t2]
        d = (beta_lad[t1] - beta_lad[t2]) * (torch.gather(E_slot, 1, s1)
                                             - torch.gather(E_slot, 1, s2))
        acc = torch.rand(B, t1.numel(), device=device) < torch.exp(d.clamp(max=0.0))
        pos[:, t1] = torch.where(acc, s2, s1)
        pos[:, t2] = torch.where(acc, s1, s2)
        n_acc[p] += acc.sum()
    return pos


def beta_from_pos(pos, beta_lad):
    """Inverse temperature of each slot. Shape (B, R, 1, 1)."""
    B, R = pos.shape
    out = torch.empty(B, R, device=device, dtype=DTYPE)
    out.scatter_(1, pos, beta_lad.expand(B, R))
    return out.view(B, R, 1, 1)


def make_pairs(R):
    return [(torch.arange(p, R - 1, 2, device=device),
             torch.arange(p, R - 1, 2, device=device) + 1) for p in (0, 1)]


def run(L, T_values, n_equil, n_prod, n_batch=1,
        swap_every=5, measure_every=5, J=1.0):
    R = T_values.numel()
    beta_lad = (1.0 / T_values).to(DTYPE)
    s = (torch.randint(0, 2, (n_batch, R, L, L), device=device,
                       dtype=DTYPE) * 2.0 - 1.0)
    buf = torch.empty_like(s)
    pos = torch.arange(R, device=device).expand(n_batch, R).clone()
    pairs = make_pairs(R)
    n_acc = torch.zeros(2, device=device)
    beta4 = beta_from_pos(pos, beta_lad)

    for k in tqdm(range(n_equil), desc=f"equil L={L}", miniters=500):
        s = metropolis(s, beta4, buf, J)
        if k % swap_every == 0:
            pos = swap_labels(compute_energies(s, J), pos, beta_lad, pairs, n_acc)
            beta4 = beta_from_pos(pos, beta_lad)

    n_meas = n_prod // measure_every
    E_out = torch.empty(n_meas, n_batch, R, device=device, dtype=DTYPE)
    M_out = torch.empty(n_meas, n_batch, R, device=device, dtype=DTYPE)
    n_acc.zero_()
    m = 0

    torch.mps.synchronize()
    t0 = time.perf_counter()
    for k in tqdm(range(n_prod), desc=f"prod  L={L}", miniters=500):
        s = metropolis(s, beta4, buf, J)
        if k % swap_every == 0:
            E_slot = compute_energies(s, J)
            pos = swap_labels(E_slot, pos, beta_lad, pairs, n_acc)
            beta4 = beta_from_pos(pos, beta_lad)
        if k % measure_every == 0 and m < n_meas:
            # gather from slot order into temperature order
            E_out[m] = torch.gather(compute_energies(s, J) / (L * L), 1, pos)
            M_out[m] = torch.gather(s.mean(dim=(-2, -1)).abs(), 1, pos)
            m += 1
    torch.mps.synchronize()

    dt = (time.perf_counter() - t0) / n_prod
    n_att = (n_prod // swap_every) * n_batch * sum(a.numel() for a, _ in pairs)
    print(f"L={L}  {dt * 1e3:.2f} ms/sweep   swap acceptance "
          f"{(n_acc.sum() / n_att).item():.3f}")
    return E_out.cpu().numpy(), M_out.cpu().numpy()


if __name__ == "__main__":
    T_values = torch.cat([
        torch.linspace(1.60, 2.05, 6),
        torch.linspace(2.10, 2.45, 22),
        torch.linspace(2.55, 3.40, 6),
    ]).to(device)

    for L in (32,48,64):
        nb = max(1, 65536 // (L * L))       # fill the GPU with independent seeds
        E, M = run(L, T_values, 10000, 100000, n_batch=nb)
        np.savez_compressed(f"ising_samples_L{L}.npz",
                            L=L, T=T_values.cpu().numpy(),
                            E=E.reshape(E.shape[0], -1, E.shape[2]),
                            M=M.reshape(M.shape[0], -1, M.shape[2]))