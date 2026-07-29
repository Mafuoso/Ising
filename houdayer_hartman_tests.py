import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


BAND = (0.15, 0.655)     # fixed for all Binder collapses
TCR = (2.24, 2.31)
NUR = (0.7, 1.4)
ERR_FLOOR = 1e-4
N_BLOCKS = 20

def block_jackknife_mean(x, n_blocks=N_BLOCKS):
    """ 
    Mean over raw per-snapshot samples with delete-block jackknife SE.
    Args x: 1D array of per-snapshot samples (e.g. energy or magnetization)
    Args n_blocks: number of contiguous blocks to use for jackknife

    Returns: (mean, SE) tuple
    """
    x = np.asarray(x, float); n = x.shape[0]; B = min(n_blocks, n)
    if B < 2: return float(x.mean()), 0.0
    m = n // B; xb = x[:B*m]
    block_sums = xb.reshape(B, m).sum(axis=1)            
    loo = (xb.sum() - block_sums) / (B*m - m)            # leave-one-block-out means
    var = (B-1)/B * np.sum((loo - loo.mean())**2)
    return float(x.mean()), float(np.sqrt(var))

#Q:What is the informative band? 
def _active(curve, band, err_floor=ERR_FLOOR):
      """Keep only points inside the informative band; floor the error bars.
      A curve = (L, T_array, O_array, err_array)."""
      L, T, P, e = curve 
      m = (P >= band[0]) & (P <= band[1]) 
      e2 = np.maximum(e, err_floor) if err_floor else e
      return L, T[m], P[m], e2[m]

#Gets the s value for a given Tc and nu. 
def collapse_S(params, curves, band, err_floor=ERR_FLOOR):
      """Bhattacharjee-Seno collapse quality; leave-one-size-out master, 1/err^2 weighted."""
      Tc, nu = params
      if nu <= 0: return np.inf
      scaled = []
      for L, T, P, e in (_active(c, band, err_floor) for c in curves):
          if T.size: scaled.append(((T - Tc) * L**(1.0/nu), P, e))
      if len(scaled) < 2: return np.inf
      S, count = 0.0, 0
      for j in range(len(scaled)):
          xj, yj, ej = scaled[j]
          xo = np.concatenate([scaled[k][0] for k in range(len(scaled)) if k != j])
          yo = np.concatenate([scaled[k][1] for k in range(len(scaled)) if k != j])
          o = np.argsort(xo); xo, yo = xo[o], yo[o] #Sort the master points by x
          if xo.size < 2: continue
          ins = (xj >= xo[0]) & (xj <= xo[-1])             # only score points the master spans
          if not ins.any(): continue
          ybar = np.interp(xj[ins], xo, yo) #interpolate master to get expected values for the left-out size
          S += np.sum(((yj[ins] - ybar) / ej[ins])**2); count += int(ins.sum()) #Compare the left-out size to the master, weighted by error bars, Chi^2 like. Normalzie the metric by how many points contributed to it, so that S is independent of the number of points in the band.
      return S / count if count else np.inf

#Scan over a grid of Tc and nu values to find the minimum S. Return the best Tc, nu, S, and the grid spacing.
def _scan(curves, tcr, nur, n, band, err_floor):
      tcs = np.linspace(*tcr, n); nus = np.linspace(*nur, n) #Create candidate Tc and nu values to scan over
      best, bestS = (np.mean(tcr), np.mean(nur)), np.inf
      for tc in tcs:
          for nu in nus:
              S = collapse_S((tc, nu), curves, band, err_floor)
              if S < bestS: bestS, best = S, (tc, nu)
      return best[0], best[1], bestS, (tcs[1]-tcs[0], nus[1]-nus[0])

#Use stats from the _scan function to zoom in on the best Tc and nu values. Return the best Tc, nu, and S.
def fit_collapse(curves, tcr, band, nur=(0.4, 1.5), err_floor=ERR_FLOOR, n_scan=41, n_zoom=4):
      """Coarse grid then zoom +-2 cells. Returns (Tc, nu, S_min)."""
      tc, nu, S, (dtc, dnu) = _scan(curves, tcr, nur, n_scan, band, err_floor)
      for _ in range(n_zoom): 
          tc, nu, S, (dtc, dnu) = _scan(curves, (tc-2*dtc, tc+2*dtc),
                                        (max(nu-2*dnu, 1e-3), nu+2*dnu), n_scan, band, err_floor) 
      return float(tc), float(nu), float(S)

def jackknife_nu(curves, tcr, band, nur=(0.4, 1.5)):
      """Leave-one-size-out errors on (Tc, nu). Needs >=3 sizes."""
      n = len(curves)
      if n < 3: return float('nan'), float('nan')
      tcs, nus = [], []
      for i in range(n):
          tc, nu, _ = fit_collapse([c for k, c in enumerate(curves) if k != i], tcr, band, nur)
          tcs.append(tc); nus.append(nu)
      tcs, nus = np.array(tcs), np.array(nus); fac = (n-1)/n
      return (np.sqrt(fac*np.sum((tcs-tcs.mean())**2)), np.sqrt(fac*np.sum((nus-nus.mean())**2)))

def jackknife_binder(M, n_blocks=N_BLOCKS):
    """U4 and its error at one temperature. Blocks are contiguous."""
    M = np.asarray(M, float).ravel()
    n = M.size
    B = min(n_blocks, n)
    if B < 2:
        return float("nan"), float("nan")
    m = n // B
    a = (M[:B * m] ** 2).reshape(B, m).sum(axis=1)
    b = (M[:B * m] ** 4).reshape(B, m).sum(axis=1)
    d = B * m - m
    m2 = (a.sum() - a) / d          # leave-one-block-out <M^2>
    m4 = (b.sum() - b) / d          # leave-one-block-out <M^4>
    loo = 1.0 - m4 / (3.0 * m2 ** 2)
    var = (B - 1) / B * np.sum((loo - loo.mean()) ** 2)
    return float(loo.mean()), float(np.sqrt(var))


def binder_curve(L):
    d = np.load(f"ising_samples_L{L}.npz")
    T = d["T"]
    M = d["M"].astype(np.float64)
    U = np.empty(T.size)
    e = np.empty(T.size)
    for j in range(T.size):
        U[j], e[j] = jackknife_binder(M[:, j])
    return (int(d["L"]), T, U, e)


TC_EXACT = 2.269185314
NU_EXACT = 1.0
NU_RANGE = (0.7, 1.4)

def load_M(L):
    d = np.load(f"ising_samples_L{L}.npz")
    M = d["M"].astype(float)
    if M.ndim == 3:                          # newer files: (n_samp, n_seed, n_T)
        M = M.reshape(-1, M.shape[-1])       # merge seeds into the time axis
    return d["T"], M                         # M is now (n_samples, n_T)


def rows_from_blocks(n, blocks, n_blocks=N_BLOCKS):
    m = n // n_blocks
    return np.concatenate([np.arange(b * m, (b + 1) * m) for b in blocks])


def get_tc(L_list, blocks):
    curves = []
    for L in L_list:
        T, M = load_M(L)
        X = M[rows_from_blocks(M.shape[0], blocks)]
        U = np.empty(T.size); e = np.empty(T.size)
        for j in range(T.size):
            U[j], e[j] = jackknife_binder(X[:, j])
        curves.append((L, T, U, e))
    tc, nu, S = fit_collapse(curves, tcr=TCR, band=BAND,
                             nur=NU_RANGE, err_floor=1e-4)
    return tc, nu

def fit_nu_fixedTc(curves, Tc, band, nur=NU_RANGE, n_scan=241, n_zoom=4):
    lo, hi = nur
    for _ in range(n_zoom + 1):
        nus = np.linspace(lo, hi, n_scan)
        S = np.array([collapse_S((Tc, nu), curves, band) for nu in nus])  # Tc frozen
        k = int(np.argmin(S)); d = nus[1] - nus[0] 
        lo, hi = max(nus[k] - 2*d, 1e-3), nus[k] + 2*d
    return float(nus[k]), float(S[k])


L_list = [16, 32, 48, 64]

tc_best, nu_best = get_tc(L_list, np.arange(N_BLOCKS))

rng = np.random.default_rng(0)
boot = np.array([get_tc(L_list, rng.integers(0, N_BLOCKS, N_BLOCKS))
                 for _ in range(200)])
s_tc, s_nu = boot.std(axis=0, ddof=1)

for name, fit, exact, s in (("Tc", tc_best, TC_EXACT, s_tc),
                            ("nu", nu_best, NU_EXACT, s_nu)):
    off = fit - exact
    print(f"{name}: fit = {fit:.6f} +- {s:.6f}   exact = {exact:.6f}   "
          f"off = {off:+.2e} = {off/s:+.2f} sigma")

    
#get many nu values for a range of T_c values and then we can calculate the spread
candidate_tc = np.linspace(TC_EXACT - 0.005, TC_EXACT + 0.005, 21)
nus = []
metrics = []
curves = [binder_curve(L) for L in L_list]
for T_c in candidate_tc:
    nu, S = fit_nu_fixedTc(curves, T_c, band=BAND, nur=NU_RANGE)
    nus.append(nu)
    metrics.append(S)

#Analyze the data
spread = np.std(nus)
print(f"Spread of nu values for T_c in [{candidate_tc[0]:.6f}, {candidate_tc[-1]:.6f}]: {spread:.6f}")
dnudt = np.gradient(nus, candidate_tc)
print(f"Estimated d(nu)/d(T_c): {dnudt.mean():.6f}")


S = np.asarray(metrics)
nus = np.asarray(nus)
ok = S <= S.min() * 2          # competitive region

fig, ax = plt.subplots(2, 1, figsize=(6.5, 6), sharex=True,
                       gridspec_kw={'hspace': 0.12})

for a in ax:
    a.axvspan(candidate_tc[ok].min(), candidate_tc[ok].max(),
              color='0.90', zorder=0)
    a.axvline(TC_EXACT, color='crimson', ls='--', lw=1.2, zorder=1)
    a.grid(alpha=0.25, lw=0.5)

ax[0].axhline(NU_EXACT, color='crimson', ls=':', lw=1.2)
ax[0].plot(candidate_tc, nus, '-', color='0.75', lw=1, zorder=2)
ax[0].plot(candidate_tc[~ok], nus[~ok], 'o', mfc='white',
           mec='0.6', ms=5, zorder=3)
ax[0].plot(candidate_tc[ok], nus[ok], 'o', color='tab:blue', ms=6, zorder=4)
ax[0].set_ylabel(r"$\nu$")

ax[1].plot(candidate_tc, S, '-', color='0.75', lw=1, zorder=2)
ax[1].plot(candidate_tc[~ok], S[~ok], 'o', mfc='white',
           mec='0.6', ms=5, zorder=3)
ax[1].plot(candidate_tc[ok], S[ok], 'o', color='tab:green', ms=6, zorder=4)
ax[1].set_yscale('log')
ax[1].set_ylabel(r"$S$")
ax[1].set_xlabel(r"$T_c$")

ax[0].plot([], [], '--', color='crimson', label=r"exact $T_c$, $\nu$")
ax[0].plot([], [], 'o', mfc='white', mec='0.6', label=r"$S > 2S_{\min}$")
ax[0].legend(frameon=False, fontsize=9, loc='lower left')

plt.savefig("nu_vs_Tc.png", dpi=500, bbox_inches='tight')







