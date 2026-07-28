import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Import all the data
L_4 = pd.read_csv('ising_results_4.csv')
L_8 = pd.read_csv('ising_results_8.csv')
L_16 = pd.read_csv('ising_results_16.csv')
L_32 = pd.read_csv('ising_results_32.csv')


ERR_FLOOR = 1e-2        # floor so near-zero error bars can't dominate S
N_BLOCKS  = 10          # delete-block jackknife bins (contiguous -> captures autocorrelation)


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


#Test collapse for nu on the 2D Ising model. The exact value is nu=1.0.

#first we have to get error bars for the magnetization data. 
def get_err_M(df):
    """Get error bars for the magnetization data using block jackknife."""
    err_M = []
    for T in df['T'].unique():
        M_samples = df[df['T'] == T]['M_samples'].values[0]  # Assuming M_samples is a list of samples
        mean, err = block_jackknife_mean(M_samples)
        err_M.append(err)
    return np.array(err_M)

def test_nu():
     curves = [(L_4['L'][0], L_4['T'].values, L_4['M'].values, L_4['err_M'].values),
               (L_8['L'][0], L_8['T'].values, L_8['M'].values, L_8['err_M'].values),
               (L_16['L'][0], L_16['T'].values, L_16['M'].values, L_16['err_M'].values),
               (L_32['L'][0], L_32['T'].values, L_32['M'].values, L_32['err_M'].values)] # (Temperature, Magnetization, Error) tuples for each system size
     band = (
