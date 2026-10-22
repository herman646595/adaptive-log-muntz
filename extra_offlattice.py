"""Experiment 4: off-lattice exponent (y = x^sqrt2, same equation as Ex1).  Checks Proposition 5
(L2 error of ANY method with range V_N is >= closed-form distance) against the classical Muntz collocation."""
import json, os
from muntz_pinn_full import *

lam0 = float(np.sqrt(2.0))
al = 0.6
P = Problem("Ex4  y=x^{sqrt2} (off-lattice)", al, 10 / 3, 1.0, False,
            lambda x: Gam(lam0, al) * x ** (lam0 - al) - J0(1 + 10 / 3 * lam0, x), lambda x: x ** lam0, 1 / 5,
            (8, 10), (np.array([1.0]), np.array([0.0]), np.array([lam0])))
print("self-test residual:", self_test(P))
rows = []
for N in range(2, 13):
    a, c, lam, t = muntz_classic(P, N, False, None)
    ei, e2 = errors(a, c, lam, P)
    d = float(muntz_distance_formula(mp.mpf(lam0), [k * mp.mpf(1) / 5 for k in range(1, N + 1)]))
    rows.append(dict(N=N, linf=ei, l2=e2, dist=d, time=t))
    print(f"N={N:2d}  L2 err={e2:.3e}  dist(Prop5)={d:.3e}  ratio={e2 / d:.2f}  Linf={ei:.3e}")
r2 = [muntz_pinn_v2(P, seed=s) for s in (0, 1, 2)]
v2 = [dict(err=errors(r['a'], r['c'], r['lam'], P), time=r['time'], lam=r['lam'].tolist(), a=r['a'].tolist(), c=r['c'].tolist(), effective_exponents=(r['lam'] + np.divide(r['c'], r['a'], out=np.zeros_like(r['c']), where=np.abs(r['a'])>1e-14)).tolist()) for r in r2]
for v in v2: print("v2:", v)
cert = []
for N in (4, 6, 8, 10):
    a, c, lam, t = muntz_classic(P, N, False, None)
    ce = certificate(P, a, c, lam)
    cert.append(dict(N=N, err=errors(a, c, lam, P)[0], bound=ce['bound']))
a, c, lam = r2[0]['a'], r2[0]['c'], r2[0]['lam']
ce = certificate(P, a, c, lam); cert.append(dict(N='v2', err=errors(a, c, lam, P)[0], bound=ce['bound']))
os.makedirs(OUT, exist_ok=True)
json.dump(dict(rows=rows, v2=v2, cert=cert), open(f"{OUT}/results_offlattice.json", "w"), indent=1)
L = [r"\begin{tabular}{rrrr}", r"\toprule", r"$N$ & $L_2$ error (collocation) & $\mathrm{dist}_{L^2}(x^{\sqrt2},V_N)$ & ratio \\", r"\midrule"]
for r in rows[::2]:
    L.append(f"{r['N']} & ${sci(r['l2'])}$ & ${sci(r['dist'])}$ & {r['l2'] / r['dist']:.2f} \\\\")
L += [r"\bottomrule", r"\end{tabular}"]
open(f"{OUT}/tab_offlattice.tex", "w").write("\n".join(L))
