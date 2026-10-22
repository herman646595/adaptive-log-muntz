"""
MUNTZ-PINN  --  comprehensive code (all experiments of the paper)

Six benchmark cases: five nonlinear problems and one linear manufactured verification

    ^C D^alpha y(x) = g(x) + kappa * x^theta * int_0^x ln(x-t) t^sigma |y(t)|^q dt ,   y(0)=y0

 Ex1 : alpha=3/5, q=10/3, outer=1 , exact y = x^{8/5}
 Ex2 : alpha=2/7, q=2   , outer=x , exact y = x^{1+alpha} ln x
 Ex3 : (our own, severely ill-conditioned) alpha=0.4, q=2, kappa=10, outer=1,
       exact y = x^{0.45} + x^{1.30} - x^{1.32}
       -> weak singularity at 0 (y' ~ x^{-0.55}), two nearly-cancelling exponents (gap 0.02),
          multiscale exponents, nonlinear term with six different exponents.

Methods compared
 (1) classical Muntz collocation (fixed lattice exponents, power basis and Muntz-Legendre basis)
 (2) standard PINN (MLP, L1 fractional derivative, product integration), manual backprop
 (3) adaptive log-Muntz : learnable exponents + log-augmented terms x^lam (a + c ln x), exact fractional operators

Only NumPy / SciPy / mpmath / matplotlib are needed. The master runner is code/run_all_benchmarks.py; run this module with [all|test] for the core cases.
Outputs (tables, figures, results.json) are written to ./paper_out/
"""
import sys, time
import numpy as np
import mpmath as mp
from dataclasses import dataclass
from typing import Callable
from scipy.special import gamma, gammaln, digamma, polygamma
from scipy.optimize import least_squares, minimize


# =============================================================================
# problem definitions
# =============================================================================
@dataclass
class Problem:
    name: str
    alpha: float
    q: float
    kappa: float
    outer_x: bool
    g: Callable
    y_exact: Callable
    lattice_step: float          # exponent lattice of the classical Muntz method
    N_list: tuple                # sizes used for the classical method
    exact_terms: tuple           # (a, c, lam) of the exact solution (for self-test only)
    kernel_t: bool = True
    y0: float = 0.0
    outer_power: float | None = None

    def outer(self, x):
        """Power multiplying the Volterra integral (legacy bool maps to power 0 or 1)."""
        p = self.outer_power if self.outer_power is not None else (1.0 if self.outer_x else 0.0)
        return np.asarray(x) ** p if p else np.ones_like(np.asarray(x), dtype=float)


def J0(m, x):   # int_0^x ln(x-t) t^m dt
    e = m + 1.0
    return x ** e / e * (np.log(x) - digamma(m + 2.0) - np.euler_gamma)


def J2(m, x):   # int_0^x ln(x-t) t^m ln^2 t dt = d^2/dm^2 J0
    e, L = m + 1.0, np.log(x)
    f = x ** e / e
    f1 = f * (L - 1 / e)
    f2 = f * ((L - 1 / e) ** 2 + 1 / e ** 2)
    h = digamma(m + 2.0) + np.euler_gamma
    return f2 * (L - h) - 2 * f1 * polygamma(1, m + 2.0) - f * polygamma(2, m + 2.0)


def Gam(lam, alpha):
    return np.exp(gammaln(lam + 1) - gammaln(lam + 1 - alpha))


def make_ex1():
    al, lam0 = 0.6, 1.6
    g = lambda x: Gam(lam0, al) * x ** (lam0 - al) - J0(1 + 10 / 3 * lam0, x)
    return Problem("Ex1  y=x^{8/5}", al, 10 / 3, 1.0, False, g, lambda x: x ** lam0, 1 / 5, (8, 10),
                   (np.array([1.0]), np.array([0.0]), np.array([lam0])))


def make_ex2():
    al = 2 / 7
    p = 1 + al
    d = digamma(p + 1) - digamma(p + 1 - al)
    g = lambda x: Gam(p, al) * x ** (p - al) * (np.log(x) + d) - x * J2(1 + 2 * p, x)
    return Problem("Ex2  y=x^{1+a} ln x, a=2/7", al, 2.0, 1.0, True, g,
                   lambda x: x ** p * np.log(x), 1 / 7, (8, 12, 16),
                   (np.array([0.0]), np.array([1.0]), np.array([p])))


def make_ex3():
    al, kap = 0.4, 10.0
    co = np.array([1.0, 1.0, -1.0]); pw = np.array([0.45, 1.30, 1.32])

    def g(x):
        lin = sum(c * Gam(p, al) * x ** (p - al) for c, p in zip(co, pw))
        nl = sum(ci * cj * J0(1 + pi + pj, x) for ci, pi in zip(co, pw) for cj, pj in zip(co, pw))
        return lin - kap * nl
    return Problem("Ex3  y=x^.45+x^1.30-x^1.32 (ill-cond.)", al, 2.0, kap, False, g,
                   lambda x: sum(c * x ** p for c, p in zip(co, pw)), 1 / 20, (10, 20, 30),
                   (co, np.zeros(3), pw))



def make_ex4():
    # Almusawi & Najafi (2024), Example 4: alpha=4/5, y=x^2-x,
    # kernel ln(x-t)*y(t)^2 (no additional factor t).
    al = 4/5
    def g(x):
        caputo = Gam(2.0, al)*x**(2-al) - Gam(1.0, al)*x**(1-al)
        integ = J0(4.0, x) - 2.0*J0(3.0, x) + J0(2.0, x)
        return caputo - integ
    return Problem("Ex4  y=x^2-x (AN 2024)", al, 2.0, 1.0, False, g,
                   lambda x: x**2-x, 1/5, (8, 10, 12),
                   (np.array([-1.0, 1.0]), np.zeros(2), np.array([1.0, 2.0])),
                   kernel_t=False)

def make_ex5(alpha=1/3):
    # Almusawi & Najafi (2024), Example 2: p(0)=1, p=1-x^(1+alpha),
    # D_C^alpha p = h_alpha(x) + x^(7/2) int_0^x ln(x-t) p(t)^3 dt.
    # h is generated exactly by substitution; no t factor occurs in the kernel.
    al = float(alpha)
    pwr = 1.0 + al
    def g(x):
        caputo = -Gam(pwr, al) * x
        integ = (J0(0.0, x) - 3.0*J0(pwr, x)
                 + 3.0*J0(2.0*pwr, x) - J0(3.0*pwr, x))
        return caputo - x**3.5 * integ
    return Problem(f"Ex5  y=1-x^(1+alpha), alpha={al:g} (AN 2024)", al, 3.0, 1.0, False, g,
                   lambda x: 1.0 - x**pwr, 1/7, (8, 10, 12),
                   (np.array([-1.0]), np.array([0.0]), np.array([pwr])),
                   kernel_t=False, y0=1.0, outer_power=3.5)

def make_ex6():
    # Headayat & Hamasalh (2026), manufactured linear verification:
    # D_C^0.6 y - int_0^x ln(x-t)y(t)dt = g(x), exact y=x^1.6.
    al, lam = 0.6, 1.6
    def g(x):
        return Gam(lam, al) * x**(lam-al) - J0(lam, x)
    return Problem("Ex6  linear verification y=x^{8/5} (HH 2026)", al, 1.0, 1.0, False,
                   g, lambda x: x**lam, 1/5, (8, 10),
                   (np.array([1.0]), np.array([0.0]), np.array([lam])),
                   kernel_t=False)

PROBLEMS = {1: make_ex1, 2: make_ex2, 3: make_ex3, 4: make_ex4, 5: make_ex5, 6: make_ex6}


# =============================================================================
# quadrature on s in (0,1), clustered at both ends (log singularity at s=1, power at s=0)
# =============================================================================
def make_quad(n=40):
    xg, wg = np.polynomial.legendre.leggauss(n)
    r, wr = (xg + 1) / 2, wg / 2
    s1 = 0.5 * r ** 3; w1 = 1.5 * r ** 2 * wr
    u2 = 0.5 * r ** 4; s2 = 1 - u2; w2 = 2.0 * r ** 3 * wr
    s = np.r_[s1, s2]; w = np.r_[w1, w2]
    wl = w * np.r_[np.log1p(-s1), np.log(u2)]
    return s, w, wl


# =============================================================================
# residual + analytic Jacobians for  y = sum_k x^{lam_k} (a_k + c_k ln x)
# =============================================================================
def resid(a, c, lam, x, P, Q, jac=True):
    al = P.alpha
    s, w, wl = Q
    G = Gam(lam, al)
    dl = digamma(lam + 1) - digamma(lam + 1 - al)
    dl1 = polygamma(1, lam + 1) - polygamma(1, lam + 1 - al)
    lx = np.log(x)
    xp = np.exp(np.outer(lx, lam - al))                         # M x K
    Aterm = a[None, :] + c[None, :] * (lx[:, None] + dl[None, :])
    D = (G * xp * Aterm).sum(1)
    X = x[:, None] * s[None, :]
    lX = np.log(X)
    Pw = np.exp(lX[:, :, None] * lam[None, None, :])            # M x J x K
    Y = P.y0 + Pw @ a + (Pw * lX[:, :, None]) @ c
    aY = np.abs(Y)
    F = P.kappa * (X if P.kernel_t else 1.0) * aY ** P.q
    pref = P.outer(x)[:, None]
    wt = pref * x[:, None] * (lx[:, None] * w[None, :] + wl[None, :])
    R = D - P.g(x) - (wt * F).sum(1)
    if not jac:
        return R
    A = wt * P.kappa * (X if P.kernel_t else 1.0) * P.q * aY ** (P.q - 1) * np.sign(Y)
    PA = np.einsum('ij,ijk->ik', A, Pw)
    PAl = np.einsum('ij,ijk->ik', A, Pw * lX[:, :, None])
    PAll = np.einsum('ij,ijk->ik', A, Pw * (lX ** 2)[:, :, None])
    Ja = G * xp - PA
    Jc = G * xp * (lx[:, None] + dl[None, :]) - PAl
    Jl = xp * (G * lx[:, None] * Aterm + G * dl[None, :] * Aterm + G * c[None, :] * dl1[None, :]) \
        - (a[None, :] * PAl + c[None, :] * PAll)
    return R, Ja, Jc, Jl


def eval_y(a, c, lam, x, y0=0.0):
    lx = np.log(np.maximum(x, 1e-300))
    return y0 + np.exp(np.outer(lx, lam)) @ a + (np.exp(np.outer(lx, lam)) * lx[:, None]) @ c


def cheb_pts(M):
    i = np.arange(1, M + 1)
    return 0.5 * (1 - np.cos(np.pi * (2 * i - 1) / (2 * M)))


XF = np.unique(np.r_[np.linspace(0, 1, 1001), np.geomspace(1e-8, 1, 400)])


def err_from_values(yv, P):
    e = yv - P.y_exact(np.maximum(XF, 1e-300))
    e[XF == 0] = yv[XF == 0] - P.y0
    return np.abs(e).max(), np.sqrt(np.trapezoid(e ** 2, XF))


def errors(a, c, lam, P):
    return err_from_values(eval_y(a, c, lam, XF, P.y0), P)


# =============================================================================
# (1) classical Muntz collocation
# =============================================================================
def muntz_ml_matrix(lam, dps=60):
    mp.mp.dps = dps
    L = [mp.mpf(float(v)) for v in lam]; N = len(L)
    C = np.zeros((N, N))
    for n in range(1, N + 1):
        for k in range(1, n + 1):
            num = mp.mpf(1)
            for j in range(1, n):
                num *= (L[k - 1] + L[j - 1] + 1)
            den = mp.mpf(1)
            for j in range(1, n + 1):
                if j != k:
                    den *= (L[k - 1] - L[j - 1])
            C[n - 1, k - 1] = float(num / den)
    return C


def muntz_classic(P, N, legendre, step=None):
    lam = np.arange(1, N + 1) * (step if step else P.lattice_step)
    Q = make_quad(40); x = cheb_pts(N)
    C = muntz_ml_matrix(lam) if legendre else np.eye(N)
    z = np.zeros(N)
    t0 = time.perf_counter()
    fun = lambda u: resid(C.T @ u, z, lam, x, P, Q, jac=False)
    jac = lambda u: resid(C.T @ u, z, lam, x, P, Q)[1] @ C.T
    sol = least_squares(fun, np.zeros(N), jac=jac, method='lm', xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=300)
    t = time.perf_counter() - t0
    return (C.T @ sol.x, z, lam, t)


# =============================================================================
# (2) standard PINN
# =============================================================================
def pinn_operators(M, alpha):
    h = 1.0 / M
    t = np.arange(M + 1) * h
    bj = np.arange(1, M + 1) ** (1 - alpha) - np.arange(0, M) ** (1 - alpha)
    L = np.zeros((M, M + 1)); c0 = h ** (-alpha) / gamma(2 - alpha)
    for n in range(1, M + 1):
        for m in range(1, n + 1):
            L[n - 1, m] += c0 * bj[n - m]; L[n - 1, m - 1] -= c0 * bj[n - m]
    U0 = lambda u: np.where(u > 0, u * np.log(np.where(u > 0, u, 1)) - u, 0.0)
    U1 = lambda u: np.where(u > 0, 0.5 * u ** 2 * np.log(np.where(u > 0, u, 1)) - 0.25 * u ** 2, 0.0)
    W = np.zeros((M, M + 1))
    for n in range(1, M + 1):
        for j in range(n):
            a_, b_ = (n - j - 1) * h, (n - j) * h
            M0, M1 = U0(b_) - U0(a_), U1(b_) - U1(a_)
            W[n - 1, j] += (M1 - a_ * M0) / h; W[n - 1, j + 1] += (b_ * M0 - M1) / h
    return t, L, W


class MLP:
    def __init__(self, sizes, seed=0):
        r = np.random.default_rng(seed)
        self.shapes = [(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)]
        self.p = np.concatenate([np.r_[r.normal(0, np.sqrt(1 / a), a * b), np.zeros(b)] for a, b in self.shapes])

    def unpack(self, p):
        out, i = [], 0
        for a, b in self.shapes:
            out.append((p[i:i + a * b].reshape(a, b), p[i + a * b:i + a * b + b])); i += a * b + b
        return out

    def forward(self, p, t):
        H = [t[:, None]]
        for k, (Wm, bv) in enumerate(self.unpack(p)):
            z = H[-1] @ Wm + bv
            H.append(z if k == len(self.shapes) - 1 else np.tanh(z))
        return H

    def backward(self, p, H, gout):
        grads, g = [], gout[:, None]
        for k in range(len(self.shapes) - 1, -1, -1):
            Wm, _ = self.unpack(p)[k]
            grads.append((H[k].T @ g, g.sum(0)))
            if k > 0:
                g = (g @ Wm.T) * (1 - H[k] ** 2)
        return np.concatenate([np.r_[gw.ravel(), gb] for gw, gb in grads[::-1]])


def pinn_standard(P, M=100, adam_it=6000, lbfgs_it=3000, seed=0):
    t, L, W = pinn_operators(M, P.alpha)
    gv = P.g(t[1:])
    pref = P.outer(t[1:])[:, None]
    net = MLP([1, 30, 30, 30, 1], seed)

    def loss_grad(p):
        H = net.forward(p, t)
        y = P.y0 + t * H[-1][:, 0]
        kernel_factor = t if P.kernel_t else np.ones_like(t)
        F = P.kappa * kernel_factor * np.abs(y) ** P.q
        dF = P.kappa * kernel_factor * P.q * np.abs(y) ** (P.q - 1) * np.sign(y)
        r = L @ y - gv - (pref * W) @ F
        Jy = L - (pref * W) * dF[None, :]
        return np.mean(r ** 2), net.backward(p, H, t * (2.0 / M) * (Jy.T @ r))

    t0 = time.perf_counter()
    p = net.p.copy(); m = np.zeros_like(p); v = np.zeros_like(p)
    for it in range(1, adam_it + 1):
        lr = 3e-3 * (0.05 ** (it / adam_it))
        l, gr = loss_grad(p)
        m = .9 * m + .1 * gr; v = .999 * v + .001 * gr ** 2
        p -= lr * (m / (1 - .9 ** it)) / (np.sqrt(v / (1 - .999 ** it)) + 1e-8)
    res = minimize(loss_grad, p, jac=True, method='L-BFGS-B',
                   options=dict(maxiter=lbfgs_it, maxfun=2 * lbfgs_it, ftol=1e-30, gtol=1e-14, maxcor=30))
    tt = time.perf_counter() - t0
    global LAST_PINN
    LAST_PINN = (net, res.x)
    yf = P.y0 + XF * net.forward(res.x, XF)[-1][:, 0]
    return err_from_values(yf, P) + (tt,)


# =============================================================================
# (3) proposed adaptive log-Muntz with learnable exponents and log-augmented terms
# =============================================================================
def muntz_pinn(P, K=36, M=50, adam_it=4000, mu=2e-4, seed=0, lam_range=(0.2, 3.5), tol=1e-9, max_terms=8,
               min_gap=0.005):
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    Q = make_quad(24); x = cheb_pts(M)
    lo_, hi_ = lam_range
    lam = lo_ + (hi_ - lo_) * (np.arange(K) + rng.uniform(0.15, 0.85, K)) / K    # generic dictionary
    a = np.zeros(K); c = np.zeros(K)
    st = {k: (np.zeros(K), np.zeros(K)) for k in "acl"}
    # ---- Stage A: Adam on (a, c, lam) with L1 sparsity -------------------------------------------
    for it in range(1, adam_it + 1):
        R, Ja, Jc, Jl = resid(a, c, lam, x, P, Q)
        gr = {'a': (2 / M) * Ja.T @ R + mu * np.sign(a), 'c': (2 / M) * Jc.T @ R + mu * np.sign(c),
              'l': (2 / M) * Jl.T @ R}
        dec = 0.1 ** (it / adam_it)
        for key, lr0, arr in (('a', 1e-2, a), ('c', 1e-2, c), ('l', 2e-3, lam)):
            m_, v_ = st[key]
            m_[:] = .9 * m_ + .1 * gr[key]; v_[:] = .999 * v_ + .001 * gr[key] ** 2
            arr -= lr0 * dec * (m_ / (1 - .9 ** it)) / (np.sqrt(v_ / (1 - .999 ** it)) + 1e-12)
        np.clip(lam, 0.15, 4.0, out=lam)
    tA = time.perf_counter() - t0
    # ---- Stage B: forward selection driven by the physics residual -------------------------------
    Q2 = make_quad(40); xv = np.geomspace(1e-3, 1, 200)

    def refit(a_, c_, l_):
        n = len(a_); x2 = cheb_pts(max(40, 4 * n))
        unpack = lambda z: (z[:n], z[n:2 * n], z[2 * n:])
        fun = lambda z: resid(*unpack(z), x2, P, Q2, jac=False)
        def jac(z):
            _, Ja, Jc, Jl = resid(*unpack(z), x2, P, Q2)
            return np.hstack([Ja, Jc, Jl])
        lmin = 0.15
        lo = np.r_[-np.inf * np.ones(2 * n), lmin * np.ones(n)]
        hi = np.r_[np.inf * np.ones(2 * n), 6.0 * np.ones(n)]
        z0 = np.r_[a_, c_, np.clip(l_, lmin + 1e-9, 6 - 1e-9)]
        sol = least_squares(fun, z0, jac=jac, bounds=(lo, hi), method='trf', xtol=1e-15, ftol=1e-15, gtol=1e-15,
                            max_nfev=150)
        a2, c2, l2 = unpack(sol.x)
        return a2, c2, l2, np.abs(resid(a2, c2, l2, xv, P, Q2, jac=False)).max()

    a_A, c_A, l_A = a.copy(), c.copy(), lam.copy()
    order = list(np.argsort(-(np.abs(a_A) + np.abs(c_A))))
    j0 = order.pop(0)
    a, c, lam, res = refit(a_A[[j0]], c_A[[j0]], l_A[[j0]])
    while res > tol and order and len(a) < max_terms:
        cand = next((j for j in order if np.all(np.abs(l_A[j] - lam) > min_gap)), None)
        if cand is None:
            break
        order.remove(cand)
        a, c, lam, res = refit(np.r_[a, a_A[cand]], np.r_[c, c_A[cand]], np.r_[lam, l_A[cand]])
    tB = time.perf_counter() - t0 - tA
    return a, c, lam, res, tA, tB, time.perf_counter() - t0


# =============================================================================
def self_test(P):
    Q = make_quad(40)
    x = np.linspace(0.05, 1, 9)
    a, c, lam = P.exact_terms
    return np.abs(resid(a, c, lam, x, P, Q, jac=False)).max()


def jac_test():
    P = make_ex2(); Q = make_quad(24); x = np.linspace(.1, 1, 6)
    a, c, lam = np.array([.3, -.2]), np.array([.5, .1]), np.array([.8, 1.7]); e = 1e-6
    R, Ja, Jc, Jl = resid(a, c, lam, x, P, Q)
    out = []
    for J, which in ((Ja, 0), (Jc, 1), (Jl, 2)):
        fd = []
        for k in range(2):
            d = np.zeros(2); d[k] = e; args = [a, c, lam]
            ap = [v.copy() for v in args]; am = [v.copy() for v in args]
            ap[which] += d; am[which] -= d
            fd.append((resid(*ap, x, P, Q, jac=False) - resid(*am, x, P, Q, jac=False)) / (2 * e))
        out.append(np.abs(J - np.column_stack(fd)).max())
    return out




# =============================================================================
# (3b) adaptive log-Muntz v2 : physics-informed exponent matching pursuit + joint LM + confluent splitting
# =============================================================================
XV = np.geomspace(1e-6, 1, 300)                      # validation grid for the sup-residual


def colloc(n):
    """Chebyshev points plus geometric points near 0 (needed to resolve x^{lam-alpha}, lam-alpha ~ 0)."""
    return np.unique(np.r_[cheb_pts(max(40, 4 * n)), np.geomspace(1e-6, 3e-4, 12)])


def refit2(P, a_, c_, l_, Q2, nfev=200):
    """Joint Levenberg-Marquardt/TRF on (a, c, lam); constraint lam >= alpha + 1e-6 (bounded D^alpha for power-log atoms)."""
    lmin = P.alpha + 1e-6
    n = len(a_); x2 = colloc(n)
    unpack = lambda z: (z[:n], z[n:2 * n], z[2 * n:])
    fun = lambda z: resid(*unpack(z), x2, P, Q2, jac=False)

    def jac(z):
        _, Ja, Jc, Jl = resid(*unpack(z), x2, P, Q2)
        return np.hstack([Ja, Jc, Jl])
    lo = np.r_[-np.inf * np.ones(2 * n), lmin * np.ones(n)]
    hi = np.r_[np.inf * np.ones(2 * n), 6.0 * np.ones(n)]
    z0 = np.r_[a_, c_, np.clip(l_, lmin + 1e-9, 6 - 1e-9)]
    sol = least_squares(fun, z0, jac=jac, bounds=(lo, hi), method='trf', xtol=1e-15, ftol=1e-15, gtol=1e-15,
                        max_nfev=nfev)
    a2, c2, l2 = unpack(sol.x)
    return a2, c2, l2, np.abs(resid(a2, c2, l2, XV, P, Q2, jac=False)).max()


def muntz_pinn_v2(P, seed=0, tol=1e-9, tolC=1e-11, max_terms=8, min_gap=0.004, split=True, patience=3, beam=4):
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    grid = np.linspace(P.alpha + 0.005, 3.8, 360)
    grid = np.maximum(grid + rng.uniform(-0.004, 0.004, grid.size), P.alpha + 0.001)   # jittered dictionary
    Q = make_quad(24); Q2 = make_quad(40); x = cheb_pts(60)
    a = np.zeros(0); c = np.zeros(0); lam = np.zeros(0)
    res = np.inf; best = None; bad = 0; history = []
    # ---- Stage A/B: exponent matching pursuit; every step is followed by a joint refit --------------
    while len(a) < max_terms:
        K = len(a); Kc = len(grid)
        aa = np.r_[a, np.zeros(Kc)]; cc = np.r_[c, np.zeros(Kc)]; ll = np.r_[lam, grid]
        R, Ja, Jc, Jl = resid(aa, cc, ll, x, P, Q)
        A1, A2 = Ja[:, K:], Jc[:, K:]
        g11 = (A1 * A1).sum(0); g22 = (A2 * A2).sum(0); g12 = (A1 * A2).sum(0)
        b1 = A1.T @ R; b2 = A2.T @ R
        det = g11 * g22 - g12 ** 2 + 1e-30 * (g11 * g22)
        gain = (b1 ** 2 * g22 - 2 * b1 * b2 * g12 + b2 ** 2 * g11) / np.maximum(det, 1e-300)
        ok = np.ones(Kc, bool)
        for l in lam:
            ok &= np.abs(grid - l) > min_gap
        gain[~ok] = -1
        peaks = [i for i in range(1, Kc - 1) if gain[i] > 0 and gain[i] >= gain[i - 1] and gain[i] >= gain[i + 1]]
        if not peaks:
            peaks = [int(np.argmax(gain))]
        peaks = sorted(peaks, key=lambda i: -gain[i])[:beam]
        best_c = None
        for j in peaks:
            Gm_ = np.array([[g11[j], g12[j]], [g12[j], g22[j]]]); bb = -np.array([b1[j], b2[j]])
            da, dc = np.linalg.solve(Gm_ + 1e-12 * np.eye(2) * np.trace(Gm_), bb)
            fit = refit2(P, np.r_[a, da], np.r_[c, dc], np.r_[lam, grid[j]], Q2)
            if best_c is None or fit[3] < best_c[3]:
                best_c = fit
        a_n, c_n, l_n, r_n = best_c
        bad = 0 if (r_n < res / 3 or len(a) == 0) else bad + 1
        a, c, lam, res = a_n, c_n, l_n, r_n
        history.append((a.copy(), c.copy(), lam.copy(), res))
        if best is None or res < best[3]:
            best = (a.copy(), c.copy(), lam.copy(), res)
        if res < tol or bad >= patience:
            break
    a, c, lam, res = best
    tA = time.perf_counter() - t0
    # ---- Stage C: merge near-coincident exponents, then confluent splitting ------------------------
    nsplit = 0
    if split and res > tolC:
        changed = True
        while changed and len(a) > 1:
            changed = False
            o = np.argsort(lam)
            for u, v in zip(o[:-1], o[1:]):
                if lam[v] - lam[u] < 0.05:
                    wu = abs(a[u]) + abs(c[u]) + 1e-30; wv = abs(a[v]) + abs(c[v]) + 1e-30
                    lb = (wu * lam[u] + wv * lam[v]) / (wu + wv)
                    am = a[u] + a[v]
                    cm = c[u] + c[v] + a[u] * (lam[u] - lb) + a[v] * (lam[v] - lb)
                    keep = (np.arange(len(a)) != u) & (np.arange(len(a)) != v)
                    a, c, lam, res = refit2(P, np.r_[a[keep], am], np.r_[c[keep], cm], np.r_[lam[keep], lb], Q2)
                    changed = True
                    break
        improved = True
        while improved and len(a) < max_terms + 2:
            improved = False
            for j in np.argsort(-np.abs(c)):
                if abs(c[j]) < 1e-10:
                    continue
                cands = []
                for dlt in (0.003, 0.006, 0.012, 0.025, 0.05, 0.1):
                    keep = np.arange(len(a)) != j
                    a_s = np.r_[a[keep], a[j] / 2 + c[j] / (2 * dlt), a[j] / 2 - c[j] / (2 * dlt)]
                    c_s = np.r_[c[keep], 0.0, 0.0]
                    l_s = np.r_[lam[keep], lam[j] + dlt, lam[j] - dlt]
                    cands.append(refit2(P, a_s, c_s, l_s, Q2))
                bst = min(cands, key=lambda t: t[3])
                if bst[3] < res / 10:
                    a, c, lam, res = bst; improved = True; nsplit += 1
                    break
        keep = (np.abs(a) + np.abs(c)) > 1e-9
        if keep.sum() < len(a):
            a, c, lam, res = refit2(P, a[keep], c[keep], lam[keep], Q2)
        history.append((a.copy(), c.copy(), lam.copy(), res))
    return dict(a=a, c=c, lam=lam, res=res, tA=tA, nsplit=nsplit, time=time.perf_counter() - t0, history=history)


# =============================================================================
# A POSTERIORI CERTIFICATE  (Theorem 1 of the paper)
# =============================================================================
def mittag_leffler(beta, z):
    """E_beta(z) = sum z^k / Gamma(beta k + 1), z >= 0, evaluated in log-space.  Returns (value or inf)."""
    if z <= 0:
        return 1.0
    if z ** (1 / beta) > 720:          # E_beta(z) >= exp(z^{1/beta})/beta overflows
        return np.inf
    kmax = int(3 * z ** (1 / beta) + 200)
    k = np.arange(kmax)
    lg = k * np.log(z) - gammaln(beta * k + 1)
    m = lg.max()
    if m > 700:
        return np.inf
    return float(np.exp(m) * np.exp(lg - m).sum())


def certificate(P, a, c, lam):
    """Compute a sampled-residual diagnostic; this is not a validated continuous certificate."""
    xs = np.unique(np.r_[np.geomspace(1e-8, 1, 500), np.linspace(1e-3, 1, 600)])
    Q = make_quad(40)
    rho = resid(a, c, lam, xs, P, Q, jac=False)
    gam = max(0.0, P.alpha - float(lam.min()))
    Rg = np.max(xs ** gam * np.abs(rho))
    eta = Rg * gamma(1 - gam) / gamma(1 + P.alpha - gam)
    B0 = np.max(np.abs(eval_y(a, c, lam, xs, P.y0)))
    best = None
    for d in np.linspace(0.02, 0.98, 49):
        beta = 1 + P.alpha - d
        r = eta; okk = True                      # smallest r with  eta * E_beta(c(B0+r)) <= r  (condition (H) of Thm 1)
        for _ in range(2000):
            cd = P.kappa * P.q * (B0 + r) ** (P.q - 1) * gamma(1 - d) / (np.e * d)
            E = mittag_leffler(beta, cd)
            rn = eta * E
            if not np.isfinite(rn) or rn > 1e3:
                okk = False; break
            if rn <= r * (1 + 1e-12):
                break
            r = rn
        if not okk:
            continue
        cd = P.kappa * P.q * (B0 + r) ** (P.q - 1) * gamma(1 - d) / (np.e * d)
        E = mittag_leffler(beta, cd)
        bound = eta * E
        if best is None or bound < best['bound']:
            best = dict(eta=eta, Rg=Rg, gamma=gam, r=max(r, bound), delta=d, c=cd, E=E, bound=bound, B0=B0)
    if best is None:
        best = dict(eta=eta, Rg=Rg, gamma=gam, r=np.nan, delta=np.nan, c=np.nan, E=np.nan, bound=np.nan, B0=B0)
    return best


# =============================================================================
# Proposition 3 check: L2 distance from x^mu to a fixed Muntz space (closed form vs high-precision LS)
# =============================================================================
def muntz_distance_formula(mu, lam):
    mu = mp.mpf(mu)
    val = 1 / mp.sqrt(2 * mu + 1)
    for l in lam:
        val *= abs(mu - l) / (mu + l + 1)
    return val


def muntz_distance_numeric(mu, lam, dps=120):
    mp.mp.dps = dps
    mu = mp.mpf(mu); lam = [mp.mpf(l) for l in lam]
    H = mp.matrix(len(lam), len(lam)); b = mp.matrix(len(lam), 1)
    for i, li in enumerate(lam):
        b[i] = 1 / (li + mu + 1)
        for j, lj in enumerate(lam):
            H[i, j] = 1 / (li + lj + 1)
    d2 = 1 / (2 * mu + 1) - (b.T * mp.lu_solve(H, b))[0]
    return mp.sqrt(d2)


def distance_check(h=mp.mpf(1) / 5, mus=("1.41421356237309504880168872420969807856967", "1.6")):
    mp.mp.dps = 120
    out = {}
    for mu_s in mus:
        mu = mp.mpf(mu_s); rows = []
        for N in range(1, 13):
            lam = [k * h for k in range(1, N + 1)]
            rows.append((N, float(muntz_distance_formula(mu, lam)), float(muntz_distance_numeric(mu, lam))))
        out[mu_s[:6]] = rows
    return out


# =============================================================================
# EXPERIMENTS, TABLES, FIGURES
# =============================================================================
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.environ.get("MP_OUT", "paper_out")
XP = np.geomspace(1e-6, 1, 400)                     # grid for pointwise-error profiles


def sci(x, d=2):
    if x is None or not np.isfinite(x):
        return r"\text{n/a}"
    if x == 0:
        return "0"
    e = int(np.floor(np.log10(abs(x)))); m = x / 10 ** e
    if round(m, d) >= 10:
        m /= 10; e += 1
    return f"{m:.{d}f}\\!\\times\\!10^{{{e}}}"


def classical_rows(P):
    rows, models = [], {}
    specs = []
    for N in P.N_list:
        for leg in (True, False):
            if leg and N > 10:
                continue
            specs.append((N, leg, None))
    for N in (10, 20):
        specs.append((N, False, 1.0))                        # integer lattice: Genocchi-type polynomial collocation
    for N, leg, step in specs:
        a, c, lam, t = muntz_classic(P, N, leg, step)
        ei, e2 = errors(a, c, lam, P)
        st = step if step else P.lattice_step
        name = (f"Polynomial (integer powers), $N={N}$" if step == 1.0 else
                f"M\\\"untz{'--Legendre' if leg else ' (power basis)'}, lattice ${st:.4g}k$, $N={N}$")
        rows.append(dict(method=name, kind='classical', linf=ei, l2=e2, time=t, nseeds=1))
        models[(N, leg, step)] = (a, c, lam)
    return rows, models


def agg(vals):
    v = np.array(vals, float)
    return float(np.median(v)), float(np.max(v))


def experiment(idx):
    P = PROBLEMS[idx]()
    print(f"\n=== {P.name} ===", flush=True)
    res = dict(name=P.name, rows=[], cert=[], profiles={})
    crow, cmod = classical_rows(P)
    res['rows'] += crow
    print("  classical done", flush=True)
    # ---- standard PINN
    for M, seeds in ((100, (0, 1, 2)), (200, (0,))):
        out = []
        for s in seeds:
            ei, e2, t = pinn_standard(P, M=M, seed=s)[:3]
            out.append((ei, e2, t))
            if M == 100 and s == 0:
                net, p = LAST_PINN
                res['profiles']['pinn'] = np.abs(P.y0 + XP * net.forward(p, XP)[-1][:, 0] - P.y_exact(XP)).tolist()
        res['rows'].append(dict(method=f"Standard PINN (L1, $M={M}$)", kind='pinn', linf=agg([o[0] for o in out])[0],
                                linf_max=agg([o[0] for o in out])[1], l2=agg([o[1] for o in out])[0],
                                time=agg([o[2] for o in out])[0], nseeds=len(seeds)))
    print("  PINN done", flush=True)
    # ---- adaptive log-Muntz v1 (Adam + L1 dictionary)
    out = []; v1_model = None
    for s in (0, 1, 2):
        a, c, lam, rs, tA, tB, tt = muntz_pinn(P, seed=s)
        ei, e2 = errors(a, c, lam, P)
        out.append((ei, e2, tt, len(a)))
        if s == 0:
            v1_model = (a, c, lam)
    res['rows'].append(dict(method="adaptive log-M\\\"untz v1 (Adam+$\\ell_1$ dictionary)", kind='v1', linf=agg([o[0] for o in out])[0],
                            linf_max=agg([o[0] for o in out])[1], l2=agg([o[1] for o in out])[0],
                            time=agg([o[2] for o in out])[0], nseeds=3, nterms=int(np.median([o[3] for o in out]))))
    print("  v1 done", flush=True)
    # ---- adaptive log-Muntz v2 (pursuit + LM + splitting) and ablation without splitting
    v2_hist = None; v2_model = None
    for tag, split in (("v2 without splitting", False), ("v2 (proposed)", True)):
        out = []
        for s in (0, 1, 2):
            r = muntz_pinn_v2(P, seed=s, split=split)
            ei, e2 = errors(r['a'], r['c'], r['lam'], P)
            out.append((ei, e2, r['time'], len(r['a']), r['res']))
            if s == 0 and split:
                v2_hist = r['history']; v2_model = (r['a'], r['c'], r['lam'])
                res['v2_terms'] = [dict(lam=float(l), a=float(x), c=float(y)) for l, x, y in
                                   sorted(zip(r['lam'], r['a'], r['c']))]
                res['v2_hist_res'] = [float(h[3]) for h in r['history']]
        res['rows'].append(dict(method=f"adaptive log-M\\\"untz {tag}", kind='v2s' if split else 'v2n',
                                linf=agg([o[0] for o in out])[0], linf_max=agg([o[0] for o in out])[1],
                                l2=agg([o[1] for o in out])[0], time=agg([o[2] for o in out])[0], nseeds=3,
                                nterms=int(np.median([o[3] for o in out]))))
    print("  v2 done", flush=True)
    # ---- certificates on a ladder of approximants
    ladder = []
    Ns = {1: (3, 4, 5, 6, 7, 8, 9, 10), 2: (4, 6, 8, 10, 12, 14, 16), 3: (6, 10, 14, 18, 22, 26, 30), 4: (4, 6, 8, 10), 5: (4, 6, 8, 10), 6: (3, 4, 5, 6, 7, 8, 9, 10)}[idx]
    for N in Ns:
        a, c, lam, t = muntz_classic(P, N, False, None)
        ladder.append((f"Muntz power N={N}", a, c, lam))
    for k, h in enumerate(v2_hist):
        ladder.append((f"v2 step {k + 1}", h[0], h[1], h[2]))
    ladder.append(("v1 final", *v1_model))
    for name, a, c, lam in ladder:
        ei = errors(a, c, lam, P)[0]
        ce = certificate(P, a, c, lam)
        res['cert'].append(dict(name=name, err=float(ei), bound=float(ce['bound']), eta=float(ce['eta']),
                                E=float(ce['E']) if np.isfinite(ce['E']) else None, c=float(ce['c']), delta=float(ce['delta'])))
    print("  certificates done", flush=True)
    # ---- profiles
    best_key = {1: (10, False, None), 2: (12, False, None), 3: (30, False, None), 4: (10, False, None), 5: (10, False, None), 6: (10, False, None)}[idx]
    a, c, lam = cmod[best_key]
    res['profiles']['muntz'] = np.abs(eval_y(a, c, lam, XP, P.y0) - P.y_exact(XP)).tolist()
    res['profiles']['v1'] = np.abs(eval_y(*v1_model[:2], v1_model[2], XP, P.y0) - P.y_exact(XP)).tolist()
    res['profiles']['v2'] = np.abs(eval_y(*v2_model[:2], v2_model[2], XP, P.y0) - P.y_exact(XP)).tolist()
    return res


# --------------------------------------------------------------------------------------------
def write_tables(allres, dist):
    os.makedirs(OUT, exist_ok=True)
    for i, R in enumerate(allres, 1):
        L = [r"\begin{tabular}{lrrr}", r"\toprule", r"Method & sampled max error & sampled $L_2$ error & Time (s) \\", r"\midrule"]
        for r in R['rows']:
            e = sci(r['linf']) if 'linf_max' not in r else sci(r['linf']) + (f" \\; ({sci(r['linf_max'])})" if r['nseeds'] > 1 else "")
            L.append(f"{r['method']} & ${e}$ & ${sci(r['l2'])}$ & {r['time']:.2f} \\\\")
        L += [r"\bottomrule", r"\end{tabular}"]
        open(f"{OUT}/tab_ex{i}.tex", "w").write("\n".join(L))
        L = [r"\begin{tabular}{lrrr}", r"\toprule", r"Approximant & reference-grid max error & sampled diagnostic & ratio \\", r"\midrule"]
        rows = [c for c in R['cert'] if np.isfinite(c['bound'])]
        rows = sorted(rows, key=lambda c: c['err'])
        sel = rows[::max(1, len(rows) // 7)][:8]
        for c in sel:
            L.append(f"{c['name']} & ${sci(c['err'])}$ & ${sci(c['bound'])}$ & {c['bound'] / max(c['err'], 1e-300):.1e} \\\\")
        L += [r"\bottomrule", r"\end{tabular}"]
        open(f"{OUT}/tab_cert{i}.tex", "w").write("\n".join(L))
    L = [r"\begin{tabular}{rrrr}", r"\toprule", r"$N$ & closed form (Prop.~\ref{prop:dist}) & 120-digit Gram matrix & relative difference " + chr(92)*2, r"\midrule"]
    mu_hp = mp.mpf("1.41421356237309504880168872420969807856967")
    for N, f_, n_ in dist['1.4142'][1::2]:
        lam_hp = [mp.mpf(k) / 5 for k in range(1, N + 1)]
        f_hp = muntz_distance_formula(mu_hp, lam_hp)
        n_hp = muntz_distance_numeric(mu_hp, lam_hp, dps=120)
        rel_hp = abs(f_hp - n_hp) / f_hp
        L.append(f"{N} & ${sci(f_, 6)}$ & ${sci(n_, 6)}$ & ${mp.nstr(rel_hp, 3)}$ " + chr(92)*2)
    L += [r"\bottomrule", r"\end{tabular}"]
    open(f"{OUT}/tab_dist.tex", "w").write("\n".join(L))
    # Ex3 recovered structure
    L = [r"\begin{tabular}{rrr}", r"\toprule", r"$\lambda_k$ & $a_k$ & $c_k$ \\", r"\midrule"]
    for t in allres[2]['v2_terms']:
        L.append(f"{t['lam']:.6f} & {t['a']:+.6f} & {t['c']:+.2e} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(f"{OUT}/tab_ex3_terms.tex", "w").write("\n".join(L))


def make_figures(allres, dist):
    os.makedirs(OUT, exist_ok=True)
    plt.rcParams.update({"font.size": 9})
    # Fig 1: bars of L_inf
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.4))
    for ax, R, ttl in zip(axs, allres, ["Example 1: $x^{8/5}$", "Example 2: $x^{1+\\alpha}\\ln x$", "Example 3 (ill-conditioned)"]):
        def pick(kind, fn=None):
            rr = [r for r in R['rows'] if r['kind'] == kind and (fn is None or fn(r))]
            return min(r['linf'] for r in rr) if kind == 'classical' else rr[0]['linf']
        poly = min(r['linf'] for r in R['rows'] if 'Polynomial' in r['method'])
        mz = min(r['linf'] for r in R['rows'] if r['kind'] == 'classical' and 'Polynomial' not in r['method'])
        vals = [poly, mz, pick('pinn', lambda r: '100' in r['method']), pick('v1'), pick('v2s')]
        labs = ["Poly", "Müntz\n(best N)", "PINN", "M-PINN\nv1", "M-PINN\nv2"]
        cols = ["#999999", "#6baed6", "#fd8d3c", "#9e9ac8", "#31a354"]
        ax.bar(range(5), vals, color=cols); ax.set_yscale("log"); ax.set_xticks(range(5)); ax.set_xticklabels(labs, fontsize=6)
        ax.set_title(ttl, fontsize=9); ax.set_ylim(1e-15, 1)
        ax.grid(axis="y", alpha=.3)
    axs[0].set_ylabel("$L_\\infty$ error")
    plt.tight_layout(); plt.savefig(f"{OUT}/fig_bars.pdf"); plt.savefig(f"{OUT}/fig_bars.jpg", dpi=300); plt.close()
    # Fig 2: certificate scatter
    fig, ax = plt.subplots(figsize=(4.4, 4))
    mk = ["o", "s", "^"]
    for R, m in zip(allres, mk):
        pts = [(c['err'], c['bound']) for c in R['cert'] if np.isfinite(c['bound'])]
        if pts:
            e, b = zip(*pts); ax.loglog(e, b, m, label=R['name'].split("  ")[0], alpha=.8)
    lo, hi = 1e-14, 1e0
    xx = np.array([lo, hi]); ax.loglog(xx, xx, "k--", lw=1, label="bound = error")
    ax.set_xlabel("true $\\|y-\\tilde y\\|_\\infty$"); ax.set_ylabel("sampled estimate (Thm. 1)"); ax.legend(fontsize=7); ax.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(f"{OUT}/fig_cert.pdf"); plt.savefig(f"{OUT}/fig_cert.jpg", dpi=300); plt.close()
    # Fig 3: Ex3 profiles + residual path
    R = allres[2]
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.4))
    for key, lab, col in (("muntz", "Müntz $k/20$, $N=30$", "#6baed6"), ("pinn", "Standard PINN", "#fd8d3c"),
                          ("v1", "adaptive log-Müntz v1", "#9e9ac8"), ("v2", "adaptive log-Müntz v2", "#31a354")):
        axs[0].loglog(XP, np.maximum(R['profiles'][key], 1e-17), label=lab, color=col)
    axs[0].set_xlabel("$x$"); axs[0].set_ylabel("$|y-\\tilde y|$"); axs[0].legend(fontsize=7); axs[0].grid(alpha=.3)
    axs[0].set_title("Ex3: pointwise error")
    hr = R['v2_hist_res']
    axs[1].semilogy(range(1, len(hr) + 1), hr, "o-", color="#31a354")
    axs[1].set_xlabel("pursuit step (last point: after merge/split)"); axs[1].set_ylabel("sup residual"); axs[1].grid(alpha=.3)
    axs[1].set_title("Ex3: residual along the pursuit")
    plt.tight_layout(); plt.savefig(f"{OUT}/fig_ex3.pdf"); plt.savefig(f"{OUT}/fig_ex3.jpg", dpi=300); plt.close()
    # Fig 4: Muntz distance
    fig, ax = plt.subplots(figsize=(4.4, 3.4))
    rows = dist['1.4142']
    Nn = [r[0] for r in rows]; ax.semilogy(Nn, [r[1] for r in rows], "-", label="closed form"); ax.semilogy(Nn, [r[2] for r in rows], "o", ms=4, label="high-precision LS")
    ax.set_xlabel("$N$"); ax.set_ylabel("$\\mathrm{dist}_{L^2}(x^{\\sqrt{2}},V_N)$"); ax.legend(fontsize=7); ax.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(f"{OUT}/fig_dist.pdf"); plt.savefig(f"{OUT}/fig_dist.jpg", dpi=300); plt.close()


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which == "test":
        print("Jacobian errors (a,c,lam):", jac_test())
        for i in (1, 2, 3, 4, 5, 6):
            print(i, self_test(PROBLEMS[i]()))
        return
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    dist = distance_check()
    allres = []
    for i in (1, 2, 3):                                  # resumable: each example is cached
        f = f"{OUT}/ex{i}.json"
        if os.path.exists(f):
            allres.append(json.load(open(f)))
        else:
            r = experiment(i)
            json.dump(r, open(f, "w"))
            allres.append(r)
    json.dump(dict(results=allres, dist=dist), open(f"{OUT}/results.json", "w"), indent=1)
    write_tables(allres, dist)
    make_figures(allres, dist)
    print(f"\nall done in {time.time() - t0:.0f}s; outputs in {OUT}/")


if __name__ == "__main__":
    main()
