"""Faithful numpy port of NMFN 2.0.1, used to check claims before asserting
them in R tests. Ports R semantics exactly, including the column/row
normalisation quirks and the final rescaling in nnmf_prob.
"""
import numpy as np

EPS_R = 2.2204e-16
MACH = np.finfo(float).eps


# ---------------------------------------------------------------- mpinv
def mpinv(X):
    Eps = 100 * MACH
    u, d, vh = np.linalg.svd(np.asarray(X, float), full_matrices=False)
    v = vh.T
    m = len(d)
    dk = d[d > Eps]
    notnull = len(dk)
    inv = np.diag(1.0 / dk) if notnull else np.zeros((0, 0))
    if notnull != m:                       # pad back to m x m with zeros
        inv = np.hstack([inv, np.zeros((notnull, m - notnull))])
        inv = np.vstack([inv, np.zeros((m - notnull, m))])
    mp = v @ inv @ u.T
    mp[np.abs(mp) < Eps] = 0.0
    return mp


def distance2(a, b):
    t = np.asarray(a, float) - np.asarray(b, float)
    return float((t * t).sum())


# ------------------------------------------------------------- nnmf_mm
def nnmf_mm(x, k, maxiter, eps=EPS_R, rng=None, trace=False):
    rng = rng or np.random.default_rng(0)
    x = np.asarray(x, float)
    n, m = x.shape
    W = np.abs(rng.standard_normal((n, k)))
    H = np.abs(rng.standard_normal((k, m)))
    errs = []
    for _ in range(maxiter):
        H = H * (W.T @ x) / ((W.T @ W) @ H + eps)
        W = W * (H @ x.T).T / (W @ (H @ H.T) + eps)
        if trace:
            errs.append(distance2(x, W @ H))
    return dict(W=W, H=H, errs=errs)


# ------------------------------------------------------------ nnmf_als
def nnmf_als(x, k, maxiter, eps=EPS_R, rng=None):
    rng = rng or np.random.default_rng(0)
    x = np.asarray(x, float)
    D, N = x.shape
    Xscale = x.sum()
    W = np.abs(rng.standard_normal((D, k)))
    H = np.abs(rng.standard_normal((k, N)))
    sqrnorm = np.sqrt((W @ H).sum() / Xscale)
    H, W = H / sqrnorm, W / sqrnorm
    for _ in range(maxiter):
        W = x @ (mpinv(H @ H.T) @ H).T
        W = (W > 0) * W
        W = W / (W.sum(axis=0)[None, :] + eps)   # column j / colSum j
        H = (W @ mpinv(W.T @ W)).T @ x
        H = H * (H > 0)
    return dict(W=W, H=H)


# ----------------------------------------------------------- nnmf_prob
def nnmf_prob(x, k, maxiter, eps=100 * MACH, rng=None, trace=False):
    rng = rng or np.random.default_rng(0)
    x = np.asarray(x, float)
    D, N = x.shape
    X_factor = x.sum()
    x = x / X_factor
    W = np.abs(rng.standard_normal((D, k)))
    W = W / W.sum(axis=0)[None, :]            # colSums -> 1
    H = np.abs(rng.standard_normal((k, N)))
    H = H / H.sum(axis=1)[:, None]            # rowSums -> 1
    P = np.full((k, 1), 1.0 / k)
    Pd = P[0, 0]                              # R's diag(diag(P),k,k) == P[1,1]*I
    W1, H1 = W.copy(), H.copy()
    errs = []
    for _ in range(maxiter):
        Qnorm = (W * Pd) @ H
        for j in range(k):
            Q = np.outer(W[:, j], H[j, :]) * P[j, 0] / (Qnorm + eps)
            XQ = x * Q
            d1 = XQ.sum(axis=1)
            W1[:, j] = d1 / d1.sum()
            d2 = XQ.sum(axis=0)
            H1[j, :] = d2 / d2.sum()
        W, H = W1.copy(), H1.copy()
        if trace:
            # the package's own convergence metric, verbatim
            errs.append(float(np.abs(x - W @ H).mean() / x.mean()))
    Wf = (W * np.sqrt(Pd)) * X_factor
    Hf = np.sqrt(Pd) * H
    return dict(W=Wf, H=Hf, errs=errs, mass_WH=float((W @ H).sum()),
                mass_x=float(x.sum()))


# ================================================================ checks
results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))


X = np.arange(1, 13, dtype=float).reshape(4, 3).T   # R matrix(1:12,3,4)
rng = lambda s=0: np.random.default_rng(s)

# A. mm non-negativity on non-negative input
r = nnmf_mm(X, 3, 200, rng=rng(1))
check("mm: W,H >= 0 for non-negative x",
      (r["W"] >= 0).all() and (r["H"] >= 0).all(),
      f"min W={r['W'].min():.3g} min H={r['H'].min():.3g}")

# B. mm on input containing a negative -> negatives leak through (no validation)
Xneg = X.copy()
Xneg[0, 0] = -5.0
rn = nnmf_mm(Xneg, 2, 200, rng=rng(2))
check("mm: negative input yields negative output (NO validation)",
      (rn["W"].min() < 0) or (rn["H"].min() < 0),
      f"min W={rn['W'].min():.3g} min H={rn['H'].min():.3g}")

# C. als normalises columns of W to sum 1
ra = nnmf_als(X, 2, 60, rng=rng(3))
cs = ra["W"].sum(axis=0)
check("als: colSums(W) == 1", np.allclose(cs, 1.0, atol=1e-8),
      f"colSums={np.round(cs, 10)}")

# C2. mm does NOT normalise W
check("mm: colSums(W) != 1 (no normalisation)",
      not np.allclose(r["W"].sum(axis=0), 1.0, atol=1e-3),
      f"colSums={np.round(r['W'].sum(axis=0), 4)}")

# D. mpinv Moore-Penrose conditions, full-rank and rank-deficient
A = np.array([[1.0, 2, 0], [0, 1, 1], [2, 0, 1]])
Ma = mpinv(A)
check("mpinv: full rank == solve(A)", np.allclose(Ma, np.linalg.inv(A)))
R2 = np.arange(1, 13, dtype=float).reshape(4, 3).T    # rank 2
M2 = mpinv(R2)
check("mpinv: X M X == X (rank-deficient)", np.allclose(R2 @ M2 @ R2, R2, atol=1e-8))
check("mpinv: M X M == M (rank-deficient)", np.allclose(M2 @ R2 @ M2, M2, atol=1e-8))
check("mpinv: dim transposed", mpinv(R2).shape == (R2.shape[1], R2.shape[0]),
      f"{R2.shape} -> {mpinv(R2).shape}")
check("mpinv: matches numpy pinv", np.allclose(M2, np.linalg.pinv(R2), atol=1e-8))

# E. prob reconstruction is on the ORIGINAL scale
rp = nnmf_prob(X, 2, 300, rng=rng(4))
rel = np.abs(X - rp["W"] @ rp["H"]).mean() / X.mean()
check("prob: W%*%H reconstructs x on original scale", rel < 0.25,
      f"relative error={rel:.4f}")
check("prob: W,H >= 0", (rp["W"] >= 0).all() and (rp["H"] >= 0).all())

# F. prob's own convergence metric compares mismatched masses -> cannot converge
rp3 = nnmf_prob(X, 3, 300, rng=rng(5), trace=True)
check("prob: internal errorx never reaches 1e-5 for k>1 (mass mismatch)",
      min(rp3["errs"]) > 1e-5,
      f"min errorx={min(rp3['errs']):.4f}; mass(W@H)={rp3['mass_WH']:.3f} "
      f"vs mass(x)={rp3['mass_x']:.3f} (k=3)")

# G. mm reduces reconstruction error
rt = nnmf_mm(X, 2, 300, rng=rng(6), trace=True)
check("mm: reconstruction error decreases", rt["errs"][-1] < rt["errs"][0],
      f"{rt['errs'][0]:.4g} -> {rt['errs'][-1]:.4g}")

# H. k greater than rank still runs
try:
    rk = nnmf_mm(X, 5, 100, rng=rng(7))
    check("mm: k > rank(x) runs without error", rk["W"].shape == (3, 5),
          f"W shape {rk['W'].shape}")
except Exception as e:
    check("mm: k > rank(x) runs without error", False, repr(e))

# I. all-zero input to prob -> NaN (divides by sum(x) == 0)
with np.errstate(invalid="ignore", divide="ignore"):
    try:
        rz = nnmf_prob(np.zeros((3, 4)), 2, 5, rng=rng(8))
        check("prob: all-zero x produces NaN", np.isnan(rz["W"]).any(),
              "divides by sum(x)=0")
    except Exception as e:
        check("prob: all-zero x produces NaN", False, repr(e))

# J. dimensions contract
check("dims: W is n x k and H is k x m",
      r["W"].shape == (3, 3) and r["H"].shape == (3, 4))

print(f"{'claim':<58} {'result':<6} detail")
print("-" * 110)
nfail = 0
for name, ok, detail in results:
    if not ok:
        nfail += 1
    print(f"{name:<58} {'PASS' if ok else 'FAIL':<6} {detail}")
print("-" * 110)
print(f"{len(results) - nfail}/{len(results)} claims verified")
