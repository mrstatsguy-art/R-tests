"""Second pass: characterise the ALS factor collapse and tighten two claims
that the first pass showed were stated wrongly.
"""
import numpy as np
import sys
sys.path.insert(0, "/tmp/claude-0/-home-claude/040684f3-c5b0-5fb4-bd5f-1cfc3911b3cc/scratchpad")
from verify_claims import nnmf_als, nnmf_mm, nnmf_prob, distance2, X

rng = lambda s: np.random.default_rng(s)

# ---- 1. ALS: are column sums always in {0, 1}?  How often does a factor die?
print("ALS column sums across seeds (matrix(1:12,3,4)):")
rows = []
for k in (1, 2, 3):
    dead_total = 0
    for seed in range(25):
        cs = nnmf_als(X, k, 60, rng=rng(seed))["W"].sum(axis=0)
        in_set = np.all([np.isclose(c, 0, atol=1e-10) or np.isclose(c, 1, atol=1e-8)
                         for c in cs])
        dead = int(sum(np.isclose(c, 0, atol=1e-10) for c in cs))
        dead_total += dead
        if not in_set:
            rows.append((k, seed, np.round(cs, 10), "NOT in {0,1}"))
    print(f"  k={k}: {dead_total} dead factors across 25 seeds x {k} cols "
          f"= {dead_total}/{25*k} ({100*dead_total/(25*k):.0f}%)")
print(f"  violations of the 0-or-1 rule: {len(rows)}")
for r in rows[:5]:
    print("   ", r)

# A bigger, denser matrix - does collapse still happen?
B = np.abs(np.random.default_rng(99).standard_normal((12, 8))) + 0.5
dead_b = 0
for seed in range(15):
    cs = nnmf_als(B, 4, 60, rng=rng(seed))["W"].sum(axis=0)
    dead_b += int(sum(np.isclose(c, 0, atol=1e-10) for c in cs))
print(f"  dense 12x8, k=4: {dead_b}/{15*4} dead factors "
      f"({100*dead_b/(15*4):.0f}%)")

# ---- 2. negative input: does mm raise?  is reconstruction bad?
print("\nNegative input to mm (no validation in the package):")
Xneg = X.copy(); Xneg[0, 0] = -5.0
try:
    rn = nnmf_mm(Xneg, 2, 200, rng=rng(2))
    rel = np.abs(Xneg - rn["W"] @ rn["H"]).mean() / np.abs(Xneg).mean()
    good = np.abs(X - nnmf_mm(X, 2, 200, rng=rng(2))["W"]
                  @ nnmf_mm(X, 2, 200, rng=rng(2))["H"]).mean() / X.mean()
    print(f"  raised no error: True")
    print(f"  min(W)={rn['W'].min():.3g}  min(H)={rn['H'].min():.3g}"
          f"   <- not meaningfully negative, so DON'T assert that")
    print(f"  relative reconstruction error on negative input = {rel:.3f}")
    print(f"  (same call on non-negative input          = {good:.3f})")
    print(f"  reconstruction is much worse: {rel > 5 * good}")
    # does the negative cell get reconstructed anywhere near correctly?
    recon = (rn["W"] @ rn["H"])[0, 0]
    print(f"  x[0,0] = {Xneg[0,0]}  reconstructed as {recon:.4f} "
          f"-> sign is wrong: {recon >= 0}")
except Exception as e:
    print("  RAISED:", repr(e))

# ---- 3. non-finite input: mm/als guard, prob does not
print("\nNon-finite input:")
Xna = X.copy(); Xna[1, 1] = np.nan
print("  mm/als contain `if (any(!is.finite(x))) stop(...)` -> error, by source")
print("  prob has NO such guard; result:")
try:
    with np.errstate(invalid="ignore", divide="ignore"):
        rp = nnmf_prob(Xna, 2, 10, rng=rng(3))
    print(f"    no error raised; W has NaN: {np.isnan(rp['W']).any()}, "
          f"H has NaN: {np.isnan(rp['H']).any()}")
except Exception as e:
    print("    RAISED:", repr(e))

# ---- 4. reproducibility: same seed -> identical result
print("\nReproducibility (same RNG state):")
a = nnmf_mm(X, 2, 50, rng=rng(11))
b = nnmf_mm(X, 2, 50, rng=rng(11))
c = nnmf_mm(X, 2, 50, rng=rng(12))
print(f"  same seed identical: {np.array_equal(a['W'], b['W'])}")
print(f"  different seed differs: {not np.allclose(a['W'], c['W'])}")

# ---- 5. mm: is error decrease monotone, or just net-downward?
rt = nnmf_mm(X, 2, 150, rng=rng(6), trace=True)
e = np.array(rt["errs"])
incr = int((np.diff(e) > 1e-12).sum())
print(f"\nmm error trace: start={e[0]:.4g} end={e[-1]:.4g}  "
      f"strictly-increasing steps={incr}/{len(e)-1}")
print(f"  monotone non-increasing: {incr == 0}  <- assert net decrease, not monotone"
      if incr else "  monotone non-increasing: True")

# ---- 6. als reconstruction quality vs mm, same k
for k in (1, 2):
    em = distance2(X, (lambda r: r["W"] @ r["H"])(nnmf_mm(X, k, 300, rng=rng(21))))
    ea = distance2(X, (lambda r: r["W"] @ r["H"])(nnmf_als(X, k, 300, rng=rng(21))))
    print(f"\nk={k}: sq.error mm={em:.4g}  als={ea:.4g}")
