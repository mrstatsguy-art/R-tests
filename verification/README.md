# How the assertions were verified

No R was available in the environment these tests were written in, so each
mathematical claim was checked against a faithful numpy port of NMFN 2.0.1
before being written as an R test.

    python3 verify_claims.py    # 16 claims
    python3 verify2.py          # characterises the ALS factor collapse

`verify_claims.py` found one claim stated wrongly: `colSums(W) == 1` for
nnmf_als. The real invariant is `colSums(W) ∈ {0, 1}`, because a factor can
collapse to zero. `verify2.py` then measured how often that happens.

These scripts are not part of the test suite and are not needed to run it.
They are here so the reasoning behind the assertions is checkable, and so the
port can be re-run if NMFN ever changes.

Caveat: the port uses numpy's RNG, not R's. Algebraic invariants carry over
exactly; observed frequencies (such as the collapse rates) do not.
