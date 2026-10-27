# Unit tests for NMFN 2.0.1

A `testthat` suite for the CRAN package [NMFN](https://cran.r-project.org/package=NMFN)
(Non-Negative Matrix Factorization, v2.0.1, Suhai Liu, GPL). 51 tests across
six files.

```bash
Rscript run-tests.R
```

Needs `testthat` and `NMFN` installed. Exits non-zero on failure, so it drops
straight into a CI step.

---

## Read this first: these tests have not been executed

I wrote them against the package source, which I read in full, and I verified
every mathematical claim numerically — but by porting the three algorithms to
Python/numpy, because the environment I worked in has no R and no way to
install it. So:

- **The assertions are verified.** Each invariant below was checked against a
  faithful port before being written as a test.
- **The R syntax is not.** Delimiters balance and the idioms are standard, but
  nothing has been through an R parser. Budget one pass for typos.

Run it once and tell me what breaks. The likeliest spots are the error-message
regexps in `test-input-validation.R`, which depend on R's exact phrasing.

---

## Why there are no exact expected values

NMF has no unique solution. If `x ≈ W H`, then for any positive diagonal `D`,
`(W D)(D⁻¹ H)` fits identically — so `W` and `H` are determined only up to
scaling and column permutation. On top of that all three algorithms initialise
from `rnorm()` and expose no seed argument, so every call differs.

A test asserting `expect_equal(z$W[1,1], 0.4213)` would therefore be pinning
one arbitrary point from a continuum of correct answers, and would break on any
R version whose RNG stream differs. Every test here asserts an **invariant**
instead: non-negativity, dimensions, reconstruction quality, the Moore-Penrose
conditions, scale conventions.

Each test sets a seed, because without one the suite is flaky by construction.

---

## What the suite covers

| File | Tests | Subject |
|---|---:|---|
| `test-mpinv.R` | 9 | Moore-Penrose pseudoinverse. The one function with a single right answer, so the one place exact assertions belong |
| `test-distance2.R` | 6 | The squared-error metric the package optimises |
| `test-nnmf-dispatch.R` | 7 | The `nnmf()` wrapper: method dispatch, defaults, reproducibility |
| `test-nnmf-mm.R` | 9 | Multiplicative updates (Lee & Seung) |
| `test-nnmf-als.R` | 7 | Alternating least squares, including the factor collapse below |
| `test-input-validation.R` | 6 | Input handling, and the gaps in it |

`helper-nmfn.R` holds the fixtures and a `quiet()` wrapper, since every
function in the package prints to stdout.

---

## Findings worth your attention

These came out of reading the source and checking behaviour. Four affect
anyone calling NMFN from application code.

### 1. ALS can silently return fewer factors than you asked for

The most practically important one. `nnmf_als` clips `W` at zero each
iteration (`W <- (W > 0) * W`) and then divides each column by its own sum.
When a column clips to all-zeros, that division is `0 / (0 + eps) = 0`, and the
column stays dead for every remaining iteration.

So `colSums(W)` values are drawn from `{0, 1}` — never anything between. The
returned `W` still has `k` columns; some may be entirely zero.

How often, from the numpy port on the package's own `matrix(1:12, 3, 4)`
example (rank 2), 25 random starts each:

| k requested | dead factors |
|---:|---:|
| 1 | 0% |
| 2 | 14% |
| 3 | 47% |

On a dense, full-rank, strictly positive 12×8 matrix with k=4: 0%. The pattern
is coherent — collapse happens when you ask for more factors than the data
supports. The rates come from numpy's RNG, so R's will differ; the mechanism
and the `{0, 1}` invariant will not.

**If you depend on getting `k` usable components** — a legend with `k` entries,
a loop over `k` clusters, `k` series on a plot — check:

```r
live <- sum(colSums(z$W) > 1e-10)
```

Note that asserting `all(colSums(W) == 1)` looks correct and is wrong; it fails
whenever a factor dies.

### 2. Nothing validates that the input is non-negative

This is Non-negative Matrix Factorization. A negative entry violates its
premise. None of the three algorithms check for one.

`nnmf_mm` returns without error or warning. Because its updates are
multiplicative it cannot represent a negative value at all: the offending cell
comes back as approximately zero — wrong sign — and the fit around it degrades
(relative error 0.066 vs 0.005 on the same call with the negative removed).
Nothing in the output marks the result as untrustworthy.

### 3. Non-finite input is handled inconsistently

`nnmf_mm` and `nnmf_als` both open with
`if (any(!is.finite(x))) stop("infinite or missing values in 'x'")`.
`nnmf_prob` has no equivalent, and returns `NaN` factors instead. The same bad
matrix raises a clean error through two methods and fails silently through the
third — decided by a string argument.

Relatedly, `nnmf_prob` divides by `sum(x)` unguarded, so an all-zero input
yields `NaN` with no error. An over-aggressive filter upstream in a Shiny app
is enough to hit this.

### 4. An unrecognised method name runs a different algorithm

`nnmf()`'s dispatch is an `if` / `else if` / `else` chain whose final `else`
runs the multiplicative update. So `nnmf(x, k, "nnmf_als_typo")` does not
error — it quietly runs `nnmf_mm`. The test proves it by showing identical
results under a fixed seed.

If a method name reaches `nnmf()` from config or user input, validate it first.

### Smaller notes

- **`nnmf()` overrides `nnmf_prob`'s own `eps` default.** `nnmf_prob` defaults
  `eps = 100 * .Machine$double.eps` (≈2.2e-14), but `nnmf()` always passes
  `2.2204e-16` — about 100× smaller. The wrapper and a direct call are
  different computations.
- **The convergence check only runs every 50 iterations.** It sits inside
  `if (iter %% 50 == 0)`, so with `maxiter < 50` it never runs: the algorithm
  always performs every iteration and can never stop early. With the default
  `maxiter = 1000` you get 20 lines of progress output per call — worth knowing
  before calling this inside a reactive.
- **`nnmf_prob` computes a `powers` vector it never uses.** Dead code, and
  `0/0 = NaN` when `maxiter == 1`. Harmless today because nothing reads it;
  there is a test guarding that.
- **`nnmf_mm` and `nnmf_als` have no defaults for `maxiter` or `eps`**, so
  calling them directly with two arguments fails. Only `nnmf()` supplies them.
- **`distance2` inherits R's recycling.** A length-3 vector against a 3×4
  matrix returns a number rather than an error.

`test-input-validation.R` includes a tested `safe_nnmf()` wrapper closing gaps
2, 3 and 4 — about fifteen lines, worth putting in front of the package.

---

## Contract tests vs characterisation tests

Deliberately separated, and the comments say which is which.

**Contract tests** assert what you want to be true and would want to know
about breaking: non-negativity, dimensions, reconstruction quality, the
Moore-Penrose conditions, reproducibility under a seed.

**Characterisation tests** pin down what the package does *today*, including
the behaviour above that is arguably wrong. Their value is as a tripwire: NMFN
was last published in 2022 and is not actively developed, so if a future
version changes any of it, these tests tell you — rather than your app
discovering it in production.

If you upgrade NMFN and a characterisation test fails, that is the test doing
its job. Read it, decide whether the new behaviour is better, update it.

---

## Using this in CI

Drop `tests/` into your project and add a step:

```yaml
      - name: Test NMFN behaviour
        run: Rscript -e 'testthat::test_dir("tests/testthat", stop_on_failure = TRUE)'
```

`stop_on_failure = TRUE` matters. Without it `testthat` prints failures and
exits successfully, so the pipeline reports green while the tests are red.

The suite runs in a few seconds; the slowest tests use `maxiter = 2000` on a
6×5 matrix.
