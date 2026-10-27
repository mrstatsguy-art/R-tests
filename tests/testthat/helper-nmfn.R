# Shared fixtures and helpers for the NMFN test suite.
#
# Two things about NMFN shape every test in this suite:
#
# 1. All three algorithms initialise W and H with `rnorm()` and take no seed
#    argument. Every call is therefore random. Each test sets a seed, and no
#    test asserts an exact factorisation value.
#
# 2. All three algorithms print to stdout via `cat()` -- the algorithm name
#    from `nnmf()`, plus a progress line every 50 iterations. `quiet()` below
#    swallows that so test output stays readable.

library(testthat)
library(NMFN)

# The package's own documented example: 3x4, and rank 2 (every column is the
# previous one plus 3). Rank 2 matters -- asking for k = 3 asks for more
# factors than the data supports, which is what makes the ALS collapse in
# test-nnmf_als.R visible.
X_SMALL <- matrix(1:12, 3, 4)

# A denser, full-rank, strictly positive matrix. Used where the degeneracy of
# X_SMALL would obscure the behaviour under test.
make_dense <- function(n = 12, m = 8, seed = 99) {
  set.seed(seed)
  matrix(abs(rnorm(n * m)) + 0.5, n, m)
}

# The default `eps` that nnmf() passes down. nnmf_mm() and nnmf_als() have no
# default of their own, so it must be supplied when calling them directly.
EPS <- 2.2204e-16

# Evaluate `code`, discard anything it prints, return its value.
quiet <- function(code) {
  utils::capture.output(res <- code)
  res
}

# Run `code` and return the error message, or NULL if it completed. Used to
# assert that a call does NOT error, without depending on
# testthat::expect_no_error(), which needs testthat >= 3.1.2.
error_message_of <- function(code) {
  tryCatch({
    utils::capture.output(code)
    NULL
  }, error = function(e) conditionMessage(e))
}

# Relative reconstruction error, the same quantity NMFN prints internally as
# `relative error`. Scale-free, so it is comparable across matrices.
rel_error <- function(x, W, H) {
  mean(abs(x - W %*% H)) / mean(abs(x))
}

# Every factorisation in this package returns list(W = ..., H = ...) with
# W being n x k and H being k x m. Asserted in one place.
expect_valid_factorisation <- function(z, x, k, tol = 0) {
  expect_type(z, "list")
  expect_named(z, c("W", "H"))
  expect_true(is.matrix(z$W))
  expect_true(is.matrix(z$H))
  expect_equal(dim(z$W), c(nrow(x), k))
  expect_equal(dim(z$H), c(k, ncol(x)))
  expect_true(all(is.finite(z$W)), info = "W contains non-finite values")
  expect_true(all(is.finite(z$H)), info = "H contains non-finite values")
  expect_true(all(z$W >= -tol), info = paste("min(W) =", min(z$W)))
  expect_true(all(z$H >= -tol), info = paste("min(H) =", min(z$H)))
  invisible(z)
}
