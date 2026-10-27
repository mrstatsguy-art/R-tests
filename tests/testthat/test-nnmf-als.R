# nnmf_als() -- alternating least squares, after Lars Kai Hansen's Matlab
# implementation.
#
# Each iteration solves for W by least squares, clips it at zero, normalises
# its columns, then solves for H and clips that. The clipping is what makes it
# a non-negative method, and it is also the source of the factor collapse
# documented below -- the most practically important behaviour in this file.

test_that("nnmf_als returns a valid non-negative factorisation", {
  set.seed(1)
  z <- quiet(nnmf_als(X_SMALL, 2, 60, EPS))
  # tol: H is produced by `H * (H > 0)`, which can leave values at exactly 0
  # but, with denormal arithmetic, occasionally a magnitude near 1e-50.
  expect_valid_factorisation(z, X_SMALL, k = 2, tol = 1e-8)
})

test_that("each column of W sums to 1, or to 0 if that factor died", {
  # This is the real invariant, and it is narrower than it first appears.
  #
  # The implementation divides each column of W by its own column sum, so a
  # live column sums to 1. But the preceding `W <- (W > 0) * W` can zero a
  # column outright, and then the division is 0 / (0 + eps) = 0, which leaves
  # that column dead for every remaining iteration. So column sums are drawn
  # from {0, 1} and nothing in between.
  #
  # Asserting that all columns sum to 1 looks right and is wrong; it fails
  # whenever a factor collapses.
  for (seed in c(1, 7, 42, 101, 2024)) {
    set.seed(seed)
    cs <- colSums(quiet(nnmf_als(X_SMALL, 2, 60, EPS))$W)
    expect_true(
      all(abs(cs - 1) < 1e-8 | abs(cs) < 1e-10),
      info = paste0("seed ", seed, ": colSums = ",
                    paste(signif(cs, 8), collapse = ", "))
    )
  }
})

test_that("nnmf_als can return fewer live factors than requested", {
  # CHARACTERISATION, and the thing to guard against in calling code. On a
  # rank-deficient input, asking for more factors than the data supports makes
  # ALS silently kill the surplus: the returned W still has k columns, but some
  # are entirely zero. Downstream code that assumes k usable components -- a
  # legend with k entries, a loop over k clusters, a plot with k series -- will
  # be working with fewer.
  #
  # The count is not asserted, because it depends on the random start. The
  # test asserts the shape contract holds regardless, and reports what it saw.
  set.seed(7)
  z <- quiet(nnmf_als(X_SMALL, 3, 60, EPS))   # rank-2 input, k = 3

  expect_equal(ncol(z$W), 3)                  # shape is always honoured
  live <- sum(colSums(z$W) > 1e-10)
  expect_gte(live, 1)
  expect_lte(live, 3)
  message("  nnmf_als: ", live, " of 3 requested factors are live on a rank-2 input")
})

test_that("a count of live factors is a usable check in calling code", {
  # If you depend on getting k usable factors, this is the check to make.
  live_factors <- function(z) sum(colSums(z$W) > 1e-10)

  set.seed(51)
  dense <- make_dense(12, 8)
  z <- quiet(nnmf_als(dense, 4, 60, EPS))

  # On a dense, full-rank, strictly positive input, collapse is rare -- but it
  # depends on the random start, so this tolerates one death rather than
  # demanding four survivors. A test that fails occasionally for a legitimate
  # reason is worse than a slightly looser bound.
  expect_gte(live_factors(z), 3)
  message("  nnmf_als: ", live_factors(z), " of 4 factors live on a dense input")
})

test_that("nnmf_als fits a known rank-2 structure", {
  set.seed(52)
  Wtrue <- matrix(abs(rnorm(6 * 2)) + 0.3, 6, 2)
  Htrue <- matrix(abs(rnorm(2 * 5)) + 0.3, 2, 5)
  Xtrue <- Wtrue %*% Htrue

  set.seed(53)
  z <- quiet(nnmf_als(Xtrue, 2, 200, EPS))
  expect_lt(rel_error(Xtrue, z$W, z$H), 0.05)
})

test_that("nnmf_als reduces reconstruction error relative to its start", {
  D <- make_dense(10, 6)
  set.seed(54)
  W0 <- matrix(abs(rnorm(10 * 3)), 10, 3)
  H0 <- matrix(abs(rnorm(3 * 6)), 3, 6)
  sqrnorm <- sqrt(sum(W0 %*% H0) / sum(D))
  start_error <- distance2(D, (W0 / sqrnorm) %*% (H0 / sqrnorm))

  set.seed(54)
  z <- quiet(nnmf_als(D, 3, 200, EPS))
  expect_lt(distance2(D, z$W %*% z$H), start_error)
})

test_that("nnmf_als handles k = 1", {
  set.seed(55)
  z <- quiet(nnmf_als(X_SMALL, 1, 60, EPS))
  expect_valid_factorisation(z, X_SMALL, k = 1, tol = 1e-8)
  expect_equal(sum(colSums(z$W)), 1, tolerance = 1e-8)
})
