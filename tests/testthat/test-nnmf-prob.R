# nnmf_prob() -- the multinomial / probabilistic variant.
#
# This algorithm works on a normalised copy of the input (x / sum(x)), keeps W
# with unit column sums and H with unit row sums throughout, and rescales both
# at the end so the returned factors reconstruct the original matrix. That
# round trip is the main thing worth testing, and it is the one the
# documentation says nothing about.

test_that("nnmf_prob returns a valid non-negative factorisation", {
  set.seed(1)
  z <- quiet(nnmf_prob(X_SMALL, 2, 200))
  expect_valid_factorisation(z, X_SMALL, k = 2, tol = 1e-12)
})

test_that("the returned factors reconstruct the ORIGINAL, unnormalised input", {
  # The algorithm divides x by sum(x) on the way in and multiplies W by
  # sum(x) on the way out. Worth asserting explicitly: it would be easy for a
  # future change to leave the result on the normalised scale, which would be
  # a silent factor-of-sum(x) error in anything consuming it.
  set.seed(61)
  z <- quiet(nnmf_prob(X_SMALL, 2, 400))
  expect_lt(rel_error(X_SMALL, z$W, z$H), 0.25)

  # Total mass is preserved, not just the shape. Stated as an explicit
  # relative comparison rather than via expect_equal(tolerance = ), whose
  # tolerance is relative in testthat 3e and easy to misread.
  mass_gap <- abs(sum(z$W %*% z$H) - sum(X_SMALL)) / sum(X_SMALL)
  expect_lt(mass_gap, 0.25)
})

test_that("reconstruction scale holds for an input whose total is far from 1", {
  # Multiply the input by 1000 and the reconstruction must follow. This is the
  # sharpest available check that the normalise/denormalise round trip works.
  X_big <- X_SMALL * 1000
  set.seed(62)
  z <- quiet(nnmf_prob(X_big, 2, 400))
  expect_lt(rel_error(X_big, z$W, z$H), 0.25)
  expect_gt(sum(z$W %*% z$H), 0.5 * sum(X_big))
})

test_that("nnmf_prob fits a known rank-2 structure", {
  set.seed(63)
  Wtrue <- matrix(abs(rnorm(6 * 2)) + 0.3, 6, 2)
  Htrue <- matrix(abs(rnorm(2 * 5)) + 0.3, 2, 5)
  Xtrue <- Wtrue %*% Htrue

  set.seed(64)
  z <- quiet(nnmf_prob(Xtrue, 2, 500))
  expect_lt(rel_error(Xtrue, z$W, z$H), 0.10)
})

test_that("nnmf_prob accepts maxiter = 1", {
  # Guards a latent division by zero. The function computes
  #   powers = 1.5 + (2.5 - 1.5) * ((1:maxiter) - 1) / (maxiter - 1)
  # which is 0/0 = NaN when maxiter is 1. It happens to be harmless because
  # `powers` is dead code -- computed once and never read. If a later version
  # starts using it, this test is where that shows up.
  set.seed(65)
  err <- error_message_of(nnmf_prob(X_SMALL, 2, 1))
  expect_null(err)

  set.seed(65)
  z <- quiet(nnmf_prob(X_SMALL, 2, 1))
  expect_true(all(is.finite(z$W)))
  expect_true(all(is.finite(z$H)))
})

test_that("nnmf_prob handles k = 1", {
  set.seed(66)
  z <- quiet(nnmf_prob(X_SMALL, 1, 200))
  expect_valid_factorisation(z, X_SMALL, k = 1, tol = 1e-12)
  # With one factor the rank-1 fit is the best possible, so expect it to be
  # no better than the rank-2 fit.
  set.seed(67)
  z2 <- quiet(nnmf_prob(X_SMALL, 2, 200))
  expect_gte(rel_error(X_SMALL, z$W, z$H),
             rel_error(X_SMALL, z2$W, z2$H) - 1e-6)
})

test_that("nnmf_prob produces NaN for an all-zero input", {
  # CHARACTERISATION of an unguarded edge case. The first thing the function
  # does is x <- x / sum(x); for an all-zero matrix that is 0/0, and NaN then
  # propagates into both returned factors. There is no error and no warning,
  # so a caller that can ever be handed an empty or all-zero slice -- an
  # over-aggressive filter in a Shiny app, say -- gets NaN rather than a
  # failure it can catch. Check sum(x) > 0 before calling.
  Z <- matrix(0, 3, 4)
  set.seed(68)
  err <- error_message_of(nnmf_prob(Z, 2, 10))
  expect_null(err)                       # it does not raise

  set.seed(68)
  z <- quiet(nnmf_prob(Z, 2, 10))
  expect_true(any(is.nan(z$W)) || any(is.na(z$W)))
})
