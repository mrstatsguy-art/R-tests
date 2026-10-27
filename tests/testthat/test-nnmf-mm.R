# nnmf_mm() -- multiplicative update rules (Lee & Seung, 2001).
#
# The defining property of this algorithm is that its update rules are
# guaranteed not to increase the Euclidean reconstruction error. That is the
# strongest thing worth asserting about any NMF implementation, and it is what
# the first two tests check.

test_that("nnmf_mm returns a valid non-negative factorisation", {
  set.seed(1)
  z <- quiet(nnmf_mm(X_SMALL, 2, 200, EPS))
  expect_valid_factorisation(z, X_SMALL, k = 2)
})

test_that("nnmf_mm reduces reconstruction error relative to its starting point", {
  # Reproduce the random initialisation the algorithm would have used, measure
  # the error there, then confirm the fitted factors do better.
  set.seed(31)
  W0 <- matrix(abs(rnorm(3 * 2)), 3, 2)
  H0 <- matrix(abs(rnorm(2 * 4)), 2, 4)
  start_error <- distance2(X_SMALL, W0 %*% H0)

  set.seed(31)
  z <- quiet(nnmf_mm(X_SMALL, 2, 300, EPS))
  end_error <- distance2(X_SMALL, z$W %*% z$H)

  expect_lt(end_error, start_error)
})

test_that("more iterations do not make the fit worse", {
  # Lee & Seung's monotonicity result. Not a strict inequality: once the
  # algorithm has converged, extra iterations change nothing, and floating
  # point can nudge the error up by an amount far below any real change.
  set.seed(32)
  short <- quiet(nnmf_mm(X_SMALL, 2, 50, EPS))
  set.seed(32)
  long <- quiet(nnmf_mm(X_SMALL, 2, 500, EPS))

  e_short <- distance2(X_SMALL, short$W %*% short$H)
  e_long <- distance2(X_SMALL, long$W %*% long$H)
  expect_lte(e_long, e_short + 1e-9)
})

test_that("nnmf_mm recovers a known rank-2 structure closely", {
  # A matrix built as the product of a non-negative 6x2 and a 2x5 has an exact
  # rank-2 non-negative factorisation. The algorithm should get close to it.
  set.seed(33)
  Wtrue <- matrix(abs(rnorm(6 * 2)) + 0.2, 6, 2)
  Htrue <- matrix(abs(rnorm(2 * 5)) + 0.2, 2, 5)
  Xtrue <- Wtrue %*% Htrue

  set.seed(34)
  z <- quiet(nnmf_mm(Xtrue, 2, 2000, EPS))
  expect_lt(rel_error(Xtrue, z$W, z$H), 0.05)
})

test_that("a larger k fits at least as well", {
  # More factors cannot fit worse, give or take optimisation noise.
  D <- make_dense(10, 6)
  set.seed(35)
  z1 <- quiet(nnmf_mm(D, 1, 400, EPS))
  set.seed(35)
  z3 <- quiet(nnmf_mm(D, 3, 400, EPS))
  expect_lte(distance2(D, z3$W %*% z3$H),
             distance2(D, z1$W %*% z1$H))
})

test_that("nnmf_mm does not normalise W or H", {
  # Unlike nnmf_als and nnmf_prob, this algorithm imposes no scale convention:
  # W and H absorb scale freely, so do not rely on column or row sums here.
  # NMF is only identified up to a diagonal rescaling W D, D^-1 H anyway.
  set.seed(36)
  z <- quiet(nnmf_mm(make_dense(8, 5), 3, 300, EPS))
  expect_false(isTRUE(all.equal(colSums(z$W), rep(1, 3), tolerance = 1e-3)))
})

test_that("nnmf_mm accepts k greater than the rank of the input", {
  # X_SMALL has rank 2. Asking for 5 factors is overparameterised but must not
  # error, and must still return the requested shape.
  set.seed(37)
  z <- quiet(nnmf_mm(X_SMALL, 5, 100, EPS))
  expect_valid_factorisation(z, X_SMALL, k = 5)
})

test_that("nnmf_mm handles a single factor and a single-column input", {
  set.seed(38)
  expect_valid_factorisation(quiet(nnmf_mm(X_SMALL, 1, 100, EPS)), X_SMALL, k = 1)

  one_col <- matrix(c(1, 2, 3), 3, 1)
  set.seed(39)
  expect_valid_factorisation(quiet(nnmf_mm(one_col, 1, 100, EPS)), one_col, k = 1)
})

test_that("progress output appears only every 50 iterations", {
  # CHARACTERISATION with a practical consequence. The internal convergence
  # check sits inside `if (iter %% 50 == 0)`, so with maxiter < 50 it never
  # runs: the algorithm always performs every iteration and can neither stop
  # early nor report. With the default maxiter = 1000 you get 20 lines of
  # output per call, which is worth knowing before calling this in a loop or
  # inside a Shiny reactive.
  set.seed(40)
  out_short <- utils::capture.output(nnmf_mm(X_SMALL, 2, 10, EPS))
  expect_length(grep("Iter =", out_short), 0)

  set.seed(40)
  out_long <- utils::capture.output(nnmf_mm(X_SMALL, 2, 150, EPS))
  expect_gt(length(grep("Iter =", out_long)), 0)
})
