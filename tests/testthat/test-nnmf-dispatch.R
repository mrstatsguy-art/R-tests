# nnmf() -- the user-facing wrapper that dispatches to one of three algorithms.
#
# The wrapper is where the package's API contract lives: the method strings it
# accepts, the defaults it supplies, and the return shape. Two of these tests
# document behaviour you would not guess from the documentation.

test_that("nnmf returns list(W, H) with the documented dimensions", {
  set.seed(1)
  z <- quiet(nnmf(X_SMALL, 3))
  expect_valid_factorisation(z, X_SMALL, k = 3)
})

test_that("nnmf accepts all three documented methods", {
  for (m in c("nnmf_mm", "nnmf_als", "nnmf_prob")) {
    set.seed(1)
    z <- quiet(nnmf(X_SMALL, 2, m, maxiter = 60))
    expect_valid_factorisation(z, X_SMALL, k = 2, tol = 1e-8)
  }
})

test_that("nnmf defaults to the multiplicative update algorithm", {
  set.seed(1)
  out <- utils::capture.output(z_default <- nnmf(X_SMALL, 2, maxiter = 50))
  expect_match(paste(out, collapse = " "), "Multiplicative Update")

  # Same seed, explicit method: identical result proves the same code path.
  set.seed(1)
  z_explicit <- quiet(nnmf(X_SMALL, 2, "nnmf_mm", maxiter = 50))
  expect_equal(z_default$W, z_explicit$W)
  expect_equal(z_default$H, z_explicit$H)
})

test_that("nnmf silently falls back to nnmf_mm for an unrecognised method", {
  # CHARACTERISATION of a real API wart. The dispatch is an if/else-if/else
  # chain whose final `else` runs the multiplicative update, so a typo in the
  # method name does not raise an error -- it quietly runs a different
  # algorithm from the one you asked for. If you pass a method name through
  # from configuration or user input, validate it yourself first.
  set.seed(1)
  out <- utils::capture.output(z_typo <- nnmf(X_SMALL, 2, "nnmf_als_TYPO", 50, EPS))
  expect_match(paste(out, collapse = " "), "Multiplicative Update")

  set.seed(1)
  z_mm <- quiet(nnmf(X_SMALL, 2, "nnmf_mm", 50, EPS))
  expect_equal(z_typo$W, z_mm$W)   # identical, so it really was nnmf_mm
})

test_that("nnmf announces which algorithm it ran", {
  set.seed(1)
  expect_output(nnmf(X_SMALL, 2, "nnmf_als", 50, EPS), "Alternating Least Squares")
  set.seed(1)
  expect_output(nnmf(X_SMALL, 2, "nnmf_prob", 50, EPS), "Multinomial")
})

test_that("results are reproducible under a fixed seed and vary without one", {
  # The algorithms take no seed argument, so set.seed() is the only control
  # over their random initialisation. Anything calling NMFN in a pipeline whose
  # output must be stable has to set a seed itself.
  set.seed(123)
  a <- quiet(nnmf(X_SMALL, 2, "nnmf_mm", 50, EPS))
  set.seed(123)
  b <- quiet(nnmf(X_SMALL, 2, "nnmf_mm", 50, EPS))
  expect_identical(a$W, b$W)
  expect_identical(a$H, b$H)

  set.seed(456)
  c2 <- quiet(nnmf(X_SMALL, 2, "nnmf_mm", 50, EPS))
  expect_false(isTRUE(all.equal(a$W, c2$W)))
})

test_that("nnmf passes its own eps default down to nnmf_prob", {
  # CHARACTERISATION. nnmf_prob's own default is 100 * .Machine$double.eps
  # (about 2.2e-14), but nnmf() always passes its own default of 2.2204e-16 --
  # roughly 100x smaller. Calling nnmf(x, k, "nnmf_prob") and calling
  # nnmf_prob(x, k, maxiter) directly therefore use different eps values.
  set.seed(9)
  via_wrapper <- quiet(nnmf(X_SMALL, 2, "nnmf_prob", 50))
  set.seed(9)
  direct_same_eps <- quiet(nnmf_prob(X_SMALL, 2, 50, 2.2204e-16))
  expect_equal(via_wrapper$W, direct_same_eps$W)

  set.seed(9)
  direct_own_default <- quiet(nnmf_prob(X_SMALL, 2, 50))
  # Both are valid factorisations; they are simply not the same computation.
  expect_valid_factorisation(direct_own_default, X_SMALL, k = 2, tol = 1e-8)
})
