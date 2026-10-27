# Input handling across the three algorithms.
#
# These tests are deliberately grouped, because the interesting finding is the
# INCONSISTENCY between them: nnmf_mm and nnmf_als guard against non-finite
# input, nnmf_prob does not, and none of the three checks that the input is
# non-negative despite that being the entire premise of the method.
#
# If your code passes user- or query-derived matrices into NMFN, these are the
# checks you have to perform yourself.

test_that("nnmf_mm and nnmf_als reject non-finite input", {
  X_na <- X_SMALL
  X_na[1, 1] <- NA
  X_inf <- X_SMALL
  X_inf[2, 2] <- Inf

  expect_error(nnmf_mm(X_na, 2, 50, EPS), "infinite or missing")
  expect_error(nnmf_als(X_na, 2, 50, EPS), "infinite or missing")
  expect_error(nnmf_mm(X_inf, 2, 50, EPS), "infinite or missing")
  expect_error(nnmf_als(X_inf, 2, 50, EPS), "infinite or missing")
})

test_that("nnmf_prob does NOT reject non-finite input", {
  # CHARACTERISATION of an inconsistency. nnmf_mm and nnmf_als both open with
  #   if (any(!is.finite(x))) stop("infinite or missing values in 'x'")
  # and nnmf_prob has no equivalent. So the same bad matrix raises a clean
  # error through two methods and silently returns NaN through the third --
  # and which one you get depends on a string argument.
  X_na <- X_SMALL
  X_na[1, 1] <- NA

  set.seed(71)
  err <- error_message_of(nnmf_prob(X_na, 2, 20))
  expect_null(err)

  set.seed(71)
  z <- quiet(nnmf_prob(X_na, 2, 20))
  expect_true(any(is.na(z$W)) || any(is.na(z$H)))
})

test_that("no algorithm validates that the input is non-negative", {
  # CHARACTERISATION, and the one most likely to bite. This is Non-negative
  # Matrix Factorisation; a negative entry violates its central assumption.
  # None of the three algorithms check. nnmf_mm returns without complaint, and
  # because its update rules are multiplicative it cannot represent a negative
  # value at all -- the offending cell comes back as approximately zero, with
  # the wrong sign, and the fit around it is degraded.
  #
  # Nothing in the output marks the result as untrustworthy. Validate with
  # min(x) >= 0 before calling.
  X_neg <- X_SMALL
  X_neg[1, 1] <- -5

  set.seed(72)
  err <- error_message_of(nnmf_mm(X_neg, 2, 200, EPS))
  expect_null(err)                                  # no error, no warning

  set.seed(72)
  z <- quiet(nnmf_mm(X_neg, 2, 200, EPS))
  recon <- (z$W %*% z$H)[1, 1]
  expect_gte(recon, -1e-6)                          # cannot go negative
  expect_gt(abs(recon - X_neg[1, 1]), 1)            # so it is simply wrong
})

test_that("a non-negativity guard is cheap to add in calling code", {
  # The wrapper worth having in front of NMFN. Supplied here as a worked
  # example, and tested, so the suite documents the fix as well as the gap.
  safe_nnmf <- function(x, k, method = "nnmf_mm", maxiter = 1000) {
    if (!is.matrix(x)) x <- as.matrix(x)
    if (!is.numeric(x)) stop("`x` must be numeric", call. = FALSE)
    if (any(!is.finite(x))) stop("`x` has missing or infinite values", call. = FALSE)
    if (any(x < 0)) stop("`x` has negative values; NMF requires x >= 0", call. = FALSE)
    if (sum(x) <= 0) stop("`x` sums to zero", call. = FALSE)
    if (k < 1 || k != as.integer(k)) stop("`k` must be a positive integer", call. = FALSE)
    if (!method %in% c("nnmf_mm", "nnmf_als", "nnmf_prob")) {
      stop("unknown `method`: ", method, call. = FALSE)   # nnmf() would not
    }
    nnmf(x, k, method, maxiter)
  }

  expect_error(safe_nnmf(matrix(c(-1, 2, 3, 4), 2, 2), 1), "requires x >= 0")
  expect_error(safe_nnmf(matrix(0, 3, 3), 1), "sums to zero")
  expect_error(safe_nnmf(X_SMALL, 2, "nnmf_typo"), "unknown `method`")
  expect_error(safe_nnmf(X_SMALL, 0), "positive integer")

  set.seed(73)
  z <- quiet(safe_nnmf(X_SMALL, 2, maxiter = 60))
  expect_valid_factorisation(z, X_SMALL, k = 2)
})

test_that("nnmf_mm and nnmf_als require maxiter and eps to be supplied", {
  # Only nnmf() has defaults for these; the algorithms themselves do not, so
  # calling them directly with two arguments fails. nnmf_prob defaults eps but
  # still requires maxiter.
  expect_error(nnmf_mm(X_SMALL, 2), "missing, with no default")
  expect_error(nnmf_als(X_SMALL, 2), "missing, with no default")
  expect_error(nnmf_prob(X_SMALL, 2), "missing, with no default")

  set.seed(74)
  err <- error_message_of(nnmf_prob(X_SMALL, 2, 20))   # eps defaulted
  expect_null(err)
})

test_that("a data frame is accepted by nnmf_mm and nnmf_als", {
  # Both call as.matrix(x) first. nnmf_prob does not, so it is not tested here.
  df <- as.data.frame(X_SMALL)
  set.seed(75)
  z <- quiet(nnmf_mm(df, 2, 60, EPS))
  expect_equal(dim(z$W), c(3L, 2L))
  expect_equal(dim(z$H), c(2L, 4L))
})
