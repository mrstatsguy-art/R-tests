# distance2(x1, x2) -- squared Euclidean distance, sum((x1 - x2)^2).
#
# Three lines of code, but it is the error metric the whole package is
# optimising, and it inherits R's recycling rules. The last test documents a
# trap rather than a guarantee.

test_that("distance2 is zero for identical inputs", {
  X <- matrix(1:12, 3, 4)
  expect_equal(distance2(X, X), 0)
})

test_that("distance2 computes the sum of squared differences", {
  a <- matrix(c(1, 2, 3, 4), 2, 2)
  b <- matrix(c(1, 2, 3, 5), 2, 2)
  expect_equal(distance2(a, b), 1)

  c1 <- matrix(c(0, 0, 0, 0), 2, 2)
  c2 <- matrix(c(1, 2, 3, 4), 2, 2)
  expect_equal(distance2(c1, c2), 1 + 4 + 9 + 16)
})

test_that("distance2 is symmetric and non-negative", {
  set.seed(5)
  a <- matrix(rnorm(12), 3, 4)
  b <- matrix(rnorm(12), 3, 4)
  expect_equal(distance2(a, b), distance2(b, a))
  expect_gte(distance2(a, b), 0)
})

test_that("distance2 works on plain vectors", {
  expect_equal(distance2(c(1, 2, 3), c(1, 2, 5)), 4)
})

test_that("distance2 errors on non-conformable matrices", {
  a <- matrix(1:12, 3, 4)
  b <- matrix(1:12, 4, 3)
  expect_error(distance2(a, b))
})

test_that("distance2 silently recycles a vector against a matrix", {
  # CHARACTERISATION, not an endorsement. `x1 - x2` follows R's recycling
  # rules, so a length-3 vector against a 3x4 matrix produces a number rather
  # than an error. If you call distance2 on anything whose shape you have not
  # checked, this is how a bug stays quiet.
  X <- matrix(0, 3, 4)
  v <- c(1, 1, 1)
  expect_equal(distance2(X, v), 12)   # 1 recycled across all 12 cells
})
