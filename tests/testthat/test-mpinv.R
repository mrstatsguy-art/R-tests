# mpinv() -- Moore-Penrose pseudoinverse via SVD.
#
# This is the one function in the package with a single correct answer, so it
# is the one place exact numerical assertions are appropriate. The four
# Moore-Penrose conditions define the pseudoinverse uniquely; tests 3 and 4
# check the two that matter in practice.

test_that("mpinv inverts a full-rank square matrix", {
  A <- matrix(c(1, 0, 2,
                2, 1, 0,
                0, 1, 1), 3, 3, byrow = TRUE)
  expect_equal(mpinv(A), solve(A), tolerance = 1e-10)
})

test_that("mpinv returns the transposed shape", {
  # A pseudoinverse of an n x m matrix is m x n.
  expect_equal(dim(mpinv(matrix(1:12, 3, 4))), c(4L, 3L))
  expect_equal(dim(mpinv(matrix(1:12, 4, 3))), c(3L, 4L))
})

test_that("mpinv satisfies X M X == X on a rank-deficient matrix", {
  X <- matrix(1:12, 3, 4)          # rank 2, not 3
  M <- mpinv(X)
  expect_equal(X %*% M %*% X, X, tolerance = 1e-8)
})

test_that("mpinv satisfies M X M == M on a rank-deficient matrix", {
  X <- matrix(1:12, 3, 4)
  M <- mpinv(X)
  expect_equal(M %*% X %*% M, M, tolerance = 1e-8)
})

test_that("mpinv agrees with MASS::ginv", {
  skip_if_not_installed("MASS")
  X <- matrix(1:12, 3, 4)
  expect_equal(mpinv(X), MASS::ginv(X), tolerance = 1e-8)

  set.seed(4)
  Y <- matrix(rnorm(30), 6, 5)
  expect_equal(mpinv(Y), MASS::ginv(Y), tolerance = 1e-8)
})

test_that("mpinv of a symmetric matrix is symmetric", {
  S <- crossprod(matrix(c(2, 1, 0, 1, 3, 1), 3, 2))
  M <- mpinv(S)
  expect_equal(M, t(M), tolerance = 1e-10)
})

test_that("mpinv of a zero matrix is a zero matrix", {
  # The mathematically correct answer, and it exercises the branch where every
  # singular value is discarded as negligible.
  Z <- matrix(0, 3, 4)
  expect_equal(mpinv(Z), matrix(0, 4, 3), tolerance = 1e-12)
})

test_that("mpinv handles a 1x1 matrix", {
  expect_equal(mpinv(matrix(4, 1, 1)), matrix(0.25, 1, 1), tolerance = 1e-12)
})

test_that("mpinv zeroes entries below its internal threshold", {
  # The implementation ends with mp[abs(mp) < 100 * .Machine$double.eps] <- 0,
  # so no returned entry may sit in the open interval (0, threshold).
  set.seed(11)
  M <- mpinv(matrix(rnorm(20), 4, 5))
  nz <- abs(M[M != 0])
  expect_true(all(nz >= 100 * .Machine$double.eps))
})
