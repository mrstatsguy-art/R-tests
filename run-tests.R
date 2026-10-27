#!/usr/bin/env Rscript
# Standalone runner, so the suite works without being inside a package.
#
#   Rscript run-tests.R
#
# Exits non-zero if anything fails, which is what a CI step needs.

if (!requireNamespace("testthat", quietly = TRUE)) {
  stop("install.packages('testthat')")
}
if (!requireNamespace("NMFN", quietly = TRUE)) {
  stop("install.packages('NMFN')")
}

library(testthat)
library(NMFN)

cat("NMFN version:", as.character(utils::packageVersion("NMFN")), "\n")
cat("R version:   ", R.version.string, "\n\n")

results <- testthat::test_dir(
  "tests/testthat",
  reporter = "summary",
  stop_on_failure = TRUE
)
