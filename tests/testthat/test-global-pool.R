# global defs
x_single <- torch_tensor(matrix(
  c(1, 2, 3, 4, 5, 6),
  nrow = 3,
  ncol = 2,
  byrow = TRUE
))

x_multi <- torch_tensor(matrix(
  c(1, 2, 3, 4, 5, 6, 7, 8),
  nrow = 4,
  ncol = 2,
  byrow = TRUE
))
batch_multi <- torch_tensor(c(1, 1, 2, 2), dtype = torch_long())

x_uneven <- torch_tensor(matrix(
  c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
  nrow = 5,
  ncol = 2,
  byrow = TRUE
))

batch_uneven <- torch_tensor(c(1, 1, 1, 2, 2), dtype = torch_long())

test_that("pool_global_add works with single graph", {
  result <- pool_global_add(x_single, batch = NULL)

  expect_equal(result$shape, c(1, 2))
  expected <- matrix(c(9, 12), nrow = 1, ncol = 2)
  expect_equal(as.array(result), expected, tolerance = 1e-6)
})

test_that("pool_global_add works with multiple graphs", {
  result <- pool_global_add(x_multi, batch_multi, size = 2)

  expect_equal(result$shape, c(2, 2))
  expected <- matrix(c(4, 6, 12, 14), nrow = 2, ncol = 2, byrow = TRUE)
  expect_equal(as.array(result), expected, tolerance = 1e-6)
})

test_that("pool_global_mean works with single graph", {
  result <- pool_global_mean(x_single, batch = NULL)

  expect_equal(result$shape, c(1, 2))
  expected <- matrix(c(3, 4), nrow = 1, ncol = 2)
  expect_equal(as.array(result), expected, tolerance = 1e-6)
})

test_that("pool_global_mean works with multiple graphs", {
  result <- pool_global_mean(x_multi, batch_multi, size = 2)

  expect_equal(result$shape, c(2, 2))
  expected <- matrix(c(2, 3, 6, 7), nrow = 2, ncol = 2, byrow = TRUE)
  expect_equal(as.array(result), expected, tolerance = 1e-6)
})

test_that("pool_global_max works with single graph", {
  result <- pool_global_max(x_single, batch = NULL)

  expect_equal(result$shape, c(1, 2))
  expected <- matrix(c(5, 6), nrow = 1, ncol = 2)
  expect_equal(as.array(result), expected, tolerance = 1e-6)
})

test_that("pool_global_max works with multiple graphs", {
  result <- pool_global_max(x_multi, batch_multi, size = 2)

  expect_equal(result$shape, c(2, 2))
  expected <- matrix(c(3, 4, 7, 8), nrow = 2, ncol = 2, byrow = TRUE)
  expect_equal(as.array(result), expected, tolerance = 1e-6)
})

test_that("pooling works with uneven graph sizes", {
  sum_result <- pool_global_add(x_uneven, batch_uneven, size = 2)
  mean_result <- pool_global_mean(x_uneven, batch_uneven, size = 2)
  max_result <- pool_global_max(x_uneven, batch_uneven, size = 2)

  expected_sum <- matrix(
    c(9, 12, 16, 18),
    nrow = 2,
    ncol = 2,
    byrow = TRUE
  )

  expected_mean <- matrix(
    c(3, 4, 8, 9),
    nrow = 2,
    ncol = 2,
    byrow = TRUE
  )

  expected_max <- matrix(
    c(5, 6, 9, 10),
    nrow = 2,
    ncol = 2,
    byrow = TRUE
  )

  expect_equal(as.array(sum_result), expected_sum, tolerance = 1e-6)
  expect_equal(as.array(mean_result), expected_mean, tolerance = 1e-6)
  expect_equal(as.array(max_result), expected_max, tolerance = 1e-6)
})
