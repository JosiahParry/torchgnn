as_sparse_matrix <- function(adj) {
  idx <- adj$indices() + 1L
  vals <- adj$values()

  Matrix::sparseMatrix(
    as_array(idx[1, ]),
    as_array(idx[2, ]),
    x = as_array(vals)
  )
}

# global adjacency matrix for tests
x_matrix <- matrix(
  c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12),
  nrow = 3,
  ncol = 4,
  byrow = TRUE
)

x <- torch::torch_tensor(
  x_matrix,
  dtype = torch::torch_float()
)

indices <- torch::torch_tensor(
  rbind(
    c(1, 1, 2, 2, 3, 3),
    c(2, 3, 1, 3, 1, 2)
  ),
  dtype = torch::torch_int64()
)

values <- torch::torch_ones(6)

# sparse matrix
adj <- torch::torch_sparse_coo_tensor(indices, values, c(3, 3))$coalesce()

# Matrix sparse matrix for us to compare against
adj_sparse <- as_sparse_matrix(adj)

test_that("SumAggregator works", {
  agg <- SumAggregator()
  expect_equal(agg@name, "sum")
  expect_false(agg@learnable)

  out <- forward(agg, adj, x)
  expect_equal(out$shape, c(3, 4))

  expected <- as.matrix(adj_sparse %*% x_matrix)
  expect_equal(as.array(out), expected, tolerance = 1e-6)
})

test_that("MeanAggregator works", {
  agg <- MeanAggregator()
  expect_equal(agg@name, "mean")
  expect_false(agg@learnable)

  out <- forward(agg, adj, x)
  expect_equal(out$shape, c(3, 4))

  adj_norm <- adj_sparse / Matrix::rowSums(adj_sparse)
  expected <- as.matrix(adj_norm %*% x_matrix)
  expect_equal(as.array(out), expected, tolerance = 1e-6)
})

test_that("MaxAggregator works", {
  agg <- MaxAggregator()
  expect_equal(agg@name, "max")
  expect_false(agg@learnable)

  out <- forward(agg, adj, x)
  expect_equal(out$shape, c(3, 4))

  expected <- matrix(
    c(9, 10, 11, 12, 9, 10, 11, 12, 5, 6, 7, 8),
    nrow = 3,
    ncol = 4,
    byrow = TRUE
  )
  expect_equal(as.array(out), expected, tolerance = 1e-6)
})

test_that("MinAggregator works", {
  agg <- MinAggregator()
  expect_equal(agg@name, "min")
  expect_false(agg@learnable)

  out <- forward(agg, adj, x)
  expect_equal(out$shape, c(3, 4))

  expected <- matrix(
    c(5, 6, 7, 8, 1, 2, 3, 4, 1, 2, 3, 4),
    nrow = 3,
    ncol = 4,
    byrow = TRUE
  )
  expect_equal(as.array(out), expected, tolerance = 1e-6)
})

test_that("ProductAggregator works", {
  agg <- ProductAggregator()
  expect_equal(agg@name, "prod")
  expect_false(agg@learnable)

  out <- forward(agg, adj, x)
  expect_equal(out$shape, c(3, 4))

  expected <- matrix(
    c(45, 60, 77, 96, 9, 20, 33, 48, 5, 12, 21, 32),
    nrow = 3,
    ncol = 4,
    byrow = TRUE
  )
  expect_equal(as.array(out), expected, tolerance = 1e-6)
})

test_that("VarAggregator works", {
  agg <- VarAggregator()
  expect_equal(agg@name, "var")
  expect_false(agg@learnable)

  out <- forward(agg, adj, x)
  expect_equal(out$shape, c(3, 4))

  expected <- matrix(
    c(4, 4, 4, 4, 16, 16, 16, 16, 4, 4, 4, 4),
    nrow = 3,
    ncol = 4,
    byrow = TRUE
  )
  expect_equal(as.array(out), expected, tolerance = 1e-6)
})

test_that("StdAggregator works", {
  agg <- StdAggregator()
  expect_equal(agg@name, "std")
  expect_false(agg@learnable)

  out <- forward(agg, adj, x)
  expect_equal(out$shape, c(3, 4))

  expected <- matrix(
    c(2, 2, 2, 2, 4, 4, 4, 4, 2, 2, 2, 2),
    nrow = 3,
    ncol = 4,
    byrow = TRUE
  )
  expect_equal(as.array(out), expected, tolerance = 1e-6)
})

test_that("LSTMAggregator works", {})

test_that("SoftmaxAggregator learnable works", {})
