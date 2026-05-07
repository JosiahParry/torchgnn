# https://github.com/mlverse/torch/issues/1460
test_that("adjacency matrix is 1 based index", {
  edges_from <- c(1, 1, 2, 3, 4, 5, 6, 7, 8, 9)
  edges_to <- c(2, 3, 1, 4, 5, 6, 7, 8, 9, 10)
  adj <- adj_from_edgelist(edges_from, edges_to)

  expect_all_true(as.integer(adj$indices()) != 0)
})
