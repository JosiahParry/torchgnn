#' Graph Pooling Functions
#'
#' @description
#' Aggregate node features to graph-level representations for graph classification.
#' These functions reduce node embeddings within each graph to a single vector.
#'
#' @param x Tensor. Node feature matrix with shape `(total_nodes, features)`.
#'   Contains features for all nodes from all graphs stacked together.
#' @param batch Tensor or NULL. Batch vector assigning each node to a graph.
#'   Values should be graph indices starting from 1 (e.g., `c(1,1,2,2,2)` for
#'   2 graphs with 2 and 3 nodes). If NULL, treats all nodes as a single graph.
#' @param size Integer or NULL. Number of graphs. Automatically calculated if NULL.
#'
#' @return Tensor with shape `(num_graphs, features)` containing graph-level embeddings.
#'
#' @details
#' These functions implement different reduction strategies:
#' - `pool_global_add`: Sum of node features per graph
#' - `pool_global_mean`: Mean of node features per graph
#' - `pool_global_max`: Element-wise maximum of node features per graph
#'
#' @examplesIf torch::torch_is_installed()
#' x <- torch::torch_tensor(
#'   matrix(c(1, 2, 3, 4, 5, 6, 7, 8), nrow = 4, byrow = TRUE)
#' )
#'
#' # Two graphs of two nodes each
#' batch <- torch::torch_tensor(c(1, 1, 2, 2), dtype = torch::torch_long())
#'
#' pool_global_add(x, batch)
#' pool_global_mean(x, batch)
#' pool_global_max(x, batch)
#'
#' # Without a batch vector every node belongs to one graph
#' pool_global_mean(x)
#'
#' @rdname pooling
#' @export
pool_global_add <- function(x, batch = NULL, size = NULL) {
  if (is.null(batch)) {
    return(x$sum(dim = 1, keepdim = TRUE))
  }

  n_features <- x$size(2)
  if (is.null(size)) {
    size <- batch$max()$item()
  }

  result <- torch_zeros(size, n_features)
  index_expanded <- batch$unsqueeze(2)$expand(c(-1, n_features))
  result$scatter_add_(1, index_expanded, x)

  result
}

#' @rdname pooling
#' @export
pool_global_mean <- function(x, batch = NULL, size = NULL) {
  if (is.null(batch)) {
    return(x$mean(dim = 1, keepdim = TRUE))
  }

  n_features <- x$size(2)
  if (is.null(size)) {
    size <- batch$max()$item()
  }

  sum_result <- torch_zeros(size, n_features)
  index_expanded <- batch$unsqueeze(2)$expand(c(-1, n_features))
  sum_result$scatter_add_(1, index_expanded, x)

  counts <- torch_zeros(size)
  counts$scatter_add_(1, batch, torch_ones_like(batch, dtype = torch_float()))

  sum_result / counts$unsqueeze(2)
}

#' @rdname pooling
#' @export
pool_global_max <- function(x, batch = NULL, size = NULL) {
  if (is.null(batch)) {
    return(x$max(dim = 1, keepdim = TRUE)[[1]])
  }

  n_features <- x$size(2)
  if (is.null(size)) {
    size <- batch$max()$item()
  }

  result <- torch_full(c(size, n_features), -Inf)
  index_expanded <- batch$unsqueeze(2)$expand(c(-1, n_features))
  result$scatter_reduce_(1, index_expanded, x, "amax", include_self = FALSE)

  result
}
