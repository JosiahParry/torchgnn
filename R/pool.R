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
#' - `global_add_pool`: Sum of node features per graph
#' - `global_mean_pool`: Mean of node features per graph
#' - `global_max_pool`: Element-wise maximum of node features per graph
#'
#' @rdname pooling
#' @export
global_add_pool <- function(x, batch = NULL, size = NULL) {
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
global_mean_pool <- function(x, batch = NULL, size = NULL) {
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
global_max_pool <- function(x, batch = NULL, size = NULL) {
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
