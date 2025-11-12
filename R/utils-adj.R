#' Create Sparse Adjacency Matrix from Edge List
#'
#' @param from Integer or character vector of source nodes
#' @param to Integer or character vector of target nodes
#' @param weight Numeric vector of edge weights
#' @param n Number of nodes
#' @param symmetric Make adjacency symmetric
#'
#' @return Sparse COO tensor with attributes `node_ids` and `id_map`
#' @export
adj_from_edgelist <- function(
  from,
  to,
  weight = NULL,
  n = NULL,
  symmetric = TRUE
) {
  node_ids <- NULL
  id_map <- NULL

  if (is.character(from) || is.character(to)) {
    node_ids <- unique(c(from, to))
    n <- if (is.null(n)) length(node_ids) else n
    id_map <- setNames(seq_along(node_ids), node_ids)
    from <- id_map[as.character(from)]
    to <- id_map[as.character(to)]
  } else {
    n <- if (is.null(n)) max(c(from, to)) else n
  }

  if (symmetric) {
    i <- from
    j <- to
    from <- c(i, j)
    to <- c(j, i)
    if (!is.null(weight)) {
      weight <- c(weight, weight)
    }
  }

  if (is.null(weight)) {
    weight <- rep(1, length(from))
  }

  indices <- torch::torch_tensor(
    rbind(from, to),
    dtype = torch::torch_int64()
  )

  values <- torch::torch_tensor(
    weight,
    dtype = torch::torch_float32()
  )

  adj <- torch::torch_sparse_coo_tensor(
    indices,
    values,
    size = c(n, n)
  )$coalesce()

  attr(adj, "node_ids") <- node_ids
  attr(adj, "id_map") <- id_map

  adj
}

#' Convert Node Features to Tensor
#'
#' @param nodes Dataframe with node features
#' @param adj Adjacency matrix with `id_map` attribute
#' @param node_id Column name for node IDs
#'
#' @return Dense tensor of shape `[n_nodes, n_features]`
#' @export
nodes_to_tensor <- function(nodes, adj = NULL, node_id = NULL) {
  if (!is.null(node_id) && !is.null(adj)) {
    id_map <- attr(adj, "id_map")
    if (!is.null(id_map)) {
      node_order <- names(id_map)
      nodes <- nodes[match(node_order, nodes[[node_id]]), ]
    }
    feature_cols <- setdiff(names(nodes), node_id)
  } else {
    feature_cols <- names(nodes)
  }

  features <- as.matrix(nodes[, feature_cols, drop = FALSE])
  torch::torch_tensor(features, dtype = torch::torch_float32())
}
