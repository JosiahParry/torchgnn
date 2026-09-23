#' Create Sparse Adjacency Matrix from Edge List
#'
#' @param from Integer or character vector of source nodes
#' @param to Integer or character vector of target nodes
#' @param weight Numeric vector of edge weights
#' @param n Number of nodes
#' @param symmetric Make adjacency symmetric
#'
#' @return A sparse COO `torch_tensor` of dimension `n x n`, carrying the
#'   attributes `node_ids` (the original node identifiers, or `NULL` when
#'   `from` and `to` are already integers) and `id_map` (a named integer
#'   vector mapping identifiers to row and column positions, or `NULL`).
#'
#' @examplesIf torch::torch_is_installed()
#' adj_from_edgelist(from = c(1, 2, 3), to = c(2, 3, 1))
#'
#' # Character node identifiers are mapped to positions
#' adj <- adj_from_edgelist(
#'   from = c("a", "b", "c"),
#'   to = c("b", "c", "a"),
#'   weight = c(0.5, 1, 2)
#' )
#' attr(adj, "id_map")
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
    id_map <- stats::setNames(seq_along(node_ids), node_ids)
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
#' @return A dense `torch_tensor` of dimension `n_nodes x n_features`. When
#'   `adj` and `node_id` are both supplied, rows are reordered to match the
#'   node ordering of `adj` and the identifier column is dropped.
#'
#' @examplesIf torch::torch_is_installed()
#' adj <- adj_from_edgelist(from = c("a", "b", "c"), to = c("b", "c", "a"))
#' nodes <- data.frame(id = c("c", "a", "b"), deg = c(2, 2, 2), pop = c(10, 20, 30))
#'
#' nodes_to_tensor(nodes, adj, node_id = "id")
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
