#' Normalize an Adjacency Matrix
#'
#' @description
#' Prepare a sparse adjacency matrix for message passing.
#'
#' - `gcn_normalize()` applies the symmetric normalization
#'   \eqn{D^{-1/2} A D^{-1/2}} of Kipf and Welling (2017).
#' - `adj_row_normalize()` applies the row normalization \eqn{D^{-1} A}, so
#'   that the weights of each node's neighbors sum to one.
#' - `add_graph_self_loops()` replaces any existing diagonal entries with
#'   unit self-loops, giving \eqn{A + I}.
#'
#' Isolated nodes have degree zero. Their normalization factor is set to zero
#' rather than being allowed to diverge, so they contribute nothing to the
#' aggregation.
#'
#' @param adj Sparse COO `torch_tensor` `n_nodes x n_nodes`. The adjacency
#'   matrix, which may be weighted.
#'
#' @return A coalesced sparse COO `torch_tensor` of the same dimension as
#'   `adj`.
#'
#' @references
#' Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with
#' graph convolutional networks. International Conference on Learning
#' Representations. <doi:10.48550/arXiv.1609.02907>
#'
#' @examplesIf torch::torch_is_installed()
#' adj <- adj_from_edgelist(from = c(1, 2, 3), to = c(2, 3, 1))
#'
#' gcn_normalize(adj)
#'
#' adj_row_normalize(adj)
#'
#' # The usual pre-processing for a GCN layer
#' gcn_normalize(add_graph_self_loops(adj))
#' @rdname adjacency
#' @export
gcn_normalize <- function(adj) {
  # Compute degree vector
  deg <- adj$sum(dim = 2)$to_dense()
  deg_inv_sqrt <- deg$pow(-0.5)
  deg_inv_sqrt[deg_inv_sqrt == Inf] <- 0

  indices <- adj$indices()
  values <- adj$values()

  row_deg <- deg_inv_sqrt[indices[1, ]]
  col_deg <- deg_inv_sqrt[indices[2, ]]

  norm_values <- row_deg * values * col_deg

  torch::torch_sparse_coo_tensor(indices, norm_values, adj$size())$coalesce()
}


#' @rdname adjacency
#' @export
adj_row_normalize <- function(adj) {
  # Compute row sums (degree vector)
  deg <- adj$sum(dim = 2)$to_dense()

  # Avoid division by zero
  deg_inv <- deg$reciprocal()
  deg_inv[deg_inv == Inf] <- 0

  # Scale nonzero values per row
  indices <- adj$indices()
  values <- adj$values()

  row_deg <- deg_inv[indices[1, ]]
  norm_values <- row_deg * values

  torch::torch_sparse_coo_tensor(indices, norm_values, adj$size())$coalesce()
}


#' @rdname adjacency
#' @export
add_graph_self_loops <- function(adj) {
  n <- adj$size()[[1]]
  indices <- adj$indices()
  values <- adj$values()

  # Identify non-self-loop edges
  mask <- indices[1, ] != indices[2, ]
  indices_no_self <- indices[, mask]
  values_no_self <- values[mask]

  # Build fresh self-loop tensor (1s on diagonal)
  self_indices <- torch::torch_stack(list(
    torch::torch_arange(1, n, dtype = torch::torch_int64()),
    torch::torch_arange(1, n, dtype = torch::torch_int64())
  ))
  self_values <- torch::torch_ones(n)

  # Concatenate non-self edges with self-loops
  new_indices <- torch::torch_cat(list(indices_no_self, self_indices), dim = 2)
  new_values <- torch::torch_cat(list(values_no_self, self_values))

  torch::torch_sparse_coo_tensor(new_indices, new_values, adj$size())$coalesce()
}
