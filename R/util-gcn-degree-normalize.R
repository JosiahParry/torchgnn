#' @rdname adjacency
#' @export
gcn_normalize <- function(adj) {
  # Compute degree vector
  deg <- adj$sum(dim = 2)$to_dense()
  deg_inv_sqrt <- deg$pow(-0.5)
  deg_inv_sqrt[deg_inv_sqrt == Inf] <- 0

  indices <- adj$indices() + 1L
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
  indices <- adj$indices() + 1L
  values <- adj$values()

  row_deg <- deg_inv[indices[1, ]]
  norm_values <- row_deg * values

  torch::torch_sparse_coo_tensor(indices, norm_values, adj$size())$coalesce()
}


#' Add self-loops to a graph
#'
#' @param adj a sparse COO tensor of the adjacency matrix. Can be weighted.
#' @rdname adjacency
#' @export
add_graph_self_loops <- function(adj) {
  n <- adj$size()[[1]]
  indices <- adj$indices() + 1L
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
