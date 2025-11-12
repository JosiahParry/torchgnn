#' Create Sparse Adjacency Matrix from Edge List
#'
#' Converts an edge list (from/to pairs) into a sparse COO adjacency matrix.
#' Edges are created exactly as specified in the input.
#'
#' @param from Integer vector. Source node indices for edges
#' @param to Integer vector. Target node indices for edges
#' @param n Integer. Number of nodes. Default: `max(c(from, to))`
#' @param symmetric Logical. Makes adjacency symmetric. Default: TRUE
#'
#' @return Sparse COO tensor `n x n`
#'
#' @examples
#' # Create a triangle graph
#' from <- c(1, 2, 3)
#' to <- c(2, 3, 1)
#' adj <- adj_from_edgelist(from, to)
#'
#' # Create a star graph (node 1 connected to nodes 2,3,4)
#' from <- c(1, 1, 1)
#' to <- c(2, 3, 4)
#' adj_from_edgelist(from, to)
#' @export
adj_from_edgelist <- function(
  from,
  to,
  n = max(c(from, to)),
  symmetric = TRUE
) {
  if (symmetric) {
    i <- from
    j <- to
    from <- c(i, j)
    to <- c(j, i)
  }

  # Build indices matrix: 2 x num_edges
  indices <- torch::torch_tensor(
    rbind(from, to),
    dtype = torch::torch_int64()
  )

  # Create values vector: all 1.0s
  values <- torch::torch_ones(
    length(from),
    dtype = torch::torch_float32()
  )

  # Create sparse COO tensor
  torch::torch_sparse_coo_tensor(
    indices,
    values,
    size = c(n, n)
  )$coalesce()
}
