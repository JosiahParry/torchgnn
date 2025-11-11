#' GCN Convolutional Layer (Kipf & Welling 2016)
#'
#' Implements the basic GCN layer from Kipf & Welling (2016):
#' \deqn{H^{(k)} = \sigma(\tilde{A} H^{(k-1)} W + b)}
#'
#' Where \eqn{\tilde{A} = \tilde{D}^{-1/2}(A + I)\tilde{D}^{-1/2}} is the symmetrically normalized adjacency
#' matrix with self-loops, and \eqn{\tilde{D}} is the degree matrix of \eqn{A + I}.
#'
#' @details
#' This is the standard GCN layer that uses:
#' - Single weight matrix \eqn{W}
#' - Symmetric normalization with self-loops
#' - Optional on-the-fly normalization
#'
#' When \code{normalize = TRUE} (default), the layer computes \eqn{\tilde{A}} on-the-fly from
#' the input adjacency matrix by adding self-loops and applying symmetric normalization.
#' When \code{normalize = FALSE}, you must pass in a pre-normalized adjacency matrix.
#'
#' Parameters:
#' - \eqn{W}: `in_features x out_features` learnable weight matrix
#' - \eqn{b}: `out_features` learnable bias term (optional)
#'
#' @param in_features Integer. Number of input features per node
#' @param out_features Integer. Number of output features per node
#' @param bias Logical. Add learnable bias. Default: TRUE
#' @param normalize Logical. Whether to add self-loops and compute symmetric normalization
#'   on-the-fly. Default: TRUE
#'
#' @section Forward pass:
#' @param x Tensor `n_nodes x in_features`. Node feature matrix
#' @param adj Tensor `n_nodes x n_nodes`. Adjacency matrix defining graph structure.
#'   Can be binary (0/1) or weighted. If \code{edge_weight} is provided, \code{adj}
#'   should be binary and weights will be applied from \code{edge_weight}
#' @param edge_weight Tensor `n_nodes x n_nodes` or NULL. Optional edge weights.
#'   If NULL, uses values from \code{adj}. When \code{normalize = TRUE}, edge weights
#'   are used to compute degrees: \eqn{\hat{d}_i = 1 + \sum_{j \in N(i)} e_{j,i}}.
#'   Default: NULL
#'
#' @return Tensor `n_nodes x out_features`. Transformed node features
#'
#' @references
#' Kipf, T. N., & Welling, M. (2016). Semi-supervised classification with
#' graph convolutional networks. arXiv preprint arXiv:1609.02907. <doi:10.48550/arXiv.1609.02907>
#' @export
gcn_conv_layer <- nn_module(
  "GCNConvLayer",

  initialize = function(
    in_features,
    out_features,
    bias = TRUE,
    normalize = TRUE
  ) {
    # Single weight matrix
    self$weight <- nn_parameter(torch_randn(in_features, out_features))
    nn_init_xavier_uniform_(self$weight)

    if (bias) {
      self$bias <- nn_parameter(torch_zeros(1, out_features))
    } else {
      self$bias <- NULL
    }

    self$normalize <- normalize
  },

  forward = function(x, adj, edge_weight = NULL) {
    # Apply symmetric normalization if requested
    if (self$normalize) {
      # Clone adjacency to avoid modifying input
      adj <- adj$clone()

      # If edge_weight not provided, extract from adjacency
      # (supports both binary and pre-weighted adjacency matrices)
      if (is.null(edge_weight)) {
        edge_weight <- adj
      } else {
        # If edge_weight provided, multiply adjacency structure by weights
        edge_weight <- adj * edge_weight
      }

      # Add self-loops: set diagonal to 1
      edge_weight$fill_diagonal_(1)

      # Compute degrees using edge weights (PyG formulation)
      # d_hat_i = 1 + sum(e_{j,i}) for j in N(i)
      # Since we already set diagonal to 1, this is just the row sum
      deg <- torch_sum(edge_weight, dim = 2)

      # D^(-1/2)
      deg_inv_sqrt <- torch_pow(deg, -0.5)
      deg_inv_sqrt[torch_isinf(deg_inv_sqrt)] <- 0

      # Symmetric normalization: e_{j,i} / sqrt(d_hat_j * d_hat_i)
      # Equivalent to: D^(-1/2) * edge_weight * D^(-1/2)
      adj <- deg_inv_sqrt$view(c(-1, 1)) *
        edge_weight *
        deg_inv_sqrt$view(c(1, -1))
    } else if (!is.null(edge_weight)) {
      # No normalization but edge_weight provided: apply weights directly
      adj <- adj * edge_weight
    }

    # Basic GCN: H = A_tilde X W
    out <- torch_mm(adj, x)
    out <- torch_mm(out, self$weight)

    if (!is.null(self$bias)) {
      out <- out + self$bias
    }

    out
  }
)

#' Generalized GCN Layer (Hamilton 2020)
#'
#' @description
#' Implements a single Graph Convolutional Network (GCN) layer following Hamilton 2020:
#'
#' \deqn{\mathbf{H}^{(k)} = \sigma\left(\mathbf{A}\mathbf{H}^{(k-1)}\mathbf{W}^{(k)}_{\text{neigh}} + \mathbf{H}^{(k-1)}\mathbf{W}^{(k)}_{\text{self}}\right)}
#'
#' This can also be written as (Guo et al. 2025):
#'
#' \deqn{\mathbf{X}^{(l)} = \sigma\left(\mathbf{D}^{-1}\mathbf{A}\mathbf{X}^{(l-1)}\boldsymbol{\Theta}^{(l)} + \mathbf{X}^{(l-1)}\boldsymbol{\Phi}^{(l)} + \boldsymbol{\Psi}^{(l)}\right)}
#'
#' This layer combines:
#' - Neighbor aggregation: \eqn{D^{-1}AX^{(l-1)}\Theta^{(l)}}
#' - Self transformation: \eqn{X^{(l-1)}\Phi^{(l)}} focal node transformation
#' - Global bias: \eqn{\Psi^{(l)}} additive bias term
#'
#' Parameters:
#' - \eqn{\Theta} (theta): `in_features x out_features` transforms aggregated neighbor features
#' - \eqn{\Phi} (phi): `in_features x out_features` transforms node's own features
#' - \eqn{\Psi} (psi): `out_features` global bias term (shared across all nodes)
#'
#' @details
#' The adjacency matrix is expected to be row-normalized \eqn{D^{-1}A} where \eqn{D}
#' is the degree matrix. This layer does NOT perform normalization internally.
#'
#' @param in_features Integer. Number of input features per node
#' @param out_features Integer. Number of output features per node
#' @param bias Logical. Add learnable bias term (\eqn{\Psi}). Default: TRUE
#'
#' @section Forward pass:
#' @param x Tensor `n_nodes x in_features`. Node feature matrix
#' @param adj Tensor `n_nodes x n_nodes`. Adjacency matrix. Expected to be row-normalized
#'   \eqn{D^{-1}A} where \eqn{D} is the degree matrix. Can be binary or weighted.
#'   This layer does NOT perform normalization internally
#' @param edge_weight Tensor `n_nodes x n_nodes` or NULL. Optional edge weights to
#'   apply to the adjacency matrix. If NULL, uses values from \code{adj}. Default: NULL
#'
#' @return Tensor `n_nodes x out_features`. Transformed node features (before activation)
#'
#' @references
#' Hamilton, W. L. (2020). Graph Representation Learning. In Synthesis Lectures
#' on Artificial Intelligence and Machine Learning. Springer International
#' Publishing. <doi:10.1007/978-3-031-01588-5>
#'
#' Guo, H., Wang, H., Zhu, D., Wu, L., Fotheringham, A. S., & Liu, Y. (2025).
#' RegionGCN: Spatial-Heterogeneity-Aware Graph Convolutional Networks. Annals
#' of the American Association of Geographers, 1–17.
#' <doi:10.1080/24694452.2025.2558661>
#' @export
gcn_layer <- nn_module(
  "GCNLayer",

  initialize = function(in_features, out_features, bias = TRUE) {
    # theta: neighbor aggregation weights
    self$theta <- nn_linear(in_features, out_features, bias = FALSE)
    # phi: self transformation weights
    self$phi <- nn_linear(in_features, out_features, bias = FALSE)

    # psi: global bias term
    if (bias) {
      self$psi <- nn_parameter(torch_randn(1, out_features))
    } else {
      self$psi <- NULL
    }
  },

  forward = function(x, adj, edge_weight = NULL) {
    # x: node features (n_nodes x in_features)
    # adj: row-normalized adjacency (n_nodes x n_nodes)
    # edge_weight: optional edge weights

    # Apply edge weights if provided
    if (!is.null(edge_weight)) {
      adj <- adj * edge_weight
    }

    neighbor_agg <- torch_mm(adj, x)
    # Transform aggregated neighbors (theta)
    neighbor_trans <- self$theta(neighbor_agg)
    # Transform self features (phi)
    self_trans <- self$phi(x)

    # Combine: neighbor + self + bias
    out <- neighbor_trans + self_trans
    if (!is.null(self$psi)) {
      out <- out + self$psi
    }
    out
  }
)
