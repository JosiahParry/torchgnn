#' Regional GCN Convolutional Layer (Guo et al. 2025)
#'
#' @description
#' Implements the Regional Graph Convolutional Network (RegConv) layer from Guo et al. (2025):
#'
#' \deqn{\mathbf{X}^{(l)} = \sigma\left(\left(\mathbf{D}^{-1}\mathbf{A}\mathbf{X}^{(l-1)}\boldsymbol{\Theta}^{(l)} + \mathbf{X}^{(l-1)}\boldsymbol{\Phi}^{(l)}\right)\boldsymbol{\Omega}_{reg}^{(l)} + \boldsymbol{\Psi}_{reg}^{(l)}\right)}
#'
#' This layer extends the standard GCN layer by introducing region-specific parameters to
#' handle spatial heterogeneity (spatial regimes). The computation has two stages:
#'
#' 1. **Base GCN transformation**: \eqn{\mathbf{D}^{-1}\mathbf{A}\mathbf{X}^{(l-1)}\boldsymbol{\Theta}^{(l)} + \mathbf{X}^{(l-1)}\boldsymbol{\Phi}^{(l)}}
#' 2. **Region-specific modulation**: Element-wise multiplication by \eqn{\boldsymbol{\Omega}_{reg}} and addition of \eqn{\boldsymbol{\Psi}_{reg}}
#'
#' Parameters:
#' - \eqn{\Theta} (theta): `in_features x out_features` transforms aggregated neighbor features (global)
#' - \eqn{\Phi} (phi): `in_features x out_features` transforms node's own features (global)
#' - \eqn{\Omega_{reg}} (omega_reg): `n_regions x out_features` region-specific weight modulation
#' - \eqn{\Psi_{reg}} (psi_reg): `n_regions x out_features` region-specific bias terms
#'
#' @details
#' The RegConv layer is designed for two-stage training:
#'
#' **Stage 1**: Train a global GCN to learn \eqn{\Theta} and \eqn{\Phi}, then freeze these parameters.
#'
#' **Stage 2**: Initialize \eqn{\Omega_{reg}} to all 1s and train region-specific parameters
#' (\eqn{\Omega_{reg}}, \eqn{\Psi_{reg}}) while keeping \eqn{\Theta} and \eqn{\Phi} fixed.
#'
#' The region-specific parameters allow the model to adjust predictions differently across
#' spatial regimes, enabling the model to capture spatial heterogeneity.
#'
#' @param in_features Integer. Number of input features per node
#' @param out_features Integer. Number of output features per node
#' @param n_regions Integer. Number of spatial regions/regimes
#'
#' @section Forward pass:
#' `layer(x, adj, region_assignments, edge_weight = NULL)`
#'
#' - `x`: Tensor `n_nodes x in_features`. Node feature matrix.
#' - `adj`: Sparse COO tensor `n_nodes x n_nodes`. Adjacency matrix, expected
#'   to be row-normalized \eqn{D^{-1}A} where \eqn{D} is the degree matrix.
#'   Can be binary or weighted.
#' - `region_assignments`: Tensor `n_nodes`. Integer vector with values in
#'   `1:n_regions` giving the region each node belongs to. Multiple nodes can
#'   belong to the same region.
#' - `edge_weight`: Tensor `n_nodes x n_nodes` or `NULL`. Optional edge
#'   weights applied to the adjacency matrix. If `NULL`, the values of `adj`
#'   are used.
#'
#' @return Tensor `n_nodes x out_features`. Transformed node features (before activation)
#'
#' @references
#' Guo, H., Wang, H., Zhu, D., Wu, L., Fotheringham, A. S., & Liu, Y. (2025).
#' RegionGCN: Spatial-Heterogeneity-Aware Graph Convolutional Networks. Annals
#' of the American Association of Geographers, 1–17.
#' <doi:10.1080/24694452.2025.2558661>
#'
#' @examplesIf torch::torch_is_installed()
#' adj <- adj_from_edgelist(from = c(1, 2, 3, 4), to = c(2, 3, 4, 1))
#' x <- torch::torch_randn(4, 8)
#'
#' # This layer expects a row-normalized adjacency matrix
#' adj_norm <- adj_row_normalize(add_graph_self_loops(adj))
#'
#' # Four nodes split across two spatial regimes
#' regions <- torch::torch_tensor(c(1, 1, 2, 2), dtype = torch::torch_long())
#'
#' layer <- layer_regconv(8, 4, n_regions = 2)
#' layer(x, adj_norm, regions)
#' @export
layer_regconv <- nn_module(
  "RegConvLayer",

  initialize = function(in_features, out_features, n_regions) {
    # Global parameters (same as base GCN)
    self$theta <- nn_linear(in_features, out_features, bias = FALSE)
    self$phi <- nn_linear(in_features, out_features, bias = FALSE)

    # Region-specific parameters
    # omega_reg: initialized to 1 (no modulation initially)
    self$omega_reg <- nn_parameter(torch_ones(n_regions, out_features))
    # psi_reg: initialized to 0 (no bias initially)
    self$psi_reg <- nn_parameter(torch_zeros(n_regions, out_features))
  },

  forward = function(x, adj, region_assignments, edge_weight = NULL) {
    # Apply edge weights if provided
    if (!is.null(edge_weight)) {
      adj <- adj * edge_weight
    }

    # Stage 1: Base GCN transformation (global parameters)
    # Compute: D^(-1)AX * Theta + X * Phi
    neighbor_agg <- torch_mm(adj, x)
    neighbor_trans <- self$theta(neighbor_agg) # theta: neighbor transformation
    self_trans <- self$phi(x) # phi: self transformation
    base_out <- neighbor_trans + self_trans

    # Stage 2: Region-specific modulation
    # Get region-specific parameters for each node
    omega <- self$omega_reg[region_assignments, ] # omega: region weight modulation
    psi <- self$psi_reg[region_assignments, ] # psi: region bias term

    # Apply: (base_out) * Omega_reg + Psi_reg
    base_out * omega + psi
  }
)
