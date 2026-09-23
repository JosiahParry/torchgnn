#' Multi-layer Generalized GCN Model (Hamilton 2020)
#'
#' Stacks multiple [layer_gcn_general()] layers, each of which keeps separate
#' weights for a node's own features and for its aggregated neighbors.
#'
#' @details
#' Architecture:
#' - L hidden generalized GCN layers with configurable activation
#' - 1 output layer with optional output activation
#' - Total layers = length(hidden_dims) + 1
#'
#' When `normalize = TRUE`, self-loops are added and the adjacency matrix is
#' row-normalized once in the forward pass, before any layer is applied.
#'
#' @param in_features Integer. Number of input features per node
#' @param hidden_dims Integer vector. Dimensions of hidden layers (length = L)
#' @param out_features Integer. Number of output features (typically 1 for regression)
#' @param activation Function. Activation for hidden layers. Default: nnf_relu
#' @param out_activation Function or NULL. Activation for output layer. Default: NULL
#' @param dropout Numeric. Dropout rate (0-1) applied after each hidden layer. Default: 0
#' @param normalize Logical. Whether to add self-loops and row-normalize the
#'   adjacency matrix. Default: TRUE
#'
#' @section Forward pass:
#' `model(x, adj)`
#'
#' - `x`: Tensor `n_nodes x in_features`. Node feature matrix.
#' - `adj`: Sparse COO tensor `n_nodes x n_nodes`. Adjacency matrix defining
#'   graph structure.
#'
#' @return Tensor `n_nodes x out_features`. Final predictions
#'
#' @references
#' Hamilton, W. L. (2020). Graph Representation Learning. In Synthesis Lectures
#' on Artificial Intelligence and Machine Learning. Springer International
#' Publishing. <doi:10.1007/978-3-031-01588-5>
#'
#' @examplesIf torch::torch_is_installed()
#' adj <- adj_from_edgelist(from = c(1, 2, 3, 4), to = c(2, 3, 4, 1))
#' x <- torch::torch_randn(4, 14)
#'
#' model <- model_gcn_general(14, c(32, 16), 1)
#' model(x, adj)
#'
#' # Supply a pre-normalized adjacency matrix instead
#' adj_norm <- adj_row_normalize(add_graph_self_loops(adj))
#' model <- model_gcn_general(14, c(32, 16), 1, normalize = FALSE)
#' model(x, adj_norm)
#' @export
model_gcn_general <- nn_module(
  "GeneralizedGCNModel",

  initialize = function(
    in_features,
    hidden_dims,
    out_features,
    activation = nnf_relu,
    out_activation = NULL,
    dropout = 0,
    normalize = TRUE
  ) {
    layers <- list()

    # Input to first hidden layer
    layers[[1]] <- layer_gcn_general(in_features, hidden_dims[1])
    layers[[1]]$normalize <- normalize

    # Additional hidden layers
    if (length(hidden_dims) > 1) {
      for (i in 2:length(hidden_dims)) {
        layers[[i]] <- layer_gcn_general(hidden_dims[i - 1], hidden_dims[i])
        layers[[i]]$normalize <- normalize
      }
    }

    # Output layer
    layers[[length(layers) + 1]] <- layer_gcn_general(
      hidden_dims[length(hidden_dims)],
      out_features
    )
    layers[[length(layers)]]$normalize <- normalize

    self$layers <- nn_module_list(layers)
    self$activation <- activation
    self$out_activation <- out_activation
    self$dropout_rate <- dropout
  },

  forward = function(x, adj) {
    if (self$layers[[1]]$normalize) {
      adj <- add_graph_self_loops(adj) |> adj_row_normalize()
    }

    for (i in seq_along(self$layers)) {
      x <- self$layers[[i]](x, adj)

      if (i < length(self$layers)) {
        x <- self$activation(x)
        if (self$training && self$dropout_rate > 0) {
          x <- nnf_dropout(x, p = self$dropout_rate)
        }
      } else if (!is.null(self$out_activation)) {
        x <- self$out_activation(x)
      }
    }
    x
  }
)
