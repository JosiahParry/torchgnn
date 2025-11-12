#' Multi-layer Generalized GCN Model (Hamilton 2020)
#'
#' Stacks multiple generalized GCN layers with neighbor/self weight separation.
#' Optional row-normalization is applied internally if `normalize = TRUE`.
#' @export
gcn_general_model <- nn_module(
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
    layers[[1]] <- gcn_general_layer(in_features, hidden_dims[1])
    layers[[1]]$normalize <- normalize

    # Additional hidden layers
    if (length(hidden_dims) > 1) {
      for (i in 2:length(hidden_dims)) {
        layers[[i]] <- gcn_general_layer(hidden_dims[i - 1], hidden_dims[i])
        layers[[i]]$normalize <- normalize
      }
    }

    # Output layer
    layers[[length(layers) + 1]] <- gcn_general_layer(
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
