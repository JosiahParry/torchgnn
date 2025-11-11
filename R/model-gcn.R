#' Multi-layer GCN Model
#'
#' Stacks multiple GCN layers to create a deep graph convolutional network.
#'
#' @details
#' Architecture:
#' - L hidden GCN layers with configurable activation
#' - 1 output GCN layer with optional output activation
#' - Total layers = length(hidden_dims) + 1
#'
#' For example, hidden_dims = c(56, 56) creates:
#' - Layer 1: in_features → 56 (activation)
#' - Layer 2: 56 → 56 (activation)
#' - Layer 3: 56 → out_features (output_activation)
#'
#' @param in_features Integer. Number of input features per node
#' @param hidden_dims Integer vector. Dimensions of hidden layers (length = L)
#' @param out_features Integer. Number of output features (typically 1 for regression)
#' @param activation Function. Activation for hidden layers. Default: nnf_relu
#' @param output_activation Function or NULL. Activation for output layer. Default: NULL
#' @param dropout Numeric. Dropout rate (0-1) applied after each hidden layer. Default: 0
#'
#' @section Forward pass:
#' @param x Tensor `n_nodes x in_features`. Node feature matrix
#' @param adj Tensor `n_nodes x n_nodes`. Adjacency matrix. Expected to be row-normalized
#'   \eqn{D^{-1}A} where \eqn{D} is the degree matrix. Can be binary or weighted
#' @param edge_weight Tensor `n_nodes x n_nodes` or NULL. Optional edge weights to
#'   apply to the adjacency matrix. Passed through to all layers. If NULL, uses values
#'   from \code{adj}. Default: NULL
#'
#' @return Tensor `n_nodes x out_features`. Final predictions
#'
#' @examples
#' \dontrun{
#' # Binary classification with sigmoid
#' model <- gcn_model(14, c(56, 56), 1, output_activation = nnf_sigmoid)
#'
#' # Multi-class with softmax
#' model <- gcn_model(
#'   14,
#'   c(32, 32),
#'   10,
#'   output_activation = function(x) nnf_softmax(x, dim = -1)
#' )
#'
#' # Regression (no output activation)
#' model <- gcn_model(14, c(64, 64), 1)
#'
#' # With dropout and tanh activation
#' model <- gcn_model(14, c(56, 56), 1,
#'                    activation = nnf_tanh,
#'                    dropout = 0.5)
#' }
#' @export
gcn_model <- nn_module(
  "GCN",

  initialize = function(
    in_features,
    hidden_dims,
    out_features,
    activation = nnf_relu,
    output_activation = NULL,
    dropout = 0
  ) {
    layers <- list()

    # Input to first hidden layer
    layers[[1]] <- gcn_layer(in_features, hidden_dims[1])

    # Additional hidden layers
    if (length(hidden_dims) > 1) {
      for (i in 2:length(hidden_dims)) {
        layers[[i]] <- gcn_layer(hidden_dims[i - 1], hidden_dims[i])
      }
    }

    # Output layer
    layers[[length(layers) + 1]] <- gcn_layer(
      hidden_dims[length(hidden_dims)],
      out_features
    )

    self$layers <- nn_module_list(layers)
    self$activation <- activation
    self$output_activation <- output_activation
    self$dropout_rate <- dropout
  },

  forward = function(x, adj, edge_weight = NULL) {
    for (i in seq_along(self$layers)) {
      x <- self$layers[[i]](x, adj, edge_weight)

      if (i < length(self$layers)) {
        # Hidden layer: apply activation + dropout
        x <- self$activation(x)

        if (self$training && self$dropout_rate > 0) {
          x <- nnf_dropout(x, p = self$dropout_rate)
        }
      } else {
        # Output layer: apply output activation if provided
        if (!is.null(self$output_activation)) {
          x <- self$output_activation(x)
        }
      }
    }
    x
  }
)
