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
#' Uses `gcn_conv_layer` which automatically handles adding self-loops and symmetric
#' normalization when `normalize = TRUE`.
#'
#' @param in_features Integer. Number of input features per node
#' @param hidden_dims Integer vector. Dimensions of hidden layers (length = L)
#' @param out_features Integer. Number of output features (typically 1 for regression)
#' @param activation Function. Activation for hidden layers. Default: nnf_relu
#' @param output_activation Function or NULL. Activation for output layer. Default: NULL
#' @param dropout Numeric. Dropout rate (0-1) applied after each hidden layer. Default: 0
#' @param normalize Logical. Whether to add self-loops and apply symmetric normalization.
#'   Default: TRUE
#'
#' @section Forward pass:
#' @param x Tensor `n_nodes x in_features`. Node feature matrix
#' @param adj Tensor `n_nodes x n_nodes`. Binary adjacency matrix (0/1) defining graph structure.
#'   When `normalize = TRUE`, self-loops are added and symmetric normalization is applied
#'   automatically
#' @param edge_weight Tensor `n_nodes x n_nodes` or NULL. Optional edge weights to
#'   apply to the adjacency structure. If NULL, treats all edges as having weight 1.
#'   Passed through to all layers. Default: NULL
#'
#' @return Tensor `n_nodes x out_features`. Final predictions
#'
#' @examples
#' \dontrun{
#' # Binary classification with sigmoid
#' model <- model_gcn(14, c(56, 56), 1, output_activation = nnf_sigmoid)
#'
#' # Multi-class with softmax
#' model <- model_gcn(
#'   14,
#'   c(32, 32),
#'   10,
#'   output_activation = function(x) nnf_softmax(x, dim = -1)
#' )
#'
#' # Regression (no output activation)
#' model <- model_gcn(14, c(64, 64), 1)
#'
#' # With dropout and tanh activation
#' model <- model_gcn(14, c(56, 56), 1,
#'                    activation = nnf_tanh,
#'                    dropout = 0.5)
#' }
#' @export
model_gcn <- nn_module(
  "GCNConvModel",

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
    layers[[1]] <- layer_gcn(
      in_features,
      hidden_dims[1],
      normalize = normalize
    )

    # Additional hidden layers
    if (length(hidden_dims) > 1) {
      for (i in 2:length(hidden_dims)) {
        layers[[i]] <- layer_gcn(
          hidden_dims[i - 1],
          hidden_dims[i],
          normalize = normalize
        )
      }
    }

    # Output layer
    layers[[length(layers) + 1]] <- layer_gcn(
      hidden_dims[length(hidden_dims)],
      out_features,
      normalize = normalize
    )

    self$layers <- nn_module_list(layers)
    self$activation <- activation
    self$out_activation <- out_activation
    self$dropout_rate <- dropout
  },

  forward = function(x, adj) {
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
