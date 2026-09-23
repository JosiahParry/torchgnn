#' GraphSAGE Layer (Hamilton et al. 2017)
#'
#' @description
#' Implements the GraphSAGE (Graph Sample and Aggregate) layer:
#'
#' \deqn{\mathbf{h}_{\mathcal{N}(v)}^{(k)} = \text{AGGREGATE}\left(\{\mathbf{h}_u^{(k-1)} : u \in \mathcal{N}(v)\}\right)}
#'
#' \deqn{\mathbf{h}_v^{(k)} = \sigma\left(\mathbf{W}^{(k)} \cdot \text{CONCAT}\left(\mathbf{h}_v^{(k-1)}, \mathbf{h}_{\mathcal{N}(v)}^{(k)}\right)\right)}
#'
#' This layer:
#' 1. Aggregates neighbor features using the specified aggregator
#' 2. Concatenates node's own features with aggregated neighbor features
#' 3. Applies linear transformation and optional normalization
#'
#' Parameters:
#' - \eqn{W}: `(in_features + aggregated_features) x out_features` learnable weight matrix
#' - \eqn{b}: `out_features` learnable bias term (optional)
#'
#' @details
#' The aggregator should be an S7 Aggregator object (e.g., `MeanAggregator()`, `MaxAggregator()`).
#' Each aggregator is responsible for its own normalization. For example, `MeanAggregator()`
#' applies row normalization internally, while `MaxAggregator()` uses the adjacency structure
#' without normalization.
#'
#' @param in_features Integer. Number of input features per node
#' @param out_features Integer. Number of output features per node
#' @param aggregator Aggregator S7 object. Default: `MeanAggregator()`
#' @param bias Logical. Add learnable bias. Default: TRUE
#' @param concat Logical. If TRUE, concatenates self and neighbor features. If FALSE, adds them. Default: TRUE
#'
#' @section Forward pass:
#' `layer(x, adj)`
#'
#' - `x`: Tensor `n_nodes x in_features`. Node feature matrix.
#' - `adj`: Sparse COO tensor `n_nodes x n_nodes`. Adjacency matrix defining
#'   graph structure.
#'
#' @return Tensor `n_nodes x out_features`. Transformed node features
#'
#' @references
#' Hamilton, W., Ying, Z., & Leskovec, J. (2017). Inductive representation learning
#' on large graphs. Advances in Neural Information Processing Systems, 30. <doi:10.48550/arXiv.1706.02216>
#'
#' @examplesIf torch::torch_is_installed()
#' adj <- adj_from_edgelist(from = c(1, 2, 3, 4), to = c(2, 3, 4, 1))
#' x <- torch::torch_randn(4, 8)
#'
#' layer <- layer_sage(8, 4)
#' layer(x, adj)
#'
#' # Any aggregator can be supplied
#' layer <- layer_sage(8, 4, aggregator = MaxAggregator())
#' layer(x, adj)
#'
#' # Add self and neighbor features instead of concatenating them
#' layer <- layer_sage(8, 4, concat = FALSE)
#' layer(x, adj)
#' @export
layer_sage <- nn_module(
  "SAGELayer",

  initialize = function(
    in_features,
    out_features,
    aggregator = MeanAggregator(),
    bias = TRUE,
    concat = TRUE
  ) {
    if (!S7::S7_inherits(aggregator, Aggregator)) {
      cli::cli_abort(
        "{.arg aggregator} must be a {.cls torchgnn::Aggregator} see {.help torchgnn::Aggregator}"
      )
    }
    self$aggregator <- aggregator
    self$concat <- concat

    # Weight matrix size depends on concatenation
    if (concat) {
      weight_in <- in_features * 2
    } else {
      weight_in <- in_features
    }

    self$weight <- nn_parameter(torch_randn(weight_in, out_features))
    nn_init_xavier_uniform_(self$weight)

    if (bias) {
      self$bias <- nn_parameter(torch_zeros(1, out_features))
    } else {
      self$bias <- NULL
    }
  },

  forward = function(x, adj) {
    neighbor_agg <- forward(self$aggregator, adj, x)

    # Combine self and neighbor features
    if (self$concat) {
      combined <- torch_cat(list(x, neighbor_agg), dim = 2)
    } else {
      combined <- x + neighbor_agg
    }

    # Linear transformation
    out <- torch_mm(combined, self$weight)

    if (!is.null(self$bias)) {
      out <- out + self$bias
    }

    out
  }
)
