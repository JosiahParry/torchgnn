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
#' @param x Tensor `n_nodes x in_features`. Node feature matrix (dense or sparse)
#' @param adj Sparse torch tensor `n_nodes x n_nodes`. Adjacency matrix defining graph structure.
#'   Must be a sparse COO tensor.
#'
#' @return Tensor `n_nodes x out_features`. Transformed node features
#'
#' @references
#' Hamilton, W., Ying, Z., & Leskovec, J. (2017). Inductive representation learning
#' on large graphs. Advances in Neural Information Processing Systems, 30. <doi:10.48550/arXiv.1706.02216>
#' @export
sage_layer <- nn_module(
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
