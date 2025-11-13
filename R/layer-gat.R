#' Graph Attention Network Layer (Veličković et al. 2018)
#'
#' @description
#' Implements the Graph Attention Network (GAT) layer:
#'
#' \deqn{\mathbf{h}_i^{(l+1)} = \sigma\left(\sum_{j \in \mathcal{N}(i)} \alpha_{ij} \mathbf{W}^{(l)} \mathbf{h}_j^{(l)}\right)}
#'
#' where the attention coefficients \eqn{\alpha_{ij}} are computed as:
#'
#' \deqn{\alpha_{ij} = \frac{\exp(\text{LeakyReLU}(\mathbf{a}^T [\mathbf{W}\mathbf{h}_i || \mathbf{W}\mathbf{h}_j]))}{\sum_{k \in \mathcal{N}(i)} \exp(\text{LeakyReLU}(\mathbf{a}^T [\mathbf{W}\mathbf{h}_i || \mathbf{W}\mathbf{h}_k]))}}
#'
#' This layer:
#' 1. Applies linear transformation to node features
#' 2. Computes attention coefficients for each edge
#' 3. Normalizes attention weights via softmax over neighbors
#' 4. Aggregates neighbor features weighted by attention
#'
#' Parameters:
#' - \eqn{W}: `in_features x out_features` learnable weight matrix
#' - \eqn{a}: `2 * out_features` learnable attention vector
#'
#' @details
#' Multi-head attention is supported via the `heads` parameter. When `heads > 1`:
#' - If `concat = TRUE`: outputs are concatenated (output size = `out_features * heads`)
#' - If `concat = FALSE`: outputs are averaged (output size = `out_features`)
#'
#' @param in_features Integer. Number of input features per node
#' @param out_features Integer. Number of output features per node (per head)
#' @param heads Integer. Number of attention heads. Default: 1
#' @param concat Logical. If TRUE, concatenate multi-head outputs. If FALSE, average them.
#'   Default: TRUE
#' @param dropout Numeric. Dropout rate (0-1) applied to attention coefficients. Default: 0
#' @param negative_slope Numeric. Negative slope for LeakyReLU. Default: 0.2
#' @param bias Logical. Add learnable bias. Default: TRUE
#'
#' @section Forward pass:
#' @param x Tensor `n_nodes x in_features`. Node feature matrix
#' @param adj Sparse torch tensor `n_nodes x n_nodes`. Adjacency matrix defining graph
#'   structure. Must be a sparse COO tensor.
#'
#' @return Tensor `n_nodes x (out_features * heads)` if concat=TRUE, else `n_nodes x out_features`
#'
#' @references
#' Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., & Bengio, Y. (2018).
#' Graph Attention Networks. International Conference on Learning Representations.
#' <doi:10.48550/arXiv.1710.10903>
#' @export
gat_layer <- nn_module(
  "GATLayer",

  initialize = function(
    in_features,
    out_features,
    heads = 1,
    concat = TRUE,
    dropout = 0,
    negative_slope = 0.2,
    bias = TRUE
  ) {
    self$in_features <- in_features
    self$out_features <- out_features
    self$heads <- heads
    self$concat <- concat
    self$dropout_rate <- dropout
    self$negative_slope <- negative_slope

    self$weight <- nn_parameter(torch_randn(heads, in_features, out_features))
    nn_init_xavier_uniform_(self$weight)

    self$att <- nn_parameter(torch_randn(heads, 2 * out_features, 1))
    nn_init_xavier_uniform_(self$att)

    if (bias) {
      if (concat) {
        self$bias <- nn_parameter(torch_zeros(heads * out_features))
      } else {
        self$bias <- nn_parameter(torch_zeros(out_features))
      }
    } else {
      self$bias <- NULL
    }
  },

  forward = function(x, adj) {
    n_nodes <- x$size(1)
    n_features <- self$out_features
    idx <- adj$indices() + 1L

    source_nodes <- idx[1, ]
    target_nodes <- idx[2, ]

    multi_head_output <- list()

    for (h in 1:self$heads) {
      h_transformed <- torch_mm(x, self$weight[h, , ])

      edge_h_i <- h_transformed[source_nodes, ]
      edge_h_j <- h_transformed[target_nodes, ]

      edge_features <- torch_cat(list(edge_h_i, edge_h_j), dim = 2)

      edge_attention <- torch_mm(edge_features, self$att[h, , ])$squeeze(2)
      edge_attention <- nnf_leaky_relu(
        edge_attention,
        negative_slope = self$negative_slope
      )

      attention_exp <- edge_attention$exp()

      attention_sum <- torch_zeros(n_nodes)
      attention_sum$scatter_add_(1, source_nodes, attention_exp)

      attention_norm <- attention_exp / attention_sum[source_nodes]

      if (self$training && self$dropout_rate > 0) {
        attention_norm <- nnf_dropout(attention_norm, p = self$dropout_rate)
      }

      weighted_features <- h_transformed[target_nodes, ] *
        attention_norm$unsqueeze(2)

      h_prime <- torch_zeros(n_nodes, n_features)
      index_expanded <- source_nodes$unsqueeze(2)$expand(c(-1, n_features))
      h_prime$scatter_add_(1, index_expanded, weighted_features)

      multi_head_output[[h]] <- h_prime
    }

    if (self$concat) {
      output <- torch_cat(multi_head_output, dim = 2)
    } else {
      output <- torch_stack(multi_head_output, dim = 1)$mean(dim = 1)
    }

    if (!is.null(self$bias)) {
      output <- output + self$bias
    }

    output
  }
)
