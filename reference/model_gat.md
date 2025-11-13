# Multi-layer Graph Attention Network Model (Veličković et al. 2018)

Stacks multiple GAT layers with multi-head attention.

## Usage

``` r
model_gat(
  in_features,
  hidden_dims,
  out_features,
  heads = 8,
  out_heads = 1,
  activation = nnf_elu,
  out_activation = NULL,
  dropout = 0.6,
  att_dropout = 0.6,
  negative_slope = 0.2
)
```

## Arguments

- in_features:

  Integer. Number of input features per node

- hidden_dims:

  Integer vector. Dimensions of hidden layers (length = L)

- out_features:

  Integer. Number of output features (typically 1 for regression)

- heads:

  Integer. Number of attention heads for hidden layers. Default: 8

- out_heads:

  Integer. Number of attention heads for output layer. Default: 1

- activation:

  Function. Activation for hidden layers. Default:
  [torch::nnf_elu](https://torch.mlverse.org/docs/reference/nnf_elu.html)

- out_activation:

  Function or NULL. Activation for output layer. Default: NULL

- dropout:

  Numeric. Dropout rate (0-1) applied to attention and features.
  Default: 0.6

- att_dropout:

  Numeric. Dropout rate for attention coefficients. Default: 0.6

- negative_slope:

  Numeric. Negative slope for LeakyReLU in attention. Default: 0.2

- x:

  Tensor `n_nodes x in_features`. Node feature matrix (dense or sparse)

- adj:

  Sparse torch tensor `n_nodes x n_nodes`. Adjacency matrix defining
  graph structure. Must be a sparse COO tensor.

## Value

Tensor `n_nodes x out_features`. Final predictions

## Details

Architecture:

- L hidden GAT layers with configurable activation

- 1 output GAT layer with optional output activation

- Total layers = length(hidden_dims) + 1

Each layer uses multi-head attention to learn importance weights for
neighbor aggregation. Hidden layers typically concatenate attention
heads, while the output layer averages them.

## Forward pass

## References

Veličković P., Cucurull, G., Casanova, A., Romero, A., Li, P., & Bengio,
Y. (2018). Graph Attention Networks. International Conference on
Learning Representations. <doi:10.48550/arXiv.1710.10903>

## Examples

``` r
if (FALSE) { # \dontrun{
# Binary classification with 8-head attention
model <- gat_model(14, c(8, 8), 1, output_activation = nnf_sigmoid)

# Multi-class with 4 heads
model <- gat_model(
  14,
  c(16, 16),
  3,
  heads = 4,
  output_activation = function(x) nnf_softmax(x, dim = -1)
)

# Regression with custom dropout
model <- gat_model(14, c(32, 32), 1, dropout = 0.5, att_dropout = 0.5)
} # }
```
