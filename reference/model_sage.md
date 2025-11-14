# Multi-layer GraphSAGE Model (Hamilton et al. 2017)

Stacks multiple GraphSAGE layers with configurable aggregation
functions.

## Usage

``` r
model_sage(
  in_features,
  hidden_dims,
  out_features,
  aggregator = MeanAggregator(),
  activation = nnf_relu,
  out_activation = NULL,
  dropout = 0,
  concat = TRUE
)
```

## Arguments

- in_features:

  Integer. Number of input features per node

- hidden_dims:

  Integer vector. Dimensions of hidden layers (length = L)

- out_features:

  Integer. Number of output features (typically 1 for regression)

- aggregator:

  Aggregator S7 object. Aggregation function for all layers. Default:
  [`MeanAggregator()`](https://josiahparry.github.io/torchgnn/reference/aggregator.md)

- activation:

  Function. Activation for hidden layers. Default: nnf_relu

- out_activation:

  Function or NULL. Activation for output layer. Default: NULL

- dropout:

  Numeric. Dropout rate (0-1) applied after each hidden layer. Default:
  0

- concat:

  Logical. If TRUE, concatenates self and neighbor features. If FALSE,
  adds them. Default: TRUE

- x:

  Tensor `n_nodes x in_features`. Node feature matrix (dense or sparse)

- adj:

  Sparse torch tensor `n_nodes x n_nodes`. Adjacency matrix defining
  graph structure. Must be a sparse COO tensor.

## Value

Tensor `n_nodes x out_features`. Final predictions

## Details

Architecture:

- L hidden SAGE layers with configurable activation

- 1 output SAGE layer with optional output activation

- Total layers = length(hidden_dims) + 1

Each layer aggregates neighbor features using the specified aggregator,
then combines with self features via concatenation or addition.

## Forward pass

## References

Hamilton, W., Ying, Z., & Leskovec, J. (2017). Inductive representation
learning on large graphs. Advances in Neural Information Processing
Systems, 30. <doi:10.48550/arXiv.1706.02216>

## Examples

``` r
if (FALSE) { # \dontrun{
# Binary classification with sigmoid and mean aggregation
model <- model_sage(14, c(56, 56), 1, output_activation = nnf_sigmoid)

# Multi-class with softmax and max aggregation
model <- model_sage(
  14,
  c(32, 32),
  10,
  aggregator = MaxAggregator(),
  output_activation = function(x) nnf_softmax(x, dim = -1)
)

# Regression with sum aggregation
model <- model_sage(14, c(64, 64), 1, aggregator = SumAggregator())

# With dropout and custom activation
model <- model_sage(
  14,
  c(56, 56),
  1,
  activation = nnf_tanh,
  dropout = 0.5
)
} # }
```
