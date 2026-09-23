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
  concat = TRUE,
  norm = NULL
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

- norm:

  `nn_module` generator or NULL. Normalization applied after each hidden
  layer, before the activation. Called once per hidden layer with that
  layer's output dimension. Default: NULL

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

`model(x, adj, batch = NULL)`

- `x`: Tensor `n_nodes x in_features`. Node feature matrix.

- `adj`: Sparse COO tensor `n_nodes x n_nodes`. Adjacency matrix
  defining graph structure.

- `batch`: Tensor or `NULL`. Batch vector assigning each node to a
  graph, using 1-based graph indices. Passed to `norm`. If `NULL`, all
  nodes are treated as a single graph.

## References

Hamilton, W., Ying, Z., & Leskovec, J. (2017). Inductive representation
learning on large graphs. Advances in Neural Information Processing
Systems, 30. <doi:10.48550/arXiv.1706.02216>

## Examples

``` r
if (FALSE) { # torch::torch_is_installed()
adj <- adj_from_edgelist(from = c(1, 2, 3, 4), to = c(2, 3, 4, 1))
x <- torch::torch_randn(4, 14)

# Binary classification with sigmoid and mean aggregation
model <- model_sage(14, c(56, 56), 1, out_activation = torch::nnf_sigmoid)
model(x, adj)

# Multi-class with softmax and max aggregation
model <- model_sage(
  14,
  c(32, 32),
  10,
  aggregator = MaxAggregator(),
  out_activation = function(x) torch::nnf_softmax(x, dim = -1)
)
model(x, adj)

# Regression with sum aggregation
model_sage(14, c(64, 64), 1, aggregator = SumAggregator())

# With dropout and custom activation
model_sage(14, c(56, 56), 1, activation = torch::torch_tanh, dropout = 0.5)

# With normalization after each hidden layer
model <- model_sage(14, c(56, 32), 1, norm = layer_layer_norm)
model(x, adj)

model_sage(14, c(56, 32), 1, norm = \(d) layer_layer_norm(d, mode = "node"))
}
```
