# Multi-layer Generalized GCN Model (Hamilton 2020)

Stacks multiple
[`layer_gcn_general()`](https://josiahparry.github.io/torchgnn/reference/layer_gcn_general.md)
layers, each of which keeps separate weights for a node's own features
and for its aggregated neighbors.

## Usage

``` r
model_gcn_general(
  in_features,
  hidden_dims,
  out_features,
  activation = nnf_relu,
  out_activation = NULL,
  dropout = 0,
  normalize = TRUE
)
```

## Arguments

- in_features:

  Integer. Number of input features per node

- hidden_dims:

  Integer vector. Dimensions of hidden layers (length = L)

- out_features:

  Integer. Number of output features (typically 1 for regression)

- activation:

  Function. Activation for hidden layers. Default: nnf_relu

- out_activation:

  Function or NULL. Activation for output layer. Default: NULL

- dropout:

  Numeric. Dropout rate (0-1) applied after each hidden layer. Default:
  0

- normalize:

  Logical. Whether to add self-loops and row-normalize the adjacency
  matrix. Default: TRUE

## Value

Tensor `n_nodes x out_features`. Final predictions

## Details

Architecture:

- L hidden generalized GCN layers with configurable activation

- 1 output layer with optional output activation

- Total layers = length(hidden_dims) + 1

When `normalize = TRUE`, self-loops are added and the adjacency matrix
is row-normalized once in the forward pass, before any layer is applied.

## Forward pass

`model(x, adj)`

- `x`: Tensor `n_nodes x in_features`. Node feature matrix.

- `adj`: Sparse COO tensor `n_nodes x n_nodes`. Adjacency matrix
  defining graph structure.

## References

Hamilton, W. L. (2020). Graph Representation Learning. In Synthesis
Lectures on Artificial Intelligence and Machine Learning. Springer
International Publishing. <doi:10.1007/978-3-031-01588-5>

## Examples

``` r
if (FALSE) { # torch::torch_is_installed()
adj <- adj_from_edgelist(from = c(1, 2, 3, 4), to = c(2, 3, 4, 1))
x <- torch::torch_randn(4, 14)

model <- model_gcn_general(14, c(32, 16), 1)
model(x, adj)

# Supply a pre-normalized adjacency matrix instead
adj_norm <- adj_row_normalize(add_graph_self_loops(adj))
model <- model_gcn_general(14, c(32, 16), 1, normalize = FALSE)
model(x, adj_norm)
}
```
