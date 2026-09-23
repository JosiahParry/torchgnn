# Graph Isomorphism Network Layer (Xu et al. 2019)

Implements the Graph Isomorphism Network (GIN) layer:

\$\$\mathbf{h}\_i^{(k)} = \text{MLP}^{(k)}\left((1 + \epsilon^{(k)})
\cdot \mathbf{h}\_i^{(k-1)} + \sum\_{j \in \mathcal{N}(i)}
\mathbf{h}\_j^{(k-1)}\right)\$\$

This layer:

1.  Aggregates neighbor features via summation

2.  Adds weighted self features using learnable epsilon

3.  Applies MLP transformation

Parameters:

- MLP: Multi-layer perceptron (typically 2 layers)

- epsilon: Learnable or fixed weight for self features

## Usage

``` r
layer_gin(in_features, out_features, eps = 0, learn_eps = FALSE)
```

## Arguments

- in_features:

  Integer. Number of input features per node

- out_features:

  Integer. Number of output features per node

- eps:

  Numeric. Initial value for epsilon. Default: 0

- learn_eps:

  Logical. Whether to learn epsilon parameter. Default: FALSE

## Value

Tensor `n_nodes x out_features`. Transformed node features

## Details

The MLP is constructed as a sequence of Linear-BatchNorm-ReLU-Linear
layers. The epsilon parameter can be learned or fixed at 0.

## Forward pass

`layer(x, adj)`

- `x`: Tensor `n_nodes x in_features`. Node feature matrix.

- `adj`: Sparse COO tensor `n_nodes x n_nodes`. Adjacency matrix
  defining graph structure.

## References

Xu, K., Hu, W., Leskovec, J., & Jegelka, S. (2019). How Powerful are
Graph Neural Networks? International Conference on Learning
Representations. <doi:10.48550/arXiv.1810.00826>

## Examples

``` r
if (FALSE) { # torch::torch_is_installed()
adj <- adj_from_edgelist(from = c(1, 2, 3, 4), to = c(2, 3, 4, 1))
x <- torch::torch_randn(4, 8)

layer <- layer_gin(8, 4)
layer(x, adj)

# Learn the weight given to a node's own features
layer <- layer_gin(8, 4, learn_eps = TRUE)
layer(x, adj)
}
```
