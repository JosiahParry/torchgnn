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

- x:

  Tensor `n_nodes x in_features`. Node feature matrix

- adj:

  Sparse torch tensor `n_nodes x n_nodes`. Adjacency matrix defining
  graph structure. Must be a sparse COO tensor.

## Value

Tensor `n_nodes x out_features`. Transformed node features

## Details

The MLP is constructed as a sequence of Linear-BatchNorm-ReLU-Linear
layers. The epsilon parameter can be learned or fixed at 0.

## Forward pass

## References

Xu, K., Hu, W., Leskovec, J., & Jegelka, S. (2019). How Powerful are
Graph Neural Networks? International Conference on Learning
Representations. <doi:10.48550/arXiv.1810.00826>
