# Generalized GCN Layer (Hamilton 2020)

Implements a single Graph Convolutional Network (GCN) layer following
Hamilton 2020:

\$\$\mathbf{H}^{(k)} =
\sigma\left(\mathbf{A}\mathbf{H}^{(k-1)}\mathbf{W}^{(k)}\_{\text{neigh}} +
\mathbf{H}^{(k-1)}\mathbf{W}^{(k)}\_{\text{self}}\right)\$\$

This can also be written as (Guo et al. 2025):

\$\$\mathbf{X}^{(l)} =
\sigma\left(\mathbf{D}^{-1}\mathbf{A}\mathbf{X}^{(l-1)}\boldsymbol{\Theta}^{(l)} +
\mathbf{X}^{(l-1)}\boldsymbol{\Phi}^{(l)} +
\boldsymbol{\Psi}^{(l)}\right)\$\$

This layer combines:

- Neighbor aggregation: \\D^{-1}AX^{(l-1)}\Theta^{(l)}\\

- Self transformation: \\X^{(l-1)}\Phi^{(l)}\\ focal node transformation

- Global bias: \\\Psi^{(l)}\\ additive bias term

Parameters:

- \\\Theta\\ (theta): `in_features x out_features` transforms aggregated
  neighbor features

- \\\Phi\\ (phi): `in_features x out_features` transforms node's own
  features

- \\\Psi\\ (psi): `out_features` global bias term (shared across all
  nodes)

## Usage

``` r
layer_gcn_general(in_features, out_features, bias = TRUE, normalize = FALSE)
```

## Arguments

- in_features:

  Integer. Number of input features per node

- out_features:

  Integer. Number of output features per node

- bias:

  Logical. Add learnable bias term (\\\Psi\\). Default: TRUE

- normalize:

  Logical. Whether to add self-loops and row-normalize the adjacency
  matrix on-the-fly. Default: FALSE

## Value

Tensor `n_nodes x out_features`. Transformed node features (before
activation)

## Details

The adjacency matrix is expected to be row-normalized \\D^{-1}A\\ where
\\D\\ is the degree matrix. This layer does NOT perform normalization
internally.

## Forward pass

`layer(x, adj)`

- `x`: Tensor `n_nodes x in_features`. Node feature matrix.

- `adj`: Sparse COO tensor `n_nodes x n_nodes`. Adjacency matrix. Unless
  `normalize = TRUE`, it is expected to be row-normalized \\D^{-1}A\\,
  where \\D\\ is the degree matrix. Can be binary or weighted.

## References

Hamilton, W. L. (2020). Graph Representation Learning. In Synthesis
Lectures on Artificial Intelligence and Machine Learning. Springer
International Publishing. <doi:10.1007/978-3-031-01588-5>

Guo, H., Wang, H., Zhu, D., Wu, L., Fotheringham, A. S., & Liu, Y.
(2025). RegionGCN: Spatial-Heterogeneity-Aware Graph Convolutional
Networks. Annals of the American Association of Geographers, 1–17.
<doi:10.1080/24694452.2025.2558661>

## Examples

``` r
if (FALSE) { # torch::torch_is_installed()
adj <- adj_from_edgelist(from = c(1, 2, 3, 4), to = c(2, 3, 4, 1))
x <- torch::torch_randn(4, 8)

# This layer expects a row-normalized adjacency matrix
adj_norm <- adj_row_normalize(add_graph_self_loops(adj))

layer <- layer_gcn_general(8, 4)
layer(x, adj_norm)
}
```
