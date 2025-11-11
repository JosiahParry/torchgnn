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
gcn_layer(in_features, out_features, bias = TRUE)
```

## Arguments

- in_features:

  Integer. Number of input features per node

- out_features:

  Integer. Number of output features per node

- bias:

  Logical. Add learnable bias term (\\\Psi\\). Default: TRUE

- x:

  Tensor `n_nodes x in_features`. Node feature matrix

- adj:

  Tensor `n_nodes x n_nodes`. Adjacency matrix. Expected to be
  row-normalized \\D^{-1}A\\ where \\D\\ is the degree matrix. Can be
  binary or weighted. This layer does NOT perform normalization
  internally

- edge_weight:

  Tensor `n_nodes x n_nodes` or NULL. Optional edge weights to apply to
  the adjacency matrix. If NULL, uses values from `adj`. Default: NULL

## Value

Tensor `n_nodes x out_features`. Transformed node features (before
activation)

## Details

The adjacency matrix is expected to be row-normalized \\D^{-1}A\\ where
\\D\\ is the degree matrix. This layer does NOT perform normalization
internally.

## Forward pass

## References

Hamilton, W. L. (2020). Graph Representation Learning. In Synthesis
Lectures on Artificial Intelligence and Machine Learning. Springer
International Publishing. <doi:10.1007/978-3-031-01588-5>

Guo, H., Wang, H., Zhu, D., Wu, L., Fotheringham, A. S., & Liu, Y.
(2025). RegionGCN: Spatial-Heterogeneity-Aware Graph Convolutional
Networks. Annals of the American Association of Geographers, 1–17.
<doi:10.1080/24694452.2025.2558661>
