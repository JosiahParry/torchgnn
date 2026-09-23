# Regional GCN Convolutional Layer (Guo et al. 2025)

Implements the Regional Graph Convolutional Network (RegConv) layer from
Guo et al. (2025):

\$\$\mathbf{X}^{(l)} =
\sigma\left(\left(\mathbf{D}^{-1}\mathbf{A}\mathbf{X}^{(l-1)}\boldsymbol{\Theta}^{(l)} +
\mathbf{X}^{(l-1)}\boldsymbol{\Phi}^{(l)}\right)\boldsymbol{\Omega}\_{reg}^{(l)} +
\boldsymbol{\Psi}\_{reg}^{(l)}\right)\$\$

This layer extends the standard GCN layer by introducing region-specific
parameters to handle spatial heterogeneity (spatial regimes). The
computation has two stages:

1.  **Base GCN transformation**:
    \\\mathbf{D}^{-1}\mathbf{A}\mathbf{X}^{(l-1)}\boldsymbol{\Theta}^{(l)} +
    \mathbf{X}^{(l-1)}\boldsymbol{\Phi}^{(l)}\\

2.  **Region-specific modulation**: Element-wise multiplication by
    \\\boldsymbol{\Omega}\_{reg}\\ and addition of
    \\\boldsymbol{\Psi}\_{reg}\\

Parameters:

- \\\Theta\\ (theta): `in_features x out_features` transforms aggregated
  neighbor features (global)

- \\\Phi\\ (phi): `in_features x out_features` transforms node's own
  features (global)

- \\\Omega\_{reg}\\ (omega_reg): `n_regions x out_features`
  region-specific weight modulation

- \\\Psi\_{reg}\\ (psi_reg): `n_regions x out_features` region-specific
  bias terms

## Usage

``` r
layer_regconv(in_features, out_features, n_regions)
```

## Arguments

- in_features:

  Integer. Number of input features per node

- out_features:

  Integer. Number of output features per node

- n_regions:

  Integer. Number of spatial regions/regimes

## Value

Tensor `n_nodes x out_features`. Transformed node features (before
activation)

## Details

The RegConv layer is designed for two-stage training:

**Stage 1**: Train a global GCN to learn \\\Theta\\ and \\\Phi\\, then
freeze these parameters.

**Stage 2**: Initialize \\\Omega\_{reg}\\ to all 1s and train
region-specific parameters (\\\Omega\_{reg}\\, \\\Psi\_{reg}\\) while
keeping \\\Theta\\ and \\\Phi\\ fixed.

The region-specific parameters allow the model to adjust predictions
differently across spatial regimes, enabling the model to capture
spatial heterogeneity.

## Forward pass

`layer(x, adj, region_assignments, edge_weight = NULL)`

- `x`: Tensor `n_nodes x in_features`. Node feature matrix.

- `adj`: Sparse COO tensor `n_nodes x n_nodes`. Adjacency matrix,
  expected to be row-normalized \\D^{-1}A\\ where \\D\\ is the degree
  matrix. Can be binary or weighted.

- `region_assignments`: Tensor `n_nodes`. Integer vector with values in
  `1:n_regions` giving the region each node belongs to. Multiple nodes
  can belong to the same region.

- `edge_weight`: Tensor `n_nodes x n_nodes` or `NULL`. Optional edge
  weights applied to the adjacency matrix. If `NULL`, the values of
  `adj` are used.

## References

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

# Four nodes split across two spatial regimes
regions <- torch::torch_tensor(c(1, 1, 2, 2), dtype = torch::torch_long())

layer <- layer_regconv(8, 4, n_regions = 2)
layer(x, adj_norm, regions)
}
```
