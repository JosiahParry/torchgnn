# Normalize an Adjacency Matrix

Prepare a sparse adjacency matrix for message passing.

- `gcn_normalize()` applies the symmetric normalization \\D^{-1/2} A
  D^{-1/2}\\ of Kipf and Welling (2017).

- `adj_row_normalize()` applies the row normalization \\D^{-1} A\\, so
  that the weights of each node's neighbors sum to one.

- `add_graph_self_loops()` replaces any existing diagonal entries with
  unit self-loops, giving \\A + I\\.

Isolated nodes have degree zero. Their normalization factor is set to
zero rather than being allowed to diverge, so they contribute nothing to
the aggregation.

## Usage

``` r
gcn_normalize(adj)

adj_row_normalize(adj)

add_graph_self_loops(adj)
```

## Arguments

- adj:

  Sparse COO `torch_tensor` `n_nodes x n_nodes`. The adjacency matrix,
  which may be weighted.

## Value

A coalesced sparse COO `torch_tensor` of the same dimension as `adj`.

## References

Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with
graph convolutional networks. International Conference on Learning
Representations. <doi:10.48550/arXiv.1609.02907>

## Examples

``` r
if (FALSE) { # torch::torch_is_installed()
adj <- adj_from_edgelist(from = c(1, 2, 3), to = c(2, 3, 1))

gcn_normalize(adj)

adj_row_normalize(adj)

# The usual pre-processing for a GCN layer
gcn_normalize(add_graph_self_loops(adj))
}
```
