# Create Sparse Adjacency Matrix from Edge List

Create Sparse Adjacency Matrix from Edge List

## Usage

``` r
adj_from_edgelist(from, to, weight = NULL, n = NULL, symmetric = TRUE)
```

## Arguments

- from:

  Integer or character vector of source nodes

- to:

  Integer or character vector of target nodes

- weight:

  Numeric vector of edge weights

- n:

  Number of nodes

- symmetric:

  Make adjacency symmetric

## Value

A sparse COO `torch_tensor` of dimension `n x n`, carrying the
attributes `node_ids` (the original node identifiers, or `NULL` when
`from` and `to` are already integers) and `id_map` (a named integer
vector mapping identifiers to row and column positions, or `NULL`).

## Examples

``` r
if (FALSE) { # torch::torch_is_installed()
adj_from_edgelist(from = c(1, 2, 3), to = c(2, 3, 1))

# Character node identifiers are mapped to positions
adj <- adj_from_edgelist(
  from = c("a", "b", "c"),
  to = c("b", "c", "a"),
  weight = c(0.5, 1, 2)
)
attr(adj, "id_map")
}
```
