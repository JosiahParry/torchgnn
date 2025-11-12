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

Sparse COO tensor with attributes `node_ids` and `id_map`
