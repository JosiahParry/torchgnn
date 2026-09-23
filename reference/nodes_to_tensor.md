# Convert Node Features to Tensor

Convert Node Features to Tensor

## Usage

``` r
nodes_to_tensor(nodes, adj = NULL, node_id = NULL)
```

## Arguments

- nodes:

  Dataframe with node features

- adj:

  Adjacency matrix with `id_map` attribute

- node_id:

  Column name for node IDs

## Value

A dense `torch_tensor` of dimension `n_nodes x n_features`. When `adj`
and `node_id` are both supplied, rows are reordered to match the node
ordering of `adj` and the identifier column is dropped.

## Examples

``` r
if (FALSE) { # torch::torch_is_installed()
adj <- adj_from_edgelist(from = c("a", "b", "c"), to = c("b", "c", "a"))
nodes <- data.frame(id = c("c", "a", "b"), deg = c(2, 2, 2), pop = c(10, 20, 30))

nodes_to_tensor(nodes, adj, node_id = "id")
}
```
