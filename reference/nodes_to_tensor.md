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

Dense tensor of shape `[n_nodes, n_features]`
