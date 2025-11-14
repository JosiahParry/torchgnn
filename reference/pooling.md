# Graph Pooling Functions

Aggregate node features to graph-level representations for graph
classification. These functions reduce node embeddings within each graph
to a single vector.

## Usage

``` r
pool_global_add(x, batch = NULL, size = NULL)

pool_global_mean(x, batch = NULL, size = NULL)

pool_global_max(x, batch = NULL, size = NULL)
```

## Arguments

- x:

  Tensor. Node feature matrix with shape `(total_nodes, features)`.
  Contains features for all nodes from all graphs stacked together.

- batch:

  Tensor or NULL. Batch vector assigning each node to a graph. Values
  should be graph indices starting from 1 (e.g., `c(1,1,2,2,2)` for 2
  graphs with 2 and 3 nodes). If NULL, treats all nodes as a single
  graph.

- size:

  Integer or NULL. Number of graphs. Automatically calculated if NULL.

## Value

Tensor with shape `(num_graphs, features)` containing graph-level
embeddings.

## Details

These functions implement different reduction strategies:

- `pool_global_add`: Sum of node features per graph

- `pool_global_mean`: Mean of node features per graph

- `pool_global_max`: Element-wise maximum of node features per graph
