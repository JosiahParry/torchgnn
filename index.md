# torchgnn

Graph Neural Networks for [`{torch}`](https://mlverse.github.io/torch/)
in R.

## Installation

Install the development version:

``` r
pak::pak("josiahparry/torchgnn")
```

## Features

The following layers are implemented

- [`gcn_conv_layer()`](https://josiahparry.github.io/torchgnn/reference/gcn_conv_layer.md):
  Standard GCN layer (Kipf & Welling, 2016)
- [`gcn_general_layer()`](https://josiahparry.github.io/torchgnn/reference/gcn_general_layer.md):
  Generalized GCN layer (Hamilton, 2020)
- [`sage_layer()`](https://josiahparry.github.io/torchgnn/reference/sage_layer.md):
  GraphSAGE layer (Hamilton, Ying, and Leskovec, 2017)
- [`regconv_layer()`](https://josiahparry.github.io/torchgnn/reference/regconv_layer.md):
  RegionConv layer for regionalized GCN (Guo et al. 2025)

### Models

[torchgnn](https://josiahparry.github.io/torchgnn/) provides utilities
to create GNN models with multiple layers.

- [`gcn_conv_model()`](https://josiahparry.github.io/torchgnn/reference/gcn_conv_model.md)
- [`gcn_general_model()`](https://josiahparry.github.io/torchgnn/reference/gcn_general_model.md)
- `model_sage()`

``` r
library(torchgnn)

gcn_general_model(
  in_features = 50,
  hidden_dims = c(32, 16),
  out_features = 1
)
```

``` R
An `nn_module` containing 4,305 parameters.

── Modules ─────────────────────────────────────────────────────────────────────
• layers: <nn_module_list> #4,305 parameters
```

### Aggregators

An S7 class
[`torchgnn::Aggregator`](https://josiahparry.github.io/torchgnn/reference/aggregator.md)
is created with a generic method with signature
`forward(x, adj, tensor)`. Only basic aggregators are implemented at
present.

- [`SumAggregator()`](https://josiahparry.github.io/torchgnn/reference/aggregator.md):
  Sum of neighbor features
- [`MeanAggregator()`](https://josiahparry.github.io/torchgnn/reference/aggregator.md):
  Mean of neighbor features (with row normalization)
- [`MaxAggregator()`](https://josiahparry.github.io/torchgnn/reference/aggregator.md):
  Element-wise maximum of neighbor features
- [`MinAggregator()`](https://josiahparry.github.io/torchgnn/reference/aggregator.md):
  Element-wise minimum of neighbor features
- [`ProductAggregator()`](https://josiahparry.github.io/torchgnn/reference/aggregator.md):
  Element-wise product of neighbor features
- [`VarAggregator()`](https://josiahparry.github.io/torchgnn/reference/aggregator.md):
  Variance of neighbor features
- [`StdAggregator()`](https://josiahparry.github.io/torchgnn/reference/aggregator.md):
  Standard deviation of neighbor features

``` r
MeanAggregator()
```

``` R
<torchgnn::MeanAggregator>
 @ name     : chr "mean"
 @ learnable: logi FALSE
```

### Data Preparation

[torchgnn](https://josiahparry.github.io/torchgnn/) works with sparse
COO torch tensors for adjacency and expects dense feature tensors. Note
that weighted is handled by incorporating it directly into the adjacency
matrix.

`adj_from_edgelist(from, to, weight)`: Create sparse adjacency matrix
from edge list (supports character IDs, too)

``` r
set.seed(0)
i <- sort(rep.int(1:10, 5))
j <- sample(1:10, 50, TRUE)

adj <- adj_from_edgelist(i, j)
adj
```

``` R
torch_tensor
[ SparseCPUFloatType{}
indices:
Columns 1 to 26 0  0  0  0  0  0  1  1  1  1  1  2  2  2  2  2  3  3  3  3  3  4  4  4  4  4
 0  1  3  5  6  8  0  1  2  4  6  1  4  5  6  9  0  4  5  6  8  1  2  3  4  8

Columns 27 to 52 4  5  5  5  5  5  5  5  5  6  6  6  6  6  6  6  6  7  7  7  7  7  8  8  8  8
 9  0  2  3  5  6  7  8  9  0  1  2  3  5  7  8  9  5  6  7  8  9  0  3  4  5

Columns 53 to 63 8  8  8  8  9  9  9  9  9  9  9
 6  7  8  9  2  4  5  6  7  8  9
[ CPULongType{2,63} ]
values:
 2
 2
 1
 1
 1
 1
 2
 2
 1
 2
 1
 1
 1
 2
 1
 3
 1
 2
... [the output was truncated (use n=-1 to disable)]
]
```

`nodes_to_tensor(nodes, adj, node_id)` will convert node features from a
dataframe to a tensor with correct ordering.

``` r
train_data <- as.data.frame(
  matrix(runif(10 * 5, max = 20), nrow = 10)
)

nodes_to_tensor(train_data, adj)
```

``` R
torch_tensor
 13.0174   9.5270  15.1417  15.5783  12.6699
  5.1603  17.8440   4.0538  15.9462   4.2642
  9.5709  17.2868  14.2224   9.1055   2.5874
 15.3262   7.7998   2.4338   8.2017   9.5624
  1.6849  15.5464   4.9098  16.2174  18.4815
 17.5064  19.2124   2.8661  12.0987  11.9752
  6.7815   8.6932   4.7926  13.0945  19.5234
 16.7888  14.2503   1.1787   7.0639  14.6358
  6.9337   7.9999  12.8458   5.4052   7.1345
  6.6755   6.5070  17.5254  19.8537   8.6295
[ CPUFloatType{10,5} ]
```

The function
[`graph_split()`](https://josiahparry.github.io/torchgnn/reference/graph_split.md)
create split **masks**. This API will likely change as how to integrate
large graphs into [torch](https://torch.mlverse.org/docs) and
[luz](https://mlverse.github.io/luz/) is figured out.

``` r
graph_split(adj)
```

``` R
<graph_split>
named list [1:4] 
$ data    :Float [1:10, 1:10]
$ train_id: int [1:6] 2 7 4 3 5 10
$ val_id  : int [1:2] 6 1
$ test_id : int [1:2] 9 8
@ prop: num [1:3] 0.6 0.2 0.2
```

## Example

Train a GCN on the PubMed Diabetes citation network for document
classification.

This example creates a GCN model

data preparation

``` r
library(torch)
library(torchgnn)
library(nanoparquet)
library(dplyr)

# read to tempfiles
# data from https://linqs.org/datasets/#pubmed-diabetes
nodes_tmp <- tempfile("nodes", fileext = ".parquet")
edges_tmp <- tempfile("edges", fileext = ".parquet")

download.file(
  "https://github.com/JosiahParry/torchgnn/raw/refs/heads/main/examples/data/pubmed-diabetes/edges.parquet",
  edges_tmp, quiet = TRUE
)
download.file(
  "https://github.com/JosiahParry/torchgnn/raw/refs/heads/main/examples/data/pubmed-diabetes/nodes.parquet",
  nodes_tmp, quiet = TRUE
)

# read in the nodes and edges
nodes <- read_parquet(nodes_tmp)
edges <- read_parquet(edges_tmp)

# create our adjacency matrix from IDs
adj <- adj_from_edgelist(edges$from, edges$to)

# create our training data tensor
X <- nodes_to_tensor(
  select(nodes, -label),
  adj,
  node_id = "paper_id"
)

# create our target class tensor
Y <- nodes_to_tensor(select(nodes, 1:2), adj, "paper_id")$to(
  dtype = torch_long()
)$squeeze()

# perform a train, test, validation split
split <- graph_split(X, seed = 42)
```

``` r
# define a 2 layer GCN model
model <- gcn_conv_model(
  # number of variables
  in_features = 500,
  # define our hidden layers
  hidden_dims = 16,
  # predicting 3 labels
  out_features = 3,
  dropout = 0.5
)
model 
```

``` R
An `nn_module` containing 8,067 parameters.

── Modules ─────────────────────────────────────────────────────────────────────
• layers: <nn_module_list> #8,067 parameters
```

``` r
# use ADAM optimizer
optimizer <- optim_adam(model$parameters, lr = 0.01)

# set our number of epochs
n_epochs <- 100

# create our training loop
for (epoch in 1:n_epochs) {
  model$train()
  optimizer$zero_grad()

  out <- model(X, adj)
  loss <- nnf_cross_entropy(out[split$train_id, ], Y[split$train_id])

  loss$backward()
  optimizer$step()

  model$eval()
  with_no_grad({
    val_out <- model(X, adj)
    val_pred <- val_out[split$val_id, ]$argmax(dim = 2)
    val_acc <- (val_pred == Y[split$val_id])$to(
      dtype = torch_float()
    )$mean()$item()
  })

  # print info every 20 epochs
  if (epoch %% 20 == 0) {
    cat(sprintf(
      "Epoch %d | Loss: %.4f | Val Acc: %.4f\n",
      epoch,
      loss$item(),
      val_acc
    ))
  }
}
```

``` R
Epoch 20 | Loss: 0.8129 | Val Acc: 0.8022
Epoch 40 | Loss: 0.5767 | Val Acc: 0.8349
Epoch 60 | Loss: 0.4758 | Val Acc: 0.8463
Epoch 80 | Loss: 0.4317 | Val Acc: 0.8590
Epoch 100 | Loss: 0.4001 | Val Acc: 0.8618
```

``` r
model$eval()
with_no_grad({
  test_out <- model(X, adj)
  test_pred <- test_out[split$test_id, ]$argmax(dim = 2)
  test_acc <- (test_pred == Y[split$test_id])$to(
    dtype = torch_float()
  )$mean()$item()
})

cat(sprintf("Test Accuracy: %.4f\n", test_acc))
```

``` R
Test Accuracy: 0.8570
```
