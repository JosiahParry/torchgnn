# Create Train/Validation/Test Split for Graph Data

Creates splits for graph neural networks following rsample's structure.
Unlike traditional ML, GCNs use the full graph during training but only
compute loss on labeled (training) nodes.

## Usage

``` r
graph_split(data, prop = c(0.6, 0.2, 0.2), seed = NULL)
```

## Arguments

- data:

  Data frame or adjacency matrix. The full dataset to split.

- prop:

  Numeric vector of length 2 or 3. Proportions for splits.

  - Length 2: c(train, test) - creates train/test split

  - Length 3: c(train, val, test) - creates train/val/test split Must
    sum to 1.0. Default: c(0.6, 0.2, 0.2)

- seed:

  Integer or NULL. Random seed for reproducibility. Default: NULL

## Value

A graph_split object (list) containing:

- data: Original data

- train_id: Integer vector of training indices

- val_id: Integer vector of validation indices (or NULL if length(prop)
  == 2)

- test_id: Integer vector of test indices

## Details

The proportions must sum to 1.0. The function creates non-overlapping
splits where each row belongs to exactly one split.

For GCN training, you use:

- Full X and A_sparse for all forward passes

- IDs to select which predictions to use for loss computation

## Examples

``` r
nodes <- data.frame(id = 1:100, feature = seq(0, 1, length.out = 100))

# Standard 60/20/20 split
split <- graph_split(nodes, seed = 42)
lengths(split[c("train_id", "val_id", "test_id")])
#> train_id   val_id  test_id 
#>       60       20       20 

# Custom split (70/15/15)
graph_split(nodes, prop = c(0.7, 0.15, 0.15), seed = 42)
#> <graph_split>
#> named list [1:4] 
#> $ data    :'data.frame': 100 obs. of  2 variables:
#>  ..$ id     : int [1:100] 1 2 3 4 5 6 7 8 9 10 ...
#>  ..$ feature: num [1:100] 0 0.0101 0.0202 0.0303 0.0404 ...
#> $ train_id: int [1:70] 49 65 25 74 18 100 47 24 71 89 ...
#> $ val_id  : int [1:15] 17 32 48 14 72 23 99 57 70 97 ...
#> $ test_id : int [1:15] 62 59 75 46 1 60 19 90 77 85 ...
#> @ prop: num [1:3] 0.7 0.15 0.15

# Two-way split (80/20 train/test)
split <- graph_split(nodes, prop = c(0.8, 0.2), seed = 42)

# The identifiers select which rows contribute to the loss
head(split$train_id)
#> [1]  49  65  25  74  18 100
```
