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
if (FALSE) { # \dontrun{
# Standard 60/20/20 split
split <- graph_split(A_sparse, seed = 42)

# Custom split (70/15/15)
split <- graph_split(A_sparse, prop = c(0.7, 0.15, 0.15))

# Two-way split (80/20 train/test)
split <- graph_split(A_sparse, prop = c(0.8, 0.2))

# Use in training
predictions <- model(X, A_sparse)
train_loss <- nnf_mse_loss(predictions[split$train_id], y[split$train_id])
} # }
```
