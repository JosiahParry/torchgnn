# torchgnn


Graph Neural Networks for [`{torch}`](https://mlverse.github.io/torch/)
in R.

## Installation

Install the development version of the package using

``` r
pak::pak("josiahparry/torchgnn")
```

## Usage

Use `gcn_model()` to create a high level model:

``` r
library(torchgnn)

gcn_model(
  in_features = 14,
  hidden_dims = c(56, 56),
  out_features = 1,
  output_activation = torch::nnf_sigmoid
)
```

    An `nn_module` containing 8,065 parameters.

    ── Modules ─────────────────────────────────────────────────────────────────────
    • layers: <nn_module_list> #8,065 parameters

Or create individual layers using `gcn_layer()`:

``` r
gcn_layer(10, 1)
```

    An `nn_module` containing 21 parameters.

    ── Modules ─────────────────────────────────────────────────────────────────────
    • theta: <nn_linear> #10 parameters
    • phi: <nn_linear> #10 parameters

    ── Parameters ──────────────────────────────────────────────────────────────────
    • psi: Float [1:1, 1:1]
