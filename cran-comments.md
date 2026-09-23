## R CMD check results

0 errors | 0 warnings | 0 notes

* This is a new release.

## Method references

The methods implemented by this package are described in the publications
cited in the `Description` field of DESCRIPTION, and in the `@references`
section of the corresponding help pages.

## Examples and tests

All examples and tests are guarded by `torch::torch_is_installed()`. The
'torch' package downloads the LibTorch runtime on first use rather than at
install time, so on a machine where that download has not happened the
guarded code is skipped rather than failing. This follows the approach used
by other packages built on 'torch', such as 'luz' and 'tabnet'.
