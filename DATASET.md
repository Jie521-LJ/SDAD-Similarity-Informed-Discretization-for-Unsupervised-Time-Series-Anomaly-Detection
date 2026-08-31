# Dataset Included in This Code Package

This package includes the SMD machine files needed to reproduce the reported validation:

`machine-1-1`, `machine-1-2`, `machine-1-3`, `machine-1-4`, `machine-1-5`,
`machine-1-6`, `machine-1-7`, `machine-2-1`, `machine-2-2`, `machine-2-3`.

Data layout:

```text
repos_paper/TranAD/data/SMD/
  train/
  test/
  labels/
```

The training split is used for unsupervised model fitting. Test labels are used only for metric evaluation.

## Data Row Example

Each data row is a comma-separated multivariate time step. For `machine-1-1`, each row contains `38` variables. The label file contains one binary value per test time step.

Train sample from `train/machine-1-1.txt`:

```text
0.032258,0.039195,0.027871,0.024390,0.000000,0.915385,0.343691,0.000000,0.020011,0.000122,0.106312,0.081081,0.027397,0.060266,0.085018,0.122516,0.000000,0.000000,0.062195,0.041221,0.043242,0.031607,0.533195,0.010224,0.011195,0.009274,0.000000,0.036625,0.000000,0.004298,0.029993,0.022131,0.000000,0.000045,0.034677,0.034747,0.000000,0.000000
```

Test sample from `test/machine-1-1.txt`:

```text
0.075269,0.065678,0.070234,0.074332,0.000000,0.933333,0.274011,0.000000,0.031081,0.000000,0.134132,0.081081,0.027397,0.067808,0.125842,0.150562,0.000000,0.000000,0.121988,0.091978,0.093960,0.074155,0.935405,0.018077,0.032010,0.016584,0.000000,0.077702,0.000000,0.008596,0.068036,0.048893,0.000386,0.000034,0.064432,0.064500,0.000000,0.000000
```

Label sample from `labels/machine-1-1.txt`:

```text
0
```
