# VIND
#### Variational Inference using Numerical Differentiation for Non-Reparameterisable Parameters

### Dependencies
Python 3.6 with following packages
```
numpy, scipy, scikit-learn, pytorch, matplotlib, sacred, pandas, tqdm
```

For reproducing experiments, please run the following files using `python {filename}` and
subsequently plot the used figures with `python create_plots.py`. 

- linear_regression.py
- mse_grad_gamma.py
- wishart_student_normal.py
- wishart_normal_normal.py

For the stationarity test on the included data, use `R` to run `stationarity.R`.


### Data
Used standard ML benchmark data sets as well as openly accessible historical data
from [yahoo finance](https://finance.yahoo.com/). For more information on the individual
files used, see [here](data/README.md).
