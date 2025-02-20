# CorrReg

Fit regression of bivariate data while fitting for correlation of the data as a function of independent variables.
Just as the value of a variable (the dependent variable) can be regressed as a function of other variables (called the independent variables),
so too one might want to regress the correlation between two dependent variables as a function of independent variables.
For example, we might hypothesize that the correlation between two measures decreases with age.
CorrReg provides a solution for that which is extremely flexible in the data it accepts: any collection of data points each which measure two dependent variables (called `y1` and `y2`) as well any number of independent variables.
In particular, when the independent variables are all distinct or have few replicate values.

CorrReg allows specifying typical regression formulas (such as `age * sex` or `cos(t) + sin(t)`) for how the mean of each `y1` and `y2` change, for how their variances change, and for how the correlation between `y1` and `y2` change.
Specifically, `log(SD(y_i))` is modelled by the given variance formula and `arctanh(rho)` is modelled by the correlation formula, where `rho` is the correlation between `y1` and `y2` and `SD(y)` is the standard deviation of the variable.
For both of these, these refer to the standard deviations and correlation of the *error term* of the variables conditional on the values of the independent variables.
For example, if we are regressing versus age, then it is quite possible that `y1` and `y2` are independent (and so have 0 correlation) even if both are increasing with age, since the correlation would be of their values at a specific age.
CorrReg is implemented using REML and assumes that these error terms are multivariate normally distributed.

## Example
``` python
import corr_reg
import pandas
import numpy as np
from numpy import cos, sin, exp, tanh, random
rng = random.default_rng(1)

# Simulate samples with a dependence on time
def true_cov(t):
    # true covariance matrix at a given value of t
    rho = tanh(0.3-0.3*cos(t)) # correlation has a cosinor dependence on time
    sigma_y1 = exp(0.5*sin(t)) # so does variance in one variable
    sigma_y2 = exp(0.5) # but not the other
    return np.array([[sigma_y1**2, sigma_y1*sigma_y2*rho], [sigma_y1*sigma_y2*rho, sigma_y2**2]])

# Generate the data for 500 different values of time
N_SAMPLES = 500
T = np.linspace(0.0, 2*np.pi, N_SAMPLES)
test_data = np.array([rng.multivariate_normal( [5+sin(time), 3+cos(time)], true_cov(time)) for time in T])
df = pandas.DataFrame(dict(
    y1 = test_data[:,0],
    y2 = test_data[:,1],
    t = T,
))

# Run the correlation model, allowing cosinor terms for all components
cr = corr_reg.CorrReg(
    data = df,
    y1 = "y1",
    y2 = "y2",
    mean_model = "cos(t) + sin(t)",
    variance_model = "cos(t) + sin(t)",
    corr_model = "cos(t) + sin(t)",
).fit()
print(cr.summary())

# Test if there is significantly worse fit when not allowing correlation to vary over time
# by first fitting a reduced model and then performing a likelihood ratio test
cr_restricted = corr_reg.CorrReg(
    data = df,
    y1 = "y1",
    y2 = "y2",
    mean_model = "cos(t) + sin(t)",
    variance_model = "cos(t) + sin(t)",
    corr_model = "1",
).fit()
print(f"P-value of the correlation model compared to the constant model:\n{cr.likelihood_ratio_test(cr_restricted)}")
```

Output:
```
Correlation Regression REML Results
--------
Dependent variables: y1 = y1   y2 = y2
Mean model: y_i ~ cos(t) + sin(t)
Variance model: log(SD(y_i)) ~ cos(t) + sin(t)
Correlation model: arctanh(rho) ~ cos(t) + sin(t)
--------
Optimization terminated successfully.
--------
Mean model:
     Term            y1         y2
Intercept      4.990096  2.9040437
   cos(t) -0.0024966225  1.0160536
   sin(t)     1.0730464 0.16497436
--------
Variance model:
     Term        y1        y2
Intercept -0.043525  0.498516
   cos(t) -0.045300 -0.018338
   sin(t)  0.304450  0.020142
--------
Correlation model:
     Term       Rho
Intercept  0.279446
   cos(t) -0.332866
   sin(t)  0.000253
P-value of the correlation model compared to the constant model:
1.1132422734390535e-06
```

## Installation

```
pip install git+https://github.com/tgbrooks/corr_reg
```

CorrReg uses `jax` for accelerating computations.
Please see the [installing JAX guide](https://docs.jax.dev/en/latest/installation.html#installation) for information about using `jax` with GPUs for more acceleration, but CorrReg also works well with just CPU acceleration.
`jax` will compile the functions at runtime and so first execution at a specific problem size and design will require compilation and will run much slower than later executions.

## Limitations

One limitations is that p-values are computed by a likelihood ratio test, which is only asymptotically correct and so is unlikely to be meaningful when there are few data points.
It is likely a good idea to supplement these p-values with alternative methods, such as permutation tests.
CorrReg is also potentially sensitive to outliers and so all results need to be examined carefully by hand.