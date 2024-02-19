import numpy as np
import patsy
import pandas
import scipy.optimize

class CorrReg:
    ''' Results of a Correlation Regression

    Correlation regression is performed on two dependent variables and any
    number of independent regressor variables. The correlation between the two
    dependent variables (called y1 and y2) is estimated through REML.

    The model is composed of three parts:
     - two linear models for the values of y1 and y2
     - two linear models for the values of log(sigma_y1), log(sigma_y2),
       which are the log standard deviations of the residuals of y1 and y2
     - a linear model for the value of arctanh(rho) where rho
       is the correlation of y1 and y2

    This model allows the correlation rho and the standard deviation sigma_y1 and sigma_y2 to
    vary as a function of the regressor variables.
    These models are all specified as in the standard formula notation familiar from R and python
    libraries like statsmodels, see the patsy library for details. Variables are evaluated in the
    provided `data` module.

    Attributes:
        data: the datatable that contains all the dependent and independent variables for use in the formulas
        y1: name of the variable in data to use as y1
        y2: name of the variable in data to use as y2
        mean_model: 1-sided formula specifying the linear model to use for both y1 and y2
        variance_model: 1-sided formula specifying the linear model of the (logs of) the standard deviations of y1 and y2
        corr_model: 1-sided formula specifying the linear model of the (arctanh of) the correlation between y1 and y2
    '''
    def __init__(
            self,
            data,
            y1:str,
            y2:str,
            mean_model: str,
            variance_model: str,
            corr_model: str,
        ):
        self.data = data
        self.dependent_data = np.vstack((
            np.asarray(self.data[y1]),
            np.asarray(self.data[y2]),
        ))
        self.y1 = y1
        self.y2 = y2

        self.mean_model = mean_model
        self.mean_model_dmat = patsy.dmatrix(self.mean_model, self.data)

        self.variance_model = variance_model
        self.variance_model_dmat = patsy.dmatrix(self.variance_model, self.data)

        self.corr_model = corr_model
        self.corr_model_dmat = patsy.dmatrix(self.corr_model, self.data)

    def reml_loglikelihood(self, cov) -> float:
        ''' Log-likelihood of the given correlation parameters after restricting to ReML

        See "Bayesian inference for variance components using only error contrasts" Harville 1974
        https://www.jstor.org/stable/2334370
        and
        https://xiuming.info/docs/tutorials/reml.pdf
        '''

        # Extract the covariance terms
        rho, sigma_y1, sigma_y2 = cov
        N_samples = rho.shape[0]

        # Number of predictors for mean model
        k = self.mean_model_dmat.shape[1]

        # Covariance matrix of y1 and y2
        H = np.moveaxis(
            np.array([[sigma_y1**2, sigma_y1*sigma_y2*rho],
                       [sigma_y1*sigma_y2*rho, sigma_y2**2]]),
            2,
            0
        )
        # Inverse correlation matrix - gives distance metric for the residuals
        H_inv = np.linalg.inv(H)

        # X is the matrix of independent regressor values
        X = np.array(self.mean_model_dmat)
        # Inner product of X with itself using H^-1 as the metric
        XHiX = np.einsum("ij,ik,ilm->jklm", X, X, H_inv) # k x k x 2 x 2
        XHiX = np.block([ # 2k x 2k
            [XHiX[:,:,0,0], XHiX[:,:,0,1]],
            [XHiX[:,:,1,0], XHiX[:,:,1,1]],
        ])
        XHiX_inv = np.linalg.inv(XHiX) # 2k x 2k
        XHiX_inv = np.array([  # 2 x 2 x k x k
            [XHiX_inv[:k,:k], XHiX_inv[:k, k:]],
            [XHiX_inv[k:,:k], XHiX_inv[k:, k:]],
        ])

        # Estimate the parameters of the models for the linear model of y1 and y2 (mean value models)
        # using the specified covariance matrix
        #G = H_inv @ X @ XHiX_inv
        G = np.einsum("ijk,il,kslm->ijm", H_inv, X, XHiX_inv)
        beta_hat = np.moveaxis(G, 0, 2) @ self.dependent_data[:, :, None]

        # Compute fit values and residual size
        Y_hat = np.einsum("ij,kjl->ki", X ,beta_hat)
        resid = (self.dependent_data - Y_hat) # Y - Y_hat
        mahalanobis = np.einsum("ji,ijk,ki->i", resid, H_inv, resid) # (Y - Y_hat)^T @ H_inv @ (Y - Y_hat)

        # Compute the full REML log likelihood
        # Normalize by the number of samples to make convergence criteria more consistent across studies
        log_like = -1/2*(
            np.sum(np.log(np.linalg.det(H)), axis=0)
            + np.log(np.linalg.det(XHiX))
            + np.sum(mahalanobis, axis=0)
        ) / N_samples
        return log_like

    def params_to_cov(self, params):
        ''' Converts parameters to cov components

        Returns: rho (correlation), sigma_y1 and sigma_y2 (variances)
        '''
        # Separate out the parts of the params
        param_part_lengths = [self.variance_model_dmat.shape[1], self.variance_model_dmat.shape[1], self.corr_model_dmat.shape[1]]
        y1_variance_params, y2_variance_params, corr_params = split_array(params, param_part_lengths)

        # The three covariance parameters (correlation and two standard deviations)
        rho = np.tanh(self.corr_model_dmat @ corr_params)
        sigma_y1 = np.exp(self.variance_model_dmat @ y1_variance_params)
        sigma_y2 = np.exp(self.variance_model_dmat @ y2_variance_params)

        return (rho, sigma_y1, sigma_y2)

    def objective(self, params):
        '''
        Computes the objective function to be minimized for given values of `params`
        '''
        cov = self.params_to_cov(params)
        return -self.reml_loglikelihood(cov)

    def fit(self):
        '''Run the REML fit to find the best parameters

        Updates the values in `self` to reflect the fit.
        Afterwards, `self.params` contains the fit parameters for the covariance matrix.
        '''

        # Initial guess for parameters
        init_params = np.concatenate((
            np.zeros(self.variance_model_dmat.shape[1]), # y1 variance
            np.zeros(self.variance_model_dmat.shape[1]), # y2 variance
            np.zeros(self.corr_model_dmat.shape[1]),
        ))

        # Perform the optimization
        # minimizing -loglikelihood (maximizing loglikelihood)
        def val(*args):
            val = self.objective(*args)
            return val
        res = scipy.optimize.minimize(
            fun = val,
            x0 = init_params,
            method = "BFGS",
            tol = 1e-2,
        )

        # Extract the parameters and error value
        self.params = res.x
        self.loglikelihood = res.fun
        self.opt_result = res
        return self

    def summary(self) -> str:
        ''' Return a summary table displaying the result '''
        param_part_lengths = [self.variance_model_dmat.shape[1], self.variance_model_dmat.shape[1], self.corr_model_dmat.shape[1]]
        sigma_y1, sigma_y2, rho = split_array(self.params, param_part_lengths)
        lines = ['Correlation Regression REML Results',
            "--------",
            f"Dependent variables: y1 = {self.y1}   y2 = {self.y2}",
            f"Mean model: y_i ~ {self.mean_model}",
            f"Variance model: log(SD(y_i)) ~ {self.mean_model}",
            f"Correlation model: arctanh(rho) ~ {self.mean_model}",
            "--------",
            self.opt_result.message,
            "--------",
            "Standard deviations model:",
            pandas.DataFrame([(name, val1, val2)
                    for name, val1, val2 in zip(self.variance_model_dmat.design_info.column_names, sigma_y1, sigma_y2)],
                columns = ["Variable", self.y1, self.y2])
                .to_string(index=False),
            "--------",
            "Correlation model:",
            pandas.DataFrame([(name, val)
                    for name, val in zip(self.corr_model_dmat.design_info.column_names, rho)],
                columns = ["Variable", "Rho"])
                .to_string(index=False),
        ]
        return '\n'.join(lines)


def split_array(array, lengths):
    ''' For a given array x, split the first axis into pieces of the given length '''
    assert array.shape[0] == sum(lengths)
    start = 0
    for length in lengths:
        yield array[start:start+length]
        start += length


from numpy import cos, sin, exp, tanh
rng = np.random.default_rng(1)
N_SAMPLES = 500
T = np.linspace(0.0, 2*np.pi, N_SAMPLES)
def true_cov(t):
    rho = tanh(0.3-0.3*cos(t)) # correlation
    sigma_y1 = exp(0.5*sin(t))
    sigma_y2 = exp(0.5)
    return np.array([[sigma_y1**2, sigma_y1*sigma_y2*rho], [sigma_y1*sigma_y2*rho, sigma_y2**2]])
test_data = np.array([rng.multivariate_normal( [5+sin(time), 3+cos(time)], true_cov(time)) for time in T])
df = pandas.DataFrame(dict(
    y1 = test_data[:,0],
    y2 = test_data[:,1],
    t = T,
))

cr = CorrReg(
    data = df,
    y1 = "y1",
    y2 = "y2",
    mean_model = "cos(t) + sin(t)",
    variance_model = "cos(t) + sin(t)",
    corr_model = "cos(t) + sin(t)",
).fit()
print(cr.opt_result)