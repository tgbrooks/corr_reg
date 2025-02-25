import copy
import patsy
import pandas
import scipy.optimize
import scipy.stats
import jax.numpy as np
import jax

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
        self.mean_model_dmat = patsy.dmatrix(self.mean_model, self.data, eval_env=1)
        self.mean_model_dmat_array = np.asarray(self.mean_model_dmat)

        self.variance_model = variance_model
        self.variance_model_dmat = patsy.dmatrix(self.variance_model, self.data, eval_env=1)
        self.variance_model_dmat_array = np.asarray(self.variance_model_dmat)

        self.corr_model = corr_model
        self.corr_model_dmat = patsy.dmatrix(self.corr_model, self.data, eval_env=1)
        self.corr_model_dmat_array = np.asarray(self.corr_model_dmat)

        self.params = None
        self.mean_model_params = None

    def _compute_beta_hat(self, cov=None):
        ''' Compute the estimated value of beta (the mean model parameters) for both y1 and y2

        This value depends upon the covariance matrix, by default used the fit value if any.
        '''
        if cov is None:
            cov = self._params_to_cov(self.params)

        # X is the matrix of independent regressor values
        X = self.mean_model_dmat_array
        Y = self.dependent_data
        _, _, beta_hat, _ = _compute_beta_H_xhix_jit(cov, X, Y)
        return beta_hat

    def reml_loglikelihood(self, cov) -> float:
        ''' Log-likelihood of the given correlation parameters after restricting to ReML

        See "Bayesian inference for variance components using only error contrasts" Harville 1974
        https://www.jstor.org/stable/2334370
        and
        https://xiuming.info/docs/tutorials/reml.pdf
        '''

        # X is the matrix of independent regressor values
        X = self.mean_model_dmat_array
        # Y is the matrix of dependent values
        Y = self.dependent_data

        return _reml_loglikelihood(cov, X, Y)

    def _params_to_cov(self, params):
        ''' Converts parameters to cov components

        Returns: rho (correlation), sigma_y1 and sigma_y2 (variances)
        '''
        return _params_to_cov(params, self.variance_model_dmat_array, self.corr_model_dmat_array)

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
        res = scipy.optimize.minimize(
            fun = _objective_and_grad_jit,
            x0 = init_params,
            method = "BFGS",
            tol = 1e-2,
            jac = True,
            args=(
                self.dependent_data,
                self.mean_model_dmat_array,
                self.variance_model_dmat_array,
                self.corr_model_dmat_array
            )
        )

        # Extract the parameters and error value
        N_samples = self.dependent_data.shape[1]
        self.params = res.x
        self.loglikelihood = -res.fun * N_samples
        self.opt_result = res
        self.mean_model_params = self._compute_beta_hat()[:,:,0]
        return self

    def summary(self) -> str:
        ''' Return a summary table displaying the result '''
        param_part_lengths = [self.variance_model_dmat.shape[1], self.variance_model_dmat.shape[1], self.corr_model_dmat.shape[1]]
        sigma_y1, sigma_y2, rho = split_array(self.params, param_part_lengths)
        lines = ['Correlation Regression REML Results',
            "--------",
            f"Dependent variables: y1 = {self.y1}   y2 = {self.y2}",
            f"Mean model: y_i ~ {self.mean_model}",
            f"Variance model: log(SD(y_i)) ~ {self.variance_model}",
            f"Correlation model: arctanh(rho) ~ {self.corr_model}",
            "--------",
            self.opt_result.message,
            "--------",
            "Mean model:",
            pandas.DataFrame([(name, val1, val2)
                    for name, val1, val2 in zip(self.mean_model_dmat.design_info.column_names, self.mean_model_params[0], self.mean_model_params[1])],
                columns = ["Term", self.y1, self.y2])
                .to_string(index=False),
            "--------",
            "Variance model:",
            pandas.DataFrame([(name, val1, val2)
                    for name, val1, val2 in zip(self.variance_model_dmat.design_info.column_names, sigma_y1, sigma_y2)],
                columns = ["Term", self.y1, self.y2])
                .to_string(index=False),
            "--------",
            "Correlation model:",
            pandas.DataFrame([(name, val)
                    for name, val in zip(self.corr_model_dmat.design_info.column_names, rho)],
                columns = ["Term", "Rho"])
                .to_string(index=False),
        ]
        return '\n'.join(lines)

    def likelihood_ratio_test(self, nested_model: "CorrReg") -> float:
        ''' Compute a p-value for a likelihood ratio test

        Parameters:
            nested_model: a model fit to the same data, with the same mean model
                but smaller variance and/or correlation models than this model

        Returns:
            p-value: Likelihood ratio test p-value
        '''
        if not np.array_equal(nested_model.mean_model_dmat, self.mean_model_dmat):
            raise ValueError("Likelihood ratio tests assumes that the mean models are the same, but provided nested model did not match")
        # TODO: check that nested_model is indeed nested?

        LR = 2*(self.loglikelihood - nested_model.loglikelihood)
        this_dof = self.corr_model_dmat.shape[1] + 2*self.variance_model_dmat.shape[1]
        nested_dof = nested_model.corr_model_dmat.shape[1] + 2*nested_model.variance_model_dmat.shape[1]
        if nested_dof >= this_dof:
            raise ValueError("Provided model is not nested within the reference model for the likelihood ratio test")

        dof = this_dof - nested_dof

        return scipy.stats.chi2(dof).sf(LR)

    def predict(self, data, confidence_intervals=None):
        """
        Give the predicted fit values corresponding to the dependent variables in `data`

        Parameters:
            data: a dataframe of the dependent variable values to predict
            confidence_intervals: value from 0-1 of which CI to compute if any (default: None)

        Returns:
            data frame containing fit (mean) values, variances, and correlation for y1 and y2
            if confidence_intervals is specified, then also contains '_lower' and '_upper' columns
            for the CI bounds of variance and correlations estimates.
        """
        N_samples = self.dependent_data.shape[1]

        # Construct the new model matrices from dependent variables
        mean_model_dmat = patsy.dmatrix(self.mean_model, data, eval_env=1)
        mean_model_dmat_array = np.asarray(mean_model_dmat)
        variance_model_dmat = patsy.dmatrix(self.variance_model, data, eval_env=1)
        variance_model_dmat_array = np.asarray(variance_model_dmat)
        corr_model_dmat = patsy.dmatrix(self.corr_model, data, eval_env=1)
        corr_model_dmat_array = np.asarray(corr_model_dmat)

        param_part_lengths = [variance_model_dmat.shape[1], variance_model_dmat.shape[1], corr_model_dmat.shape[1]]

        y1_variance_params, y2_variance_params, corr_params = split_array(self.params, param_part_lengths)

        # The three covariance parameters (correlation and two standard deviations)
        raw_rho = corr_model_dmat @ corr_params
        rho = np.tanh(raw_rho)
        raw_sigma_y1 = variance_model_dmat @ y1_variance_params
        sigma_y1 = np.exp(raw_sigma_y1)
        raw_sigma_y2 = variance_model_dmat @ y2_variance_params
        sigma_y2 = np.exp(raw_sigma_y2)

        # Mean model parameters
        beta_H = self._compute_beta_hat()

        y1_fit = mean_model_dmat_array @ beta_H[0]
        y2_fit = mean_model_dmat_array @ beta_H[1]
        results = pandas.DataFrame(data)
        results[f'{self.y1}_fit'] = y1_fit
        results[f'{self.y2}_fit'] = y2_fit
        results[f'{self.y1}_variance'] = sigma_y1
        results[f'{self.y2}_variance'] = sigma_y2
        results[f'correlation'] = rho

        if confidence_intervals is not None:
            # _objective is logLikelihood / N, so we have to remultiply by N
            hess = _objective_hess(self.params, self.dependent_data, self.mean_model_dmat, self.variance_model_dmat, self.corr_model_dmat) * N_samples
            # NOTE: jax.numpy doens't throw if the matrix isn't invertible, it just gives infinities or bad results
            inv_hess = np.linalg.inv(hess)
            lower_cutoff, upper_cutoff = scipy.stats.norm().interval(confidence_intervals)

            # X @ Hess @ X^T gives the standard errors for the two REML parts
            # We build three different 'X's for y1 var, y2 var, and correlation
            # leaving the others as zero
            y1_var_dmat = np.hstack((
                variance_model_dmat_array,
                np.zeros_like(variance_model_dmat_array),
                np.zeros_like(corr_model_dmat_array)
            ))
            y2_var_dmat = np.hstack((
                np.zeros_like(variance_model_dmat_array),
                variance_model_dmat_array,
                np.zeros_like(corr_model_dmat_array)
            ))
            corr_dmat = np.hstack((
                np.zeros_like(variance_model_dmat_array),
                np.zeros_like(variance_model_dmat_array),
                corr_model_dmat_array
            ))
            # Standard errors:
            y1_var_se = np.sqrt(np.einsum("ij,jk,ik->i", y1_var_dmat, inv_hess, y1_var_dmat))
            y2_var_se = np.sqrt(np.einsum("ij,jk,ik->i", y2_var_dmat, inv_hess, y2_var_dmat))
            corr_se = np.sqrt(np.einsum("ij,jk,ik->i", corr_dmat, inv_hess, corr_dmat))

            results[f'{self.y1}_variance_lower'] = np.exp(raw_sigma_y1 + lower_cutoff * y1_var_se)
            results[f'{self.y1}_variance_upper'] = np.exp(raw_sigma_y1 + upper_cutoff * y1_var_se)
            results[f'{self.y2}_variance_lower'] = np.exp(raw_sigma_y2 + lower_cutoff * y2_var_se)
            results[f'{self.y2}_variance_upper'] = np.exp(raw_sigma_y2 + upper_cutoff * y2_var_se)
            results['correlation_lower'] = np.tanh(raw_rho + lower_cutoff * corr_se)
            results['correlation_upper'] = np.tanh(raw_rho + upper_cutoff * corr_se)
        return results

def split_array(array, lengths):
    ''' For a given array x, split the first axis into pieces of the given length '''
    assert array.shape[0] == sum(lengths)
    start = 0
    for length in lengths:
        yield array[start:start+length]
        start += length

def log_det(pos_def_matrix):
    ''' Compute the log of the determinant for a positive definite matrix via Cholesky decomposition

    Avoids overflow that commonly occurs for np.log(np.linalg.det(pos_def_matrix))
    '''
    A = np.linalg.cholesky(pos_def_matrix)
    return 2*np.sum(np.log(np.diag(A)))

@jax.jit
def _compute_beta_H_xhix(cov, X, Y):
    ''' Inner function to compute most of the values needed for REML likelihood '''
    rho, sigma_y1, sigma_y2 = cov

    # Number of predictors for mean model
    k = X.shape[1]

    # Covariance matrix of y1 and y2
    H = np.moveaxis(
        np.array([[sigma_y1**2, sigma_y1*sigma_y2*rho],
                    [sigma_y1*sigma_y2*rho, sigma_y2**2]]),
        2,
        0
    )
    # Inverse correlation matrix - gives distance metric for the residuals
    H_inv = np.linalg.inv(H)
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
    beta_hat = np.moveaxis(G, 0, 2) @ Y[:, :, None]
    return H, H_inv, beta_hat, XHiX

@jax.jit
def _reml_loglikelihood(cov, X, Y) -> float:
    ''' inner function to compute the reml loglikelihood '''

    # Obatin:
    # H - the covariance matrix
    # H_inv - its inverse
    # beta_hat - the mean model parameters for both y1 and y2
    # XHiX - the inner product of X and itself under the H_inv metric
    H, H_inv, beta_hat, XHiX = _compute_beta_H_xhix(cov, X, Y)

    # Compute fit values and residual size
    Y_hat = np.einsum("ij,kjl->ki", X ,beta_hat)
    resid = (Y - Y_hat) # Y - Y_hat
    mahalanobis = np.einsum("ji,ijk,ki->i", resid, H_inv, resid) # (Y - Y_hat)^T @ H_inv @ (Y - Y_hat)

    # Compute the full REML log likelihood
    log_like = -1/2*(
        np.sum(np.log(np.linalg.det(H)), axis=0)
        + log_det(XHiX)
        + np.sum(mahalanobis, axis=0)
    )
    return log_like

def _params_to_cov(params, variance_model_dmat, corr_model_dmat):
    ''' Inner function extracting covariance into parameters
    '''
    # Separate out the parts of the params
    param_part_lengths = [variance_model_dmat.shape[1], variance_model_dmat.shape[1], corr_model_dmat.shape[1]]
    y1_variance_params, y2_variance_params, corr_params = split_array(params, param_part_lengths)

    # The three covariance parameters (correlation and two standard deviations)
    rho = np.tanh(corr_model_dmat @ corr_params)
    sigma_y1 = np.exp(variance_model_dmat @ y1_variance_params)
    sigma_y2 = np.exp(variance_model_dmat @ y2_variance_params)

    return (rho, sigma_y1, sigma_y2)

def _objective(params, Y, mean_model_dmat, variance_model_dmat, corr_model_dmat):
    '''
    Inner function for the objective function to be minimized for given values of `params`
    '''
    cov = _params_to_cov(params, variance_model_dmat, corr_model_dmat)
    N_samples = cov[0].shape[0]
    return -_reml_loglikelihood(cov, mean_model_dmat, Y) / N_samples

_compute_beta_H_xhix_jit = jax.jit(_compute_beta_H_xhix)
_objective_and_grad = jax.value_and_grad(_objective, argnums=0)
_objective_and_grad_jit = jax.jit(_objective_and_grad)
_objective_hess = jax.hessian(_objective, argnums=0)

def multi_corr_reg(data, dependent_vars, mean_model, variance_model, corr_model):
    '''
    Runs the same CorrReg model for each pair of variables in `dependent_vars`.

    Somewhat faster than running each of these separately.

    Parameters:
        data: a dataframe giving all variables used in the model
        dependent_vars: a list of variables in `data` to pairwise compare
        mean_model: a one-sided formula string giving the formula for the means
        variance_model: a one-sided formula string giving the formula for the variances
        corr_model: a one-sided formula string giving the formula for the correlation
    '''
    assert len(dependent_vars) >= 2, "Must provide at least two dependent variables to compare"
    base_corr_reg = CorrReg(
        data = data,
        y1 = dependent_vars[0],
        y2 = dependent_vars[1],
        mean_model = mean_model,
        variance_model = variance_model,
        corr_model = corr_model,
    )
    results = []
    for i, y1 in enumerate(dependent_vars):
        for j, y2 in enumerate(dependent_vars):
            if i <= j:
                continue
            cr = copy.copy(base_corr_reg)
            cr.y1 = y1
            cr.y2 = y2
            cr.dependent_data = np.vstack((
                np.asarray(data[y1]),
                np.asarray(data[y2]),
            ))
            cr.fit()
            results.append({
                "y1": y1,
                "y2": y2,
                "res": cr,
            })
    return results
