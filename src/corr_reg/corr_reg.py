import numpy as np
import patsy
import pandas
import scipy.optimize

class CorrReg:
    ''' Results of a Correlation Regression '''
    def __init__(
            self,
            data,
            y1_var:str,
            y2_var:str,
            mean_model: str,
            variance_model: str,
            corr_model: str,
        ):
        self.data = data
        self.dependent_data = np.vstack((
            np.asarray(self.data[y1_var]),
            np.asarray(self.data[y2_var]),
        ))
        self.y1_var = y1_var
        self.y2_var = y2_var

        self.mean_model = mean_model
        self.mean_model_dmat = patsy.dmatrix(self.mean_model, self.data)

        self.variance_model = variance_model
        self.variance_model_dmat = patsy.dmatrix(self.variance_model, self.data)

        self.corr_model = corr_model
        self.corr_model_dmat = patsy.dmatrix(self.corr_model, self.data)

        #self.params, self.loglikelihood = self.fit()

    def reml_loglikelihood(self, cov):
        ''' Log-likelihood of the given correlation parameters after restricting to ReML
         
        See "Bayesian inference for variance components using only error contrasts" Harville 1974
        https://www.jstor.org/stable/2334370
        and
        https://xiuming.info/docs/tutorials/reml.pdf
        '''

        rho, sigma_y1, sigma_y2 = cov
        N_samples = rho.shape[0]

        k = self.mean_model_dmat.shape[1] # Number of predictors for mean model

        H = np.moveaxis(
            np.array([[sigma_y1**2, sigma_y1*sigma_y2*rho],
                       [sigma_y1*sigma_y2*rho, sigma_y2**2]]),
            2,
            0
        )
        H_inv = np.linalg.inv(H)

        X = np.array(self.mean_model_dmat)
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
        #G = H_inv @ X @ XHiX_inv
        G = np.einsum("ijk,il,kslm->ijm", H_inv, X, XHiX_inv)
        beta_hat = np.moveaxis(G, 0, 2) @ self.dependent_data[:, :, None]
        Y_hat = np.einsum("ij,kjl->ki", X ,beta_hat)
        resid = (self.dependent_data - Y_hat) # Y - Y_hat
        mahalanobis = np.einsum("ji,ijk,ki->i", resid, H_inv, resid) # (y - y_hat)^T @ H_inv @ (y - y_hat)
        log_like = -1/2*(
            np.sum(np.log(np.linalg.det(H)), axis=0)
            + np.log(np.linalg.det(XHiX))
            + np.sum(mahalanobis, axis=0)
        ) / N_samples
        return log_like

    def params_to_cov(self, params):
        ''' converts parameters to cov components
         
        Returns: rho (correlation), sigma_y1 and sigma_y2 (variances) '''
        # Separate out the parts of the params
        param_part_lengths = [self.variance_model_dmat.shape[1], self.variance_model_dmat.shape[1], self.corr_model_dmat.shape[1]]
        y1_variance_params, y2_variance_params, corr_params = split_array(params, param_part_lengths)
        rho = np.tanh(self.corr_model_dmat @ corr_params)
        sigma_y1 = np.exp(self.variance_model_dmat @ y1_variance_params)
        sigma_y2 = np.exp(self.variance_model_dmat @ y2_variance_params)
        cov_structure = (rho, sigma_y1, sigma_y2)
        return cov_structure

    def objective(self, params):
        cov = self.params_to_cov(params)
        return -self.reml_loglikelihood(cov)

    def fit(self):
        ''' given data and metadata T determine the best fit parameters '''

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

def split_array(array, lengths):
    ''' For a given array x, split the first axis into pieces of the given length '''
    assert array.shape[0] == sum(lengths)
    start = 0
    for length in lengths:
        yield array[start:start+length]
        start += length


rng = np.random.default_rng(1)
N_SAMPLES = 500
T = np.linspace(0.0, 2*np.pi, N_SAMPLES)
def true_cov(t):
    rho = np.tanh(0.3-0.3*np.cos(t)) # correlation
    sigma_y1 = np.exp(0.5*np.sin(t))
    sigma_y2 = np.exp(0.5)
    return np.array([[sigma_y1**2, sigma_y1*sigma_y2*rho], [sigma_y1*sigma_y2*rho, sigma_y2**2]])
test_data = np.array([rng.multivariate_normal( [5+np.sin(time), 3+np.cos(time)], true_cov(time)) for time in T])
df = pandas.DataFrame(dict(
    y1 = test_data[:,0],
    y2 = test_data[:,1],
    t = T,
))

cr = CorrReg(
    data = df,
    y1_var = "y1",
    y2_var = "y2",
    mean_model = "np.cos(t) + np.sin(t)",
    variance_model = "np.cos(t) + np.sin(t)",
    corr_model = "np.cos(t) + np.sin(t)",
).fit()
print(cr.opt_result)