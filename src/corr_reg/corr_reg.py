import numpy as np
import patsy
import pandas
import scipy.optimize

class CorrReg:
    ''' Results of a Correlation Regression '''
    def __init__(
            self,
            data,
            x_var:str,
            y_var:str,
            mean_model: str,
            variance_model: str,
            corr_model: str,
        ):
        self.data = data
        self.dependent_data = np.vstack((
            np.asarray(self.data[x_var]),
            np.asarray(self.data[y_var]),
        ))
        self.x_var = x_var
        self.y_var = y_var

        self.mean_model = mean_model
        self.mean_model_dmat = patsy.dmatrix(self.mean_model, self.data)

        self.variance_model = variance_model
        self.variance_model_dmat = patsy.dmatrix(self.variance_model, self.data)

        self.corr_model = corr_model
        self.corr_model_dmat = patsy.dmatrix(self.corr_model, self.data)

        #self.params, self.loglikelihood = self.fit()

    def reml_loglikelihood(self, fit_corr):
        ''' Log-likelihood of the given correlation parameters after restricting to ReML
         
        See "Bayesian inference for variance components using only error contrasts" Harville 1974
        https://www.jstor.org/stable/2334370
        and
        https://xiuming.info/docs/tutorials/reml.pdf
        '''

        rho = fit_corr[0,:]
        sigma_x = fit_corr[1,:]
        sigma_y = fit_corr[2,:]
        N_samples = rho.shape[0]

        k = self.mean_model_dmat.shape[1] # Number of predictors for mean model

        H = np.moveaxis(
            np.array([[sigma_x**2, sigma_x*sigma_y*rho],
                       [sigma_x*sigma_y*rho, sigma_y**2]]),
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

    def param_to_fit(self, params):
        ''' converts parameters to fit values and fit correlation matrix '''
        # Separate out the parts of the params
        param_part_lengths = [self.variance_model_dmat.shape[1], self.variance_model_dmat.shape[1], self.corr_model_dmat.shape[1]]
        x_variance_params, y_variance_params, corr_params = split_array(params, param_part_lengths)
        rho = np.tanh(self.corr_model_dmat @ corr_params)
        sigma_x = np.exp(self.variance_model_dmat @ x_variance_params)
        sigma_y = np.exp(self.variance_model_dmat @ y_variance_params)
        cov_structure = np.array([rho, sigma_x, sigma_y])
        return cov_structure

    def objective(self, params):
        fit_corr = self.param_to_fit(params)
        return -self.reml_loglikelihood(fit_corr)

    def fit(self):
        ''' given data and metadata T determine the best fit parameters '''

        # Initial guess for parameters
        init_params = np.concatenate((
            np.zeros(self.variance_model_dmat.shape[1]), # x variance
            np.zeros(self.variance_model_dmat.shape[1]), # y variance
            np.zeros(self.corr_model_dmat.shape[1]),
        ))

        def val(*args):
            val = self.objective(*args)
            return val
        res = scipy.optimize.minimize(
            fun = val,
            x0 = init_params,
            method = "BFGS",
            tol = 1e-2,
        )

        # parameters and error value
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
    sigma_x = np.exp(0.5*np.sin(t))
    sigma_y = np.exp(0.5)
    return np.array([[sigma_x**2, sigma_x*sigma_y*rho], [sigma_x*sigma_y*rho, sigma_y**2]])
test_data = np.array([rng.multivariate_normal( [5+np.sin(time), 3+np.cos(time)], true_cov(time)) for time in T])
df = pandas.DataFrame(dict(
    x = test_data[:,0],
    y = test_data[:,1],
    t = T,
))

cr = CorrReg(
    data = df,
    x_var = "x",
    y_var = "y",
    mean_model = "np.cos(t) + np.sin(t)",
    variance_model = "np.cos(t) + np.sin(t)",
    corr_model = "np.cos(t) + np.sin(t)",
).fit()
print(cr.opt_result)