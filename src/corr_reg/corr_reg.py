import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy.optimize
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
        self.dependent_data = jnp.vstack((
            jnp.asarray(self.data[x_var]),
            jnp.asarray(self.data[y_var]),
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

        H = jnp.moveaxis(
            jnp.array([[sigma_x**2, sigma_x*sigma_y*rho],
                       [sigma_x*sigma_y*rho, sigma_y**2]]),
            2,
            0
        )
        H_inv = jnp.linalg.inv(H)

        X = jnp.array(self.mean_model_dmat)
        XHiX = jnp.einsum("ij,ik,ilm->jklm", X, X, H_inv) # k x k x 2 x 2
        XHiX = jnp.block([ # 2k x 2k
            [XHiX[:,:,0,0], XHiX[:,:,0,1]],
            [XHiX[:,:,1,0], XHiX[:,:,1,1]],
        ])
        XHiX_inv = jnp.linalg.inv(XHiX) # 2k x 2k
        XHiX_inv = jnp.array([  # 2 x 2 x k x k
            [XHiX_inv[:k,:k], XHiX_inv[:k, k:]],
            [XHiX_inv[k:,:k], XHiX_inv[k:, k:]],
        ])
        #G = H_inv @ X @ XHiX_inv
        G = jnp.einsum("ijk,il,kslm->ijm", H_inv, X, XHiX_inv)
        beta_hat = jnp.moveaxis(G, 0, 2) @ self.dependent_data[:, :, None]
        Y_hat = jnp.einsum("ij,jkl->ji", X ,beta_hat)
        resid = (self.dependent_data - Y_hat) # Y - Y_hat
        mahalanobis = jnp.einsum("ji,ijk,ki->i", resid, H_inv, resid) # (y - y_hat)^T @ H_inv @ (y - y_hat)
        log_like = -1/2*(
            jnp.sum(jnp.log(jnp.linalg.det(H)), axis=0)
            + jnp.log(jnp.linalg.det(XHiX))
            + jnp.sum(mahalanobis, axis=0)
        ) / N_samples
        return log_like

    def param_to_fit(self, params):
        ''' converts parameters to fit values and fit correlation matrix '''
        # Separate out the parts of the params
        param_part_lengths = [self.variance_model_dmat.shape[1], self.variance_model_dmat.shape[1], self.corr_model_dmat.shape[1]]
        x_variance_params, y_variance_params, corr_params = split_array(params, param_part_lengths)
        rho = jnp.tanh(self.corr_model_dmat @ corr_params)
        sigma_x = jnp.exp(self.variance_model_dmat @ x_variance_params)
        sigma_y = jnp.exp(self.variance_model_dmat @ y_variance_params)
        cov_structure = jnp.array([rho, sigma_x, sigma_y])
        return cov_structure

    def objective(self, params):
        fit_corr = self.param_to_fit(params)
        return -self.reml_loglikelihood(fit_corr)

    objective_and_grad = jax.value_and_grad(objective, argnums=[1])
    hessian = jax.hessian(objective, argnums=[1])

    def fit(self):
        ''' given data and metadata T determine the best fit parameters '''

        # Initial guess for parameters
        init_params = jnp.concatenate((
            jnp.zeros(self.variance_model_dmat.shape[1]), # x variance
            jnp.zeros(self.variance_model_dmat.shape[1]), # y variance
            jnp.zeros(self.corr_model_dmat.shape[1]),
        ))

        def val_and_jac(*args):
            val, (jac,) = self.objective_and_grad(*args)
            return val, jac
        def hess(*args):
            return self.hessian(*args)
        def val(*args):
            val = self.objective(*args)
            return val
        res = scipy.optimize.minimize(
            fun = val_and_jac,
            #fun = val,
            x0 = init_params,
            method = "BFGS",
            #method = "Newton-CG",
            jac = True,
            #hess = hess,
        )

        #res = jax.scipy.optimize.minimize(
        #    fun = self.objective,
        #    x0 = init_params,
        #    method = "BFGS",
        #)
        #print(res)

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
N = 1000
T = jnp.linspace(0.0,1.0,1001)
def true_cov(t):
    rho = jnp.tanh(0.3-0.3*t)
    sigma_x = jnp.exp(0.5)
    sigma_y = jnp.exp(0.0)
    return jnp.array([[sigma_x**2, sigma_x*sigma_y*rho], [sigma_x*sigma_y*rho, sigma_y**2]])
test_data = jnp.array([rng.multivariate_normal( [0,0], true_cov(time)) for time in T])
df = pandas.DataFrame(dict(x = test_data[:,0], y = test_data[:,1], t = T))

cr = CorrReg(
    data = df,
    x_var = "x",
    y_var = "y",
    mean_model = "t",
    variance_model = "1",
    corr_model = "t",
).fit()
print(cr.opt_result)