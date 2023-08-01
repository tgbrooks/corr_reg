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
        self.independent_data = jnp.vstack((
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

    def compute_loglikelihood(self, fit_data, fit_corr):
        ''' Log-likelihood of the given fit '''
        diff = self.independent_data - fit_data
        rho = fit_corr[0,:]
        sigma_x = fit_corr[1,:]
        sigma_y = fit_corr[2,:]
        # mean of 1/(1-rho^2) [(x - mu_x)^2 - 2 rho (x - mu_x)(y - mu_y) + (y - mu_y)^2]
        error = 1/2*jnp.mean(
            (
                diff[0]**2/sigma_x**2
                + diff[1]**2/sigma_y**2
                - 2*rho*diff[0]*diff[1]/sigma_x/sigma_y
            )
            / (1 - rho**2)
        )
        # the factor outside the exp[...] part of the likelihood (after going to log-likelihood)
        factor = jnp.mean(jnp.log(jnp.sqrt(1 - rho**2) * sigma_x * sigma_y))
        return factor + error

    def param_to_fit(self, params):
        ''' converts parameters to fit values and fit correlation matrix '''
        # Separate out the parts of the params
        param_part_lengths = [self.mean_model_dmat.shape[1], self.mean_model_dmat.shape[1], self.variance_model_dmat.shape[1], self.variance_model_dmat.shape[1], self.corr_model_dmat.shape[1]]
        x_mean_params, y_mean_params, x_variance_params, y_variance_params, corr_params = split_array(params, param_part_lengths)
        mean_fit = jnp.vstack((
            self.mean_model_dmat @ x_mean_params,
            self.mean_model_dmat @ y_mean_params,
        ))
        rho = jnp.tanh(self.corr_model_dmat @ corr_params)
        sigma_x = jnp.exp(self.variance_model_dmat @ x_variance_params)
        sigma_y = jnp.exp(self.variance_model_dmat @ y_variance_params)
        cov_structure = jnp.array([rho, sigma_x, sigma_y])
        return mean_fit, cov_structure

    def objective(self, params):
        fit_data, fit_corr = self.param_to_fit(params)
        return self.compute_loglikelihood(fit_data, fit_corr)

    objective_and_grad = jax.value_and_grad(objective, argnums=[1])

    def fit(self):
        ''' given data and metadata T determine the best fit parameters '''

        # Initial guess for parameters
        init_params = jnp.concatenate((
            jnp.zeros(self.mean_model_dmat.shape[1]), # x mean
            jnp.zeros(self.mean_model_dmat.shape[1]), # y mean
            jnp.zeros(self.variance_model_dmat.shape[1]), # x variance
            jnp.zeros(self.variance_model_dmat.shape[1]), # y variance
            jnp.zeros(self.corr_model_dmat.shape[1]),
        ))

        def val_and_jac(*args):
            val, (jac,) = self.objective_and_grad(*args)
            return val, jac
        res = scipy.optimize.minimize(
            fun = val_and_jac,
            x0 = init_params,
            method = "BFGS",
            #jac = True,
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
T = jnp.linspace(-0.5,0.5,1001)
def true_cov(t):
    rho = jnp.tanh(t)
    sigma_x = jnp.exp(0.5)
    sigma_y = jnp.exp(0.5)
    return jnp.array([[sigma_x**2, sigma_x*sigma_y*rho], [sigma_x*sigma_y*rho, sigma_y**2]])
test_data = jnp.array([rng.multivariate_normal( [0,0], true_cov(time)) for time in T])
df = pandas.DataFrame(dict(x = test_data[:,0], y = test_data[:,1], t = T))

cr = CorrReg(
    data = df,
    x_var = "x",
    y_var = "y",
    mean_model = "1",
    variance_model = "1",
    corr_model = "t",
).fit()