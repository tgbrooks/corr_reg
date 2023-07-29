import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy.optimize
import scipy.optimize

rng = np.random.default_rng(1)
N = 1000
T = jnp.linspace(-0.5,0.5,1001)
def true_cov(t):
    rho = jnp.tanh(t)
    sigma_x = jnp.exp(0.5*t)
    sigma_y = jnp.exp(0.5)
    return jnp.array([[sigma_x**2, sigma_x*sigma_y*rho], [sigma_x*sigma_y*rho, sigma_y**2]])
test_data = jnp.array([rng.multivariate_normal( [0,0], true_cov(time)) for time in T])

def error(data, fit_data, fit_corr):
    ''' MLE of the fit correlation matrix '''
    diff = data - fit_data
    rho = fit_corr[0,:]
    sigma_x = fit_corr[1,:]
    sigma_y = fit_corr[2,:]
    # mean of 1/(1-rho^2) [(x - mu_x)^2 - 2 rho (x - mu_x)(y - mu_y) + (y - mu_y)^2]
    error = 1/2*jnp.mean(
        (
            diff[:,0]**2/sigma_x**2
            + diff[:,1]**2/sigma_y**2
            - 2*rho*diff[:,0]*diff[:,1]/sigma_x/sigma_y
        )
          / (1 - rho**2)
    )
    # the factor outside the exp[...] part of the likelihood (after going to log-likelihood)
    factor = jnp.mean(jnp.log(jnp.sqrt(1 - rho**2) * sigma_x * sigma_y))
    return factor + error

def param_to_fit(params, T):
    ''' converts parameters to fit values and fit correlation matrix '''
    mean_fit = jnp.zeros((len(T), 2))
    rho_a,rho_b, sigma_xa, sigma_xb, sigma_ya, sigma_yb = params
    rho = jnp.tanh(rho_a * T + rho_b)
    sigma_x = jnp.exp(sigma_xa * T + sigma_xb)
    sigma_y = jnp.exp(sigma_ya * T + sigma_yb)
    cov_structure = jnp.array([rho, sigma_x, sigma_y])
    return mean_fit, cov_structure

def objective(params, data, T):
    fit_data, fit_corr = param_to_fit(params, T)
    return error(data, fit_data, fit_corr)

objective_and_grad = jax.value_and_grad(objective, argnums=[0])
objective_grad = jax.grad(objective, argnums=[0])
objective_hessian = jax.hessian(objective, argnums=[0])

def update(params, m, v, i, data, optimizer_config):
    ''' ADAM optimizer to minimize the objective function '''
    beta1 = optimizer_config['beta1']
    beta2 = optimizer_config['beta2']
    alpha = optimizer_config['alpha']
    epsilon = optimizer_config['epsilon']
    # Adam optimizer
    error, (grad,) = objective_and_grad(params, data, T)
    m = [beta1 * mX + (1 - beta1) * gradX for mX, gradX in zip(m, grad)]
    v = [beta2 * vX + (1 - beta2) * gradX**2 for vX, gradX in zip(v, grad)]
    mHat = [mX / (1 - beta1**i) for mX in m]
    vHat = [vX / (1 - beta2**i) for vX in v]
    new_params = jnp.array([param - alpha * mHatX / jnp.sqrt(vHatX + epsilon)
                            for param, mHatX, vHatX in zip(params, mHat, vHat)])
    return new_params, m, v, error

def fit(data, T):
    ''' given data and metadata T determine the best fit parameters '''

    # Initial guess for parameters
    init_params = jnp.array([
        0., 0., #rho
        0., 1., #sigma_x
        0., 1., #sigma_y
    ])

    #def val_and_jac(*args):
    #    val, (jac,) = objective_and_grad(*args)
    #    return val, jac
    #res = scipy.optimize.minimize(
    #    fun = val_and_jac,
    #    x0 = init_params,
    #    args = (data, T),
    #    method = "Newton-CG",
    #    jac = True,
    #    hess = objective_hessian,
    #    options = {
    #        "eps": 1e-3,
    #    }
    #)

    res = jax.scipy.optimize.minimize(
        fun = objective,
        x0 = init_params,
        args = (data, T),
        method = "BFGS",
    )
    print(res)

    # parameters and error value
    return res.x, res.fun

test_params, test_error = fit(test_data, T)