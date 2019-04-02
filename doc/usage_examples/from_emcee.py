import emcee
import numpy as np
import arviz as az
import matplotlib.pyplot as plt

J = 8
y_obs = np.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0])
sigma = np.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])

def log_prior_8school(theta, J):
    mu, tau, eta = theta[0], theta[1], theta[2:]
    # Half-cauchy prior, hwhm=25
    if tau < 0:
        return -np.inf
    prior_tau = -np.log(tau ** 2 + 25 ** 2)
    prior_mu = -(mu / 10) ** 2  # normal prior, loc=0, scale=10
    prior_eta = -np.sum(eta ** 2)  # normal prior, loc=0, scale=1
    return prior_mu + prior_tau + prior_eta

def log_likelihood_8school(theta, y, sigma):
    mu, tau, eta = theta[0], theta[1], theta[2:]
    return -((mu + tau * eta - y) / sigma) ** 2

def lnprob_8school(theta, J, y, sigma):
    prior = log_prior_8school(theta, J)
    like_vect = log_likelihood_8school(theta, y, sigma)
    like = np.sum(like_vect)
    return like + prior

nwalkers, draws = 50, 2000
ndim = J + 2
pos = np.random.normal(size=(nwalkers, ndim))
pos[:, 1] = np.absolute(pos[:, 1])
sampler = emcee.EnsembleSampler(
    nwalkers,
    ndim,
    lnprob_8school,
    args=(J, y_obs, sigma),
)
sampler.run_mcmc(pos, draws)

# define variable names, it cannot be inferred from emcee
var_names = ['mu','tau']+['eta{}'.format(i) for i in range(J)]
emcee_data = az.from_emcee(sampler, var_names=var_names)

az.plot_posterior(emcee_data)
plt.show()
