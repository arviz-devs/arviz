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
    return like + prior, like_vect

nwalkers, draws = 50, 4000
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

# swap axis to have chain as 1st axis and apply burnin
burnin = 500
thin = 10
chain = sampler.chain[:, burnin::thin, :]
blobs = np.swapaxes(sampler.blobs, 0, 1)[:, burnin::thin, :]

# create posterior group
emcee_result = az.convert_to_inference_data(
    {"mu": chain[:, :, 0], "tau": chain[:, :, 1], "eta": chain[:, :, 2:]},
    coords={"school": range(8)},
    dims={"eta": ["school"]},
)
# create sample_stats group with log_likelihood
emcee_stats = az.convert_to_inference_data(
    {"log_likelihood": blobs},
    group="sample_stats",
    coords={"school": range(8)},
    dims={"log_likelihood": ["school"]},
)
# combine all emcee data in a single InferenceData object
emcee_data = az.concat(emcee_result, emcee_stats)

# calculate reff and loo
if emcee.__version__[0] == '3':
    ess = (draws-burnin) / sampler.get_autocorr_time(quiet=True, discard=burnin, thin=thin)
else:
    # in emcee2, if the chain is too short, the autocorr_time raises an error and there is
    # no way to avoid that, so to prevent the docs crashing, the ess value is hardcoded.
    ess = (draws-burnin)/30
reff = np.mean(ess) / (nwalkers * chain.shape[1])
loo_stats = az.loo(emcee_data, reff=reff, pointwise=True)

# plot results
az.plot_khat(loo_stats.pareto_k)
plt.show()
