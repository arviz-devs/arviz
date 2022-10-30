# Plotting and reporting Bayes Factor given idata, var name, prior distribution and reference value
def plot_bf(trace, var_name, prior, family = 'normal',  ref_val=0, xlim=None, ax=None):
    # grab trace, a variable name to compute difference and prior.
    # ref_val is the parameter we want to compare
    from scipy import stats
    import matplotlib.pyplot as plt
    import numpy as np
    # test some elemtns
    # varName should be string
    if not isinstance(var_name, str):
        print('varName is not a string')
    # BFs based on density estimation (using kernel smoothing instead of spline)
    # stack trace posterior
    post = extract(idata, var_names=var_name)
    if prior is None:
        # grab prior from the data in case it wasn't defined by the user
        prior = extract(idata, var_names=var_name, group="prior")
    post = tr[var_name]
    if post.ndim > 1:
        print("Posterior distribution has {post.ndim} dimensions")
    if family=='normal':
        # generate vector
        if xlim is None:
            x = np.linspace(np.min(prior), np.max(prior),prior.shape[0])
        else:
            x = np.linspace(xlim[0], xlim[1],prior.shape[0])
        #x = np.linspace(np.min(post), np.max(post),prior.shape[0])
        my_pdf = stats.gaussian_kde(post)
        prior_pdf = stats.gaussian_kde(prior)
    elif family!='normal':
        # for now and error of notImplemented
        print("The method for {family} distribution is not implemented yet")
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(
        x, my_pdf(x), "--", lw=2.5, alpha=0.6, label="Posterior"
    )  # distribution function
    ax.plot(x, prior_pdf(x), "r-", lw=2.5, alpha=0.6, label="Prior")
    if ref_val>np.max(post) | ref_val<np.min(post):
        print('Reference value is out of bounds of posterior')
    else:
        posterior = my_pdf(ref_val) # this gives the pdf at ref_val
    prior = prior_pdf(ref_val)
    BF10 = posterior / prior
    BF01 = prior / posterior
    print("the Bayes Factor 10 is %.3f" % (BF10))
    print("the Bayes Factor 01 is %.3f" % (BF01))
    ax.plot(ref_val, posterior, "ko", lw=1.5, alpha=1)
    ax.plot(ref_val, prior, "ko", lw=1.5, alpha=1)
    plt.xlabel("Delta")
    plt.ylabel("Density")
    plt.legend(loc="upper left")
    plt.show()
    return {'BF10': BF10, 'BF01':BF01}, ax
# end