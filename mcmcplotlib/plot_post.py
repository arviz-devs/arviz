from __future__ import division
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from hpd import hpd_grid


def plot_post(sample, alpha=0.05, show_mode=True, kde_plot=True, bins=50, 
    ROPE=None, comp_val=None, roundto=2):
    """Plot posterior and HPD

    Parameters
    ----------

    sample : Numpy array or python list
        An array containing MCMC samples
    alpha : float
        Desired probability of type I error (defaults to 0.05)
    show_mode: Bool
        If True the legend will show the mode(s) value(s), if false the mean(s)
        will be displayed
    kde_plot: Bool
        If True the posterior will be displayed using a Kernel Density Estimation
        otherwise an histogram will be used
    bins: integer
        Number of bins used for the histogram, only works when kde_plot is False
    ROPE: list or numpy array
        Lower and upper values of the Region Of Practical Equivalence
    comp_val: float
        Comparison value
        

    Returns
    -------

    post_summary : dictionary
        Containing values with several summary statistics

    """       

    post_summary = {'mean':0,'median':0,'mode':0, 'alpha':0,'hpd_low':0,
                   'hpd_high':0, 'comp_val':0, 'pc_gt_comp_val':0, 'ROPE_low':0,
                   'ROPE_high':0, 'pc_in_ROPE':0}

    post_summary['mean'] = np.mean(sample)
    post_summary['median'] = np.median(sample)
    post_summary['alpha'] = alpha

    # Compute the hpd, KDE and mode for the posterior
    hpd, x, y, modes = hpd_grid(sample, alpha, roundto)
    post_summary['hpd'] = hpd
    post_summary['mode'] = modes

    ## Plot KDE.
    if kde_plot:
            plt.plot(x, y, color='k', lw=2)
    ## Plot histogram.
    else:
        plt.hist(sample, normed=True, bins=bins, facecolor='b', 
        edgecolor='w')

    ## Display mode or mean:
    if show_mode:
        string = '{:.{roundto}g} ' * len(post_summary['mode'])
        plt.plot(0, label='mode =' + string.format(*post_summary['mode'], roundto=roundto).rstrip(), alpha=0)
    else:
        plt.plot(0, label='mean = {:.{roundto}g}'.format(post_summary['mean'], roundto=roundto), alpha=0)

    ## Display the hpd.
    hpd_label = ''
    for value in hpd:
        plt.plot(value, [0, 0], linewidth=10, color='b')
        hpd_label = hpd_label +  '{:.{roundto}g} {:.{roundto}g}'.format(value[0], value[1], roundto=roundto) 
    plt.plot(0, 0, linewidth=4, color='b', label='hpd {:.{roundto}g}%\n{}'.format((1-alpha)*100, hpd_label, roundto=roundto))
    ## Display the ROPE.
    if ROPE is not None:
        pc_in_ROPE = np.sum((sample > ROPE[0]) & (sample < ROPE[1]))/len(sample)*100
        plt.plot(ROPE, [0, 0], linewidth=20, color='r', alpha=0.75)
        plt.plot(0, 0, linewidth=4, color='r', label='{:.{roundto}g}% in ROPE'.format(pc_in_ROPE, roundto=roundto))
        post_summary['ROPE_low'] = ROPE[0] 
        post_summary['ROPE_high'] = ROPE[1] 
        post_summary['pc_in_ROPE'] = pc_in_ROPE
    ## Display the comparison value.
    if comp_val is not None:
        pc_gt_comp_val = 100 * np.sum(sample > comp_val)/len(sample)
        pc_lt_comp_val = 100 - pc_gt_comp_val
        plt.axvline(comp_val, ymax=.75, color='g', linewidth=4, alpha=0.75,
            label='{:.{roundto}g}% < {:.{roundto}g} < {:.{roundto}g}%'.format(pc_lt_comp_val, 
                                                                              comp_val, pc_gt_comp_val, 
                                                                              roundto=roundto))
        post_summary['comp_val'] = comp_val
        post_summary['pc_gt_comp_val'] = pc_gt_comp_val

    plt.legend(loc=0, framealpha=1)
    frame = plt.gca()
    frame.axes.get_yaxis().set_ticks([])
    return post_summary
    

