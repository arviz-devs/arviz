from __future__ import division
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from hpd import hpd


def plot_post(param_sample_vec, cred_mass=0.95, comp_val=None,
              ROPE=None, ylab='', xlab='parameter', fontsize=14, labelsize=14,
              title='', framealpha=1, show_mode=True, bins=50, kde_plot=True, 
              roundto=3):
    """
    Write me!
    """        
    # colors taken from the default seaborn color pallete
    blue, green, red = [(0.2980392156862745, 0.4470588235294118, 
    0.6901960784313725), (0.3333333333333333, 0.6588235294117647, 
    0.40784313725490196), (0.7686274509803922, 0.3058823529411765, 
    0.3215686274509804)]
    ## Compute HDI
    HDI = hpd(param_sample_vec, 1-cred_mass)

    post_summary = {'mean':0,'median':0,'mode':0, 'hdi_mass':0,'hdi_low':0,
                   'hdi_high':0, 'comp_val':0, 'pc_gt_comp_val':0, 'ROPE_low':0,
                   'ROPE_high':0, 'pc_in_ROPE':0}

    post_summary['mean'] = round(np.mean(param_sample_vec), roundto)
    post_summary['median'] = round(np.median(param_sample_vec), roundto)
    post_summary['mode'] = round(stats.mode(param_sample_vec)[0], roundto)
    post_summary['hdi_mass'] = cred_mass
    post_summary['hdi_low'] = round(HDI[0], roundto)
    post_summary['hdi_high'] = round(HDI[1], roundto)

    ## Plot KDE.
    if kde_plot:
            density = stats.kde.gaussian_kde(param_sample_vec)
            l = np.min(param_sample_vec)
            u = np.max(param_sample_vec)
            x = np.linspace(0, 1, 100) * (u - l) + l
            plt.plot(x, density(x), color=blue)
    ## Plot histogram.
    else:
        plt.hist(param_sample_vec, normed=True, bins=bins, facecolor=blue, 
        edgecolor='w')


    ## Display mean or mode:
    if show_mode:
        plt.plot(0, label='mode = %.2f' % post_summary['mode'], alpha=0)
    else:
        plt.plot(0, label='mean = %.2f' % post_summary['mean'], alpha=0)
    ## Display the comparison value.
    if comp_val is not None:
        pc_gt_comp_val = 100 * np.sum(param_sample_vec > comp_val)/len(param_sample_vec)
        pc_lt_comp_val = 100 - pc_gt_comp_val
        plt.axvline(comp_val, ymax=.75, color=green,
                 linestyle='--', linewidth=4,
                 label='%.1f%% < %.1f < %.1f%%'
                 % (pc_lt_comp_val, comp_val, pc_gt_comp_val))
        post_summary['comp_val'] = comp_val
        post_summary['pc_gt_comp_val'] = pc_gt_comp_val
    ## Display the ROPE.
    if ROPE is not None:
        rope_col = 'darkred'
        pc_in_ROPE = round(np.sum((param_sample_vec > ROPE[0]) & (param_sample_vec < ROPE[1]))/len(param_sample_vec)*100)
        plt.axvline(ROPE[0], ymax=.75, color=red, linewidth=4,
                label='%.1f%% in ROPE' % pc_in_ROPE)
        plt.axvline(ROPE[1], ymax=.75, color=red, linewidth=4)
        post_summary['ROPE_low'] = ROPE[0] 
        post_summary['ROPE_high'] = ROPE[1] 
        post_summary['pc_in_ROPE'] = pc_in_ROPE

    ## Display the HDI.
    plt.plot(HDI, [0, 0], linewidth=8, color='k', label='HDI %.1f%% %.3f-%.3f' % (cred_mass*100, HDI[0], HDI[1]))

    plt.xlabel(xlab, fontsize=fontsize)
    plt.ylabel(ylab, fontsize=fontsize)
    plt.title(title, fontsize=fontsize)
    plt.legend(loc='upper left', fontsize=labelsize, framealpha=framealpha)
    frame = plt.gca()
    frame.axes.get_yaxis().set_ticks([])
    return post_summary

