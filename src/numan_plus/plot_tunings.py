import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from statistics import mean, stdev, sqrt
import scipy.stats as stats
from sklearn.utils import shuffle
from tqdm.notebook import tqdm

def get_tuning_matrix(Q, R, pref_num):
    # 1.Calculate average tuning curve of each unit
    tuning_curves = average_tuning_curves(Q, R)

    # 2.Calculate population tuning curves for each preferred numerosity
    tuning_mat = np.array([np.mean(tuning_curves[:,pref_num==q], axis=1) for q in np.array([0,1,2,3,4])]) # one row for each pref numerosity
    tuning_err = np.array([np.std(tuning_curves[:,pref_num==q], axis=1) / np.sqrt(np.sum(pref_num==q)) # standard error for each point on each tuning curve
                            for q in np.array([0,1,2,3,4])])

    # 3.Normalize population tuning curves to the 0-1 range
    tmmin = tuning_mat.min(axis=1)[:,None]
    tmmax = tuning_mat.max(axis=1)[:,None]
    tuning_mat = (tuning_mat-tmmin) / (tmmax-tmmin)
    tuning_err = tuning_err / (tmmax-tmmin) # scale standard error to be consistent with above normalization

    return tuning_mat, tuning_err

def plot_tunings(tuning_mat, tuning_err, save_name=None):
    # Plot population tuning curves on linear scale
    plt.figure(figsize=(9,4))
    plt.title(save_name)
    plt.subplot(1,2,1)
    for i, (tc, err) in enumerate(zip(tuning_mat, tuning_err)):
        plt.errorbar(Qrange, tc, err, color=colors[i])
        plt.xticks(ticks=np.array([0,1,2,3,4]), labels=np.array([1,2,3,4,5]))
    plt.xlabel('Numerosity')
    plt.ylabel('Normalized Neural Activity')
    # Plot population tuning curves on log scale
    plt.subplot(1,2,2)
    for i, (tc, err) in enumerate(zip(tuning_mat, tuning_err)):
        plt.errorbar(np.array([1,2,3,4,5]), tc, err, color=colors[i]) # offset x axis by one to avoid taking the log of zero
    plt.xscale('log', base=2)
    plt.gca().xaxis.set_major_formatter(ScalarFormatter())
    plt.xticks(ticks=Qrange+1, labels=Qrange+1)
    plt.xlabel('Numerosity')
    plt.ylabel('Normalized Neural Activity')
    plt.show()
    # save figure
    if not (save_name is None):
        plt.savefig('./processed/spots/anova/imaris/'+ save_name + '.png')

def plot_selective_cells_histo(Q, R, save_name=None):
    pref_num = preferred_numerosity(Q, R)
    hist = [np.sum(pref_num==q) for q in np.array([0,1,2,3,4])]
    perc  = hist/np.sum(hist)

    # plot number neurons percentages and absolute distance tuning
    plt.figure(figsize=(4,4))
    plt.bar(Qrange, hist, width=0.8, color=colors)
    for x, y, p in zip(Qrange, hist, perc):
        plt.text(x, y, str(y)+'\n'+str(round(p*100,1))+'%')
    plt.axhline(y=chance_lev, color='k', linestyle='--')
    plt.xticks(np.array([0,1,2,3,4]),[1,2,3,4,5])
    plt.xlabel('Preferred Numerosity')
    plt.ylabel('Number of cells')
    plt.title(save_name)
    plt.show()
    # save figure
    if not (save_name is None):
        plt.savefig('./processed/spots/anova/imaris/'+ save_name + '.png')

def abs_dist_tunings(tuning_mat, absolute_dist=0, save_name=None):
    if absolute_dist == 1:
        distRange = [0, 1, 2, 3, 4]
    else:
        distRange = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
    dist_tuning_dict = {}
    for i in distRange:
        dist_tuning_dict[str(i)]=[]
    for pref_n in Qrange:
        for n in Qrange:
            if absolute_dist == 1:
                dist_tuning_dict[str(abs(n - pref_n))].append(tuning_mat[pref_n][n])
            else:
                dist_tuning_dict[str(n - pref_n)].append(tuning_mat[pref_n][n])
            
    dist_avg_tuning = [mean(dist_tuning_dict[key]) for key in dist_tuning_dict.keys()]
    dist_err_tuning = []
    for key in dist_tuning_dict.keys():
        if len(dist_tuning_dict[key])>1:
            dist_err_tuning.append(np.nanstd(dist_tuning_dict[key])/sqrt(len(dist_tuning_dict[key])))
        else:
            dist_err_tuning.append(0)

    #plot
    plt.figure(figsize=(4,4))
    plt.errorbar(distRange, dist_avg_tuning, dist_err_tuning, color='black')
    plt.xticks(distRange)
    plt.xlabel('Absolute numerical distance')
    plt.ylabel('Normalized Neural Activity')
    plt.title(save_name)
    plt.show
    # save figure
    if not (save_name is None):
        plt.savefig('./processed/spots/anova/imaris/'+ save_name + '.png')
    
    #statistics t-test comparisons
    for i in distRange:
        if i==4:
            continue
        print('Comaparison absolute distances '+ str(i)+ ' and '+ str(i+1)+ ':')
        print(stats.ttest_ind(a=dist_tuning_dict[str(i)], b=dist_tuning_dict[str(i+1)], equal_var=False))
    
    return dist_avg_tuning, dist_err_tuning
