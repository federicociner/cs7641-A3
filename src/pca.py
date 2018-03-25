import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from helpers import get_abspath, save_array
from matplotlib import cm
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn


def pca_experiment(X, name, dims, evp):
    """Run PCA on specified dataset and saves dataset with components that
    explain at least 85% of total variance.

    Args:
        X (Numpy.Array): Attributes.
        name (str): Dataset name.
        dims (int): Number of components.
        evp (float): Explained variance percentage threshold.
    Returns:
        res (Pandas.DataFrame)
    """
    pca = PCA(random_state=0, svd_solver='full', n_components=dims)
    comps = pca.fit_transform(X)  # get principal components

    # cumulative explained variance greater than threshold
    r = range(1, dims + 1)
    ev = pd.Series(pca.explained_variance_, index=r, name='ev')
    evr = pd.Series(pca.explained_variance_ratio_, index=r, name='evr')
    evrc = evr.rename('evr_cum').cumsum()
    res = comps[:, :evrc.where(evrc > evp).idxmin()]
    evars = pd.concat((ev, evr, evrc), axis=1)

    # save results as CSV
    resdir = 'results/DR/PCA'
    evfile = get_abspath('{}_variances.csv'.format(name), resdir)
    resfile = get_abspath('{}_projected.csv'.format(name), resdir)
    save_array(array=res, filename=resfile, subdir=resdir)
    evars.to_csv(evfile, index_label='comp_no')


def generate_variance_plot(name, evp):
    """Plots explained variance and cumulative explained variance ratios as a
    function of principal components.

    Args:
        name (str): Dataset name.
        evp (float): Explained variance percentage threshold.

    """
    resdir = 'results/DR/PCA'
    df = pd.read_csv(get_abspath('{}_variances.csv'.format(name), resdir))

    # get figure and axes
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 3))

    # plot explained variance and cumulative explain variance ratios
    x = df['comp_no']
    evr = df['evr']
    evr_cum = df['evr_cum']
    ax.plot(x, evr, marker='.', color='b', label='EVR')
    ax.plot(x, evr_cum, marker='.', color='g', label='Cumulative EVR')
    ax.set_title('PCA Explained Variance by PC ({})'.format(name))
    ax.set_ylabel('Explained Variance')
    ax.set_xlabel('Principal Component')
    vmark = evr_cum.where(evr_cum > evp).idxmin() + 1
    ax.axvline(x=vmark, linestyle='--', color='r')
    ax.grid(color='grey', linestyle='dotted')

    # change layout size, font size and width
    fig.tight_layout()
    for ax in fig.axes:
        ax_items = [ax.title, ax.xaxis.label, ax.yaxis.label]
        for item in (ax_items + ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(8)

    # save figure
    plotdir = 'plots/DR/PCA'
    plotpath = get_abspath('{}_explvar.png'.format(name), plotdir)
    plt.savefig(plotpath)
    plt.clf()


def main():
    """Run code to generate PCA results.

    """
    winepath = get_abspath('winequality.csv', 'data/experiments')
    seismicpath = get_abspath('seismic_bumps.csv', 'data/experiments')
    wine = np.loadtxt(winepath, delimiter=',')
    seismic = np.loadtxt(seismicpath, delimiter=',')

    # split data into X and y
    wX = wine[:, :-1]
    wY = wine[:, -1]
    sX = seismic[:, :-1]
    sY = seismic[:, -1]
    wDims = wX.shape[1]
    sDims = sX.shape[1]
    evp = 0.85

    # generate PCA results
    pca_experiment(wX, 'winequality', dims=wDims, evp=evp)
    pca_experiment(sX, 'seismic-bumps', dims=sDims, evp=evp)

    # generate PCA explained variance plots
    generate_variance_plot('winequality', evp=evp)
    generate_variance_plot('seismic-bumps', evp=evp)


if __name__ == '__main__':
    main()
