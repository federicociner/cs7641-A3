import pandas as pd
import numpy as np
import timeit
from clustering import get_cluster_data, generate_validation_plots
from clustering import clustering_experiment, generate_cluster_plots
from clustering import generate_component_plots
from helpers import get_abspath, save_array
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
    resdir = 'results/PCA'
    evfile = get_abspath('{}_variances.csv'.format(name), resdir)
    resfile = get_abspath('{}_projected.csv'.format(name), resdir)
    save_array(array=res, filename=resfile, subdir=resdir)
    evars.to_csv(evfile, index_label='n')


def generate_variance_plot(name, evp):
    """Plots explained variance and cumulative explained variance ratios as a
    function of principal components.

    Args:
        name (str): Dataset name.
        evp (float): Explained variance percentage threshold.

    """
    resdir = 'results/PCA'
    df = pd.read_csv(get_abspath('{}_variances.csv'.format(name), resdir))

    # get figure and axes
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 3))

    # plot explained variance and cumulative explain variance ratios
    x = df['n']
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
        for item in ax_items + ax.get_xticklabels() + ax.get_yticklabels():
            item.set_fontsize(8)

    # save figure
    plotdir = 'plots/PCA'
    plotpath = get_abspath('{}_explvar.png'.format(name), plotdir)
    plt.savefig(plotpath)
    plt.clf()


def run_clustering(wY, sY, rdir, pdir):
    """Re-run clustering experiments on datasets after dimensionality
    reduction.

    Args:
        wY (Numpy.Array): Labels for winequality.
        sY (Numpy.Array): Labels for seismic-bumps.
        rdir (str): Input file directory.
        pdir (str): Output directory.

    """
    winepath = get_abspath('winequality_projected.csv', rdir)
    seismicpath = get_abspath('seismic-bumps_projected.csv', rdir)
    wX = np.loadtxt(winepath, delimiter=',')
    sX = np.loadtxt(seismicpath, delimiter=',')
    rdir = rdir + '/clustering'
    pdir = pdir + '/clustering'

    # re-run clustering experiments after applying PCA
    clusters = [2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 18, 20, 25, 30, 45, 80, 120]
    clustering_experiment(wX, wY, 'winequality', clusters, rdir=rdir)
    clustering_experiment(sX, sY, 'seismic-bumps', clusters, rdir=rdir)

    # generate 2D data for cluster visualization
    get_cluster_data(wX, wY, 'winequality', km_k=15, gmm_k=15, rdir=rdir)
    get_cluster_data(sX, sY, 'seismic-bumps', km_k=20, gmm_k=15, rdir=rdir)

    # generate component plots (metrics to choose size of k)
    generate_component_plots(name='winequality', rdir=rdir, pdir=pdir)
    generate_component_plots(name='seismic-bumps', rdir=rdir, pdir=pdir)

    # # generate validation plots (relative performance of clustering)
    generate_validation_plots(name='winequality', rdir=rdir, pdir=pdir)
    generate_validation_plots(name='seismic-bumps', rdir=rdir, pdir=pdir)

    # generate validation plots (relative performance of clustering)
    df_wine = pd.read_csv(get_abspath('winequality_2D.csv', rdir))
    df_seismic = pd.read_csv(get_abspath('seismic-bumps_2D.csv', rdir))
    generate_cluster_plots(df_wine, name='winequality', pdir=pdir)
    generate_cluster_plots(df_seismic, name='seismic-bumps', pdir=pdir)


def main():
    """Run code to generate results.

    """
    print 'Running PCA experiments'
    start_time = timeit.default_timer()

    winepath = get_abspath('winequality.csv', 'data/experiments')
    seismicpath = get_abspath('seismic-bumps.csv', 'data/experiments')
    wine = np.loadtxt(winepath, delimiter=',')
    seismic = np.loadtxt(seismicpath, delimiter=',')

    # set explained variance threshold
    evp = 0.85

    # split data into X and y
    wX = wine[:, :-1]
    wY = wine[:, -1]
    sX = seismic[:, :-1]
    sY = seismic[:, -1]
    wDims = wX.shape[1]
    sDims = sX.shape[1]
    rdir = 'results/PCA'
    pdir = 'plots/PCA'

    # generate PCA results
    pca_experiment(wX, 'winequality', dims=wDims, evp=evp)
    pca_experiment(sX, 'seismic-bumps', dims=sDims, evp=evp)

    # generate PCA explained variance plots
    generate_variance_plot('winequality', evp=evp)
    generate_variance_plot('seismic-bumps', evp=evp)

    # re-run clustering experiments
    run_clustering(wY, sY, rdir, pdir)

    # calculate and print running time
    end_time = timeit.default_timer()
    elapsed = end_time - start_time
    print "Completed PCA experiments in {} seconds".format(elapsed)


if __name__ == '__main__':
    main()
