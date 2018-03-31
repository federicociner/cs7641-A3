from collections import defaultdict
from itertools import product
import timeit
import pandas as pd
import numpy as np
from clustering import get_cluster_data, generate_validation_plots
from clustering import clustering_experiment, generate_cluster_plots
from clustering import generate_component_plots
from helpers import get_abspath, save_array
from helpers import reconstruction_error, pairwise_dist_corr
from sklearn.random_projection import SparseRandomProjection
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn


def rp_experiment(X, name, dims):
    """Run Randomized Projections on specified dataset and saves reconstruction
    error and pairwise distance correlation results as CSV file.

    Args:
        X (Numpy.Array): Attributes.
        name (str): Dataset name.
        dims (list(int)): List of component number values.

    """
    re = defaultdict(dict)
    pdc = defaultdict(dict)

    for i, dim in product(range(10), dims):
        rp = SparseRandomProjection(random_state=i, n_components=dim)
        rp.fit(X)
        re[dim][i] = reconstruction_error(rp, X)
        pdc[dim][i] = pairwise_dist_corr(rp.transform(X), X)

    re = pd.DataFrame(pd.DataFrame(re).T.mean(axis=1))
    re.rename(columns={0: 'recon_error'}, inplace=True)
    pdc = pd.DataFrame(pd.DataFrame(pdc).T.mean(axis=1))
    pdc.rename(columns={0: 'pairwise_dc'}, inplace=True)
    metrics = pd.concat((re, pdc), axis=1)

    # save results as CSV
    resdir = 'results/RP'
    resfile = get_abspath('{}_metrics.csv'.format(name), resdir)
    metrics.to_csv(resfile, index_label='n')


def save_rp_results(X, name, dims):
    """Run RP and save projected dataset as CSV.

    Args:
        X (Numpy.Array): Attributes.
        name (str): Dataset name.
        dims (int): Number of components.

    """
    # transform data using ICA
    rp = SparseRandomProjection(random_state=0, n_components=dims)
    res = rp.fit_transform(X)

    # save results file
    resdir = 'results/RP'
    resfile = get_abspath('{}_projected.csv'.format(name), resdir)
    save_array(array=res, filename=resfile, subdir=resdir)


def generate_plots(name):
    """Plots reconstruction error and pairwise distance correlation as a
    function of number of components.

    Args:
        name (str): Dataset name.

    """
    resdir = 'results/RP'
    df = pd.read_csv(get_abspath('{}_metrics.csv'.format(name), resdir))

    # get figure and axes
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 3))

    # plot metrics
    x = df['n']
    re = df['recon_error']
    pdc = df['pairwise_dc']
    ax1.plot(x, re, marker='.', color='g')
    ax1.set_title('Reconstruction Error ({})'.format(name))
    ax1.set_ylabel('Reconstruction error')
    ax1.set_xlabel('# Components')
    ax1.grid(color='grey', linestyle='dotted')

    ax2.plot(x, pdc, marker='.', color='b')
    ax2.set_title('Pairwise Distance Correlation ({})'.format(name))
    ax2.set_ylabel('Pairwise distance correlation')
    ax2.set_xlabel('# Components')
    ax2.grid(color='grey', linestyle='dotted')

    # change layout size, font size and width
    fig.tight_layout()
    for ax in fig.axes:
        ax_items = [ax.title, ax.xaxis.label, ax.yaxis.label]
        for item in ax_items + ax.get_xticklabels() + ax.get_yticklabels():
            item.set_fontsize(8)

    # save figure
    plotdir = 'plots/RP'
    plotpath = get_abspath('{}_metrics.png'.format(name), plotdir)
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

    # generate validation plots (relative performance of clustering)
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
    print 'Running RP experiments'
    start_time = timeit.default_timer()

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
    rdir = 'results/RP'
    pdir = 'plots/RP'

    # generate PCA results
    rp_experiment(wX, 'winequality', dims=range(1, wDims + 1))
    rp_experiment(sX, 'seismic-bumps', dims=range(1, sDims + 1))

    # generate PCA explained variance plots
    generate_plots(name='winequality')
    generate_plots(name='seismic-bumps')

    # save ICA results with best # of components
    save_rp_results(wX, 'winequality', dims=7)
    save_rp_results(sX, 'seismic-bumps', dims=10)

    # re-run clustering experiments
    run_clustering(wY, sY, rdir, pdir)

    # calculate and print running time
    end_time = timeit.default_timer()
    elapsed = end_time - start_time
    print "Completed RP experiments in {} seconds".format(elapsed)

if __name__ == '__main__':
    main()
