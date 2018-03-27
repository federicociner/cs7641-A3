import pandas as pd
import numpy as np
from clustering import get_cluster_data, generate_validation_plots
from clustering import clustering_experiment, generate_cluster_plots
from clustering import generate_component_plots
from helpers import get_abspath, save_array
from sklearn.decomposition import FastICA
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn


def ica_experiment(X, name, dims):
    """Run ICA on specified dataset and saves mean kurtosis results as CSV
    file.

    Args:
        X (Numpy.Array): Attributes.
        y (Numpy.Array): Labels.
        name (str): Dataset name.
        dims (list(int)): List of component number values.

    """
    ica = FastICA(random_state=0, max_iter=5000)
    kurt = {}

    for dim in dims:
        ica.set_params(n_components=dim)
        tmp = ica.fit_transform(X)
        df = pd.DataFrame(tmp)
        df = df.kurt(axis=0)
        kurt[dim] = df.abs().mean()

    res = pd.DataFrame.from_dict(kurt, orient='index')
    res.rename(columns={0: 'kurtosis'}, inplace=True)

    # save results as CSV
    resdir = 'results/ICA'
    resfile = get_abspath('{}_kurtosis.csv'.format(name), resdir)
    res.to_csv(resfile, index_label='comp_no')


def save_ica_results(X, name, dims):
    """Run ICA and save projected dataset as CSV.

    Args:
        X (Numpy.Array): Attributes.
        name (str): Dataset name.
        dims (int): Number of components.

    """
    # transform data using ICA
    ica = FastICA(random_state=0, max_iter=5000, n_components=dims)
    res = ica.fit_transform(X)

    # save results file
    resdir = 'results/ICA'
    resfile = get_abspath('{}_projected.csv'.format(name), resdir)
    save_array(array=res, filename=resfile, subdir=resdir)


def generate_kurtosis_plot(name):
    """Plots mean kurtosis as a function of number of components.

    Args:
        name (str): Dataset name.

    """
    resdir = 'results/ICA'
    df = pd.read_csv(get_abspath('{}_kurtosis.csv'.format(name), resdir))

    # get figure and axes
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 3))

    # plot explained variance and cumulative explain variance ratios
    x = df['comp_no']
    kurt = df['kurtosis']
    ax.plot(x, kurt, marker='.', color='g')
    ax.set_title('ICA Mean Kurtosis ({})'.format(name))
    ax.set_ylabel('Mean Kurtosis')
    ax.set_xlabel('# Components')
    ax.grid(color='grey', linestyle='dotted')

    # change layout size, font size and width
    fig.tight_layout()
    for ax in fig.axes:
        ax_items = [ax.title, ax.xaxis.label, ax.yaxis.label]
        for item in (ax_items + ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(8)

    # save figure
    plotdir = 'plots/ICA'
    plotpath = get_abspath('{}_kurtosis.png'.format(name), plotdir)
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
    get_cluster_data(wX, wY, 'winequality', km_k=15, gmm_k=12, rdir=rdir)
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
    rdir = 'results/ICA'
    pdir = 'plots/ICA'

    # generate PCA results
    ica_experiment(wX, 'winequality', dims=range(1, wDims + 1))
    ica_experiment(sX, 'seismic-bumps', dims=range(1, sDims + 1))

    # generate PCA explained variance plots
    generate_kurtosis_plot('winequality')
    generate_kurtosis_plot('seismic-bumps')

    # save ICA results with best # of components
    save_ica_results(wX, 'winequality', dims=8)
    save_ica_results(sX, 'seismic-bumps', dims=17)

    # re-run clustering experiments
    run_clustering(wY, sY, rdir, pdir)


if __name__ == '__main__':
    main()
