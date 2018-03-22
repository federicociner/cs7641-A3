import pandas as pd
import numpy as np
from helpers import cluster_acc, get_abspath
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture as GMM
from collections import defaultdict
from sklearn.metrics import adjusted_mutual_info_score as ami
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn


def clustering_experiment(X, y, name, clusters):
    """Generate results CSVs for given datasets using the K-Means and EM
    clustering algorithms.

    Args:
        X (Numpy.Array): Attributes.
        y (Numpy.Array): Labels.
        name (str): Dataset name.
        clusters (list[int]): List of k values.

    """
    sse = defaultdict(dict)  # sum of squared errors
    logl = defaultdict(dict)  # log-likelihood
    bic = defaultdict(dict)  # BIC for EM
    silhouette = defaultdict(dict)  # silhouette score
    acc = defaultdict(lambda: defaultdict(dict))  # accuracy scores
    adjmi = defaultdict(lambda: defaultdict(dict))  # adjusted mutual info
    km = KMeans(random_state=0)  # K-Means
    gmm = GMM(random_state=0)  # Gaussian Mixture Model (EM)

    # start loop for given values of k
    for k in clusters:
        km.set_params(n_clusters=k)
        gmm.set_params(n_components=k)
        km.fit(X)
        gmm.fit(X)

        # calculate SSE, log-likelihood, accuracy, and adjusted mutual info
        sse[k][name] = km.score(X)
        logl[k][name] = gmm.score(X)
        acc[k][name]['K-Means'] = cluster_acc(y, km.predict(X))
        acc[k][name]['GMM'] = cluster_acc(y, gmm.predict(X))
        adjmi[k][name]['K-Means'] = ami(y, km.predict(X))
        adjmi[k][name]['GMM'] = ami(y, gmm.predict(X))

        # calculate silhouette score for K-Means
        km_silhouette = silhouette_score(X, km.predict(X))
        silhouette[k][name] = km_silhouette

        # calculate BIC for EM
        bic[k][name] = gmm.bic(X)

    # generate output dataframes
    sse = (-pd.DataFrame(sse)).T
    sse.rename(columns={name: 'SSE'}, inplace=True)
    logl = pd.DataFrame(logl).T
    logl.rename(columns={name: 'Log-likelihood'}, inplace=True)
    bic = pd.DataFrame(bic).T
    bic.rename(columns={name: 'BIC'}, inplace=True)
    silhouette = pd.DataFrame(silhouette).T
    silhouette.rename(columns={name: 'Silhouette Score'}, inplace=True)
    acc = pd.Panel(acc)
    adjmi = pd.Panel(adjmi)
    outdir = 'results/clustering'

    # SSE
    ssefile = get_abspath('{}_sse.csv'.format(name), outdir)
    sse.to_csv(ssefile, index_label='k')

    # Log-likelihood
    loglfile = get_abspath('{}_logl.csv'.format(name), outdir)
    logl.to_csv(loglfile, index_label='k')

    # BIC
    bicfile = get_abspath('{}_bic.csv'.format(name), outdir)
    bic.to_csv(bicfile, index_label='k')

    # Silhouette Score
    silfile = get_abspath('{}_silhouette.csv'.format(name), outdir)
    silhouette.to_csv(silfile, index_label='k')

    # save accuracy results
    accfile = get_abspath('{}_acc.csv'.format(name), outdir)
    acc.loc[:, :, name].T.to_csv(accfile, index_label='k')

    # save adjusted mutual info results
    adjmifile = get_abspath('{}_adjmi.csv'.format(name), outdir)
    adjmi.loc[:, :, name].T.to_csv(adjmifile, index_label='k')


def generate_component_plots(name):
    """Generates plots of result files for given dataset.

    Args:
        name (str): Dataset name.

    """
    resdir = 'results/clustering'
    sse = pd.read_csv(get_abspath('{}_sse.csv'.format(name), resdir))
    logl = pd.read_csv(get_abspath('{}_logl.csv'.format(name), resdir))
    bic = pd.read_csv(get_abspath('{}_bic.csv'.format(name), resdir))
    sil = pd.read_csv(get_abspath('{}_silhouette.csv'.format(name), resdir))

    # get figure and axes
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1,
                                             ncols=4,
                                             figsize=(15, 3))

    # plot SSE for K-Means
    k = sse['k']
    metric = sse['SSE']
    ax1.plot(k, metric, marker='o', markersize=5, color='g')
    ax1.set_title('K-Means SSE ({})'.format(name))
    ax1.set_ylabel('Sum of squared error')
    ax1.set_xlabel('Number of clusters (k)')
    ax1.grid(color='grey', linestyle='dotted')

    # plot Silhoutte Score for K-Means
    metric = sil['Silhouette Score']
    ax2.plot(k, metric, marker='o', markersize=5, color='b')
    ax2.set_title('K-Means Avg Silhouette Score ({})'.format(name))
    ax2.set_ylabel('Mean silhouette score')
    ax2.set_xlabel('Number of clusters (k)')
    ax2.grid(color='grey', linestyle='dotted')

    # plot log-likelihood for EM
    metric = logl['Log-likelihood']
    ax3.plot(k, metric, marker='o', markersize=5, color='r')
    ax3.set_title('GMM Log-likelihood ({})'.format(name))
    ax3.set_ylabel('Log-likelihood')
    ax3.set_xlabel('Number of clusters (k)')
    ax3.grid(color='grey', linestyle='dotted')

    # plot BIC for EM
    metric = bic['BIC']
    ax4.plot(k, metric, marker='o', markersize=5, color='k')
    ax4.set_title('GMM BIC ({})'.format(name))
    ax4.set_ylabel('BIC')
    ax4.set_xlabel('Number of clusters (k)')
    ax4.grid(color='grey', linestyle='dotted')

    # change layout size, font size and width between subplots
    fig.tight_layout()
    for ax in fig.axes:
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(8)
    plt.subplots_adjust(wspace=0.3)

    # save figure
    plotdir = 'plots/clustering'
    plotpath = get_abspath('{}_components.png'.format(name), plotdir)
    plt.savefig(plotpath)
    plt.clf()


def generate_validation_plots(name):
    """Generates plots of validation metrics (accuracy, adjusted mutual info)
    for both datasets.

    Args:
        name (str): Dataset name.

    """
    resdir = 'results/clustering'
    acc = pd.read_csv(get_abspath('{}_acc.csv'.format(name), resdir))
    adjmi = pd.read_csv(get_abspath('{}_adjmi.csv'.format(name), resdir))

    # get figure and axes
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))

    # plot accuracy
    k = acc['k']
    km = acc['K-Means']
    gmm = acc['GMM']
    ax1.plot(k, km, marker='o', markersize=5, color='b', label='K-Means')
    ax1.plot(k, gmm, marker='o', markersize=5, color='g', label='GMM')
    ax1.set_title('Accuracy Score ({})'.format(name))
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Number of clusters (k)')
    ax1.grid(color='grey', linestyle='dotted')
    ax1.legend(loc='best')

    # plot adjusted mutual info
    km = adjmi['K-Means']
    gmm = adjmi['GMM']
    ax2.plot(k, km, marker='o', markersize=5, color='r', label='K-Means')
    ax2.plot(k, gmm, marker='o', markersize=5, color='k', label='GMM')
    ax2.set_title('Adjusted Mutual Info ({})'.format(name))
    ax2.set_ylabel('Adjusted mutual information score')
    ax2.set_xlabel('Number of clusters (k)')
    ax2.grid(color='grey', linestyle='dotted')
    ax2.legend(loc='best')

    # change layout size, font size and width between subplots
    fig.tight_layout()
    for ax in fig.axes:
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(8)
    plt.subplots_adjust(wspace=0.3)

    # save figure
    plotdir = 'plots/clustering'
    plotpath = get_abspath('{}_validation.png'.format(name), plotdir)
    plt.savefig(plotpath)
    plt.clf()


def generate_cluster_plots():
    """Visualizes clusters using t-SNE to decompose to 2D for both datasets.

    """
    print 'hi'


def main():
    """Run code to generate clustering results.

    """
    winepath = get_abspath('winequality.csv', 'data/experiments')
    seismicpath = get_abspath('seismic_bumps.csv', 'data/experiments')
    wine = np.loadtxt(winepath, delimiter=',')
    seismic = np.loadtxt(seismicpath, delimiter=',')

    # split data into X and y
    wineX = wine[:, :-1]
    wineY = wine[:, -1]
    seismicX = seismic[:, :-1]
    seismicY = seismic[:, -1]

    # run clustering experiments
    clusters = [2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 18, 20, 25, 30, 45, 80, 120]
    clustering_experiment(wineX, wineY, 'winequality', clusters)
    clustering_experiment(seismicX, seismicY, 'seismic-bumps', clusters)

    # generate component plots (metrics to choose size of k)
    generate_component_plots(name='winequality')
    generate_component_plots(name='seismic-bumps')

    # generate validation plots (relative performance of clustering)
    generate_validation_plots(name='winequality')
    generate_validation_plots(name='seismic-bumps')

if __name__ == '__main__':
    main()
