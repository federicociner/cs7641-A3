import pandas as pd
import numpy as np
import timeit
from helpers import cluster_acc, get_abspath, save_dataset, save_array
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


def clustering_experiment(X, y, name, clusters, rdir):
    """Generate results CSVs for given datasets using the K-Means and EM
    clustering algorithms.

    Args:
        X (Numpy.Array): Attributes.
        y (Numpy.Array): Labels.
        name (str): Dataset name.
        clusters (list[int]): List of k values.
        rdir (str): Output directory.

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
        acc[k][name]['km'] = cluster_acc(y, km.predict(X))
        acc[k][name]['gmm'] = cluster_acc(y, gmm.predict(X))
        adjmi[k][name]['km'] = ami(y, km.predict(X))
        adjmi[k][name]['gmm'] = ami(y, gmm.predict(X))

        # calculate silhouette score for K-Means
        km_silhouette = silhouette_score(X, km.predict(X))
        silhouette[k][name] = km_silhouette

        # calculate BIC for EM
        bic[k][name] = gmm.bic(X)

    # generate output dataframes
    sse = (-pd.DataFrame(sse)).T
    sse.rename(columns={name: 'sse'}, inplace=True)
    logl = pd.DataFrame(logl).T
    logl.rename(columns={name: 'log-likelihood'}, inplace=True)
    bic = pd.DataFrame(bic).T
    bic.rename(columns={name: 'bic'}, inplace=True)
    silhouette = pd.DataFrame(silhouette).T
    silhouette.rename(columns={name: 'silhouette_score'}, inplace=True)
    acc = pd.Panel(acc)
    acc = acc.loc[:, :, name].T.rename(
        lambda x: '{}_acc'.format(x), axis='columns')
    adjmi = pd.Panel(adjmi)
    adjmi = adjmi.loc[:, :, name].T.rename(
        lambda x: '{}_adjmi'.format(x), axis='columns')

    # concatenate all results
    dfs = (sse, silhouette, logl, bic, acc, adjmi)
    metrics = pd.concat(dfs, axis=1)
    resfile = get_abspath('{}_metrics.csv'.format(name), rdir)
    metrics.to_csv(resfile, index_label='k')


def generate_component_plots(name, rdir, pdir):
    """Generates plots of result files for given dataset.

    Args:
        name (str): Dataset name.
        rdir (str): Input file directory.
        pdir (str): Output directory.

    """
    metrics = pd.read_csv(get_abspath('{}_metrics.csv'.format(name), rdir))

    # get figure and axes
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1,
                                             ncols=4,
                                             figsize=(15, 3))

    # plot SSE for K-Means
    k = metrics['k']
    metric = metrics['sse']
    ax1.plot(k, metric, marker='o', markersize=5, color='g')
    ax1.set_title('K-Means SSE ({})'.format(name))
    ax1.set_ylabel('Sum of squared error')
    ax1.set_xlabel('Number of clusters (k)')
    ax1.grid(color='grey', linestyle='dotted')

    # plot Silhoutte Score for K-Means
    metric = metrics['silhouette_score']
    ax2.plot(k, metric, marker='o', markersize=5, color='b')
    ax2.set_title('K-Means Avg Silhouette Score ({})'.format(name))
    ax2.set_ylabel('Mean silhouette score')
    ax2.set_xlabel('Number of clusters (k)')
    ax2.grid(color='grey', linestyle='dotted')

    # plot log-likelihood for EM
    metric = metrics['log-likelihood']
    ax3.plot(k, metric, marker='o', markersize=5, color='r')
    ax3.set_title('GMM Log-likelihood ({})'.format(name))
    ax3.set_ylabel('Log-likelihood')
    ax3.set_xlabel('Number of clusters (k)')
    ax3.grid(color='grey', linestyle='dotted')

    # plot BIC for EM
    metric = metrics['bic']
    ax4.plot(k, metric, marker='o', markersize=5, color='k')
    ax4.set_title('GMM BIC ({})'.format(name))
    ax4.set_ylabel('BIC')
    ax4.set_xlabel('Number of clusters (k)')
    ax4.grid(color='grey', linestyle='dotted')

    # change layout size, font size and width between subplots
    fig.tight_layout()
    for ax in fig.axes:
        ax_items = [ax.title, ax.xaxis.label, ax.yaxis.label]
        for item in ax_items + ax.get_xticklabels() + ax.get_yticklabels():
            item.set_fontsize(8)
    plt.subplots_adjust(wspace=0.3)

    # save figure
    plotpath = get_abspath('{}_components.png'.format(name), pdir)
    plt.savefig(plotpath)
    plt.clf()


def generate_validation_plots(name, rdir, pdir):
    """Generates plots of validation metrics (accuracy, adjusted mutual info)
    for both datasets.

    Args:
        name (str): Dataset name.
        rdir (str): Input file directory.
        pdir (str): Output directory.

    """
    metrics = pd.read_csv(get_abspath('{}_metrics.csv'.format(name), rdir))

    # get figure and axes
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))

    # plot accuracy
    k = metrics['k']
    km = metrics['km_acc']
    gmm = metrics['gmm_acc']
    ax1.plot(k, km, marker='o', markersize=5, color='b', label='K-Means')
    ax1.plot(k, gmm, marker='o', markersize=5, color='g', label='GMM')
    ax1.set_title('Accuracy Score ({})'.format(name))
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Number of clusters (k)')
    ax1.grid(color='grey', linestyle='dotted')
    ax1.legend(loc='best')

    # plot adjusted mutual info
    km = metrics['km_adjmi']
    gmm = metrics['gmm_adjmi']
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
        ax_items = [ax.title, ax.xaxis.label, ax.yaxis.label]
        for item in ax_items + ax.get_xticklabels() + ax.get_yticklabels():
            item.set_fontsize(8)
    plt.subplots_adjust(wspace=0.3)

    # save figure
    plotpath = get_abspath('{}_validation.png'.format(name), pdir)
    plt.savefig(plotpath)
    plt.clf()


def get_cluster_data(X, y, name, km_k, gmm_k, rdir, perplexity=30):
    """Generates 2D dataset that contains cluster labels for K-Means and GMM,
    as well as the class labels for the given dataset.

    Args:
        X (Numpy.Array): Attributes.
        y (Numpy.Array): Labels.
        name (str): Dataset name.
        perplexity (int): Perplexity parameter for t-SNE.
        km_k (int): Number of clusters for K-Means.
        gmm_k (int): Number of components for GMM.
        rdir (str): Folder to save results CSV.

    """
    # generate 2D X dataset
    X2D = TSNE(n_iter=5000, perplexity=perplexity).fit_transform(X)

    # get cluster labels using best k
    km = KMeans(random_state=0).set_params(n_clusters=km_k)
    gmm = GMM(random_state=0).set_params(n_components=gmm_k)
    km_cl = np.atleast_2d(km.fit(X2D).labels_).T
    gmm_cl = np.atleast_2d(gmm.fit(X2D).predict(X2D)).T
    y = np.atleast_2d(y).T

    # create concatenated dataset
    cols = ['x1', 'x2', 'km', 'gmm', 'class']
    df = pd.DataFrame(np.hstack((X2D, km_cl, gmm_cl, y)), columns=cols)

    # save as CSV
    filename = '{}_2D.csv'.format(name)
    save_dataset(df, filename, sep=',', subdir=rdir, header=True)


def generate_cluster_plots(df, name, pdir):
    """Visualizes clusters using pre-processed 2D dataset.

    Args:
        df (Pandas.DataFrame): Dataset containing attributes and labels.
        name (str): Dataset name.
        pdir (str): Output folder for plots.

    """
    # get cols
    x1 = df['x1']
    x2 = df['x2']
    km = df['km']
    gmm = df['gmm']
    c = df['class']

    # plot cluster scatter plots
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(12, 3))
    ax1.scatter(x1, x2, marker='x', s=20, c=km, cmap='gist_rainbow')
    ax1.set_title('K-Means Clusters ({})'.format(name))
    ax1.set_ylabel('x1')
    ax1.set_xlabel('x2')
    ax1.grid(color='grey', linestyle='dotted')

    ax2.scatter(x1, x2, marker='x', s=20, c=gmm, cmap='gist_rainbow')
    ax2.set_title('GMM Clusters ({})'.format(name))
    ax2.set_ylabel('x1')
    ax2.set_xlabel('x2')
    ax2.grid(color='grey', linestyle='dotted')

    # change color map depending on dataset
    cmap = None
    if name == 'winequality':
        cmap = 'hsv'
    else:
        cmap = 'summer'
    ax3.scatter(x1, x2, marker='o', s=20, c=c, cmap=cmap)
    ax3.set_title('Class Labels ({})'.format(name))
    ax3.set_ylabel('x1')
    ax3.set_xlabel('x2')
    ax3.grid(color='grey', linestyle='dotted')

    # change layout size, font size and width between subplots
    fig.tight_layout()
    for ax in fig.axes:
        ax_items = [ax.title, ax.xaxis.label, ax.yaxis.label]
        for item in ax_items + ax.get_xticklabels() + ax.get_yticklabels():
            item.set_fontsize(8)
    plt.subplots_adjust(wspace=0.3)

    # save figure
    plotdir = pdir
    plotpath = get_abspath('{}_clusters.png'.format(name), plotdir)
    plt.savefig(plotpath)
    plt.clf()

def nn_cluster_datasets(X, name, km_k, gmm_k):
    """Generates datasets for ANN classification by appending cluster label to
    original dataset.

    Args:
        X (Numpy.Array): Original attributes.
        name (str): Dataset name.
        km_k (int): Number of clusters for K-Means.
        gmm_k (int): Number of components for GMM.

    """
    km = KMeans(random_state=0).set_params(n_clusters=km_k)
    gmm = GMM(random_state=0).set_params(n_components=gmm_k)
    km.fit(X)
    gmm.fit(X)

    # add cluster labels to original attributes
    km_x = np.concatenate((X, km.labels_[:, None]), axis=1)
    gmm_x = np.concatenate((X, gmm.predict(X)[:, None]), axis=1)

    # save results
    resdir = 'results/NN'
    kmfile = get_abspath('{}_km_labels.csv'.format(name), resdir)
    gmmfile = get_abspath('{}_gmm_labels.csv'.format(name), resdir)
    save_array(array=km_x, filename=kmfile, subdir=resdir)
    save_array(array=gmm_x, filename=gmmfile, subdir=resdir)


def main():
    """Run code to generate clustering results.

    """
    print 'Running base clustering experiments'
    start_time = timeit.default_timer()

    winepath = get_abspath('winequality.csv', 'data/experiments')
    seismicpath = get_abspath('seismic-bumps.csv', 'data/experiments')
    wine = np.loadtxt(winepath, delimiter=',')
    seismic = np.loadtxt(seismicpath, delimiter=',')
    rdir = 'results/clustering'
    pdir = 'plots/clustering'

    # split data into X and yreduced
    wX = wine[:, :-1]
    wY = wine[:, -1]
    sX = seismic[:, :-1]
    sY = seismic[:, -1]

    # run clustering experiments
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

    # generate neural network datasets with cluster labels
    nn_cluster_datasets(wX, name='winequality', km_k=15, gmm_k=15)
    nn_cluster_datasets(sX, name='seismic-bumps', km_k=20, gmm_k=15)

    # calculate and print running time
    end_time = timeit.default_timer()
    elapsed = end_time - start_time
    print "Completed clustering experiments in {} seconds".format(elapsed)

if __name__ == '__main__':
    main()
