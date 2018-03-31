import os
from collections import Counter
import numpy as np
import scipy.sparse as sps
from scipy.linalg import pinv
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.utils.class_weight import compute_sample_weight


def save_array(array, filename, sep=',', subdir='data'):
    """Saves a Numpy array as a delimited text file.

    Args:
        array (Numpy.Array): Input array.
        filename (str): Output file name.
        sep (str): Delimiter.
        subdir (str): Parent directory path for output file.

    """
    tdir = os.path.join(os.getcwd(), os.pardir, subdir, filename)
    np.savetxt(fname=tdir, X=array, delimiter=sep, fmt='%.20f')


def save_dataset(df, filename, sep=',', subdir='data', header=True):
    """Saves Pandas data frame as a CSV file.

    Args:
        df (Pandas.DataFrame): Data frame.
        filename (str): Output file name.
        sep (str): Delimiter.
        subdir (str): Project directory to save output file.
        header (Boolean): Specify inclusion of header.

    """
    tdir = os.path.join(os.getcwd(), os.pardir, subdir, filename)
    df.to_csv(path_or_buf=tdir, sep=sep, header=header, index=False)


def get_abspath(filename, filepath):
    """Gets absolute path of specified file within the project directory. The
    filepath has to be a subdirectory within the main project directory.

    Args:
        filename (str): Name of specified file.
        filepath (str): Subdirectory of file.
    Returns:
        fullpath (str): Absolute filepath.

    """
    p = os.path.abspath(os.path.join(os.curdir, os.pardir))
    fullpath = os.path.join(p, filepath, filename)

    return fullpath


def balanced_accuracy(labels, predictions):
    """Modifies the standard accuracy scoring function to account for
    potential imbalances in class distributions.

    Args:
        labels (numpy.array): Actual class labels.
        predictions (numpy.array): Predicted class labels.
    Returns:
        Modified accuracy scoring function.

    """
    weights = compute_sample_weight('balanced', labels)
    return accuracy_score(labels, predictions, sample_weight=weights)


def cluster_acc(Y, clusterY):
    """Calculates accuracy of labels in each cluster by comparing to the
    actual Y labels.

    Args:
        Y (Numpy.Array): Actual labels.
        clusterY (Numpy.Array): Predicted labels per cluster.
    Returns:
        score (float): Accuracy score for given cluster labels.

    """
    assert Y.shape == clusterY.shape
    pred = np.empty_like(Y)
    for label in set(clusterY):
        mask = clusterY == label
        sub = Y[mask]
        target = Counter(sub).most_common(1)[0][0]
        pred[mask] = target
    return balanced_accuracy(Y, pred)


def reconstruction_error(projections, x):
    """Calculates reconstruction error on a given set of projected data based
    on the original dataset.

    Args:
        projections (Numpy.Array): Random matrix used for projections.
        x (Numpy.Array): Original dataset.
    Returns:
        errors (Numpy.Array): Reconstruction error.

    """
    W = projections.components_
    if sps.issparse(W):
        W = W.todense()
    p = pinv(W)
    reconstructed = np.dot(np.dot(p, W), x.T).T  # Unproject projected data
    errors = np.square(x - reconstructed)
    return np.nanmean(errors)


def pairwise_dist_corr(x1, x2):
    """Calculates the pairwise distance correlation between two arrays.

    Args:
        x1 (Numpy.Array): First array.
        x2 (Numpy.Array): Second array.
    Returns:
        Numpy.Array of pairwise distance correlations.

    """
    assert x1.shape[0] == x2.shape[0]

    d1 = pairwise_distances(x1)
    d2 = pairwise_distances(x2)
    return np.corrcoef(d1.ravel(),d2.ravel())[0,1]
