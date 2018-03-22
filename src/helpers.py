import os
import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score


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


def cluster_acc(Y, clusterY):
    """ Calculates accuracy of labels in each cluster by comparing to the
    actual Y labels.

    Args:
        Y (Numpy.Array): Actual labels.
        clusterY (Numpy.Array): Predicted labels per cluster.
    Returns:
        score (float): Accuracy score for given cluster labels.

    """
    assert (Y.shape == clusterY.shape)
    pred = np.empty_like(Y)
    for label in set(clusterY):
        mask = clusterY == label
        sub = Y[mask]
        target = Counter(sub).most_common(1)[0][0]
        pred[mask] = target

    return accuracy_score(Y, pred)
