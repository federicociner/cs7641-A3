from scipy.io import arff
from helpers import get_abspath, save_dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os


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


def get_splits(X, y, dname, filepath='data/experiments'):
    """ Splits X and y datasets into training, validation, and test data sets
    and then saves them as CSVs

    Args:
        X (Numpy.Array): Attributes.
        y (Numpy.Array): Classes.
        dname (str): Dataset name.
        filepath (str): Output folder.

    """
    # get train and test splits
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0, stratify=y)

    # combine datasets
    np_train = np.concatenate((X_train, y_train[:, np.newaxis]), axis=1)
    train = pd.DataFrame(np_train)

    np_test = np.concatenate((X_test, y_test[:, np.newaxis]), axis=1)
    test = pd.DataFrame(np_test)

    # save datasets to CSV
    output_path = 'data/experiments'
    trainfile = '{}_train.csv'.format(dname)
    testfile = '{}_test.csv'.format(dname)
    save_dataset(train, trainfile, subdir=output_path, header=False)
    save_dataset(test, testfile, subdir=output_path, header=False)


def preprocess_winequality():
    """Cleans and generates wine quality dataset for experiments as a
    CSV file.

    """
    # get file paths
    sdir = 'data/winequality'
    tdir = 'data/experiments'
    wr_file = get_abspath('winequality-red.csv', sdir)
    ww_file = get_abspath('winequality-white.csv', sdir)

    # load as data frame
    wine_red = pd.read_csv(wr_file, sep=';')
    wine_white = pd.read_csv(ww_file, sep=';')

    # encode artifical label to determine if wine is red or not
    wine_red['red'] = 1
    wine_white['red'] = 0

    # combine datasets and format column names
    df = wine_red.append(wine_white)
    df.columns = ['_'.join(col.split(' ')) for col in df.columns]
    df.rename(columns={'quality': 'class'}, inplace=True)

    # split out X data and scale (Gaussian zero mean and unit variance)
    X = df.drop(columns='class').as_matrix()
    y = df['class'].as_matrix()
    X_scaled = StandardScaler().fit_transform(X)
    data = np.concatenate((X_scaled, y[:, np.newaxis]), axis=1)

    # save to CSV
    save_array(array=data, filename='winequality.csv', subdir=tdir)


def preprocess_seismic():
    """Cleans and generates seismic bumps dataset for experiments as a
    CSV file. Uses one-hot encoding for categorical features.

    """
    # get file path
    sdir = 'data/seismic-bumps'
    tdir = 'data/experiments'
    seismic_file = get_abspath('seismic-bumps.arff', sdir)

    # read arff file and convert to record array
    rawdata = arff.loadarff(seismic_file)
    df = pd.DataFrame(rawdata[0])

    # apply one-hot encoding to categorical features using Pandas get_dummies
    cat_cols = ['seismic', 'seismoacoustic', 'shift', 'ghazard']
    cats = df[cat_cols]
    onehot_cols = pd.get_dummies(cats, prefix=cat_cols)

    # replace 0s with -1s to improve NN performance
    onehot_cols.replace(to_replace=[0], value=[-1], inplace=True)

    # drop original categorical columns and append one-hot encoded columns
    df.drop(columns=cat_cols, inplace=True)
    df = pd.concat((onehot_cols, df), axis=1)

    # drop columns that have only 1 unique value (features add no information)
    for col in df.columns:
        if len(np.unique(df[col])) == 1:
            df.drop(columns=col, inplace=True)

    # cast class column as integer
    df['class'] = df['class'].astype(int)

    # split out X data and scale (Gaussian zero mean and unit variance)
    X = df.drop(columns='class').as_matrix()
    y = df['class'].as_matrix()
    X_scaled = StandardScaler().fit_transform(X)
    data = np.concatenate((X_scaled, y[:, np.newaxis]), axis=1)

    # save to CSV
    save_array(array=data, filename='seismic_bumps.csv', subdir=tdir)


def main():
    # run preprocessing functions
    preprocess_winequality()
    preprocess_seismic()


if __name__ == '__main__':
    main()
