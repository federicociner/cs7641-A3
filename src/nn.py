import timeit
import os
import numpy as np
from helpers import get_abspath, balanced_accuracy
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split


def create_ann(name):
    """ Construct the multi-layer perceptron classifier object for a given
    dataset based on A1 best parameters.

    Args:
        name (str): Name of dataset.
    Returns:
        ann (sklearn.MLPClassifier): Neural network classifier.

    """
    ann = MLPClassifier(activation='relu', max_iter=5000,
                        solver='adam', learning_rate='adaptive')

    if name == 'winequality':
        hls = (500, 500, 500)
        alpha = 0.0316
        ann.set_params(hidden_layer_sizes=hls, alpha=alpha)
    elif name == 'seismic-bumps':
        hls = (250, 250)
        alpha = 316.22
        ann.set_params(hidden_layer_sizes=hls, alpha=alpha)

    return ann


def ann_experiment(X, y, name, ann):
    """Run ANN experiment and generate accuracy and timing score

    Args:
        X (Numpy.Array): Attributes.
        y (Numpy.Array): Labels.

    """
    # get training and test splits
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0, stratify=y)

    # train model
    start_time = timeit.default_timer()
    ann.fit(X_train, y_train)
    end_time = timeit.default_timer()
    elapsed = end_time - start_time

    # get predicted labels using test data and score
    y_pred = ann.predict(X_test)
    acc = balanced_accuracy(y_test, y_pred)

    return acc, elapsed


def main():
    """Run code to generate results.

    """
    combined = get_abspath('combined_results.csv', 'results/NN')

    try:
        os.remove(combined)
    except:
        pass

    with open(combined, 'a') as f:
        f.write('dataset,algorithm,accuracy,elapsed_time\n')

    names = ['winequality', 'seismic-bumps']
    dimred_algos = ['PCA', 'ICA', 'RP', 'RF']
    cluster_algos = ['km', 'gmm']

    # generate results
    for name in names:
        # get labels
        filepath = get_abspath('{}.csv'.format(name), 'data/experiments')
        data = np.loadtxt(filepath, delimiter=',')
        X = data[:, :-1]
        y = data[:, -1]

        # save base dataset results
        ann = create_ann(name=name)
        acc, elapsed = ann_experiment(X, y, name, ann)
        with open(combined, 'a') as f:
            f.write('{},{},{},{}\n'.format(name, 'base', acc, elapsed))

        for d in dimred_algos:
            # get attributes
            resdir = 'results/{}'.format(d)
            filepath = get_abspath('{}_projected.csv'.format(name), resdir)
            X = np.loadtxt(filepath, delimiter=',')

            # train ANN and get test score, elapsed time
            ann = create_ann(name=name)
            acc, elapsed = ann_experiment(X, y, name, ann)
            with open(combined, 'a') as f:
                f.write('{},{},{},{}\n'.format(name, d, acc, elapsed))

        for c in cluster_algos:
            # get attributes
            resdir = 'results/NN'
            filepath = get_abspath('{}_{}_labels.csv'.format(name, c), resdir)
            X = np.loadtxt(filepath, delimiter=',')

            # train ANN and get test score, elapsed time
            ann = create_ann(name=name)
            acc, elapsed = ann_experiment(X, y, name, ann)
            with open(combined, 'a') as f:
                f.write('{},{},{},{}\n'.format(name, c, acc, elapsed))


if __name__ == '__main__':
    main()
