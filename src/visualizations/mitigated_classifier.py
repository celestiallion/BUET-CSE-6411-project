import os
import argparse
import h5py

import graphviz
from sklearn import ensemble
from sklearn import tree

from joblib import dump
from sklearn.metrics import matthews_corrcoef

import numpy as np


def main(args):
    data_file_path = os.path.join(args.data_root, args.data_file)
    hf = h5py.File(data_file_path, 'r')
    X, y = hf['X'][:], hf['y'][:]
    split_point = int(0.03 * len(y))
    X_train, X_test = X[:split_point], X[split_point:]
    y_train, y_test = y[:split_point], y[split_point:]

    X_test_mitigated, y_test_mitigated = [], []
    for X, y in zip(X_test, y_test):
        if y == 5 or y == 7:
            continue
        continue_flag = False
        for _X in X_train:
            if all(X == _X):
                continue_flag = True
                break
        if continue_flag:
            continue
        else:
            X_test_mitigated.append(X)
            y_test_mitigated.append(y)
    X_test_mitigated, y_test_mitigated = np.array(X_test_mitigated), np.array(y_test_mitigated)

    # clf = ensemble.RandomForestClassifier(n_estimators=5)
    # clf.fit(X_train, y_train)
    # print(clf.score(X_test_mitigated, y_test_mitigated))
    #
    # # dump(clf, '/home/adnan/PycharmProjects/BUET-CSE-6411-project/data/models/random_forest_classifier_{}.joblib'.format(test_acc))
    # #
    # for count, estimator_ in enumerate(clf.estimators_):
    #     dot_data = tree.export_graphviz(estimator_, out_file=None)
    #     graph = graphviz.Source(dot_data)
    #     graph.render('estimator_{}'.format(count))
    #     print(count)

    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    print(clf.score(X_test_mitigated, y_test_mitigated))

    dot_data = tree.export_graphviz(clf, out_file=None)
    graph = graphviz.Source(dot_data)
    graph.render('Decision_tree')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/home/adnan/Datasets/')
    parser.add_argument('--data_file', type=str, default='neighbor_rank_histogram_dataset_oversampled_window_3.h5')
    parser_args = parser.parse_args()

    main(parser_args)
