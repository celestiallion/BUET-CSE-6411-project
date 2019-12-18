import os
import argparse
import h5py

import graphviz
from sklearn import ensemble
from sklearn import tree

from joblib import dump


def main(args):
    data_file_path = os.path.join(args.data_root, args.data_file)
    hf = h5py.File(data_file_path, 'r')
    X, y = hf['X'][:], hf['y'][:]
    split_point = int(0.8 * len(y))
    X_train, X_test = X[:split_point], X[split_point:]
    y_train, y_test = y[:split_point], y[split_point:]

    clf = ensemble.RandomForestClassifier(n_estimators=5)
    clf.fit(X_train, y_train)
    test_acc = clf.score(X_test, y_test)
    print(test_acc)

    dump(clf, '/home/adnan/PycharmProjects/BUET-CSE-6411-project/data/models/random_forest_classifier_{}.joblib'.format(test_acc))

    for count, estimator_ in enumerate(clf.estimators_):
        dot_data = tree.export_graphviz(estimator_, out_file=None)
        graph = graphviz.Source(dot_data)
        graph.render('estimator_{}'.format(count))
        print(count)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/home/adnan/Datasets/')
    parser.add_argument('--data_file', type=str, default='neighbor_rank_histogram_dataset_oversampled_window_5.h5')
    parser_args = parser.parse_args()

    main(parser_args)
