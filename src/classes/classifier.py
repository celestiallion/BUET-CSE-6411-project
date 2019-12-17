import os
import argparse
import h5py
from sklearn import ensemble


def main(args):
    data_file_path = os.path.join(args.data_root, args.data_file)
    hf = h5py.File(data_file_path, 'r')
    X, y = hf['X'][:], hf['y'][:]
    split_point = int(0.8 * len(y))
    X_train, X_test = X[:split_point], X[split_point:]
    y_train, y_test = y[:split_point], y[split_point:]

    clf = ensemble.RandomForestClassifier(n_estimators=5)
    clf.fit(X_train, y_train)
    print(clf.score(X_test, y_test))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/home/adnan/Datasets/')
    parser.add_argument('--data_file', type=str, default='neighbor_rank_histogram_dataset_oversampled_window_5.h5')
    parser_args = parser.parse_args()

    main(parser_args)
