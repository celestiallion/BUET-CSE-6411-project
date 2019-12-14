import os
import argparse
import h5py

from sklearn import decomposition

from sklearn import linear_model
from sklearn import svm
from sklearn import naive_bayes
from sklearn import tree
from sklearn import ensemble


def main(args):
    data_file_path = os.path.join(args.data_root, args.data_file)
    hf = h5py.File(data_file_path, 'r')
    X, y = hf['X'][:], hf['y'][:]
    split_point = int(0.8 * len(y))
    X_train, X_test = X[:split_point], X[split_point:]
    y_train, y_test = y[:split_point], y[split_point:]

    # svd = decomposition.TruncatedSVD(n_components=64, n_iter=10, random_state=42)
    # X_train = svd.fit_transform(X_train)
    # X_test = svd.fit_transform(X_test)
    # nmf = decomposition.NMF(n_components=64, init='random', random_state=0)
    # X_train = nmf.fit_transform(X_train)
    # X_test = nmf.fit_transform(X_test)

    # clf = RidgeClassifier().fit(X_train, y_train)
    # clf = linear_model.SGDClassifier(max_iter=1000, tol=1e-3)
    # clf = svm.SVC(kernel='rbf')
    clf_nb = naive_bayes.GaussianNB()
    clf_dt = tree.DecisionTreeClassifier()
    clf_et = ensemble.ExtraTreesClassifier(n_estimators=5)
    clf_rf = ensemble.RandomForestClassifier(n_estimators=10)
    clf = ensemble.VotingClassifier(estimators=[('nb', clf_nb), ('dt', clf_dt), ('et', clf_et), ('rf', clf_rf)], voting='soft')
    clf.fit(X_train, y_train)
    print(clf.score(X_test, y_test))

    # import graphviz
    # dot_data = tree.export_graphviz(clf, out_file=None)
    # graph = graphviz.Source(dot_data)
    # graph.render('toy_neighbor_rank_histogram')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/home/adnan/CSE 6411 Materials/data/')
    parser.add_argument('--data_file', type=str, default='new_class_analysis_data_20191212_normalized.h5')
    parser_args = parser.parse_args()

    main(parser_args)
