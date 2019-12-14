import os
import argparse
import h5py
from sklearn import manifold
from sklearn import decomposition
from sklearn.cluster import AffinityPropagation
from sklearn.metrics import homogeneity_completeness_v_measure

n_components = 2
n_neighbors = 10


def main(args):
    hf = h5py.File(args.data_root, 'r')
    X, y = hf['X'][:], hf['y'][:]

    # X = manifold.Isomap(n_neighbors=n_neighbors, n_components=n_components).fit_transform(X)
    # X = manifold.TSNE(n_components=n_components, init='random', random_state=0, perplexity=30).fit_transform(X)
    # X = manifold.SpectralEmbedding(n_components=n_components, affinity='nearest_neighbors', n_neighbors=n_neighbors).fit_transform(X)
    # X = manifold.MDS(n_components=n_components).fit_transform(X)
    X = decomposition.PCA(n_components=n_components).fit_transform(X)

    labels_pred = AffinityPropagation(damping=0.75).fit_predict(X)
    print(homogeneity_completeness_v_measure(labels_true=y, labels_pred=labels_pred))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/home/adnan/CSE 6411 Materials/data/new_class_analysis_data_20191212_normalized.h5')
    parser_args = parser.parse_args()

    main(parser_args)
