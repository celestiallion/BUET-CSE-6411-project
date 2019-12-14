import os
import argparse
import h5py
from sklearn import decomposition
from sklearn import manifold
import matplotlib.pyplot as plt

n_components = 2
n_neighbors = 10


def visualize(X, y, embedder, title) -> None:
    red = y == 1
    sienna = y == 2
    gold = y == 3
    olivedrab = y == 4
    slategray = y == 5
    royalblue = y == 6
    plum = y == 7
    maroon = y == 8
    lightsteelblue = y == 9

    X = embedder.fit_transform(X)

    one = plt.scatter(X[red, 0], X[red, 1], c="red")
    two = plt.scatter(X[sienna, 0], X[sienna, 1], c="sienna")
    three = plt.scatter(X[gold, 0], X[gold, 1], c="gold")
    four = plt.scatter(X[olivedrab, 0], X[olivedrab, 1], c="olivedrab")
    five = plt.scatter(X[slategray, 0], X[slategray, 1], c="slategray")
    six = plt.scatter(X[royalblue, 0], X[royalblue, 1], c="royalblue")
    seven = plt.scatter(X[plum, 0], X[plum, 1], c="plum")
    eight = plt.scatter(X[maroon, 0], X[maroon, 1], c="maroon")
    nine = plt.scatter(X[lightsteelblue, 0], X[lightsteelblue, 1], c="lightsteelblue")

    plt.legend((one, two, three, four, five, six, seven, eight, nine), ('1', '2', '3', '4', '5', '6', '7', '8', '9'), scatterpoints=1, ncol=3, fontsize=12)
    plt.title(title)

    plt.show()


def main(args):
    hf = h5py.File(args.data_root, 'r')
    X, y = hf['X'][:], hf['y'][:]

    # pca = decomposition.PCA(n_components=n_components)
    # nmf = decomposition.NMF(n_components=n_components, init='random', random_state=0)
    svd = decomposition.TruncatedSVD(n_components=n_components, n_iter=10, random_state=0)
    # tsne = manifold.TSNE(n_components=n_components, init='random', random_state=0, perplexity=30)
    # isomap = manifold.Isomap(n_neighbors=n_neighbors, n_components=n_components)
    # spectral_embedding = manifold.SpectralEmbedding(n_components=n_components, affinity='nearest_neighbors', n_neighbors=n_neighbors)
    # mds = manifold.MDS(n_components=n_components)
    # methods = ['standard', 'ltsa', 'hessian', 'modified']
    # labels = ['LLE', 'LTSA', 'Hessian LLE', 'Modified LLE']
    # lle = manifold.LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=n_components, eigen_solver='auto', method='modified')
    visualize(X, y, svd, 'SVD')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/home/adnan/CSE 6411 Materials/data/new_class_analysis_data_20191212_normalized.h5')
    parser_args = parser.parse_args()

    main(parser_args)
