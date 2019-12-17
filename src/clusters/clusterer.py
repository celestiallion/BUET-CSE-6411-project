import argparse
import h5py
from sklearn.cluster import AffinityPropagation
from sklearn.metrics import homogeneity_completeness_v_measure


def main(args):
    hf = h5py.File(args.data_root, 'r')
    X, y = hf['X'][:2000], hf['y'][:2000]

    labels_pred = AffinityPropagation(damping=0.75).fit_predict(X)
    print(homogeneity_completeness_v_measure(labels_true=y, labels_pred=labels_pred))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/home/adnan/Datasets/neighbor_rank_histogram_dataset_oversampled_window_5.h5')
    parser_args = parser.parse_args()

    main(parser_args)
