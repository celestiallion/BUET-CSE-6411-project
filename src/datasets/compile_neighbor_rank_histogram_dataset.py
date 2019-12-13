import os
import argparse
import h5py
import numpy as np
from random import sample as random_sample

from project_configs import DATASET_ROOT
from src.features.neighbor_rank_histogram import NeighborRankHistogram


class CompileNeighborRankHistogram:
    def __init__(self, dataset_root_path=DATASET_ROOT, K=3, bases='ACGT', right_neighbor_count=3, left_neighbor_count=3):
        self.dataset_root_path = dataset_root_path
        self.neighbor_rank_histogram = NeighborRankHistogram(dataset_root_path, K, bases, right_neighbor_count, left_neighbor_count)

    def compile_entire_dataset(self, write_path) -> None:
        X, y = [], []
        for dataset_dir in os.listdir(self.dataset_root_path):
            label = int(dataset_dir[-1])
            vectors, labels = self.compile_dataset(dataset_dir, label)
            X.extend(vectors)
            y.extend(labels)
        X, y = zip(*random_sample(list(zip(X, y)), len(X)))

        hf = h5py.File(write_path, 'w')
        hf.create_dataset('X', data=np.array(X))
        hf.create_dataset('y', data=np.array(y))

    def compile_dataset(self, dataset_dir, label, write_path=None):
        fasta_dir_path = os.path.join(self.dataset_root_path, dataset_dir)
        fasta_file_paths = [os.path.join(fasta_dir_path, fasta_file) for fasta_file in os.listdir(fasta_dir_path)]

        X = []
        for count, fasta_file_path in enumerate(fasta_file_paths):
            try:
                neighbor_rank_histogram_vector = self.neighbor_rank_histogram.get_neighbor_rank_histogram_vector(fasta_file_path, normalize=True)
                X.append(neighbor_rank_histogram_vector)
                print('{} are yet to be processed ...'.format(len(fasta_file_paths) - count))
            except ValueError:
                continue

        y = [label] * len(X)
        if write_path is not None:
            hf = h5py.File(write_path, 'w')
            hf.create_dataset('X', data=np.array(X))
            hf.create_dataset('y', data=np.array(y))
        else:
            return X, y


def main(args):
    compile_neighbor_rank_histogram = CompileNeighborRankHistogram(args.dataset_root_path, args.K, args.bases, args.right_neighbor_count, args.left_neighbor_count)
    compile_neighbor_rank_histogram.compile_entire_dataset(args.write_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root_path', type=str, default=DATASET_ROOT)
    parser.add_argument('--K', type=int, default=3)
    parser.add_argument('--bases', type=str, default='ACGT')
    parser.add_argument('--right_neighbor_count', type=int, default=3)
    parser.add_argument('--left_neighbor_count', type=int, default=3)
    parser.add_argument('--write_path', type=str, default='/home/adnan/PycharmProjects/BUET-CSE-6411-project/data/neighbor_rank_histogram_dataset_imbalanced.h5')
    parser_args = parser.parse_args()

    main(parser_args)