import os
import h5py
import operator
import argparse


class AugmentDataset:
    def __init__(self, data_write_path, features, targets):
        self.data_write_path = data_write_path
        self.features = features
        self.targets = targets

    def get_distinct_element_counts(self):
        distinct_target_counts = {}
        for item in self.targets:
            if item not in distinct_target_counts:
                distinct_target_counts[str(item)] = 1
            else:
                distinct_target_counts[str(item)] += 1
        max_count = max(distinct_target_counts.items(), key=operator.itemgetter(1))[0]
        return max_count, distinct_target_counts

    def oversample_dataset(self):
        pass


def main(args):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/home/adnan/PycharmProjects/BUET-CSE-6411-project/data/', help='Path to the directory containing the dataset.')
    parser.add_argument('--data_file', type=str, default='neighbor_rank_histogram_dataset_imbalanced.h5', help='Dataset file.')
    parser_args = parser.parse_args()

    main(parser_args)
