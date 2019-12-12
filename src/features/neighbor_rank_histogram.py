import numpy as np

from project_configs import DATASET_ROOT
from src.utilities.sequence_reader import SequenceReader


class NeighborRankHistogram:
    def __init__(self, dataset_root_path=DATASET_ROOT, K=3, bases='ACGT', right_neighbor_count=3, left_neighbor_count=3):
        self.dataset_root_path = dataset_root_path
        self.K = K
        self.score_fraction = 1. / K
        self.bases = bases
        self.right_neighbor_count = right_neighbor_count
        self.left_neighbor_count = left_neighbor_count
        self.seq_reader = SequenceReader(dataset_root_path, K, bases)
        self.codon_identifiers = self.seq_reader.get_codon_identifiers()

    def get_neighbors(self, codon_seq):
        len_codon_seq = len(codon_seq)
        neighbor_seq = []
        for idx in range(len_codon_seq):
            left_neighbors = codon_seq[max(0, idx - self.left_neighbor_count):idx]
            right_neighbors = codon_seq[min(1+idx, len_codon_seq):min(len_codon_seq, idx+self.right_neighbor_count)]
            center_word = codon_seq[idx]
            neighbor_seq.append((center_word, left_neighbors, right_neighbors))

        return neighbor_seq

    def get_cooccurence_vector(self, codon_seq):
        cooccurence_matrix = np.zeros((len(self.bases) ** self.K, len(self.bases) ** self.K), dtype=np.float32)

        center_neighbor_pairs = self.get_neighbors(codon_seq)

        for center_word, left_neighbors, right_neighbors in center_neighbor_pairs:
            row = center_word
            cooccurence_matrix[row][row] = 1.
            for c, neighbor in enumerate(left_neighbors):
                col = neighbor
                cooccurence_matrix[row][col] += c * self.score_fraction
            for c, neighbor in enumerate(right_neighbors):
                col = neighbor
                cooccurence_matrix[row][col] += 1. - c * self.score_fraction

        cooccurence_vector = cooccurence_matrix.flatten()

        return cooccurence_vector

    def get_neighbor_rank_histogram(self, fasta_file_path):
        seq = self.seq_reader.read_sequence(fasta_file_path)
        seq = self.seq_reader.get_base_sequence(seq)
        codon_seq = self.seq_reader.get_base_seq_as_codon_seq(seq, self.codon_identifiers)

        return self.get_cooccurence_vector(codon_seq)
