import numpy as np

from project_configs import DATASET_ROOT
from src.utilities.sequence_reader import SequenceReader


class NeighborRankHistogram:
    def __init__(self, dataset_root_path=DATASET_ROOT, K=3, bases='ACGT', right_neighbor_count=3, left_neighbor_count=3):
        self.dataset_root_path = dataset_root_path
        self.K = K
        self.left_neighbor_count = left_neighbor_count
        self.right_neighbor_count = right_neighbor_count
        self.left_score_fraction = 1. / left_neighbor_count
        self.right_score_fraction = 1. / right_neighbor_count
        self.bases = bases
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

    def get_cooccurence_vector(self, codon_seq, normalize=False):
        cooccurence_matrix = np.zeros((len(self.bases) ** self.K, len(self.bases) ** self.K), dtype=np.float32)
        center_neighbor_pairs = self.get_neighbors(codon_seq)
        for center_word, left_neighbors, right_neighbors in center_neighbor_pairs:
            row = center_word
            cooccurence_matrix[row][row] = 1.
            for c, neighbor in enumerate(left_neighbors):
                col = neighbor
                cooccurence_matrix[row][col] += c * self.left_score_fraction
            for c, neighbor in enumerate(right_neighbors):
                col = neighbor
                cooccurence_matrix[row][col] += 1. - c * self.right_score_fraction
        cooccurence_vector = cooccurence_matrix.flatten()
        if normalize:
            max_elem = np.amax(cooccurence_vector)
            cooccurence_vector /= max_elem
        return cooccurence_vector

    def get_neighbor_rank_histogram_vector(self, fasta_file_path, normalize=False):
        seq = self.seq_reader.read_sequence(fasta_file_path)
        if seq != '':
            seq = self.seq_reader.get_base_sequence(seq)
            codon_seq = self.seq_reader.get_base_seq_as_codon_seq(seq, self.codon_identifiers)
            return self.get_cooccurence_vector(codon_seq, normalize)
        else:
            raise ValueError
