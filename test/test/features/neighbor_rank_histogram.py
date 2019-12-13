import unittest
from pprint import pprint

from src.features.neighbor_rank_histogram import NeighborRankHistogram


class TestNeighborRankHistogram(unittest.TestCase):
    def setUp(self):
        self.neighbor_rank_histogram = NeighborRankHistogram()

    def test_get_neighbor_rank_histogram(self):
        fasta_file_path = '/home/adnan/CSE 6411 Materials/Dataset/FastaFiles1/aMSX7lvrZeCBwE9kUnDf.fasta'

        neighbor_rank_histogram_vector = self.neighbor_rank_histogram.get_neighbor_rank_histogram_vector(fasta_file_path)

        pprint(neighbor_rank_histogram_vector)
        self.assertIsNotNone(neighbor_rank_histogram_vector)


if __name__ == '__main__':
    unittest.main()
