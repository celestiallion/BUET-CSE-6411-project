import unittest

from src.datasets.compile_neighbor_rank_histogram_dataset import CompileNeighborRankHistogram


class TestCompileNeighborRankHistogram(unittest.TestCase):
    def setUp(self):
        self.compile_neighbor_histogram = CompileNeighborRankHistogram()

    def test_compile_dataset(self):
        dataset_dir = 'FastaFiles8'
        label = 8
        write_path = '/home/adnan/PycharmProjects/BUET-CSE-6411-project/data/neighbor_rank_histogram_{}.h5'.format(label)

        self.compile_neighbor_histogram.compile_dataset(dataset_dir, label, write_path)


if __name__ == '__main__':
    unittest.main()
