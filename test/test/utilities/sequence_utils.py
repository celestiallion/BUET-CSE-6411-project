import unittest

from src.utilities.sequence_utils import seq_into_k_mers


class TestSequenceUtils(unittest.TestCase):
    def test_seq_into_k_mers(self):
        seq = 'AACTTAGTCGC'
        K = 3
        k_mers = seq_into_k_mers(seq, K)

        self.assertListEqual(k_mers, ['AAC', 'TTA', 'GTC', 'GC'])

if __name__ == '__main__':
    unittest.main()
