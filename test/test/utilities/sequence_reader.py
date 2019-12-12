import unittest
from pprint import pprint

from src.utilities.sequence_reader import SequenceReader


class TestSequenceReader(unittest.TestCase):
    def setUp(self):
        self.sequence_reader = SequenceReader()

    def test_get_codon_identifiers(self):
        codon_identifiers = self.sequence_reader.get_codon_identifiers()

        pprint(codon_identifiers)
        self.assertIsNotNone(codon_identifiers)

    def test_read_sequence(self):
        fasta_file_path = '/home/adnan/CSE 6411 Materials/Dataset/FastaFiles1/aMSX7lvrZeCBwE9kUnDf.fasta'
        seq = self.sequence_reader.read_sequence(fasta_file_path)

        self.assertIsNotNone(seq)

    def test_get_base_sequence(self):
        fasta_file_path = '/home/adnan/CSE 6411 Materials/Dataset/FastaFiles1/aMSX7lvrZeCBwE9kUnDf.fasta'
        seq = self.sequence_reader.read_sequence(fasta_file_path)
        seq = self.sequence_reader.get_base_sequence(seq)

        self.assertIsNotNone(seq)

    def test_get_base_seq_as_codon_seq(self):
        fasta_file_path = '/home/adnan/CSE 6411 Materials/Dataset/FastaFiles1/aMSX7lvrZeCBwE9kUnDf.fasta'

        codon_identifiers = self.sequence_reader.get_codon_identifiers()
        seq = self.sequence_reader.read_sequence(fasta_file_path)
        seq = self.sequence_reader.get_base_sequence(seq)
        codon_seq = self.sequence_reader.get_base_seq_as_codon_seq(seq, codon_identifiers)

        pprint(codon_seq)

        self.assertIsNotNone(codon_seq)
        self.assertEqual(type(codon_seq), list)


if __name__ == '__main__':
    unittest.main()
