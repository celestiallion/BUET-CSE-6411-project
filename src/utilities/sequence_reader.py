import itertools

from Bio import SeqIO

from project_configs import DATASET_ROOT
from src.utilities.sequence_utils import seq_into_k_mers


class SequenceReader:
    def __init__(self, dataset_root_path=DATASET_ROOT, K=3, bases='ACGT'):
        self.dataset_root_path = dataset_root_path
        self.K = K
        self.bases = bases

    def get_codon_identifiers(self):
        def generate_codons():
            yield from itertools.product(*([self.bases] * self.K))

        codon_identifiers = {}
        for count, x in enumerate(generate_codons()):
            codon_identifiers[''.join(x)] = count

        return codon_identifiers

    def read_sequence(self, fasta_file_path):
        try:
            seq = str(SeqIO.read(fasta_file_path, 'fasta').seq)
            return seq
        except UnicodeDecodeError:
            return ''

    def get_base_sequence(self, seq):
        new_seq = []
        for item in seq:
            if item in self.bases:
                new_seq.append(item)
        return ''.join(new_seq)

    def get_base_seq_as_codon_seq(self, seq, codon_identifiers):
        k_mers = seq_into_k_mers(seq, self.K)
        return [codon_identifiers[k_mer] for k_mer in k_mers if len(k_mer) == self.K]
