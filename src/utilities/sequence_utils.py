def seq_into_k_mers(seq, K):
    return [seq[idx:min(idx+K, len(seq))] for idx in range(0, len(seq), K)]
