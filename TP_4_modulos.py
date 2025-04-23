import random
import numpy as np

def generate_dna_sequences(num_sequences, sequence_length=1000, position=4, seed=None, balance = 0.5):
    if seed is not None:
        random.seed(seed)

    bases = ['A', 'T', 'C', 'G']
    
    sequences = []
    target = []

    for i in range(num_sequences):
        
        random_sequence = [random.choice(bases) for _ in range(sequence_length)]
        
        if np.random.uniform() > balance:
            random_sequence[position] = 'A'
            target.append(1)
        else:
            random_sequence[position] = random.choice(['C', 'G', 'T'])
            target.append(0)
        
        
        random_sequence = ''.join(random_sequence)
        sequences.append(random_sequence)
    
    return sequences, target


def one_hot_encoding(sequence: str) -> np.ndarray:

    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seq_len = len(sequence)
    encoding = np.zeros((4, seq_len), dtype=np.float32)
    sequence_upper = sequence.upper()

    indices = np.array([mapping.get(base, 4) for base in sequence_upper])

    valid_indices = indices[indices < 4]  # Exclude 'N' positions
    encoding[valid_indices, np.arange(seq_len)[indices < 4]] = 1.0

    n_positions = (indices == 4)
    encoding[:, n_positions] = 0.25

    return encoding


def dinucleotide_shuffle(one_hot_sequence):

    nucleotide_map = ['A', 'C', 'G', 'T']
    seq_len = one_hot_sequence.shape[1]
    decoded_sequence = ''.join([nucleotide_map[np.argmax(one_hot_sequence[:, i])] for i in range(seq_len)])
    dinucleotides = [decoded_sequence[i:i+2] for i in range(0, seq_len, 2)]
    np.random.shuffle(dinucleotides)
    shuffled_sequence_str = ''.join(dinucleotides)
    
    return one_hot_encoding(shuffled_sequence_str)

def mutation(one_hot_sequence, n_mutations = 100):
    nucleotide_map = ['A', 'C', 'G', 'T']
    choices = {nt: [x for x in nucleotide_map if x != nt] for nt in nucleotide_map}
    seq_len = one_hot_sequence.shape[1]
    decoded_sequence = ''.join([nucleotide_map[np.argmax(one_hot_sequence[:, i])] for i in range(seq_len)])

    mutation_indices = np.random.choice(seq_len, size=n_mutations, replace=False)

    mutated_sequence = ''.join([
        np.random.choice(choices[nt]) if i in mutation_indices else nt
        for i, nt in enumerate(decoded_sequence)
    ])

    return one_hot_encoding(mutated_sequence)

def read_fasta_as_list(file_path):
    sequences = []
    with open(file_path, 'r') as file:
        current_seq = []
        for line in file:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if current_seq:
                    sequences.append(''.join(current_seq))
                    current_seq = []
            else:
                current_seq.append(line)
        if current_seq:
            sequences.append(''.join(current_seq))
    return sequences


