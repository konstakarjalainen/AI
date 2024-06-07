import numpy as np


def letter_encoder(filename):

    raw_text = open(filename, 'r', encoding='utf-8').read()
    raw_text = raw_text.lower()

    # create mapping of unique chars to integers
    chars = sorted(list(set(raw_text)))
    print(chars)
    char_to_int = dict((c, i) for i, c in enumerate(chars))
    int_to_char = dict((i, c) for i, c in enumerate(chars))
    n_vocab = len(chars)

    zeros_arr = np.zeros(n_vocab)
    data_matrix = []
    for char in raw_text:
        zeros_copy = zeros_arr.copy()
        zeros_copy[char_to_int[char]] = 1.0
        data_matrix.append(zeros_copy)
    # summarize the loaded data

    return data_matrix, n_vocab, char_to_int, int_to_char


filename = "abcde.txt"
data_matrix, n_vocab, char_to_int, int_to_char = letter_encoder(filename)

print("Total Characters: ", len(data_matrix))
print("Total Vocab: ", n_vocab)
print(char_to_int)
print(int_to_char)
print("First encoded character: ", data_matrix[0])
