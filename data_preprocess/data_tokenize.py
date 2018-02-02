# encoding=utf-8
import pickle
import numpy as np

word_size = 700
char_size = 5

with open('../cache/all_text.pkl', 'rb') as file:
    all_text = pickle.load(file)
with open('../cache/all_code.pkl', 'rb') as file:
    all_code = pickle.load(file)
with open('../cache/code_dict.pkl', 'rb') as file:
    code_dict = pickle.load(file)
with open('../cache/word_dict.pkl', 'rb') as file:
    word_dict = pickle.load(file)
with open('../cache/char_dict.pkl', 'rb') as file:
    char_dict = pickle.load(file)

n = len(all_text)
class_num = len(code_dict)

all_word_tokens = []
all_char_tokens = []
for row in all_text:
    token_row = []
    char_row = []
    for word in row:
        if word in word_dict:
            token_row.append(word_dict[word])
            char_in_word = []
            for c in word:
                if c in char_dict:
                    char_in_word.append(char_dict[c])
            char_row.append(char_in_word)
    all_word_tokens.append(token_row)
    all_char_tokens.append(char_row)

word_tokens_matrix = np.zeros([n, word_size], dtype=np.int32)
char_tokens_matrix = np.zeros([n, word_size, char_size], dtype=np.int32)

for sample_i in range(n):
    for word_j in range(word_size):
        if word_j >= len(all_word_tokens[sample_i]): break
        word_tokens_matrix[sample_i, word_j] = all_word_tokens[sample_i][word_j]
        raw_word = all_text[sample_i][word_j]
        for char_k in range(char_size):
            if char_k >= len(all_char_tokens[sample_i][word_j]): break
            char_tokens_matrix[sample_i, word_j, char_k] = all_char_tokens[sample_i][word_j][char_k]

code_tokens_matrix = []
for codes in all_code:
    token_row = []
    for code in codes:
        token_row.append(code_dict[code])
    code_tokens_matrix.append(token_row)

mark = list(range(len(word_tokens_matrix)))
np.random.shuffle(mark)

np.save('../cache/word_tokens_matrix.npy', word_tokens_matrix[mark])
np.save('../cache/char_tokens_matrix.npy', char_tokens_matrix[mark])
np.save('../cache/code_tokens_matrix.npy', np.array(code_tokens_matrix)[mark])
