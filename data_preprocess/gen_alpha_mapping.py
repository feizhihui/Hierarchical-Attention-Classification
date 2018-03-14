# encoding=utf-8
import pickle
import numpy as np
import pandas as pd

train_eval_rate = 0.95863

with open('../cache/alphas.pkl', 'rb') as file:
    alphas = pickle.load(file)

alphas = alphas.squeeze()

word_tokens_matrix = np.load('../cache/word_tokens_matrix.npy')
train_eval_line = int(train_eval_rate * int(len(word_tokens_matrix)))
test_W = word_tokens_matrix[train_eval_line:]

with open('../cache/word_dict.pkl', 'rb') as file:
    word_dict = pickle.load(file)

ix_to_word = {ix: word for word, ix in word_dict.items()}
raw_word = []
for line in test_W:
    raw_sentence = []
    for ix in line:
        raw_sentence.append(ix_to_word[ix])
    raw_word.append(raw_sentence)

raw_word = np.array(raw_word)

print(raw_word.shape)
print(alphas.shape)

data1 = pd.DataFrame(raw_word)
data2 = pd.DataFrame(alphas)

data1.to_csv('../data/test_data.csv', header=False, index=False)
data2.to_csv('../data/test_weight.csv', header=False, index=False)
