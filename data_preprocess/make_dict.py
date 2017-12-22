# encoding=utf-8
import pickle
import numpy as np

embeddings = 128

char_dict = {'#PADDING#': 0}
char_embeddings = [[0] * embeddings]
with open('../data/char_embeddings.128', 'r') as file:
    for i, line in enumerate(file.readlines()[1:]):
        columns = line.split()
        c = columns[0]
        feature = columns[1:]
        assert len(feature) == 128
        char_dict[c] = i + 1
        char_embeddings.append(feature)
assert len(char_embeddings) == len(char_dict)
char_embeddings = np.array(char_embeddings, dtype=np.float32)
print(char_embeddings.shape, type(char_embeddings))

print('char vocab size:', len(char_dict))
with open('../cache/char_dict.pkl', 'wb') as file:
    pickle.dump(char_dict, file)
np.save('../cache/char_embeddings.npy', char_embeddings)

word_dict = {'#PADDING#': 0}
word_embeddings = [[0] * embeddings]
with open('../data/word_embeddings.128', 'r') as file:
    for i, line in enumerate(file.readlines()[1:]):
        columns = line.split()
        c = columns[0]
        feature = columns[1:]
        assert len(feature) == 128
        word_dict[c] = i + 1
        word_embeddings.append(feature)

word_embeddings = np.array(word_embeddings, dtype=np.float32)

print('word vocab size:', len(word_dict))
with open('../cache/word_dict.pkl', 'wb') as file:
    pickle.dump(word_dict, file)
np.save('../cache/word_embeddings.npy', word_embeddings)
print(word_embeddings.shape, type(word_embeddings))
