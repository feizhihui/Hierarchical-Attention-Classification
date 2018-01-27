# encoding=utf-8
import numpy as np
import pickle

train_eval_rate = 0.9
# may be change
num_classes = 488


class DataLoader(object):
    def __init__(self):
        word_tokens_matrix = np.load('../cache/word_tokens_matrix.npy')
        char_tokens_matrix = np.load('../cache/char_tokens_matrix.npy')
        code_tokens_matrix = np.load('../cache/code_tokens_matrix.npy')

        self.char_embeddings = np.load('../cache/char_embeddings.npy')
        self.word_embeddings = np.load('../cache/word_embeddings.npy')

        # tnum = np.sum(self.mapping_label(code_tokens_matrix))
        # print(tnum)
        train_eval_line = int(train_eval_rate * int(len(word_tokens_matrix)))

        self.train_W = word_tokens_matrix[:train_eval_line]
        self.train_C = char_tokens_matrix[:train_eval_line]
        self.train_Y = code_tokens_matrix[:train_eval_line]
        self.train_size = len(self.train_W)
        print('training size is', self.train_size)

        self.test_W = word_tokens_matrix[train_eval_line:]
        self.test_C = char_tokens_matrix[train_eval_line:]
        self.test_Y = code_tokens_matrix[train_eval_line:]
        self.test_size = len(self.test_W)
        print('testing size is', self.test_size)

    def shuffle(self):
        mark = list(range(self.train_size))
        np.random.shuffle(mark)
        self.train_W = self.train_W[mark]
        self.train_C = self.train_C[mark]
        self.train_Y = self.train_Y[mark]

    def mapping_label(self, batch_Y):
        batch_Y_ = np.zeros([len(batch_Y), num_classes])
        for i, codes in enumerate(batch_Y):
            for code in codes:
                batch_Y_[i, code] = 1
        return batch_Y_


if __name__ == '__main__':
    DataLoader()
