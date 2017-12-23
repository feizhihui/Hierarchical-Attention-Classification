# coding=utf8

import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import layers

vocab_size = 7906
char_size = 1941
num_classes = 488
embedding_size = 128
hidden_size = 50
learning_rate = 0.05
grad_clip = 5

threshold = 0.3

max_word_num = 400
max_word_length = 5


def length(sequences):
    # 动态展开
    used = tf.sign(tf.reduce_max(tf.abs(sequences), reduction_indices=2))
    seq_len = tf.reduce_sum(used, reduction_indices=1)
    return tf.cast(seq_len, tf.int32)


class DeepHan():
    def __init__(self, word_embeddings, char_embeddings):
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.char_embeddings = char_embeddings
        self.word_embeddings = word_embeddings

        with tf.name_scope('placeholder'):
            # self.batch_size = tf.placeholder(tf.int32, name='batch_size')
            # x的shape为[batch_size, 句子数， 句子长度(单词个数)]，但是每个样本的数据都不一样，，所以这里指定为空
            # y的shape为[batch_size, num_classes]
            self.input_c = tf.placeholder(tf.int32, [None, None, None], name='input_c')
            self.input_w = tf.placeholder(tf.int32, [None, None], name='input_w')
            self.input_y = tf.placeholder(tf.float32, [None, num_classes], name='input_y')

        # 构建模型
        char_embedded = self.char2vec()  # 构建词向量矩阵，返回对应的词词向量 [None, None, None]=>[None, None, None,embedding_size]
        word_vec = self.word2vec(char_embedded)
        doc_vec = self.doc2vec(word_vec)
        out = self.classifer(doc_vec)
        self.out = out
        ones_t = tf.ones_like(out)
        zeros_t = tf.zeros_like(out)
        self.predict = tf.cast(tf.where(tf.greater(tf.sigmoid(out), threshold), ones_t, zeros_t),
                               tf.int32)

        self.back_propagate()

    def back_propagate(self):
        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_y,
                                                                               logits=self.out,
                                                                               name='loss'))

        global_step = tf.Variable(0, trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        # RNN中常用的梯度截断，防止出现梯度过大难以求导的现象
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), grad_clip)
        grads_and_vars = tuple(zip(grads, tvars))
        self.train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    def char2vec(self):
        with tf.name_scope("char2vec"):
            embedding_mat = tf.Variable(self.char_embeddings)
            # shape为[batch_size, word_in_doc, char_in_word, embedding_size]
            char_embedded = tf.nn.embedding_lookup(embedding_mat, self.input_c)
        return char_embedded

    def word2vec(self, word_embedded):
        with tf.name_scope("word2vec"):
            # GRU的输入tensor是[batch_size, max_time, ...].在构造句子向量时max_time应该是每个句子的长度，所以这里将
            # batch_size * sent_in_doc当做是batch_size.这样一来，每个GRU的cell处理的都是一个单词的词向量
            # 并最终将一句话中的所有单词的词向量融合（Attention）在一起形成句子向量

            # shape为[batch_size*sent_in_doc, word_in_sent, embedding_size]
            word_embedded = tf.reshape(word_embedded, [-1, max_word_length, self.embedding_size])
            # shape为[batch_size*sent_in_doce, word_in_sent, hidden_size*2]
            word_encoded = self.BidirectionalGRUEncoder(word_embedded, name='word_encoder')
            # shape为[batch_size*sent_in_doc, hidden_size*2]
            sent_vec = self.AttentionLayer(word_encoded, name='word_attention')
            return sent_vec

    def doc2vec(self, word_vecs):
        with tf.name_scope("doc2vec"):
            word_vecs = tf.reshape(word_vecs, [-1, max_word_num, self.hidden_size * 2])
            # shape为[batch_size, sent_in_doc, hidden_size*2]
            doc_encoded = self.BidirectionalGRUEncoder(word_vecs, name='sent_encoder')
            # shape为[batch_szie, hidden_szie*2]
            doc_vec = self.AttentionLayer(doc_encoded, name='sent_attention')
            return doc_vec

    def classifer(self, doc_vec):
        with tf.name_scope('doc_classification'):
            out = layers.fully_connected(inputs=doc_vec, num_outputs=self.num_classes, activation_fn=None)
            return out

    def BidirectionalGRUEncoder(self, inputs, name):
        # 输入inputs的shape是[batch_size*sent_in_doc, word_in_sent, embedding_size]
        with tf.variable_scope(name):
            GRU_cell_fw = rnn.GRUCell(self.hidden_size)
            GRU_cell_bw = rnn.GRUCell(self.hidden_size)
            # fw_outputs和bw_outputs的size都是[batch_size*sent_in_doc, word_in_sent, embedding_size]
            #  tuple of (outputs, output_states)
            ((fw_outputs, bw_outputs), (_, _)) = tf.nn.bidirectional_dynamic_rnn(cell_fw=GRU_cell_fw,
                                                                                 cell_bw=GRU_cell_bw,
                                                                                 inputs=inputs,
                                                                                 sequence_length=length(inputs),
                                                                                 dtype=tf.float32)
            # outputs的size是[batch_size*sent_in_doc, max_time, hidden_size*2]
            outputs = tf.concat((fw_outputs, bw_outputs), 2)
            return outputs

    # 输出的状态向量按权值相加
    def AttentionLayer(self, inputs, name):
        # inputs是GRU的输出，size是[batch_size, max_time, encoder_size(hidden_size * 2)]
        with tf.variable_scope(name):
            # u_context是上下文的重要性向量，用于区分不同单词/句子对于句子/文档的重要程度,
            # 因为使用双向GRU，所以其长度为2×hidden_szie
            # 一个context记录了所有的经过全连接后的word或者sentence的权重
            u_context = tf.Variable(tf.truncated_normal([self.hidden_size * 2]), name='u_context')
            # 使用一个全连接层编码GRU的输出的到期隐层表示,输出u的size是[batch_size, max_time, hidden_size * 2]
            h = layers.fully_connected(inputs, self.hidden_size * 2, activation_fn=tf.nn.tanh)
            # alpha shape为[batch_size, max_time, 1]
            alpha = tf.nn.softmax(tf.reduce_sum(tf.multiply(h, u_context), axis=2, keep_dims=True), dim=1)
            # reduce_sum之前shape为[batch_szie, max_time, hidden_szie*2]，之后shape为[batch_size, hidden_size*2]
            atten_output = tf.reduce_sum(tf.multiply(inputs, alpha), axis=1)
            return atten_output
