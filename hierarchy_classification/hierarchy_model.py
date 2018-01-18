# coding=utf8

import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import layers

# =============== may be change  total 1177 488
num_classes = 1177
# =====================
embedding_size = 128
hidden_size = 100

grad_clip = 5
init_learning_rate = 0.005  # CNN 0.001  # GR,U 0.002  # LST,M 0.005
threshold = 0.20

max_word_num = 400
max_word_length = 5

filter_num = 64
filter_sizes = [1, 3, 5]

use_skip_gram = True


class DeepHan():
    def __init__(self, word_embeddings, char_embeddings, decay_steps=96, decay_rate=0.98):
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate

        self.char_embeddings = char_embeddings
        self.word_embeddings = word_embeddings

        with tf.name_scope('placeholder'):
            # self.batch_size = tf.placeholder(tf.int32, name='batch_size')
            # x的shape为[batch_size, 句子数， 句子长度(单词个数)]，但是每个样本的数据都不一样，，所以这里指定为空
            # y的shape为[batch_size, num_classes]
            self.input_c = tf.placeholder(tf.int32, [None, None, None], name='input_c')
            self.input_w = tf.placeholder(tf.int32, [None, None], name='input_w')
            self.input_y = tf.placeholder(tf.float32, [None, num_classes], name='input_y')
            self.keep_prob = tf.placeholder(tf.float32)

        # 构建模型
        char_embedded = self.char2vec()  # 构建词向量矩阵，返回对应的词词向量 [None, None, None]=>[None, None, None,embedding_size]
        # word_vec1 = self.word2vec(char_embedded)
        word_vec1 = self.vanilla_rnn(tf.reshape(char_embedded, [-1, max_word_length, self.embedding_size]),
                                     name='char_encode')
        word_vec1 = tf.reshape(word_vec1, [-1, max_word_num, self.hidden_size * 2])

        # use_skip_gram = False
        if use_skip_gram:
            word_vec2 = self.skip_gram()
            word_embedded = tf.concat([word_vec1, word_vec2], axis=2)
            # word_embedded = word_vec2
        else:
            # word_embedded = word_vec1  # only char-embedding
            word_embedded = self.skip_gram()  # only skip_gram

        # doc_vec = self.doc2vec_rnn(word_embedded)
        # doc_vec = self.doc2vec_cnn(word_embedded)
        # doc_vec = self.doc2vec_cbow(word_embedded)
        doc_vec = self.vanilla_rnn(word_embedded, name='doc_embedding')
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
        learning_rate = tf.train.exponential_decay(init_learning_rate, global_step, self.decay_steps, self.decay_rate,
                                                   staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        # RNN中常用的梯度截断，防止出现梯度过大难以求导的现象
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), grad_clip)
        grads_and_vars = tuple(zip(grads, tvars))
        self.train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    def char2vec(self):
        with tf.name_scope("char2vec"):
            self.char_embedding_mat = tf.Variable(self.char_embeddings)
            # shape为[batch_size, word_in_doc, char_in_word, embedding_size]
            char_embedded = tf.nn.embedding_lookup(self.char_embedding_mat, self.input_c)
        return char_embedded

    def skip_gram(self):
        with tf.name_scope('word2vec_of_skipgram'):
            self.word_embedding_mat = tf.Variable(self.word_embeddings)
            word_embedded = tf.nn.embedding_lookup(self.word_embedding_mat, self.input_w)
        return word_embedded

    def word2vec(self, char_embedded):
        with tf.name_scope("word2vec"):
            # LSTM的输入tensor是[batch_size, max_time, ...].在构造句子向量时max_time应该是每个句子的长度，所以这里将
            # batch_size * sent_in_doc当做是batch_size.这样一来，每个LSTM的cell处理的都是一个单词的词向量
            # 并最终将一句话中的所有单词的词向量融合（Attention）在一起形成句子向量

            char_embedded = tf.reshape(char_embedded, [-1, max_word_length, self.embedding_size])
            # shape为[batch_size*sent_in_doce, word_in_sent, hidden_size*2]
            word_encoded = self.BidirectionalLSTMEncoder(char_embedded, name='word_encoder')
            # shape为[batch_size*sent_in_doc, hidden_size*2]
            word_vec = self.AttentionLayer(word_encoded, name='word_attention')

            word_embedded = tf.reshape(word_vec, [-1, max_word_num, self.hidden_size * 2])

            return word_embedded

    def doc2vec_rnn(self, word_vecs):
        with tf.name_scope("doc2vec"):
            # shape为[batch_size, sent_in_doc, hidden_size*2]
            doc_encoded = self.BidirectionalLSTMEncoder(word_vecs, name='sent_encoder')
            # shape为[batch_szie, hidden_szie*2]
            doc_vec = self.AttentionLayer(doc_encoded, name='sent_attention')
            return doc_vec

    def classifer(self, doc_vec):
        with tf.name_scope('doc_classification'):
            out = layers.fully_connected(inputs=doc_vec, num_outputs=self.num_classes, activation_fn=None)
            return out

    def BidirectionalLSTMEncoder(self, inputs, name):
        # 输入inputs的shape是[batch_size*sent_in_doc, word_in_sent, embedding_size]
        with tf.variable_scope(name):
            LSTM_cell_fw = rnn.LSTMCell(self.hidden_size)
            LSTM_cell_bw = rnn.LSTMCell(self.hidden_size)
            # fw_outputs和bw_outputs的size都是[batch_size*sent_in_doc, word_in_sent, embedding_size]
            #  tuple of (outputs, output_states)
            ((fw_outputs, bw_outputs), (_, _)) = tf.nn.bidirectional_dynamic_rnn(cell_fw=LSTM_cell_fw,
                                                                                 cell_bw=LSTM_cell_bw,
                                                                                 inputs=inputs,
                                                                                 sequence_length=self.length(inputs),
                                                                                 dtype=tf.float32)
            # outputs的size是[batch_size*sent_in_doc, max_time, hidden_size*2]
            outputs = tf.concat((fw_outputs, bw_outputs), 2)
            return outputs

    # 输出的状态向量按权值相加
    def AttentionLayer(self, inputs, name):
        # inputs是LSTM的输出，size是[batch_size, max_time, encoder_size(hidden_size * 2)]
        with tf.variable_scope(name):
            # u_context是上下文的重要性向量，用于区分不同单词/句子对于句子/文档的重要程度,
            # 因为使用双向LSTM，所以其长度为2×hidden_szie
            # 一个context记录了所有的经过全连接后的word或者sentence的权重
            u_context = tf.Variable(tf.truncated_normal([self.hidden_size * 2]), name='u_context')
            # 使用一个全连接层编码LSTM的输出的到期隐层表示,输出u的size是[batch_size, max_time, hidden_size * 2]
            h = layers.fully_connected(inputs, self.hidden_size * 2, activation_fn=tf.nn.tanh)
            # alpha shape为[batch_size, max_time, 1]
            alpha = tf.nn.softmax(tf.reduce_sum(tf.multiply(h, u_context), axis=2, keep_dims=True), dim=1)
            # reduce_sum之前shape为[batch_szie, max_time, hidden_szie*2]，之后shape为[batch_size, hidden_size*2]
            atten_output = tf.reduce_sum(tf.multiply(inputs, alpha), axis=1)
            return atten_output

    def length(self, sequences):
        # 动态展开
        used = tf.sign(tf.reduce_max(tf.abs(sequences), reduction_indices=2))
        seq_len = tf.reduce_sum(used, reduction_indices=1)
        self.seq_len = tf.cast(seq_len, tf.int32)
        return self.seq_len

    def doc2vec_cnn(self, word_vecs):
        width = word_vecs.get_shape().as_list()[-1]
        weights = {
            'wc1': tf.Variable(
                tf.truncated_normal([filter_sizes[0], width, filter_num], stddev=0.1)),
            'wc2': tf.Variable(
                tf.truncated_normal([filter_sizes[1], width, filter_num], stddev=0.1)),
            'wc3': tf.Variable(
                tf.truncated_normal([filter_sizes[2], width, filter_num], stddev=0.1))
        }

        biases = {
            'bc1': tf.Variable(tf.truncated_normal([filter_num], stddev=0.1)),
            'bc2': tf.Variable(tf.truncated_normal([filter_num], stddev=0.1)),
            'bc3': tf.Variable(tf.truncated_normal([filter_num], stddev=0.1))
        }

        def conv1d(x, W, b):
            x = tf.reshape(x, shape=[-1, max_word_num, width])
            x = tf.nn.conv1d(x, W, 1, padding='SAME')
            x = tf.nn.bias_add(x, b)
            # shape=(n,time_steps,filter_num)
            h = tf.nn.relu(x)
            pooled = tf.reduce_max(h, axis=1)

            return pooled

        def multi_conv(x, weights, biases):
            # Convolution Layer
            conv1 = conv1d(x, weights['wc1'], biases['bc1'])
            conv2 = conv1d(x, weights['wc2'], biases['bc2'])
            conv3 = conv1d(x, weights['wc3'], biases['bc3'])
            #  n*time_steps*(3*filter_num)
            convs = tf.concat([conv1, conv2, conv3], 1)
            return convs

        input = tf.reshape(word_vecs, [-1, max_word_num, width])
        x_convs = multi_conv(input, weights, biases)
        x_convs = tf.reshape(x_convs, [-1, 3 * filter_num])
        x_convs = tf.nn.dropout(x_convs, self.keep_prob)

        return x_convs

    def doc2vec_cbow(self, word_embedded):
        return tf.reduce_mean(word_embedded, axis=1)

    def vanilla_rnn(self, inputs, name='vanilla_rnn'):
        # 输入inputs的shape是[batch_size*sent_in_doc, word_in_sent, embedding_size]
        with tf.variable_scope(name):
            LSTM_cell_fw = rnn.LSTMCell(self.hidden_size)
            LSTM_cell_bw = rnn.LSTMCell(self.hidden_size)
            # fw_outputs和bw_outputs的size都是[batch_size*sent_in_doc, word_in_sent, embedding_size]
            #  tuple of (outputs, output_states)
            ((_, _), (fw_state, bw_state)) = tf.nn.bidirectional_dynamic_rnn(cell_fw=LSTM_cell_fw,
                                                                             cell_bw=LSTM_cell_bw,
                                                                             inputs=inputs,
                                                                             sequence_length=self.length(inputs),
                                                                             dtype=tf.float32)
            outputs = tf.concat((fw_state.h, bw_state.h), 1)
            return outputs
