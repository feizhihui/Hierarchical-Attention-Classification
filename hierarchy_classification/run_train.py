# encoding=utf-8
import tensorflow as tf
from dataloader import DataLoader
from hierarchy_model import DeepHan
import numpy as np
import sklearn.metrics as metrics
import pickle
import os

# LD_LIBRARY_PATH   	/usr/local/cuda-8.0/lib64:$LD_LIBRARY_PATH
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

batch_size = 128
eval_batch_size = 1024

epoch_num = 70  # 50

keep_pro = 0.75  # 0.75

loader = DataLoader()
model = DeepHan(loader.word_embeddings, loader.char_embeddings, decay_steps=loader.train_size / batch_size)


def validataion(localize=False):
    # model.prediction_fused
    print('begin to eval:')
    outputs = []
    logits = []
    for i in range(0, loader.test_size, eval_batch_size):
        batch_W = loader.test_W[i:i + eval_batch_size]
        batch_C = loader.test_C[i:i + eval_batch_size]
        y_pred, y_logit = sess.run([model.predict, model.logit],
                                   feed_dict={model.input_w: batch_W, model.input_c: batch_C, model.keep_prob: 1.})
        outputs.append(y_pred)
        logits.append(y_logit)
    outputs = np.concatenate(outputs, axis=0)
    logits = np.concatenate(logits, axis=0)

    MiP, MiR, MiF, P_NUM, T_NUM, hamming_loss = micro_score(outputs, loader.mapping_label(loader.test_Y))
    print(">>>>>>>> Final Result:  PredictNum:%.2f, TrueNum:%.2f" % (P_NUM, T_NUM))
    print(">>>>>>>> Micro-Precision:%.3f, Micro-Recall:%.3f, Micro-F Measure:%.3f, Hamming Loss:%.5f" % (
        MiP, MiR, MiF, hamming_loss))
    if localize:
        with open('../cache/scores.pkl', 'wb') as file:
            pickle.dump((logits, loader.mapping_label(loader.test_Y)), file)


def micro_score(output, label):
    N = len(output)
    total_P = np.sum(output)
    total_R = np.sum(label)
    TP = float(np.sum(output * label))
    MiP = TP / max(total_P, 1e-12)
    MiR = TP / max(total_R, 1e-12)
    MiF = 2 * MiP * MiR / max((MiP + MiR), 1e-12)
    hamming_loss = metrics.hamming_loss(label, output)
    return MiP, MiR, MiF, total_P / N, total_R / N, hamming_loss


config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    print('begin training:')
    for epoch in range(epoch_num):
        loader.shuffle()
        for iter, indices in enumerate(range(0, loader.train_size, batch_size)):
            batch_W = loader.train_W[indices:indices + batch_size]
            batch_C = loader.train_C[indices:indices + batch_size]
            batch_Y = loader.mapping_label(loader.train_Y[indices:indices + batch_size])
            y_pred, loss, _ = sess.run(
                [model.predict, model.loss, model.train_op],
                feed_dict={model.input_w: batch_W, model.input_c: batch_C,
                           model.input_y: batch_Y, model.keep_prob: keep_pro})
            if iter % 10 == 0:
                print("===Result===")
                MiP, MiR, MiF, P_NUM, T_NUM, hamming_loss = micro_score(y_pred, batch_Y)
                print("epoch:%d  iter:%d, mean loss:%.3f,  PNum:%.2f, TNum:%.2f" % (
                    epoch + 1, iter + 1, loss, P_NUM, T_NUM))
                print("Micro-Precision:%.3f, Micro-Recall:%.3f, Micro-F Measure:%.3f, Hamming Loss:%.5f" % (
                    MiP, MiR, MiF, hamming_loss))

        if epoch >= epoch_num / 4:
            validataion()

    print('======= final =======')
    validataion(localize=True)
