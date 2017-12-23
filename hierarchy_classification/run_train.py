# encoding=utf-8
import tensorflow as tf
from dataloader import DataLoader
from hierarchy_model import DeepHan
import numpy as np
import os

# LD_LIBRARY_PATH	/usr/local/cuda-8.0/lib64:$LD_LIBRARY_PATH
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

batch_size = 128
eval_batch_size = 1024

epoch_num = 7

keep_pro = 0.9

loader = DataLoader()
model = DeepHan(loader.word_embeddings, loader.char_embeddings)


def validataion():
    # model.prediction_fused
    print('begin to eval:')
    outputs = []
    for i in range(0, loader.test_size, eval_batch_size):
        batch_W = loader.test_W[i:i + eval_batch_size]
        batch_C = loader.test_C[i:i + eval_batch_size]
        y_pred = sess.run(model.predict, feed_dict={model.input_w: batch_W, model.input_c: batch_C})
        outputs.append(y_pred)
    outputs = np.concatenate(outputs, axis=0)

    MiP, MiR, MiF, P_NUM, T_NUM = micro_score(outputs, loader.mapping_label(loader.test_Y))
    print(">>>>>>>> Final Result:  PredictNum:%.2f, TrueNum:%.2f" % (P_NUM, T_NUM))
    print(">>>>>>>> Micro-Precision:%.3f, Micro-Recall:%.3f, Micro-F Measure:%.3f" % (MiP, MiR, MiF))


def micro_score(output, label):
    N = len(output)
    total_P = np.sum(output)
    total_R = np.sum(label)
    TP = float(np.sum(output * label))
    MiP = TP / max(total_P, 1e-12)
    MiR = TP / max(total_R, 1e-12)
    MiF = 2 * MiP * MiR / max((MiP + MiR), 1e-12)
    return MiP, MiR, MiF, total_P / N, total_R / N


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print('begin training:')
    for epoch in range(epoch_num):
        loader.shuffle()
        for iter, indices in enumerate(range(0, loader.train_size, batch_size)):
            batch_W = loader.train_W[indices:indices + batch_size]
            batch_C = loader.train_C[indices:indices + batch_size]
            batch_Y = loader.mapping_label(loader.train_Y[indices:indices + batch_size])
            y_pred, loss, _ = sess.run([model.predict, model.loss, model.train_op],
                                       feed_dict={model.input_w: batch_W, model.input_c: batch_C,
                                                  model.input_y: batch_Y})
            if iter % 10 == 0:
                print("===Result===")
                MiP, MiR, MiF, P_NUM, T_NUM = micro_score(y_pred, batch_Y)
                print("epoch:%d  iter:%d, mean loss:%.3f,  PNum:%.2f, TNum:%.2f" % (
                    epoch + 1, iter + 1, loss, P_NUM, T_NUM))
                print("Micro-Precision:%.3f, Micro-Recall:%.3f, Micro-F Measure:%.3f" % (MiP, MiR, MiF))

        validataion()
