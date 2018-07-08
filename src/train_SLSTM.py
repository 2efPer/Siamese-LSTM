# encoding=utf-8
"""
训练主要模型
"""

from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Input, Embedding, LSTM
from sklearn.model_selection import train_test_split
from time import time
from utils import make_w2v_embeddings
from utils import split_and_zero_padding
from utils import ManDist
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
matplotlib.use('Agg')
TRAIN_CSV = '../data/corpus.csv'
train_df = pd.read_csv(TRAIN_CSV)
train_df.columns = ['id', 's1', 's2','tag']
for q in ['s1', 's2']:
    train_df[q + '_n'] = train_df[q]
embedding_dim = 20
max_seq_length = 5
train_df, embeddings = make_w2v_embeddings(train_df, embedding_dim=embedding_dim)
validation_size = int(len(train_df) * 0.1)
training_size = len(train_df) - validation_size

X = train_df[['s1_n', 's2_n']]
Y = train_df['tag']
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size)
X_train = split_and_zero_padding(X_train, max_seq_length)
X_validation = split_and_zero_padding(X_validation, max_seq_length)
Y_train = Y_train.values
Y_validation = Y_validation.values
assert X_train['left'].shape == X_train['right'].shape
assert len(X_train['left']) == len(Y_train)


x = Sequential()
x.add(Embedding(len(embeddings), embedding_dim,
                weights=[embeddings], input_shape=(max_seq_length,), trainable=False))
n_hidden = 50
x.add(LSTM(n_hidden))
shared_model = x
left_input = Input(shape=(max_seq_length,), dtype='int32')
right_input = Input(shape=(max_seq_length,), dtype='int32')

malstm_distance = ManDist()([shared_model(left_input), shared_model(right_input)])
model = Model(inputs=[left_input, right_input], outputs=[malstm_distance])
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.SGD(), metrics=['accuracy'])
model.summary()
shared_model.summary()

batch_size = 1024 * 2
n_epoch = 50
training_start_time = time()
malstm_trained = model.fit([X_train['left'], X_train['right']], Y_train,
                           batch_size=batch_size, epochs=n_epoch,
                           validation_data=([X_validation['left'], X_validation['right']], Y_validation))
training_end_time = time()
print("Training time finished.\n%d epochs in %12.2f" % (n_epoch,
                                                        training_end_time - training_start_time))
model.save('../data/model.h5')

#========
plt.subplot(211)
plt.plot(malstm_trained.history['acc'])
plt.plot(malstm_trained.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.subplot(212)
plt.plot(malstm_trained.history['loss'])
plt.plot(malstm_trained.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')

plt.tight_layout(h_pad=1.0)


print(str(malstm_trained.history['val_acc'][-1])[:6] +
      "(max: " + str(max(malstm_trained.history['val_acc']))[:6] + ")")
print("Done.")