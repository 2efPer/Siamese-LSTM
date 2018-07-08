
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import gensim
import numpy as np
import itertools

def make_w2v_embeddings(df, embedding_dim=20):
    vocabs = {}
    vocabs_cnt = 0
    vocabs_not_w2v = {}
    vocabs_not_w2v_cnt = 0
    print("Loading word2vec model(it may takes 2-3 mins) ...")
    word2vec = gensim.models.word2vec.Word2Vec.load("../data/Atec.w2v").wv
    for index, row in df.iterrows():
        # Print the number of embedded sentences.
        if index != 0 and index % 1000 == 0:
            print("{:,} sentences embedded.".format(index))
        # Iterate through the text of both questions of the row
        for sentence in ['s1', 's2']:
            q2n = []  # q2n -> question numbers representation
            for word in str(row[sentence]).split(","):
                # If a word is missing from word2vec model.
                if word not in word2vec.vocab:
                    if word not in vocabs_not_w2v:
                        vocabs_not_w2v_cnt += 1
                        vocabs_not_w2v[word] = 1
                # If you have never seen a word, append it to vocab dictionary.
                if word not in vocabs:
                    vocabs_cnt += 1
                    vocabs[word] = vocabs_cnt
                    q2n.append(vocabs_cnt)
                else:
                    q2n.append(vocabs[word])
            # Append question as number representation
            df.at[index, sentence + '_n'] = q2n

    embeddings = 1 * np.random.randn(len(vocabs) + 1, embedding_dim)  # This will be the embedding matrix
    embeddings[0] = 0  # So that the padding will be ignored
    # Build the embedding matrix
    for word, index in vocabs.items():
        if word in word2vec.vocab:
            embeddings[index] = word2vec.word_vec(word)
    del word2vec

    return df, embeddings


def split_and_zero_padding(df, max_seq_length):
    x = {'left': df['s1_n'], 'right': df['s2_n']}
    for dataset, side in itertools.product([x], ['left', 'right']):
        dataset[side] = pad_sequences(dataset[side], padding='pre', truncating='post', maxlen=max_seq_length)
    return dataset


class ManDist(Layer):
    def __init__(self, **kwargs):
        self.result = None
        super(ManDist, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ManDist, self).build(input_shape)

    def call(self, x, **kwargs):
        self.result = K.exp(-K.sum(K.abs(x[0] - x[1]), axis=1, keepdims=True))
        return self.result

    def compute_output_shape(self, input_shape):
        return K.int_shape(self.result)