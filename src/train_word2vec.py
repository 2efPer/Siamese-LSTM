# encoding=utf-8
'''
训练word2vec模型
'''
import gensim
import pandas as pd


def extract_questions():
    df = pd.read_csv("~/PycharmProjects/sentence_similarity/data/corpus.csv", header=None)
    df.columns = ['id', 's1', 's2','tag']
    for i, row in df.iterrows():
        if row['s1']:
            yield str(row['s1']).split(",")
        if row['s2']:
            yield str(row['s2']).split(",")


documents = list(extract_questions())
print(len(documents))
model = gensim.models.Word2Vec(documents, size=20)
model.train(documents, total_examples=len(documents), epochs=10)
model.save("../data/Atec.w2v")