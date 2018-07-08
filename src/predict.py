# encoding=utf-8
"""
输入数据： 行号/t分词后的句子1/t分词后的句子2/r/n
输出 ： 行号\t预测结果(0/1)

步骤：
1.word2vec向量化输入数据
2.SLSTM得到结果
"""
import sys
from pretreat import process
import pandas as pd
from utils import make_w2v_embeddings
from utils import split_and_zero_padding
from utils import ManDist
import tensorflow as tf

if __name__ == "__main__":
    # 预处理成中间文件tmp.csv
    input_file_name = sys.argv[1]
    output_file_name = sys.argv[2]
    mid_file_name = "./tmp.csv"
    input_file = open(input_file_name, "r")
    mid_file = open(mid_file_name, "w")
    for num, line in enumerate(input_file):
        pretreated_content = process(num + 1, line)
        mid_file.write(pretreated_content.encode("utf8"))
    input_file.close()
    mid_file.close()
    # 加载模型
    test_df = pd.read_csv(mid_file_name, header=None)
    test_df.columns = ['id', 's1', 's2','tag']
    for q in ['s1', 's2']:
        test_df[q + '_n'] = test_df[q]
    embedding_dim = 20
    max_seq_length = 5
    test_df, embeddings = make_w2v_embeddings(test_df, embedding_dim=embedding_dim)
    X_test = split_and_zero_padding(test_df, max_seq_length)
    model = tf.keras.models.load_model('../data/model.h5', custom_objects={'ManDist': ManDist})
    prediction = model.predict([X_test['left'], X_test['right']])
    print(prediction)
    with open(output_file_name, 'w') as fout:
        for no, line in enumerate(prediction):
            if float(line[0]) > 0.5:
                fout.write(str(no + 1) + '\t1\n')
            else:
                fout.write(str(no + 1) + '\t0\n')