# encoding=utf-8

"""
评测数据格式：  行号\t句1\t句2
本程序作用： 将每行句子分词处理
"""

import jieba
stopwords = [word.decode("utf8").strip() for word in open("../dictionary/stopwords.txt", 'r').readlines()]
jieba.load_userdict("../dictionary/dic.txt")


def remove_stopwords(segmented_list):
    out_str = ""
    for word in segmented_list:
        if word not in stopwords and word != " ":
            out_str += word
            out_str += ","
    return out_str[:-1]


def pretreat_sentence(sentence):
    seg_list = jieba.cut(sentence)
    final_seg_str = remove_stopwords(seg_list)
    return final_seg_str


def process(new_num, content):
    (_, s1, s2, tag) = content.decode("utf8").split("\t")
    new_num = '"' + str(new_num) + '",'
    new_s1 = '"' + pretreat_sentence(s1) + '",'
    new_s2 = '"' + pretreat_sentence(s2) + '",'
    tag = '"' + str(tag).strip() + '"\n'
    return new_num + new_s1 + new_s2 + tag

def process_no_tag_file(new_num, content):
    (_, s1, s2) = content.decode("utf8").split("\t")
    new_num = '"' + str(new_num) + '",'
    new_s1 = '"' + pretreat_sentence(s1) + '",'
    new_s2 = '"' + pretreat_sentence(s2) + '"\n'
    return new_num + new_s1 + new_s2


if __name__ == "__main__":
    input_file = open("../data/data.csv", "r")
    output_file = open("../data/corpus.csv", "w")
    for num, line in enumerate(input_file):
        pretreated_content = process(num + 1, line)
        output_file.write(pretreated_content.encode("utf8"))
    input_file.close()
    output_file.close()
