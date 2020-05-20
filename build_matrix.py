#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   build_matrix.py
@Time    :   2020/05/20 11:53:00
@Author  :   JohnsonGuo
@Contact :   gwx0320@foxmail.com
@License :   (C)Copyright 2019-2020, johnsongwx
@Desc    :   None
'''


from data_processing import *
from build_one_hot import *
import time

dic_size = 500  # 词典大小


def construct_high_freq_word_database():
    path = 'data/data_raw/1998-01-2003版-带音.txt'
    count_list = countword(readfiles(path, coding='gbk'))
    cnt = 0
    with open("./data/high_freq_words", "w", encoding='utf-8') as f:
        for i in range(dic_size - 1):
            word, count = count_list[i]
            if len(word) > 2 and word[-2] == 'n' and word[-1] == 't':
                cnt += 1
            f.write(word + '\n')
        #     print("{:<10}{:>10}".format(word, count))
        print("在已建立的词典中，共有%d个命名实体词" % (cnt))


def construct_dictionary():
    with open("./data/high_freq_words", "r", encoding='utf-8') as f:
        text = f.read().split()
        word_dic = {}
        text_sum = len(text)
        arr = np.zeros(dic_size, dtype=int)
        for i in range(text_sum):
            arr[i] = 1
            word_dic[text_process(text[i])] = list(arr.copy())
            arr[i] = 0

        arr[dic_size - 1] = 1
        word_dic["Unknown"] = list(arr.copy())
        arr[dic_size - 1] = 0

        # # 检查向量建设
        # res = list(word_dic.items())
        # for i in range(4):
        #     word, array = res[i]
        #     print(word)
        #     print(array)
        return word_dic


if __name__ == 'main':
    # 训练集、验证集、测试集构造
    start = time.time()
    # build_set('data/data_raw/1998-01-2003版-带音.txt', coding='gbk')
    build_set('data/data_raw/data_raw.txt')
    elapsed = (time.time() - start)
    print("数据集分拆完毕，时间：", elapsed)

    #################################

    start = time.time()
    # 建立高频词词表
    construct_high_freq_word_database()
    # 建立字典，dic为一字典，每一词为索引，对应其one-hot向量
    dic = construct_dictionary()

    elapsed = (time.time() - start)
    print("数据预处理完毕，时间：", elapsed)

    #################################

    # 训练数据准备
    ts_path = "data/training_set"
    vs_path = "data/validation_set"

    # 建立训练集的词向量
    start = time.time()

    x, y = build_one_hot(ts_path, dic)  # x是个列表
    matrix = torch.tensor(x).t().float()
    matrix_y = torch.tensor(y).float()

    elapsed = (time.time() - start)
    print("训练集数据建立完毕，时间：", elapsed)

    print("数据保存中……")
    start = time.time()
    torch.save(matrix, "./tensor/matrix")
    print("数据保存中……")
    torch.save(matrix_y, "./tensor/matrix_y")
    elapsed = (time.time() - start)
    print("训练集数据保存完毕，时间：", elapsed)

    # 建立验证集的词向量
    start = time.time()

    x_val, y_v = build_one_hot(vs_path, dic)  # x_val是个列表
    matrix_vali = torch.tensor(x_val).t().float()
    y_val = torch.tensor(y_v).float()

    elapsed = (time.time() - start)
    print("验证集数据建立完毕，时间：", elapsed)

    print("数据保存中……")
    start = time.time()
    torch.save(matrix_vali, "./tensor/matrix_vali")
    print("数据保存中……")
    torch.save(y_val, "./tensor/y_val")
    elapsed = (time.time() - start)
    print("验证集数据保存完毕，时间：", elapsed)
