#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   build_one_hot.py
@Time    :   2020/05/20 11:53:00
@Author  :   JohnsonGuo
@Contact :   gwx0320@foxmail.com
@License :   (C)Copyright 2019-2020, johnsongwx
@Desc    :   None
'''


import torch
import numpy as np


# 参数为一字词，返回值为去掉了[]和/词性的纯词汇，建立词典函数那里也要改一下
def text_process(ori_word):
    start = 0
    if ori_word[0] == '[':
        start = 1
    end = ori_word.find(']')
    if end == -1:
        end = len(ori_word)

    return ori_word[start:end]


def build_one_hot(path, dic):
    x = []  # 最终，x是一个存储若干torch向量的数组
    y = []  # y是一个普通数组，里面每一位都是[0]或[1]，便于后续操作
    # matrix = np.zeros(0)
    with open(path, "r") as f:
        lines = f.readlines()
        # cnt = 0
        for line in lines:
            # print("正在读第%d行" % cnt)
            # cnt += 1

            text = line.split()  # 一句话里若干个词语构成的列表
            text_align = []  # 每一位对应text[]里的每一个词语
            decide = []  # 记录每一个词是不是被判断过了

            word_cnt = len(text)  # 这句话里总共多少词
            for i in range(word_cnt):  # 将每个词标记为0
                text_align.append(0)
                decide.append(False)

            for i in range(word_cnt):
                if decide[i]:  # 如果这个词被判断过了，就下一个词，防止在[]中的词被覆盖判断
                    continue

                if text[i].find('nt') != -1:
                    text_align[i] = 1

                if text[i].find('[') != -1:
                    start_id = i
                    flag = 0
                    j = i
                    error = 0  # 数据集里有只存在左括号无右括号的情况，如果发现则忽略，error=1
                    while j < word_cnt:
                        if text[j].find(']') != -1:
                            end_id = j
                            if text[j].find('nt') != -1:
                                flag = 1
                            break
                        if j == word_cnt - 1 and text[j].find(']') == -1:
                            error = 1
                        j += 1
                    if error == 0:
                        for k in range(start_id, end_id + 1):
                            text_align[k] = flag
                            decide[k] = True
            # print("第%d行" % cnt)
            # print(text_align)
            # cnt += 1

            for i in range(word_cnt):
                if i + 2 >= word_cnt:
                    break
                tmp_x = []
                # tmp_y = []  # 注意这里为了后续操作方便，tmp_y是数组

                tmp_x += dic.get(text_process(text[i]), dic.get("Unknown"))
                tmp_x += dic.get(text_process(text[i+1]), dic.get("Unknown"))
                tmp_x += dic.get(text_process(text[i+2]), dic.get("Unknown"))

                x.append(tmp_x)
                y.append(text_align[i + 1])
    return x, y

# dic = {}
# arr = np.zeros(5, dtype=int)
# arr[0] = 1
# dic['天津/ns'] = list(arr.copy())
# arr[0] = 0
# arr[1] = 1
# dic['ＳＯＳ/nz'] = list(arr.copy())
# arr[1] = 0
# arr[2] = 1
# dic['的/ud'] = list(arr.copy())
# arr[2] = 0
# arr[3] = 1
# dic['记者/n'] = list(arr.copy())
# arr[3] = 0
# arr[4] = 1
# dic['Unknown'] = list(arr.copy())
# arr[4] = 0
# m, d, my = build_one_hot("/Users/gwx/OneDrive/学习/大二下/知识工程/作业1 命名实体识别/code/code_test_set_1", dic)
# print(m)
# print(d)
# print(my)

# print(text_process("  集团/n]nt"))
