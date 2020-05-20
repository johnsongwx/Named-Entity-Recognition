#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   data_processing.py
@Time    :   2020/05/20 11:53:00
@Author  :   JohnsonGuo
@Contact :   gwx0320@foxmail.com
@License :   (C)Copyright 2019-2020, johnsongwx
@Desc    :   None
'''


# 读文件函数，传入参数textrange缺省时，读取全文，否则读取前textrange行
# 可以指定编码方式，默认为utf-8
# 返回文章
def readfiles(path, coding='utf-8'):
    with open(path, 'r', encoding=coding) as txt:
        content = txt.read()
    return content


# 按要求将原始数据集数据分成三个文件
def build_set(path, coding='utf-8'):
    with open(path, 'r', encoding=coding) as txt:
        content = txt.readlines()
        cnt = 0
        with open("data/training_set", "w", encoding='utf-8') as training:
            with open("data/validation_set", "w", encoding='utf-8') as validation:
                with open("data/test_set", "w", encoding='utf-8') as test:
                    for line in content:
                        if cnt < 14935:
                            training.write(line)
                        elif 14935 <= cnt < 19966:
                            validation.write(line)
                        elif 19966 <= cnt < 23269:
                            test.write(line)
                        else:
                            break
                        cnt += 1


# 统计名词词性词频，返回按词频排好序的数组
def countword(text):
    freq = {}
    words = text.split()
    for word in words:
        if word.find('/w', 0, len(word)) == -1 and word != '':  # 不是标点符号，也不为空
            if word.find('nt', 0, len(word)) == -1:  # 不是命名实体词
                freq[word] = freq.get(word, 0) + 1
            else:
                freq[word] = freq.get(word, 0) + 20  # 是命名实体词，增加词频权重
    count = list(freq.items())
    count.sort(key=lambda x: x[1], reverse=True)
    return count


# # 将三个数据集进行分词，注意结果里会多空行，再后续使用词的时候，readline()后要判断是否为空，为空则continue
# def split_set():
#     with open("data/data_raw/training_set_raw", "r", encoding='utf-8') as training:
#         words = training.read().split()
#         # words = re.split(r" (?![^\[]*\])", training.read())
#         with open("data/training_set", "w", encoding='utf-8') as ts:
#             for word in words:
#                 ts.write(word + '\n')
#     with open("data/data_raw/validation_set_raw", "r", encoding='utf-8') as validation:
#         words = validation.read().split()
#         # words = re.split(r" (?![^\[]*\])", validation.read())
#         with open("data/validation_set", "w", encoding='utf-8') as val:
#             for word in words:
#                 val.write(word + '\n')
#     with open("data/data_raw/test_set_raw", "r", encoding='utf-8') as test:
#         words = test.read().split()
#         # words = re.split(r" (?![^\[]*\])", test.read())
#         with open("data/test_set", "w", encoding='utf-8') as tt:
#             for word in words:
#                 tt.write(word + '\n')
