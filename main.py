#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   main.py
@Time    :   2020/05/20 11:53:00
@Author  :   JohnsonGuo
@Contact :   gwx0320@foxmail.com
@License :   (C)Copyright 2019-2020, johnsongwx
@Desc    :   None
'''


import matplotlib.pyplot as plt
from build_matrix import *

dic_size = 500  # 词典大小

if __name__ == '__main__':

    start = time.time()
    matrix = torch.load("./tensor/matrix")
    matrix_y = torch.load("./tensor/matrix_y")
    elapsed = (time.time() - start)
    print("训练集数据读取完毕，时间：", elapsed)

    start = time.time()
    matrix_vali = torch.load("./tensor/matrix_vali")
    y_val = torch.load("./tensor/y_val")
    print("验证集数据读取完毕，时间：", elapsed)

    #################################

    # x和y构造完毕，开始【小批量梯度下降】

    batch_matrix = []  # 分批训练x矩阵
    batch_y = []  # 分批训练y向量
    startmatrix = 0
    # batchsize为矩阵分块大小
    batchsize = 1024
    dim_x = matrix.shape[1]
    while startmatrix < dim_x:
        if startmatrix + batchsize < dim_x:
            batch_matrix.append(matrix[:, startmatrix:startmatrix + batchsize])
            batch_y.append(matrix_y[startmatrix:startmatrix + batchsize])
        else:
            batch_matrix.append(matrix[:, startmatrix:dim_x])  # 切片范围是左闭右开
            batch_y.append(matrix_y[startmatrix:dim_x])
        startmatrix += batchsize

    learning_rate = 1
    epoch_num = 501  # 最大训练轮数

    theta = torch.rand(1, dic_size * 3, requires_grad=True).float()

    pltX = []
    pltL = []  # 训练集损失函数值
    pltY = []
    pltX10 = []

    for i in range(epoch_num):
        pltX.append(i)

        start = time.time()

        # 分小批量梯度下降
        batch = len(batch_matrix)
        for j in range(batch):
            g = torch.sigmoid(torch.mm(theta, batch_matrix[j]))
            loss = torch.mean(batch_y[j] * torch.log(g) + (1 - batch_y[j]) * torch.log(1 - g))
            loss.backward()
            with torch.no_grad():
                theta += learning_rate * theta.grad
                theta.grad.zero_()

        # 本轮训练完成，计算本轮训练的loss
        epoch_g = torch.sigmoid(torch.mm(theta, matrix))
        epoch_loss = torch.mean(matrix_y * torch.log(epoch_g) + (1 - matrix_y) * torch.log(1 - epoch_g))
        pltL.append(epoch_loss)
        print("第%d轮训练Loss：" % i)
        print(epoch_loss)
        elapsed = (time.time() - start)
        print("本轮训练时间：", elapsed)

        if i % 10 == 0:
            num = 0  # 预测总数
            correct = 0  # 准确数
            nt_sum = 0

            vali = torch.mm(theta, matrix_vali)  # vali为1*n的矩阵

            k = 0
            for yi in vali[0]:
                if yi > 0.5:
                    num += 1
                    if y_val[k] == 1:
                        correct += 1
                if y_val[k] == 1:
                    nt_sum += 1
                k += 1

            if num != 0:
                accuracy = correct / num
            else:
                accuracy = 0

            if nt_sum != 0:
                complete = correct / nt_sum
            else:
                complete = 0

            if complete + accuracy != 0:
                f1measure = 2 * accuracy * complete / (accuracy + complete)
            else:
                f1measure = 0

            print("F1-measure = ", f1measure)
            print("查准率：", accuracy)
            print("查全率：", complete)

            print("----------")
            pltX10.append(i)
            pltY.append(f1measure)

    plt.figure()
    plt.title("1F1-Measure: Learning Rate = 1")
    plt.plot(pltX10, pltY)
    plt.savefig("1[f1]lr=1.png", dpi=220)
    plt.show()

    plt.figure()
    plt.title("1Loss: Learning Rate = 1")
    plt.plot(pltX, pltL)
    plt.savefig("1[glo_loss]lr=1.png", dpi=220)
    plt.show()
