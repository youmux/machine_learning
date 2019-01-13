import numpy as np
import matplotlib.pyplot as plt
from linear_regression import LinearRegression
import random

if __name__ == '__main__':
    # 生成数据
    X_train = np.arange(1, 30, 1)
    X_train = X_train.reshape(1, X_train.shape[0])
    Y_train = np.array([15, 11, 10, 32, 16, 20, 27, 44, 41, 48, 53, 39, 41, 40, 49, 36, 49, 94, 57, 45, 72, 96, 43, 81, 70, 99, 80, 91, 70])
    X_test = np.arange(1, 30, 1)
    X_test = X_test.reshape(1, X_test.shape[0])

    N = 200 # 迭代次数

    # 线性回归
    lr = LinearRegression()
    lr.init(X_train, Y_train)
    lr.train(0.005, N)
    predicted = lr.predict(X_test)

    # 显示
    # 测试集,模型参数w、b的函数
    plt.subplot(1,2,1)
    Y_test = predicted
    plt.scatter(X_train, Y_train)
    X_test = X_test.reshape(X_test.shape[1], 1)
    Y_test = Y_test.reshape(Y_test.shape[1], 1)
    plt.plot(X_test, Y_test, color = 'red')
    # 迭代代价图
    plt.subplot(1,2,2)
    x = [xx for xx in range(N)]
    plt.plot(x, lr.costs)
    plt.show()