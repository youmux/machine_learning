import numpy as np
import matplotlib.pyplot as plt

from neural_networks_logistic_regression import NeuralNetworksLogisticRegression
from logistic_regression import LogisticRegression

# 绘制决策边界
def plot_decision_boundary(model, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=np.squeeze(y), cmap=plt.cm.Spectral)    # show dataset points; y -> np.squeeze(y)


# 加载数据
def load_planar_dataset():
    np.random.seed(1)
    m = 400  # 样本数量
    N = int(m / 2)  # 每个类别的样本量
    D = 2  # 维度数
    X = np.zeros((m, D))  # 初始化X
    Y = np.zeros((m, 1), dtype='uint8')  # 初始化Y
    a = 4  # 花儿的最大长度

    for j in range(2):
        ix = range(N * j, N * (j + 1))
        t = np.linspace(j * 3.12, (j + 1) * 3.12, N) + np.random.randn(N) * 0.2  # theta
        r = a * np.sin(4 * t) + np.random.randn(N) * 0.2  # radius
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        Y[ix] = j

    X = X.T
    Y = Y.T

    return X, Y


if __name__ == "__main__":
    # 加载数据
    X, Y = load_planar_dataset()
    
    N = 20000   # 迭代次数

    # 1.逻辑回归
    lr = LogisticRegression()
    lr.fit(X, Y)
    lr.train(0.05, N)
    predicted = lr.predict(X)   # 预测结果
    print(f'逻辑回归的准确率:{np.mean(np.equal(Y, predicted)) * 100}%')
    # 边界图
    plt.subplot(2,2,1)
    plot_decision_boundary(lambda x:lr.predict(x.T), X, Y)   # 边界是用等高线绘制的
    # 迭代代价图
    plt.subplot(2,2,2)
    x = [xx for xx in range(N)]
    plt.plot(x, lr.costs)



    # 2.神经网络的逻辑回归
    nn_lr = NeuralNetworksLogisticRegression()
    nn_lr.fit(X, Y, n_h=4)
    nn_lr.train(learning_rate=0.05, num_iter=N)
    predicted = nn_lr.predict(X)
    print(f'神经网络逻辑归回的准确率为:{np.mean(np.equal(Y, predicted)) * 100} %')

    # 边界图
    plt.subplot(2,2,3)
    plot_decision_boundary(lambda x:nn_lr.predict(x.T), X, Y)   # 边界是用等高线绘制的
    # 迭代代价图
    plt.subplot(2,2,4)
    x = [xx for xx in range(N)]
    plt.plot(x, nn_lr.costs)
    plt.show()