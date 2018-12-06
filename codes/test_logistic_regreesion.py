# 识别猫
import numpy as np
import matplotlib.pyplot as plt
import h5py
from logistic_regreesion import LogisticRegreesion

if __name__ == '__main__':
    # 加载数据
    # train_set_x_orig ：保存的是训练集里面的图像数据（本训练集有209张64x64的图像）。
    # train_set_y_orig ：保存的是训练集的图像对应的分类值（【0 | 1】，0表示不是猫，1表示是猫）。
    # test_set_x_orig ：保存的是测试集里面的图像数据（本训练集有50张64x64的图像）。
    # test_set_y_orig ： 保存的是测试集的图像对应的分类值（【0 | 1】，0表示不是猫，1表示是猫）。
    # classes ： 保存的是以bytes类型保存的两个字符串数据，数据为：[b’non-cat’ b’cat’]。
    train_dataset_path = '../datasets/train_catvnoncat.h5'
    test_dataset_path = '../datasets/test_catvnoncat.h5'
    def load_dataset():
        train_dataset = h5py.File(train_dataset_path, 'r')
        train_set_x_orig = np.array(train_dataset["train_set_x"][:])
        train_set_y_orig = np.array(train_dataset["train_set_y"][:])
        
        test_dataset = h5py.File(test_dataset_path, 'r')
        test_set_x_orig = np.array(test_dataset["test_set_x"][:])
        test_set_y_orig = np.array(test_dataset["test_set_y"][:])

        classes = np.array(test_dataset["list_classes"][:])

        train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
        test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

        return train_set_x_orig, train_set_y_orig[0], test_set_x_orig, test_set_y_orig[0], classes

    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
    # 预览猫图
    # plt.imshow(train_set_x_orig[25])
    # plt.title(classes[train_set_y[25]].decode('utf-8'))
    # plt.show()

    # 处理数据
    # 1.一维化
    train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T   # -1表示剩下的列自动算（每一列为一个图，每列的每个元素相当于一个参数）
    test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
    # 2.数据预处理 - 转为[0-1]区间的浮点数
    train_set_x = train_set_x_flatten / 255
    test_set_x = test_set_x_flatten / 255

    # 逻辑回归
    lr = LogisticRegreesion()
    lr.fit(train_set_x, train_set_y)
    lr.train(0.005, 2000)
    predicted = lr.predict(test_set_x)
    print(predicted)
    print(test_set_y)

    # 显示代价函数迭代
    x = [xx for xx in range(1, 2001)]
    plt.plot(x, lr.costs)
    plt.show()