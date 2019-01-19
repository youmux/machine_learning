# 数字分类
# 本例子,仅对数字0识别

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from logistic_regression import LogisticRegression

if __name__ == '__main__':
    # 加载数据
    train_dataset_path = '../datasets/digital_datasets/train_images/'
    test_dataset_path = '../datasets/digital_datasets/test_images/'

    def load_dataset(dataset_path):
        images = []
        targets = []

        path_root = os.listdir(dataset_path)
        for path_root_dir in path_root: # 目录下
            path_root_x = dataset_path + path_root_dir + '/'
            path_root_root = os.listdir(path_root_x)    # 子目录下
            train_sets_path = [path_root_x + filename for filename in path_root_root]
            for path in train_sets_path:
                image = Image.open(path)    # 加载图片
                image_array = np.array(image)   # 转为矩阵
                image_array_ravel = image_array.ravel() # 改变形状
                image_array_ravel_scale = image_array_ravel / 255   # 缩放
                images.append(image_array_ravel_scale)

                l = path.split('/')
                targets.append(1.0 if l[-2] == '0' else 0.0) # 不属于数字0的目录下的图片，结果都设为0
                
        X = np.stack(images)                 # 取出训练数据
        Y = np.array(targets, ndmin=2).T     # 取出数据集结果
        return X.T, Y.T

    # 加载数据
    train_set_x, train_set_y = load_dataset(train_dataset_path)
    test_set_x, test_set_y = load_dataset(test_dataset_path)

    # 逻辑回归
    N = 2000
    lr = LogisticRegression()
    lr.initialize_parameters(train_set_x, train_set_y)
    lr.train(0.003, N)
    predicted = lr.predict(test_set_x)

    # 显示结果对比、准确率
    print('预测结果：', end = '')
    print(predicted)
    print('测试集结果：', end = '')
    print(test_set_y)
    print(f'准确率:{np.mean(np.equal(test_set_y, predicted)) * 100}%')

    # 显示代价函数迭代
    plt.plot([x for x in range(N)], lr.costs)
    plt.show()