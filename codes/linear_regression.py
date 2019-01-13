import numpy as np

class LinearRegression:
    '''
    线性回归

    参数:
        X - 训练集(需要将训练集的特征缩放到0~1,将参数以列向量重排)
        Y - 训练集的结果(取值为0|1)
        W - 假设函数的多参数组成矩阵(w1、w2、w3 ...)(W_j对应theta_j,j=1,2,3...)
        b - 假设函数的参数(x0 = 1的值)(b对应theta_0)
        learning_rate - 学习速率
        num_iter - 迭代次数
        costs - 代价函数值的集合(非必须操作)
    
    使用:
        lg = LinearRegression()
        lg.init(X_train, Y_train)
        lg.train(0.001, 2000)
        predicted = lg.predict(X_test)
    '''
    X = 0
    Y = 0
    W = 0
    b = 0
    learning_rate = 0
    num_iter = 0
    costs = []


    # 初始化变量
    def init(self, X, Y):
        '''
        加载训练集,并设置一些初始值

        参数:
            X - 训练集
            Y - 训练集的结果
        '''
        self.X = X
        self.Y = Y
        self.W = np.zeros(shape = (X.shape[0], 1))
        self.b = 0
        self.costs = []


    # 对代价函数J求导
    # h(x) = W * x + b
    def partial_derivative(self):
        '''
        对梯度下降公式后半部分的求导(手动计算)数值

        返回:
            dW,db - 假设函数的参数的偏导值
        '''

        m = self.X.shape[1]

        # 假设函数的激活值(正向传播)
        H = np.dot(self.W.T, self.X) + self.b

        # 计算代价,记录代价(非必须操作,只是便于观察梯度下降的效果)
        cost = (1 / (2 * m)) * np.sum(H - self.Y) ** 2
        self.costs.append(cost)

        # 求偏导(反向传播)
        dW = 1 / m * np.dot(self.X, (H - self.Y).T) # 注:矩阵点乘后,已经求过和
        db = 1 / m * np.sum(H - self.Y)             # 注:没有进行过矩阵乘法运算,需要手动求和

        return dW, db


    # 梯度下降
    # temp0 = W - alpha * partial_derivative(J0(W, b))
    # temp1 = b - alpha * partial_derivative(J1(W, b))
    # ...
    def gradient_descent(self):
        '''
        进行梯度下降的运算,公式:W_j = W_j - alpha * partial_derivative(J_j(W_j, b)), j = 1,2,3...
        '''

        for i in range(self.num_iter):
            dW, db = self.partial_derivative()
            
            # 梯度下降,优化参数W、b
            self.W = self.W - self.learning_rate * dW
            self.b = self.b - self.learning_rate * db


    # 开始训练
    def train(self, learning_rate = 0, num_iter = 0):
        '''
        开始训练
        参数:
            learning_rate - 学习速率
            num_iter - 迭代次数
        '''
        self.learning_rate = learning_rate
        self.num_iter = num_iter

        self.gradient_descent()


    # 预测
    def predict(self, X):
        '''
        预测X数据集
        参数:
            X - 测试数据集
        返回:
            predicted - 对于测试数据集X的预测结果
        '''
        # 带入参数w、b预测测试集
        predicted = np.dot(self.W.T, X) + self.b

        return predicted