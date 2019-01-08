import numpy as np

class NeuralNetworksLogisticRegression:
    '''
    逻辑回归
    # !不同于线性回归的地方,用'# !'标注了

    参数:
        X - 训练集(需要将训练集的特征缩放到合适范围,并将参数以列向量重排)
        Y - 训练集的结果(取值为0|1)
        W - 假设函数的多参数组成矩阵(w1、w2、w3 ...)(W_j对应theta_j,j=1,2,3...)
        b - 假设函数的参数(x0 = 1的值)(b对应theta_0)
        learning_rate - 学习速率
        num_iter - 迭代次数
        costs - 代价函数值的集合(非必须操作)
        n_x - 输入层的数量
        n_h - 隐藏层的数量
        n_y - 输出层的数量
    
    使用:
        lg = NeuralNetworksLogisticRegression()
        lg.fit(X_train, Y_train)
        lg.train(0.001, 2000)
        predicted = lg.predict(X_test)
    '''
    X = 0
    Y = 0
    m = 0
    W1 = 0
    W2 = 0
    b1 = 0
    b2 = 0
    learning_rate = 0
    num_iter = 0
    costs = []
    n_x = 0
    n_h = 0
    n_y = 0


    # 初始化变量
    def fit(self, X, Y, n_h=1):
        '''
        1.加载训练集,并设置一些初始值
        2.设置神经网络的结构，对每层设置数量
        网络结构如下:(这里仅有1个隐藏层,且隐藏层为4个节点)
        [输入层] [隐藏层] [输出层]
                    o
            o       o       o
            o       o
                    o
        3.用随机数矩阵初始化权重矩阵，用零矩阵初始化偏移

        参数:
            X - 训练集
            Y - 训练集的结果
        
        注意:
            神经网络中,不能简单将每层的权重设为0
        '''
        self.X = X
        self.Y = Y
        self.costs = []
        self.m = X.shape[1]

        # 设置神经网络的结构
        self.n_x = self.X.shape[0]
        self.n_h = n_h
        self.n_y = self.Y.shape[0]

        # 初始化参数
        self.W1 = np.random.randn(self.n_h, self.n_x) * 0.01
        self.b1 = np.zeros(shape = (self.n_h, 1))
        self.W2 = np.random.randn(self.n_y, self.n_h) * 0.01
        self.b2 = np.zeros(shape = (self.n_y, 1))

    
    # 前向传播
    def forward_propagation(self, X):
        '''
        计算前向传播

        公式:
            A1 = X
            Z2 = W1 * A1
            A2 = g(Z2)      (add A2_0)
            Z3 = W2 * A2
            A3 = g(Z3)
        '''
        A1 = X
        Z2 = np.dot(self.W1, A1) + self.b1
        A2 = np.tanh(Z2)    # 隐藏层的激活函数，采用的是g(x) = tanh(x)
        Z3 = np.dot(self.W2, A2) + self.b2
        A3 = 1 / (1 + np.exp(-Z3))

        cache = {
            'Z2': Z2,
            'A2': A2,
            'Z3': Z3,
            'A3': A3
        }
        return cache
    
    
    # 向后传播
    def backward_propagation(self, cache):
        '''
        计算向后传播

        公式:
            delta3 = A3 - Y
            dW2 = 1/m * delta3 * A2.T
            db2 = 1/m * sum(delta3)
            delta2 = W2.T * delta3 * g2'(delta2)
            dW1 = 1/m * delta2 * X.T
            db1 = 1/m * sum(delta2)
        '''
        m = self.m

        A2 = cache['A2']
        A3 = cache['A3']

        delta3 = A3 - self.Y
        dW2 = 1/m * np.dot(delta3, A2.T)
        db2 = 1/m * np.sum(delta3, axis=1, keepdims=True)
        # 由于激活函数是: g(x) = tanh(x)
        # 求导: g'(x) = 1 - tanh(x) ** 2
        delta2 = np.dot(self.W2.T, delta3) * (1 - A2 ** 2)
        dW1 = 1/m * np.dot(delta2, self.X.T)
        db1 = 1/m * np.sum(delta2, axis=1, keepdims=True)

        grads = {
            'dW1': dW1,
            'db1': db1,
            'dW2': dW2,
            'db2': db2
        }
        return grads


    # 梯度下降
    def gradient_descent(self):
        '''
        进行梯度下降的运算,公式:W_j = W_j - alpha * partial_derivative(J_j(W_j, b)), j = 1,2,3...
        '''
        for i in range(self.num_iter):
            cache = self.forward_propagation(self.X)
            grads = self.backward_propagation(cache)    # 需要前向传播的参数计算
            dW1, db1 = grads['dW1'], grads['db1']
            dW2, db2 = grads['dW2'], grads['db2']

            # 梯度下降,更新参数W、b
            self.W1 = self.W1 - self.learning_rate * dW1
            self.b1 = self.b1 - self.learning_rate * db1
            self.W2 = self.W2 - self.learning_rate * dW2
            self.b2 = self.b2 - self.learning_rate * db2

            # 计算代价
            A_output = cache['A3']  # A3即是output层
            self.compute_cost(A_output)   # 计算代价


    # 代价计算
    def compute_cost(self, A_output):
        '''
        计算代价
        '''
        cost = (-1 / self.m) * np.sum(self.Y * np.log(A_output) + (1 - self.Y) * np.log(1 - A_output))
        self.costs.append(cost)


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
        # 复用前向传播计算
        cache = self.forward_propagation(X)
        predicted = cache['A3']     # A3即是output层

        # 转为0|1
        predicted = np.round(predicted)
        predicted = predicted.astype(np.int)
        
        return predicted