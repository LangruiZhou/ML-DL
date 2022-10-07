import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split


class LinearRegression:
    def __init__(self):
        self.w = None
        self.n_features = None  # 样本的属性数量

    # 模型训练
    def fit(self, X, y):
        # 验证X与y的匹配度
        assert isinstance(X, np.ndarray) and isinstance(y, np.ndarray)
        assert X.ndim == 2 and y.ndim == 1
        assert X.shape[0] == y.shape[0]
        # 扩展X
        n_samples = X.shape[0]  # 样本数 = X的行数
        self.n_features = X.shape[1]  # 样本的属性数量 = X的列数
        extra = np.ones((n_samples,))  # X最右侧全置为1的列
        X = np.c_[X, extra]  # 把X和extra配对
        # 计算w  w = (X^T*X)^(-1)*X^T*y
        if self.n_features < n_samples:
            self.w = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
        else:
            raise ValueError('Do not have enough samples.')

    # 预测函数
    def predict(self, X):
        n_samples = X.shape[0]  # 样本数 = X的行数
        self.n_features = X.shape[1]  # 样本的属性数量 = X的列数
        extra = np.ones((n_samples,))  # X最右侧全置为1的列
        X = np.c_[X, extra]
        # 确保w非空
        if self.w is None:
            raise RuntimeError('Cannot predict before fit.')
        y_predict = X.dot(self.w)
        return y_predict


# 计算均方误差E
def meanSquaredError(y, y_predict):
    return np.mean(np.square(y - y_predict))


if __name__ == '__main__':
    # 导入Boston房价数据集
    boston = load_boston()
    data = boston.data
    target = boston.target
    # 划分训练集/测试集, 20%用作测试集
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2)

    # 测试自己的模型
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_predict = lr.predict(X_test)
    print(meanSquaredError(y_test, y_predict))






