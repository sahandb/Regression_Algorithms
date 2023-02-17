# imports
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


class MultipleRegression(object):
    def __init__(self):
        self.coefficients = []

    def fit(self, X, y):
        if len(X.shape) == 1:
            X = self._reshape_x(X)

        X = self._concatenate_ones(X)
        self.coefficients = np.linalg.inv(X.transpose().dot(X)).dot(X.transpose()).dot(y)

    def predict(self, entry):
        b0 = self.coefficients[0]
        other_betas = self.coefficients[1:]
        prediction = b0

        for xi, bi in zip(entry, other_betas):
            prediction += (bi * xi)
        return prediction

    @staticmethod
    def _reshape_x(X):
        return X.reshape(-1, 1)

    @staticmethod
    def _concatenate_ones(X):
        ones = np.ones(shape=X.shape[0]).reshape(-1, 1)
        return np.concatenate((ones, X), 1)


class RidgeRegression(object):
    def __init__(self):
        self.coefficients = []

    def fit(self, X, y, lambdaaa=0):
        if len(X.shape) == 1:
            X = self._reshape_x(X)

        X = self._concatenate_ones(X)
        G = lambdaaa * np.eye(X.shape[1])
        self.coefficients = np.linalg.inv((X.transpose().dot(X)) + (G.transpose().dot(G))).dot(X.transpose()).dot(y)

    def predict(self, entry):
        b0 = self.coefficients[0]
        other_betas = self.coefficients[1:]
        prediction = b0

        for xi, bi in zip(entry, other_betas): prediction += (bi * xi)
        return prediction

    @staticmethod
    def _reshape_x(X):
        return X.reshape(-1, 1)

    @staticmethod
    def _concatenate_ones(X):
        ones = np.ones(shape=X.shape[0]).reshape(-1, 1)
        return np.concatenate((ones, X), 1)


class LassoRegression(object):
    def __init__(self, lr, iter, lambdaa):
        self.iter = iter
        self.lr = lr
        self.lambdaa = lambdaa

    def calc_cost(self, X, Y, theta):
        predictions = np.dot(X, theta)
        cost = 0.5 * np.sum((predictions - Y) ** 2) + self.lambdaa * np.sum(np.absolute(theta))
        return cost

    def calc_gradient(self, X, Y, predictions, theta, index):
        gd = 0.0
        gd = np.dot(predictions - Y, X[:, index])
        gd = (gd + self.lambdaa * np.sign(theta[index]))
        return gd

    def grad_desc(self, X, Y, theta):
        cost_hist = []
        m = len(Y)
        itr_hist = []
        prev_cost = 0
        for it in range(self.iter + 1):
            predictions = np.dot(X, theta)
            for i in range(3):
                theta[i] = theta[i] - self.lr * self.calc_gradient(X, Y, predictions, theta, i)
            curr_cost = self.calc_cost(X, Y, theta)
            itr_hist.append(it)
            cost_hist.append(curr_cost)
            # if it % 3 == 0:
            #     print("Iteration :", it, "  Cost:", curr_cost)
            if abs(curr_cost - prev_cost) < 0.01:
                # print("Final Cost:", curr_cost)
                break
            prev_cost = curr_cost
        return theta, itr_hist, cost_hist


class KernelRidgeRegression(object):

    # Inputs dataset, lamda, sigma for gaussian, and type of kernel you want implemented
    def __init__(self, xTrain, yTrain, Lamda=.01, Sigma=1, Type=0, deg=2):
        self.xTrain = xTrain
        self.yTrain = yTrain
        self.Lamda = Lamda
        self.Sigma = Sigma
        self.Type = Type
        self.Deg = deg
        self.N = xTrain.shape[0]

    # Returns the kernel value given x and x'
    # t determines the type of filter, 0 = polynomial, 1 = gaussian
    def kernel(self, x1, x2, t, sig=1, deg=2):
        t = self.Type
        deg = self.Deg
        if t == 0:
            return (np.dot(np.transpose(x1), x2) + 1) ** deg
        elif t == 1:
            diff = np.clip(np.linalg.norm(x1 - x2), -500, 500)
            return np.exp(-diff ** 2 / (2 * sig ** 2))

    # Trains the dataset
    def fit(self, t, deg):
        self.Type = t
        self.Deg = deg
        # Calculates K matrix
        K = np.zeros((self.N, self.N))
        for i in range(0, self.N):
            for j in range(0, self.N):
                K[i, j] = self.kernel(self.xTrain[i], self.xTrain[j], self.Type, self.Sigma, self.Deg)

        # Trains the algorithm on the training data
        I = np.identity(K.shape[0])
        self.a = np.dot(np.linalg.inv(K + self.Lamda * I), self.yTrain)

    # Predicts the output given an input array
    def predict(self, xTest):
        predictY = np.zeros((xTest.shape[0], 1))

        k = np.zeros((self.N, 1))
        for i in range(0, xTest.shape[0]):
            for j in range(0, self.N):
                k[j, 0] = self.kernel(self.xTrain[j], xTest[i], self.Type, self.Sigma, self.Deg)
            predictY[i] = np.dot(np.transpose(k), self.a)

        return predictY


def generateA():
    I4 = np.identity(4)
    gaussianD = np.random.multivariate_normal([0, 0, 0, 0], I4, 100)
    eps = np.random.normal(0, 1, 1)
    Y = (gaussianD[:, 0] / (0.5 + (gaussianD[:, 1] + 1.5) ** 2)) + ((1 + gaussianD[:, 1]) ** 2) + (0.5 * eps)
    return gaussianD, Y


def generateB():
    I10 = np.identity(10)
    gaussianD = np.random.multivariate_normal([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], I10, 100)
    eps = np.random.normal(0, 1, 1)
    Y = ((gaussianD[:, 0] ** 2) * eps) / 2
    return gaussianD, Y


def generateC():
    I4 = np.identity(4)
    gaussianD = np.random.multivariate_normal([0, 0, 0, 0], I4, 100)
    eps = np.random.normal(0, 1, 1)
    Y = (gaussianD[:, 0] ** 2) + (gaussianD[:, 1]) + (0.5 * eps)
    return gaussianD, Y


def generateD():
    I10 = np.identity(10)
    gaussianD = np.random.multivariate_normal([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], I10, 100)
    eps = np.random.normal(0, 1, 1)
    Y = np.cos(3 * (gaussianD[:, 0]) / 2) + (((gaussianD[:, 1]) ** 3) / 2) + (0.5 * eps)
    return gaussianD, Y


def runMulti(data, iter, saveFile):
    RMSE = list()
    for i in range(iter):
        x_D, y_D = data
        x_train, x_test, y_train, y_test = train_test_split(x_D, y_D, test_size=0.3)
        model = MultipleRegression()
        model.fit(x_train, y_train)
        # coef = model.coefficients
        # tr1 = model.predict(x_train[0])
        yPred = []
        for row in x_test:
            yPred.append(model.predict(row))
        # ggg = pd.DataFrame({'actual': y_test,
        #                     'predicted': np.ravel(yPred)})
        # print(ggg)
        rmse = np.sqrt(mean_squared_error(y_test, yPred))
        # print("RMSE for Multiple Regression: ", rmse)
        RMSE.append(rmse)
    print("mean RMSE for Multiple Regression: ", np.mean(RMSE))
    saveFile.write("mean RMSE for Multiple Regression: " + str(np.mean(RMSE)) + "\n")


def runRidge(lambdaa, data, iter, saveFile):
    for lamb in lambdaa:
        RMSE = list()
        for i in range(iter):
            x_D, y_D = data
            x_train, x_test, y_train, y_test = train_test_split(x_D, y_D, test_size=0.3)
            model = RidgeRegression()
            model.fit(x_train, y_train, lamb)
            # coef = model.coefficients
            # tr1 = model.predict(x_train[0])
            yPred = []
            for row in x_test:
                yPred.append(model.predict(row))
            # ggg = pd.DataFrame({'actual': y_test,
            #                     'predicted': np.ravel(yPred)})
            # print(ggg)
            rmse = np.sqrt(mean_squared_error(y_test, yPred))
            # print("RMSE for Ridge Regression: ", rmse)
            RMSE.append(rmse)
        print("for lambda:", lamb)
        saveFile.write("for lambda:" + str(lamb) + "\n")
        print("mean RMSE for Ridge Regression: ", np.mean(RMSE))
        saveFile.write("mean RMSE for Ridge Regression: " + str(np.mean(RMSE)) + "\n")


def runLasso(lr, iterLasso, lambdaa, data, iter, saveFile):
    for lamb in lambdaa:
        RMSE = list()
        for i in range(iter):
            x_D, y_D = data
            x_train, x_test, y_train, y_test = train_test_split(x_D, y_D, test_size=0.3)
            model = LassoRegression(lr, iterLasso, lamb)
            thetaa = np.random.rand(x_train.shape[1])
            ntheta, ith, coh = model.grad_desc(x_train, y_train, thetaa)
            # print(ntheta)
            y_pred = np.dot(x_test, ntheta)
            # print("for lambda:", lamb)
            # print("RMSE for Lasso Regression: ", np.sqrt(mean_squared_error(y_test, y_pred)))
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            RMSE.append(rmse)
        print("for lambda:", lamb)
        saveFile.write("for lambda:" + str(lamb) + "\n")
        print("mean RMSE for Lasso Regression: ", np.mean(RMSE))
        saveFile.write("mean RMSE for Lasso Regression: " + str(np.mean(RMSE)) + "\n")


def runKernelRidge(lambdaa, sigg=1, kernelT=None, degree=None, data=None, iter=50, saveFile=None):
    for lamb in lambdaa:
        for degg in degree:
            for tt in kernelT:
                RMSE = list()
                for i in range(iter):
                    x_D, y_D = data
                    x_train, x_test, y_train, y_test = train_test_split(x_D, y_D, test_size=0.3)

                    model = KernelRidgeRegression(x_train, y_train, Lamda=lamb, Sigma=sigg, Type=tt, deg=degg)
                    model.fit(t=0, deg=3)

                    yPred = model.predict(x_test)

                    # ggg = pd.DataFrame({'actual': y_test,
                    #                     'predicted': np.ravel(yPred)})
                    # print(ggg)
                    rmse = np.sqrt(mean_squared_error(y_test, yPred))
                    # print("RMSE for Kernel Ridge Regression: ", rmse)
                    RMSE.append(rmse)
                print("for lambda:", lamb)
                saveFile.write("for lambda:" + str(lamb) + "\n")
                if tt == 0:
                    print("for kernel poly")
                    saveFile.write("for kernel poly" + "\n")
                else:
                    print("for kernel gaussian")
                    saveFile.write("for kernel gaussian" + "\n")
                if tt == 0:
                    print("kernel degree is:", degg)
                    saveFile.write("kernel degree is:" + str(degg) + "\n")
                print("mean RMSE for Kernel Ridge Regression:", np.mean(RMSE))
                saveFile.write("mean RMSE for Kernel Ridge Regression:" + str(np.mean(RMSE)) + "\n")
                print("\n")
                saveFile.write("\n")


def main():
    models = {"A": generateA(), "B": generateB(), "C": generateC(), "D": generateD()}
    lambdaList = [0.5, 1, 10, 100, 1000]
    degList = [2, 5, 10]
    kernelType = [0, 1]
    file = open('report.txt', 'a')

    for key, value in models.items():
        if key == "A":
            print("============================ with dataset A ============================")
            file.write("============================ with dataset A ============================" + "\n")
            runMulti(value, 50, saveFile=file)
            print("\n")
            file.write("\n")
            runRidge(lambdaList, value, 50, saveFile=file)
            print("\n")
            file.write("\n")
            runLasso(0.000001, 200, lambdaList, value, 50, saveFile=file)
            print("\n")
            file.write("\n")
            runKernelRidge(lambdaList, sigg=1, kernelT=kernelType, degree=degList, data=value, saveFile=file)
        elif key == "B":
            print("\n\n============================ with dataset B ============================")
            file.write("\n\n============================ with dataset B ============================" + "\n")
            runMulti(value, 50, saveFile=file)
            print("\n")
            file.write("\n")
            runRidge(lambdaList, value, 50, saveFile=file)
            print("\n")
            file.write("\n")
            runLasso(0.000001, 200, lambdaList, value, 50, saveFile=file)
            print("\n")
            file.write("\n")
            runKernelRidge(lambdaList, sigg=1, kernelT=kernelType, degree=degList, data=value, saveFile=file)
        elif key == "C":
            print("\n\n============================ with dataset C ============================")
            file.write("\n\n============================ with dataset C ============================" + "\n")
            runMulti(value, 50, saveFile=file)
            print("\n")
            file.write("\n")
            runRidge(lambdaList, value, 50, saveFile=file)
            print("\n")
            file.write("\n")
            runLasso(0.000001, 200, lambdaList, value, 50, saveFile=file)
            print("\n")
            file.write("\n")
            runKernelRidge(lambdaList, sigg=1, kernelT=kernelType, degree=degList, data=value, saveFile=file)
        elif key == "D":
            print("\n\n============================ with dataset D ============================")
            file.write("\n\n============================ with dataset D ============================" + "\n")
            runMulti(value, 50, saveFile=file)
            print("\n")
            file.write("\n")
            runRidge(lambdaList, value, 50, saveFile=file)
            print("\n")
            file.write("\n")
            runLasso(0.000001, 200, lambdaList, value, 50, saveFile=file)
            print("\n")
            file.write("\n")
            runKernelRidge(lambdaList, sigg=1, kernelT=kernelType, degree=degList, data=value, saveFile=file)
    file.close()

    # runRidge(0.1, generateD())
    # runLasso(0.000001, 200, 0.1, generateA())
    # runKernelRidge(lamb=0.01, sigg=1, kernelT=0, degg=2, data=generateD())


if __name__ == '__main__':
    main()
