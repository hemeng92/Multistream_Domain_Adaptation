import numpy as np
import math as m
import scipy.sparse.linalg as lg
from classificationModel import Model
from scipy.stats import zscore
import matlab.engine




class Jda(object):
    def __init__(self, jdaParK, jdaParLambda, jdaParGamma, jdaParIt, jdaEng):
        # set predefined parameters
        self.jdaParK = jdaParK
        self.jdaParLambda = jdaParLambda
        self.jdaParGamma = jdaParGamma
        self.jdaParIt = jdaParIt
        self.jdaEng=jdaEng

    def JDA(self, Xs, Xt, Ys, Yt0):
        Xs = np.matrix(Xs)
        Xt = np.matrix(Xt)
        print(Xs.shape)
        X = np.hstack((Xs, Xt))
        s = np.sum(np.power(X,2), axis=0)
        sq = np.sqrt(s)
        di = np.diag((1./sq).A1)
        X = X * di
        m,n = X.shape
        ns = Xs.shape[1]
        nt = Xt.shape[1]
        C = len(set(Ys))

        # build MMD matrix
        e = np.vstack((1. / ns * np.ones((ns,1)), -1. / nt * np.ones((nt,1))))
        M = e * np.transpose(e) * C

        if len(Yt0) != 0 and len(Yt0) == nt:
            for c in list(set(Ys)):
                e = np.zeros((n,1))
                e[np.where(np.array(Ys) == c)[0]] = 1. / len(np.where(np.array(Ys) == c)[0])
                if len(np.where(np.array(Yt0) == c)[0]) == 0:
                    e[ns + np.where(np.array(Yt0) == c)[0]] = 0
                else:
                    e[ns + np.where(np.array(Yt0) == c)[0]] = -1. / len(np.where(np.array(Yt0) == c)[0])
                #e[np.where(np.isinf(e))[0]] = 0
                    + e * np.transpose(e)

        # construct centering matrix
        M = M / np.linalg.norm(M, 'fro')
        H = np.eye(n) - 1. / (n * np.ones((n,n)))

        # print(X * M * X.transpose() + self.jdaParLambda)
        # print(X * H * X.transpose())
        f1 = X * M * X.transpose() + self.jdaParLambda * np.eye(m)
        # np.savetxt('f1.csv', f1, delimiter=',')
        f2 = X * H * X.transpose()
        f1 = matlab.double(f1.tolist())
        f2 = matlab.double(f2.tolist())
        # Joint Distribution Adaptation
        A, D = self.jdaEng.eigs(f1, f2, self.jdaParK, 'SM', nargout=2)
        A = np.matrix(A)
        Z = A.transpose() * X
        print('JDA completed!\n\n')
        return Z, A

    """
    Iteratively refine the projection matrix
    """

    def run_JDA(self, Xs, Xt, Ys):
        Cls = []
        for t in range(self.jdaParIt):
            print('======================Iteration' + str(t) + '=====================\n')
            Z, A = self.JDA(Xs, Xt, Ys, Cls)
            Z = Z * np.diag((1. / np.sqrt(np.sum(np.power(Z, 2), axis=0))).A1)
            Zs = Z[:, :Xs.shape[1]]
            Zt = Z[:, Xs.shape[1]:]
            print('Zs shape')
            print(Zs.shape)
            ClfModel = Model()
            ClfModel.train(Zs.transpose(), Ys)
            Cls = ClfModel.test(Zt.transpose())
            #print(Cls)
            # acc = float(len(np.where(np.array(Cls) == np.array(Yt))[0])) / len(Yt)
            # print('SVM+JDA Accuracy: %f' % acc)


        return Zs, Zt,A,ClfModel


if __name__ == '__main__':
    jdaPark = 100  # base dimension, less than original dimension
    jdaParLambda = 0.1
    jdaParGamma = 1
    jdaParIt = 10 # iterations
    jdaEng = matlab.engine.start_matlab()
    jda = Jda(jdaPark, jdaParLambda, jdaParGamma, jdaParIt, jdaEng)

    data = np.genfromtxt('/Users/taohemeng/Downloads/English_Atrocity/EnglishAtrocityTransemb.txt', delimiter=',')
    lbl = np.genfromtxt('/Users/taohemeng/Downloads/English_Atrocity/EnglishAtrocityTranslabel.txt')
    print(len(lbl))
    print(len(data[0]))

    tData = np.genfromtxt('/Users/taohemeng/Downloads/Spanish_Protest/Spanish2emb.txt', delimiter=',')
    tLbl = np.genfromtxt('/Users/taohemeng/Downloads/Spanish_Protest/Spanish2Label.txt')


    Xs = data.transpose()
    Ys = lbl
    Xt = tData.transpose()
    Yt = tLbl

    jda.run_JDA(Xs, Xt, Ys, Yt) # input data set: Xs, Xt (each column is an instance)


