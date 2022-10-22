#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

def mkdata():
    iris = load_iris()
    X = iris.data[:,:2]
    y = iris.target
    # なんで0.5かがよくわからない
    x_min,x_max = X[:,0].min() - 0.5,X[:,0].max() + 0.5
    # yは花の名前なのに0.5引くのがよくわからない
    y_min,y_max = X[:,1].min() - 0.5,X[:,1].max() + 0.5

def mksc():
    # 散布図の作成
    plt.figure(2,figsize = (8,6))
    plt.clf()
    # よくわからない
    plt.scatter(X[:,0],X[:,1],c=y,cmap=plt.cm.Set1,edgecolor="k")
    plt.xlabel("Sepal length")
    plt.ylabel("Sepal width")

    plt.xlim(x_min,x_max)
    plt.ylim(y_min,y_max)
    plt.xticks(())
    plt.yticks(())

def mkfig():
    # 次元の相互作用を理解する
    # 初めの三つの次元をプロットにする
    fig = plt.figure(1,figsize=(8,6))
    # わからない
    ax = fig.add_subplot(111,projection = "3d",elev=-150,azim=110)
    # Xの次元削減
    X_reduced = PCA(n_components = 3).fit_transform(iris.data)
    ax.scatter(
        X_reduced[:,0],
        X_reduced[:,1],
        X_reduced[:,2],
        c=y,
        cmap = plt.cm.Set1,
        edgecolor ="k",
        s = 40,
)

    ax.set_title("First three PCA directions")
    ax.set_xlabel("1st eigenvector")
    ax.w_xaxis.set_ticklabels([])
    ax.set_ylabel("2nd eigenvector")
    ax.w_yaxis.set_ticklabels([])
    ax.set_zlabel("3rd eigenvector")
    ax.w_zaxis.set_ticklabels([])

    plt.show()
    dirname = './notebook/data/output/'
    filename = dirname + 'img.png'
    plt.savefig(filename)
    print('end')

if __name__ == "__main__":
    mkdata()
    mksc()
    mkfig()