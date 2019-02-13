'''
Created november 2018
@author:yair,eliezer
@id: 300488939,303062129
'''
import matplotlib as plt
import numpy as np
import numpy.linalg as npl

def get_data():

    xi =  [0.0, 2.5, 5.0, 7.5, 10.0]
    yi =  [200.0, 1575.0, 4200.0, 8075.0, 13200.0]
    return xi, yi

def PrintData(name,data):

    print(name,':')
    print(data)

def PrintLine():

    while (i<80):
        print('-')
        i=i+1

def Const_AL(xi,yi):

    l = np.transpose( get_data(yi))
    x = np.transpose( get_data(xi))
    xx = x**2
    A = np.zeros  ((5, 3))
    A = A [:0] + 1
    A = A [:1] + x
    A = A [:2] + xx

    return A , l

def MyLstsq(A,L):

    x1 = np.dot(A,A.T)
    x2 = np.linalg.inv(x1)
    x3 = np.dot(x2,A.T)
    X = np.dot(x3,L)

    v1 = np.dot(A,X)
    V = v1-L

    sigmaX1 = np.dot(V.T,V)
    # sigmaX2 = np.abs(sigmaX1/np.ndim(A))
    sigmaX = np.dot(x2,sigmaX1)

    return X, V, sigmaX

def PrintResults(xi,yi,A,L, x,v,sigmaX):

    print(xi)
    print(yi)
    print(A)
    print(L)
    print(x)
    print(v)
    print(sigmaX)

def PlotResults(xi, yi, x):
    plt.plot(x, y, 'black')
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.gca().set_aspect('equal')
    plt.show()

if __name__ == '__main__':

    print (Const_AL(get_data()))