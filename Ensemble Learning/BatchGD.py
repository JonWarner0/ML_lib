import numpy as np
from numpy import linalg as LA
import sys
import math

class entry:
    def __init__(self, _x, _y):
        self.x = _x
        self.y = _y


def Calc_Gradient(X,w,j):
    total = 0
    for ex in X:
        total += (ex.y - w.T @ ex.x) * ex.x[j]
    return -total


def GradientDescent(X,w,r,threshold):
    while True:
        J = []
        for j in range(len(w)):
            J.append(Calc_Gradient(X,w,j))
        new_w = w - r*np.array(J)
        error = LA.norm(w-new_w, ord=2)
        #print(error)
        if error <= threshold:
            return w
        w = new_w


def Eval(f, tests):
    sq_error = 0
    for t in tests:
        prediction = f @ t.x
        sq_error += (prediction-t.y)**2
    return sq_error/2


def ReadFile(file):
    vals = []
    with open(file,'r') as f:
        for line in f:
            ex = line.strip().split(',')
            x = np.array([float(v) for v in ex[:-1]])
            np.insert(x,0,1)
            y = float(ex[-1])
            vals.append(entry(x,y))
    return vals


if __name__ == '__main__':  
    training_f = sys.argv[1]
    testing_f = sys.argv[2]

    train = ReadFile(training_f)
    tests = ReadFile(testing_f)

    w = np.zeros(len(train[0].x))
    r = 0.0125
    et = 0.00001

    function = GradientDescent(train, w, r, et)
    error = Eval(function, tests)
    print(function)
    print("Cost:", error)

