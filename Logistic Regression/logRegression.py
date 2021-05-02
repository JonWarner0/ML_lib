import sys
import numpy as np
import random as rand
import math


def Shuffle(data):
    for i in range(len(data)):
        j = rand.randint(0, len(data)-1)
        temp = data[i]
        data[i] = data[j]
        data[j] = temp
    return data


def Gamma(g, t, d):
    return g/(1+(g/d)*t)


def MAP(s, w, v, M):
    sigmoid = 1/(1+math.exp(-1*s[1]*w@s[0]))
    return -1*(1-sigmoid)*M*s[1]*s[0] + w/(v**2)


def MLE(s, w, v, M):
    sigmoid = 1/(1+math.exp(-1*s[1]*w@s[0]))
    return -1*(1-sigmoid)*M*s[1]*s[0]


def MAP_Loss(s, w, v, M):
    return math.log(1+math.exp(-s[1]*w@s[0])) * M + (w@w)/(v**2)


def SGD(Data, g, d, var, func=MAP):
    w = np.zeros(len(Data[0][0]))
    M = len(Data)
    for t in range(100):
        data = Shuffle(Data)
        for ex in data:
            #print('Loss: ', MAP_Loss(ex,w,var,M))
            w = w - Gamma(g,t,d)*func(ex,w,var,M)
    return w


def testModel(tests, w):
    correct = 0
    incorrect = 0
    for x,y in tests:
        if y*w@x > 0:
            correct += 1
        else:
            incorrect += 1
    return correct, incorrect


def readFile(input_file):
    values = []
    with open(input_file) as f:
        for line in f:
            v = line.strip().split(',')
            data = np.array([1]+[float(n) for n in v[:-1]])
            label = 2*int(v[-1])-1
            values.append((data,label))
    return values


def LatexOutput(model, train, test):
    ctest, itest = testModel(test, model)
    ctrain, itrain = testModel(train, model)
    tr = (itrain/(ctrain+itrain))
    te = (itest/(itest+ctest))
    print(func.__name__, '&', v,'&','%.5f' % tr,'&', '%.5f' % te, '\\\\ \hline')


if __name__ == '__main__':
    train = readFile(sys.argv[1])
    test = readFile(sys.argv[2])
    estimation = sys.argv[3]

    func = None
    if estimation == '--map':
        func = MAP
    elif estimation == '--mle':
        func = MLE
    else:
        print('Unknown command sequence')
        exit()

    var = [0.01,0.1,0.5,1,3,5,10,100]
    gamma = 0.0001 # MLE 0.0001
    d = 0.5 # MLE: 0.5

    train_error_models = []
    print("----- Evaluated on Test Data -----")
    for v in var:
        model = SGD(train, gamma, d, v, func=func)
        c, i = testModel(test, model)
        print('Estimation ->',func.__name__, '\tVariance',v,  
            '\tCorrect:', c , '\tIncorrect:', i, '\tError:', i/(c+i))
        train_error_models.append((model,v))

    print('\n ----- Evaluated on Training Data')
    for m,v in train_error_models:
        c, i = testModel(train, m)
        print('Estimation ->',func.__name__, '\tVariance',v,  
            '\tCorrect:', c , '\tIncorrect:', i, '\tError:', i/(c+i))

    