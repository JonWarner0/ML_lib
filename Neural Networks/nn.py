import sys
import numpy as np
import math
import random as rand

DLZ_CACHE = []
DLW_CACHE = []


def Shuffle(data):
    for i in range(len(data)):
        j = rand.randint(0, len(data)-1)
        temp = data[i]
        data[i] = data[j]
        data[j] = temp
    return data

def Gamma(g, t, d):
    return g/(1+(g/d)*t)


def OuputLayer_BP(w, z, ex, y):
    """array of weights, z's, the current example, prediction y"""
    x, y_star = ex
    dLy = (y-y_star)

    # only use first few weights, but np shape needs preserving
    dLw = [dLy*z[i] for i in range(len(z))] + [0 for _ in range(len(z), len(w))]
    dLz = [dLy*w[i] for i in range(len(z))]

    DLW_CACHE.append(dLw)
    DLZ_CACHE.append(dLz)


def DL_DZ(w,z,dimNodes,i):
    dLz = []
    for q in range(dimNodes):
        dzz = []
        for j in range(dimNodes):
            w_set_for_prev_z = [w[k] for k in range(j,len(w),dimNodes)]
            s = Sigmoid(w_set_for_prev_z, z)
            dzz.append(s*(1-s)*w[q*dimNodes+j])
        dLz_prev = DLZ_CACHE[-1]
        dlz_next = np.array(dzz) @ np.array(dLz_prev)
        dLz.append(dlz_next)
    DLZ_CACHE.append(dLz)


def DL_DW(w,z,dimNodes):
    sigmoid_cache = []
    for j in range(dimNodes):
        w_set = np.array([w[k] for k in range(j,len(w),dimNodes)])
        s = Sigmoid(w_set,z)
        sigmoid_cache.append(s)
    dLw = []
    for j in range(len(w)):
        s = sigmoid_cache[j%dimNodes]
        # flatten array - consume all weights for a given node with z idx
        dLw.append(DLZ_CACHE[-1][j%dimNodes]*s*(1-s)*z[j//dimNodes])
    DLW_CACHE.append(dLw)


def BackPropigation(weights, z_vals, ex, prediction, dimNodes, dimLayers):
    OuputLayer_BP(weights[-1],z_vals[-1],ex,prediction) # special case
    for i in range(dimLayers-1,1,-1): # navigate backwards
        w = weights[i]
        z = np.array([1]+z_vals[i])
        DL_DW(w,z,dimNodes)
        DL_DZ(w,z,dimNodes, i)
    # input layer doesnt update z
    DL_DW(weights[0][:len(ex[0])*dimNodes], z_vals[0], dimNodes)
    needed_size = len(DLW_CACHE[-2])
    actual_size = len(DLW_CACHE[-1]) # adjustment for NP array
    DLW_CACHE[-1] += [0 for _ in range(needed_size-actual_size)]

def Sigmoid(w,z):
    return 1/(1+math.exp(w@z))


def ForwardPass(ex, weights, dimNodes, dimLayers):
    x,y = ex
    Z = [[v for v in ex[0]]]
    zi = []
    # input layer
    for j in range(dimNodes): 
            w = np.array([weights[0][k] for k in range(j,len(ex[0])*dimNodes,dimNodes)])
            zi.append(Sigmoid(w,x))
    x = np.array([1]+zi)
    Z.append(zi)

    # Hidden layers
    for i in range(1,dimLayers-1):
        zi = []
        for j in range(dimNodes):
            w = np.array([weights[i][k] for k in range(j,len(weights[i]),dimNodes)])
            zi.append(Sigmoid(w,x))
        x = np.array([1]+zi)
        Z.append(zi)

    # output layer
    prediction = x @ weights[-1][:dimNodes+1] 
    return prediction, Z
    

def SGD(data, dimNodes, dimLayers):
    global DLW_CACHE
    global DLZ_CACHE
    w = np.array([np.zeros(dimNodes*(dimNodes+1)) for _ in range(dimLayers)])
    for i in range(dimLayers): #adjust the bias term to 1
        w[i][:dimNodes+1] = 1
    for t in range(10):
        data = Shuffle(data)
        for ex in data:
            prediction, Z = ForwardPass(ex,w,dimNodes,dimLayers)
            BackPropigation(w, Z, ex, prediction, dimNodes, dimLayers)
            d = np.array([np.array(v) for v in DLW_CACHE]) 
            w = w - Gamma(0.00001, t, 0.5)*d
            DLW_CACHE.clear()
            DLZ_CACHE.clear()
    return w


def testModel(model, tests, dimNodes, dimLayers):
    correct = 0
    incorrect = 0
    for t in tests:
        y, z = ForwardPass(t,model,dimNodes,dimLayers)
        if y*t[1] >= 0:
            correct += 1
        else:
            incorrect += 1
    return  correct, incorrect


def readFile(input_file):
    values = []
    with open(input_file) as f:
        for line in f:
            v = line.strip().split(',')
            data = np.array([float(n) for n in v[:-1]])
            label = 2*int(v[-1])-1
            values.append((data, label))
    return values



def hw_Validation_Data():
    data = [(np.array([1,1,1]),1)]
    w = np.array([[-1,1,-2,2,-3,3,1,1,1],
                [-1,1,-2,2,-3,3,1,1,1],
                [-1,2,-1.5,0,0,0,0,0,0]])
    return data, w


if __name__ == '__main__':
    train = readFile(sys.argv[1])
    test = readFile(sys.argv[2])
    dimNodes = int(sys.argv[3])
    dimLayers = int(sys.argv[4])

    model = SGD(train, dimNodes, dimLayers)

    print('training')
    c,i = testModel(model, train, dimNodes, dimLayers)
    print('NN: ', 'Width', dimNodes, '\tCorrect: ', c, '\tIncorrect: ', i, '\tError: ',i/(c+i))

    print('\ntesting')
    c,i = testModel(model, test, dimNodes, dimLayers)
    print('NN: ', 'Width', dimNodes, '\tCorrect: ', c, '\tIncorrect: ', i, '\tError: ',i/(c+i))
