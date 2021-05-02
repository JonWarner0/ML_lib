import sys
import numpy as np
import math

DLZ_CACHE = []
DLW_CACHE = []


def OuputLayer_BP(w, z, ex, y):
    """array of weights, z's, the current example, prediction y"""
    x, y_star = ex
    dLy = (y-y_star)
    dLw = [dLy*wij for wij in w]
    dLz = [dLy*zij for zij in z]

    DLW_CACHE.append(dLw)
    DLZ_CACHE.append(dLz)


def DL_DZ(w,z,dimNodes):
    dzz = []
    dLz = []
    for q in range(dimNodes):
        for j in range(dimNodes):
            w_set_for_prev_z = [w[k] for k in range(len(w)) if (q+k+j) % dimNodes == 0]
            s = Sigmoid(w_set_for_prev_z, z)
            dzz.append(s*(1-s)*w[q*dimNodes+j])
        dLz_prev = DLZ_CACHE[-1]
        dlz_next = np.array(dzz) @ np.array(dLz_prev)
        dLz.append(dlz_next)
    DLZ_CACHE.append(dLz)


def DL_DW(w,z,dimNodes,i):
    dLw = []
    for j in range(len(w)):
        w_set = [w[k] for k in range(len(w)) if (k+j) % dimNodes == 0]
        s = Sigmoid(w_set,z)
        # flatten array - consume all weights for a given node with z idx
        dLw.append(DLZ_CACHE[-1][j%dimNodes]*s*(1-s)*z[j//dimNodes])
    DLW_CACHE.append(dLw)


def BackPropigation(weights, z_vals, ex, prediction, dimLayers, dimNodes):
    OuputLayer_BP(weights[-1],z_vals[-1],ex,prediction) # special case
    for i in range(dimLayers-1,1,-1): # navigate backwards
        w = weights[i]
        z = z_vals[i]
        DL_DW(w,z,dimNodes,i)
        DL_DZ(w,z,dimNodes)
    DL_DW(weights[0], z_vals[0], dimNodes, 0) # input layer special case
    

def Sigmoid(w,z):
    return 1/(1+math.exp(w@z))

#FIXME: linearization is not correct. This one may be right so update BP ones

def ForwardPass(ex, weights, dimNodes, dimLayers):
    x,y = ex
    Z = []
    for i in range(dimLayers-1):
        zi = []
        for j in range(dimNodes):
            w = np.array([weights[k] for k in range(j,len(weights),dimNodes-1)])
            zi.append(Sigmoid(w,x))
        x = np.array(zi)
        Z.append(zi)
    prediction = x @ weights[-1]
    return prediction, Z
    

def SGD(data, dimNodes, dimLayers):
    # +dimNodes accounts for the bias term
    w = np.array([np.zeros(dimNodes*2+dimNodes) for _ in range(dimLayers)])
    # +1 accounts for the bias node in each hidden layerr
    #z = np.array([np.zeros(dimNodes+1) for _ in range(dimLayers)])
    for ex in data:
        prediction, Z = ForwardPass(ex,w,dimNodes,dimLayers)
        BackPropigation(w, Z, train[0], prediction, dimNodes, dimLayers)
        w = w - 0.0001*np.array(DLW_CACHE)
    return w


def testModel(model, tests, dimNodes, dimLayers):
    correct = 0
    incorrect = 0
    for t in tests:
        y, z = ForwardPass(t,model,dimNodes,dimLayers)
        if y*t[1] > 0:
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


if __name__ == '__main__':
    train = readFile(sys.argv[1])
    test = readFile(sys.argv[2])
    dimNodes = int(sys.argv[3])
    dimLayers = int(sys.argv[4])

    model = SGD(train, dimNodes, dimLayers)
    c,i = testModel(model, test, dimNodes, dimLayers)
    print('NN: ', '\tCorrect: ', c, '\tIncorrect: ', i, '\tError: ',i/(c+i))
    #print(DLW_CACHE)

