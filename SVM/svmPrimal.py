import sys
import numpy as np
import random as rand

#--- Constants ---
G_INIT = 0.1
T = 100
C_set = [(100/873), (500/873), (700/873)]


def Shuffle(data):
	for i in range(len(data)):
		j = rand.randint(0,len(data)-1)
		temp = data[i]
		data[i] = data[j]
		data[j] = temp
	return data


def Gamma(t):
	return G_INIT/(1+(G_INIT/C)*t)
	#return G_INIT/(1+t)


def StochasticSVM(C,N,S):
	w_not = np.zeros(len(S[0][0])-1)
	w = np.zeros(len(S[0][0]))
	for t in range(T):
		data = Shuffle(S)
		gamma = Gamma(t)
		for ex in data:
			if ex[1] * w.T @ ex[0] <= 1:
				w_temp = np.append(w_not,0) 
				w = w-gamma*w_temp+gamma*C*N*ex[1]*ex[0]
			else:
				w_not = (1-gamma)*w_not
	return w


def TestModel(w, tests):
	correct = 0
	incorrect = 0
	for t in tests:
		if w.T @ t[0] * t[1] > 0:
			correct += 1
		else:
			incorrect += 1
	return correct, incorrect


def ReadFile(file):
	values = []
	with open(file) as f:
		for line in f:
			v = line.strip().split(',')
			nums = np.array([1]+[float(n) for n in v[:-1]])
			label = 2*int(v[-1])-1
			values.append((nums, label))
	return values


#FIXME: use augmented vectors or not?
if __name__ == '__main__':
	train = ReadFile(sys.argv[1])
	tests = ReadFile(sys.argv[2])
	C = C_set[int(sys.argv[3])]
	w = StochasticSVM(C,len(train),train)
	c,i = TestModel(w, tests)
	print("Stochastic SVM Primal With C =", C, "=> Correct:", c, " Incorrect:", i, " Error:", i/(c+i))
	print('Weight Vector: ', w)