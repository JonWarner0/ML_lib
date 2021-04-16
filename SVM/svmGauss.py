import sys
import numpy as np
from numpy import linalg as LA
import random as rand
from scipy import optimize	

#--- Constants ---
C_set = [(100/873), (500/873), (700/873)]
Gamma_set=[0.1,0.5,1,5,100]
M = 0
N = 0
C = 0
Gamma = 0
matrix = None
y_vector = None


def Kernel(x1, x2):
	num = LA.norm(x1-x2)**2
	return np.exp(-(num/Gamma))


def objective(a):
	total = 0
	for i in range(N):
		for j in range(M):
			total += matrix[i,j]*a[i]*a[j]
	return 0.5 * total - sum(a)


def C_constraint(a):
	return C - sum(a)


def alpha_y_constraint(a):
	return a @ y_vector


def Minimizer(): 
	x0 = np.zeros(N)
	bounds_C = [(0,C) for _ in range(N)]
	const_alpha = {'type':'eq', 'fun' : alpha_y_constraint}
	return optimize.minimize(fun=objective, x0=x0, method='SLSQP', constraints=const_alpha, bounds=bounds_C, tol=1e-9)
	


def GaussSVM(s):
	alphas = Minimizer()
	# print('Success:', alphas['success'],'\nMessage:', alphas['message'])
	a = alphas['x']
	#print(a)
	w = sum([a[i]*s[i][1]*s[i][0] for i in range(N)])
	b = sum([s[i][1] - w.T @ s[i][0] for i in range(N)])/N
	return w, b, a


# matrix_ij =  yi * yj * xi *xj 
def Preprocessing(s):
	m = []
	for i in range(N):
		temp = []
		for j in range(M):
			yij = s[i][1]*s[j][1]
			xij = Kernel(s[i][0], s[j][0])
			temp.append(yij*xij)
		m.append(temp)
	return np.array(m)


def TestModel(w,b,a,s,tests):
	correct = 0
	incorrect = 0
	for t in tests:
		v = sum([a[i]*s[i][1]*Kernel(s[i][0],t[0])+b for i in range(N)])
		if np.sign(v) == np.sign(t[1]):
			correct += 1
		else:
			incorrect += 1
	return correct, incorrect


def ReadFile(file):
	values = []
	with open(file) as f:
		for line in f:
			v = line.strip().split(',')
			nums = np.array([float(n) for n in v[:-1]])
			label = 2*int(v[-1])-1
			values.append((nums, label))
	return values


if __name__ == '__main__':
	train = ReadFile(sys.argv[1])
	tests = ReadFile(sys.argv[2])
	C = C_set[int(sys.argv[3])]
	Gamma = Gamma_set[int(sys.argv[4])]
	N = len(train)
	M = len(train[0][0])
	matrix = Preprocessing(train)
	y_vector = np.array([s[1] for s in train])

	for i in range(4):
		print()
		Gamma = Gamma_set[i]
		w,b,a = GaussSVM(train)
		sv1 = {v for v in a if v != 0}

		Gamma = Gamma_set[i+1]	
		w,b,a = GaussSVM(train)
		sv2 = {v for v in a if v != 0}

		print(Gamma_set[i], Gamma_set[i+1],'=>', len(sv1.intersection(sv2)), '\n')

	exit()
	
	w,b,a = GaussSVM(train)
	c,i = TestModel(w, b, a, train,tests)
	print("Dual Gaussian SVM With C =", C, "Gamma =", Gamma, "=> Correct:", c, " Incorrect:", i, " Error:", i/(c+i))
	print('Weight Vector: ', w)
	print('Bias: ', b)