import sys
import numpy as np
import random as rand
from scipy import optimize	

#--- Constants ---
C_set = [(100/873), (500/873), (700/873)]
M = 0
N = 0
C = 0
matrix = None
y_vector = None

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
	


def DualSVM(s):
	alphas = Minimizer()
	# print('Success:', alphas['success'],'\nMessage:', alphas['message'])
	a = alphas['x']
	w = sum([a[i]*s[i][1]*s[i][0] for i in range(N)])
	b = sum([s[i][1] - w.T @ s[i][0] for i in range(N)])/N
	return w, b


# matrix_ij =  yi * yj * xi *xj 
def Preprocessing(s):
	m = []
	for i in range(N):
		temp = []
		for j in range(M):
			yij = s[i][1]*s[j][1]
			xij = s[i][0].T @ s[j][0]
			temp.append(yij*xij)
		m.append(temp)
	return np.array(m)


def TestModel(w,b,tests):
	correct = 0
	incorrect = 0
	for t in tests:
		if (w.T @ t[0] + b) * t[1] > 0:
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
	N = len(train)
	M = len(train[0][0])
	matrix = Preprocessing(train)
	y_vector = np.array([s[1] for s in train])
	w,b = DualSVM(train)
	c,i = TestModel(w, b, tests)
	print("Dual SVM With C =", C, "=> Correct:", c, " Incorrect:", i, " Error:", i/(c+i))
	print('Weight Vector: ', w)
	print('Bias: ', b)