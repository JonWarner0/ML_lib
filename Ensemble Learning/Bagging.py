import numpy as np
import math as m
import sys
import random 
import threading

#-----HELPER CLASSES AND FUNCTIONS------
class node:
    def __init__(self, _value, _nextNode=None,  _terminal=False):
        self.value = _value
        self.children = []
        self.terminal = _terminal
        self.nextNode = _nextNode
    
    def addChild(self, _child):
        self.children.append(_child)


class entry:
    def __init__(self, atttributes, label):
        """Attributes is an array of the attribute values"""
        self.attr = atttributes
        self.label = label

# work around for lambdas not retaining median values
MEDIAN_MAP = dict()


def Entropy(p):
    """Multilabel Entropy with log base 2"""
    if p == 0:
        return 0
    return p*m.log(p,2)


def Probabilies(S,Av,L,i):
    distribution = []
    cardinality_sv = 0
    if type(Av) != str:
        for l in L: # Av should be a lambda expression containing the median
            temp = [s for s in S if Av(s.attr[i], MEDIAN_MAP[i]) and s.label == l]
            distribution.append(len(temp)) 
            cardinality_sv += len(temp)
    else:    
        for l in L: # accumulate entries in attribute Av that have label l
            temp = [s for s in S if s.attr[i] == Av and s.label == l]
            distribution.append(len(temp))
            cardinality_sv += len(temp)
    return distribution, cardinality_sv


#------------HEURISTICS-------------
def InformationGain(S, A, L):
    #--Entropy(S)--
    sizeS = len(S)
    entropy_S = []
    for l in L: 
        p_label = len([ex for ex in S if ex.label == l])/sizeS # p_distribution of l on S
        entropy_S.append(Entropy(p_label))
    entropy_S = -1*sum(entropy_S) # negative summation for multilabel
    #-- Find hightest gaining attribute --
    gains_of_A = []
    for i in A.keys(): # each attribute index
        H_Av = []
        weights = []
        for Av in A[i]:
            distribution, cardinality_sv = Probabilies(S,Av,L,i)
            d = [Entropy(v/cardinality_sv) for v in distribution if cardinality_sv != 0] 
            H_Av.append(-1*sum(d)) #entropy of attribute value
            weights.append(cardinality_sv/sizeS) # sv/s
        gain_Av = entropy_S-(np.array(weights) @ np.array(H_Av)) #entropy(S)-expected entropy
        gains_of_A.append((i, gain_Av))
    return max(gains_of_A, key=lambda k: k[1])[0]


#-------------TREE CONSTRUCTION--------------
def ID3(S, A, L, depth, _gain=InformationGain):
    """Create a decision tree based off of the input data 
        S=examples, A=attributes, L=labels
        depth = tree depth, _gain = splitting heuristic
    """
    purity = {ex.label for ex in S}
    if len(purity) == 0 or depth < 0 or len(A.keys())==0:
        freq = []
        for l in L:
            freq.append((l, len([ex for ex in S if ex.label == l])))
        most_common = max(freq, key=lambda k: k[1])[0]
        return node(_value=most_common, _terminal=True) # return most common label
    if len(purity) == 1:    
        return node(_value=purity.pop(),  _terminal=True) # return the pure label

    A_split = _gain(S,A,L) # Determine the attribute to split on
    root = node(_value=A_split) 

    for v in A[A_split]:
        subset = []
        if type(v) != str: # evaluate lambda
            subset = [ex for ex in S if v(ex.attr[A_split], MEDIAN_MAP[A_split])]
        else:
            subset = [ex for ex in S if v in ex.attr[A_split]]

        if len(subset) == 0:
            freq = []
            for l in L:
                freq.append((l, len([ex for ex in S if ex.label == l])))
            most_common = max(freq, key=lambda k: k[1])[0]
            # branch is terminal
            root.addChild(node(_value=v,_nextNode=node(_value=most_common,_terminal=True)))
        else: 
            next_A = { a:A[a] for a in A.keys() if a != A_split }
            next_node = ID3(subset, next_A, L, depth-1, _gain)
            root.addChild(node(_value=v, _nextNode=next_node)) 
    return root


#--------Bagging---------
def Bagging(S,A,L,T):
    m = len(S)
    max_depth = len(A.keys())
    tree_bag = []
    for _ in range(T):
        samples = [] # m'
        for _ in range(m):
            samples.append(S[random.randint(0,m-1)]) # prevent upper inclusivity
        root = ID3(samples,A,L,max_depth,InformationGain)
        tree_bag.append(root)
    return tree_bag


def EvalBagging(tree_bag, test_set, T):
    correct = 0
    incorrect = 0
    for test in test_set:
        votes = 0
        for tree in tree_bag:
            votes += Decision(tree, test)
        hypothesis = 1 if votes >=0 else -1
        if test.label  == hypothesis:
            correct += 1
        else:
            incorrect += 1
    return correct, incorrect

#---------Bias and Variance---------
def Bias_Var_Decomp(S,A,L,T,t_start, t_end):
    m = 1000
    max_depth = len(A.keys())
    tree_bag = []
    for i in range(t_start, t_end):
        no_replace_S = S.copy()
        samples = [] # m'
        for _ in range(m): # no replacement
                idx = random.randint(0, len(no_replace_S)-1)
                ex = no_replace_S[idx]
                no_replace_S.remove(ex)
                samples.append(ex)            
        sub_bag = []
        for _ in range(T):
            root = ID3(samples,A,L,max_depth,InformationGain)
            sub_bag.append(root)
        THREAD_RESULTS[i] = sub_bag


def Single_Bias_Var(tree_bag, test_set):
    first_trees = [t[0] for t in tree_bag]
    n = len(first_trees)
    general_bias = 0
    general_var = 0
    general_squared_error = 0
    all_bias_var = []
    for test in test_set:
        predictions = []
        for tree in first_trees:
            predictions.append(Decision(tree,test))
        avg = sum(predictions)/n
        bias = (avg - test.label)**2
        variance = 1/(n-1)*sum((x-avg)**2 for x in predictions)
        general_bias += bias
        general_var += variance
        all_bias_var.append((bias,variance))
    return general_bias/n, general_var/n


def Calc_Bias_Var(tree_bag, test_set):
    n = len(tree_bag)*len(tree_bag[0])
    general_bias = 0
    general_var = 0
    general_squared_error = 0
    all_bias_var = []
    for test in test_set:
        predictions = []
        for sub_bag in tree_bag:
            for tree in sub_bag:
                predictions.append(Decision(tree,test))
        avg = sum(predictions)/n
        bias = (avg - test.label)**2
        variance = 1/(n-1)*sum((x-avg)**2 for x in predictions)
        general_bias += bias
        general_var += variance
    return general_bias/n, general_var/n


#---------ITERATING OVER TREE----------
def Decision(root, entry):
    if root.terminal:
        return root.value
    for v in root.children:
        if type(v.value) != str and  v.value(entry.attr[root.value], MEDIAN_MAP[root.value]):
           return Decision(v.nextNode, entry)
        elif v.value == entry.attr[root.value]:
            return Decision(v.nextNode, entry)
    return None # only occurs if issue with tree construction



#-------------LOAD DATA--------------
def Use_data_As_Is(trainFile, testFile):
    """Q1 Specific"""
    S = []
    Attr = dict()
    Labels = set()
    with open(trainFile, 'r') as f:
        for line in f:
            term = line.strip().split(',')
            for i in range(len(term)-1): # build dictionary of indexable attributes
                if i in Attr.keys():
                    Attr[i].add(term[i])
                else:
                    Attr[i] = {term[i]}
            Labels.add(term[-1]) # build set of labels
            S.append(entry(term[:-1], term[-1])) # build list of examples using entry objects

    tests = []
    with open(testFile, 'r') as f:
            for line in f:
                term = line.strip().split(',')
                tests.append(entry(term[:-1], term[-1]))

    return S, Attr, Labels, tests


def Use_Numeric_Median(trainFile, testFile, replace=False):
    """Q2 Specific"""
    S = []
    Attr = dict()
    Labels = set()
    numerics = set()
    unknowns = set()
    
    with open(trainFile, 'r') as f:
        for line in f:
            term = line.strip().split(',')
            for i in range(len(term)-1): 
                try:
                    # handles numeric values positive and negative
                    term[i] = int(term[i])
                    numerics.add(i)
                    if i in Attr.keys():
                        Attr[i].append(term[i])
                    else:
                        Attr[i] = [term[i]] # needs duplicatates for median
                except:
                    if replace and term[i] == 'unknown':
                        unknowns.add(i)
                    if i in Attr.keys():
                        Attr[i].add(term[i])
                    else:
                        Attr[i] = {term[i]}
            l = 1 if term[-1] == 'yes' else -1
            Labels.add(l) 
            S.append(entry(term[:-1], l)) 

    for i in numerics: # index into Attr to replace list of numbers with media
        temp = sorted(Attr[i])
        if len(temp) % 2 == 0: # no direct median. Average the two middle values
            MEDIAN_MAP[i] = (temp[int(len(S)/2)] + temp[int((len(S)-1)/2)])/2
            Attr[i] = [lambda v,m: v <= m, lambda v,m: v > m] 
        else:
            MEDIAN_MAP[i] = temp[len(temp)/2]
            Attr[i] = [lambda v,m: v <= m, lambda v,m: v > m] 

    if replace:
        unkn_record = dict()
        for i in unknowns:
            counts = {v:0 for v in Attr[i] if v != 'unknown'}
            for s in S:
                if s.attr[i] != 'unknown':
                    counts[s.attr[i]] += 1
            common = max(counts.keys(), key=lambda k: counts[k])
            unkn_record[i] = common
            for s in S:
                if s.attr[i] == 'unknown':
                    s.attr[i] = common

    tests = []
    with open(testFile, 'r') as f:
            for line in f:
                term = line.strip().split(',')
                if replace:
                    for i in range(len(term)):
                        if term[i] == 'unknown':
                            term[i] = unkn_record[i]
                for i in numerics:
                    term[i] = int(term[i])
                l = 1 if term[-1] == 'yes' else -1
                tests.append(entry(term[:-1], l))

    return S, Attr, Labels, tests


def runBiasVariance(S,A,L,T,tests):
    threads = []
    for i in range(0, 100, 20):
        t = threading.Thread(target=Bias_Var_Decomp, args=(S,Attr,Labels,T,i,i+20))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()

    single_gb, single_gv = Single_Bias_Var(THREAD_RESULTS, tests)
    print("------Bias/Variance-----")
    print("Single learner")
    print("General Bias:", single_gb, " General Var:", single_gv, " General SE:", single_gb+single_gv)

    gb, gv = Calc_Bias_Var(THREAD_RESULTS, tests)
    print("Tree Bag")
    print("General Bias:", gb, " General Var:", gv, " General SE:", gb+gv)

    exit()


THREAD_RESULTS = [None for _ in range(100)]

#-------------ENTRY POINT--------------
if __name__ == "__main__":
    training = sys.argv[1]
    testing = sys.argv[2]
    T = int(sys.argv[3])

    S = []
    Attr = dict()
    Labels = set()
    tests = []

    if len(sys.argv) == 4:
        S, Attr, Labels, tests = Use_data_As_Is(training, testing)
    elif sys.argv[4] == '-num': 
        S, Attr, Labels, tests = Use_Numeric_Median(training, testing)
        if len(sys.argv) == 6:
            runBiasVariance(S,Attr,Labels,T,tests)
    elif sys.argv[4] == '-unkn':
        S, Attr, Labels, tests = Use_Numeric_Median(training, testing, True)
    else:
        print("Unkown command sequence")
        exit()

    tree_bag = Bagging(S,Attr, Labels, T)
    c, i = EvalBagging(tree_bag,tests,T)
    print("\nIteration: ", T)
    print("Bagging: " , " Correct: ", c, " Incorrect:", i, " Error:", i/len(tests))
    print('--------------------------------------')
