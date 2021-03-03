import numpy as np
import math as m
import sys

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
        self.weight = 0


# ---- work around for lambdas not retaining median values
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
            weighted_sum = sum(k.weight for k in temp)
            distribution.append(weighted_sum) 
            cardinality_sv += weighted_sum
    else:    
        for l in L: # accumulate entries in attribute Av that have label l
            temp = [s for s in S if s.attr[i] == Av and s.label == l]
            weighted_sum = sum(k.weight for k in temp)
            distribution.append(weighted_sum)
            cardinality_sv += weighted_sum
    return distribution, cardinality_sv


# ---- Boosting Functions -----
def ComputeError(S, h):
    error = 0
    for s in S:
        prediction = Decision(h, s)
        if s.label != prediction: 
            error += s.weight
    return error
    

def ComputeAlpha(error):
    if error == 0:
        return 0 
    return 0.5*m.log((1-error)/error, 2)


def ComputeWeights(S, Dt, alpha, h):
    """Update entry weigths"""
    Dtt = []
    Z = sum(Dt)
    for s in S:
        s.weight = (s.weight/Z) * m.exp(-alpha * s.label * Decision(h,s))
        Dtt.append(s.weight)
    return Dtt


#------------HEURISTIC-------------
def InformationGain(S, A, L):
    sizeS = len(S)
    entropy_S = []
    for l in L: 
        p_label = len([ex for ex in S if ex.label == l])/sizeS 
        entropy_S.append(Entropy(p_label))
    entropy_S = -1*sum(entropy_S)
    gains_of_A = []
    for i in A.keys():
        H_Av = []
        weights = []
        for Av in A[i]:
            distribution, cardinality_sv = Probabilies(S,Av,L,i)
            d = [Entropy(v/cardinality_sv) for v in distribution if cardinality_sv != 0] 
            H_Av.append(-1*sum(d)) 
            weights.append(cardinality_sv/sizeS)
        gain_Av = entropy_S-(np.array(weights) @ np.array(H_Av))
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


#----------ADA Boosting-----------
def AdaBoost(S,A,L,T):
    d = 1/len(S)
    D = [d for _ in S]
    for s in S: #initialization
        s.weight = D[0]

    H_final = []
    for _ in range(T):
        h = ID3(S, A, L, 2, InformationGain) 
        e = ComputeError(S,h)
        a = ComputeAlpha(e)
        D = ComputeWeights(S, D, a, h)
        H_final.append((a,h))
    return H_final


def EvalBoost(H, tests):
    correct = []
    incorrect = []
    for t in tests:
        hypothesis = 0
        for a, h in H:
            hypothesis += a * Decision(h,t)
        l = 1 if hypothesis >= 0 else -1
        if l == t.label:
            correct.append(t)
        else:
            incorrect.append(t)
    return correct, incorrect


#---------ITERATING OVER TREE----------
def Decision(root, entry):
    """Evaluate Hypothesis"""   
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



#-------------ENTRY POINT--------------
if __name__ == "__main__":
    training = sys.argv[1]
    testing = sys.argv[2]
    iterations = int(sys.argv[3])

    S = []
    Attr = dict()
    Labels = set()
    tests = []

    if len(sys.argv) == 3:
        S, Attr, Labels, tests = Use_data_As_Is(training, testing)
    elif sys.argv[4] == '-num': 
        S, Attr, Labels, tests = Use_Numeric_Median(training, testing)
    elif sys.argv[4] == '-unkn':
        S, Attr, Labels, tests = Use_Numeric_Median(training, testing, True)
    else:
        print("Unkown command sequence")
        exit()

    # Run boosting with stumps of depth 2
    H_final = AdaBoost(S, Attr, Labels, iterations)
    c, i = EvalBoost(H_final, tests)
    print("AdaBoosting: " , " Correct: ", len(c), " Incorrect:", len(i), " Error:", len(i)/len(tests))

    # Run tests and print results
    # print("Runing testing on file:", testing)

    # c,i = TestDecisionTree(Info_tree, tests)
    # print("InformationGain", " Correct:", len(c), " Incorrect:", len(i), " Error:", len(i)/len(tests))

    #Easy output for LaTeX tables
    # c,i = TestDecisionTree(Info_tree, tests)
    # print(len(i1)/len(tests), len(i2)/len(tests), len(i3)/len(tests))