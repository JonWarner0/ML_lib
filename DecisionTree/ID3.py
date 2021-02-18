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

# work around for lambdas not retaining median values
MEDIAN_MAP = dict()

#------------HEURISTICS-------------
def InformationGain(S, A, L):
    """ Returns the attribute with the largest information gain """
    #--get Entropy(S)--
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
            d = [v/cardinality_sv for v in distribution if cardinality_sv != 0] 
            H_Av.append(-1*sum([Entropy(p) for p in d])) #entropy of attribute value
            weights.append(cardinality_sv/sizeS) # sv/s
        gain_Av = entropy_S-(np.array(weights) @ np.array(H_Av)) #entropy(S)-expected entropy
        gains_of_A.append((i, gain_Av))
    return max(gains_of_A, key=lambda k: k[1])[0]


def MajorityErrorGain(S, A, L):
    sizeS = len(S)
    #-- Majority Error of S --
    d = []
    for l in L:
        d.append(len([ex for ex in S if ex.label == l]))
    d.remove(max(d))
    ME_S = sum(d)/sizeS 
    ME_totals = []
    for i in A.keys():
        ME_Av = []
        weights = []
        #-- Majority Error for each Sv --
        for Av in A[i]: #
            distribution, cardinality_sv = Probabilies(S,Av,L,i)
            max_p = max(distribution) 
            distribution.remove(max_p) # remove max value so minorities are left
            if cardinality_sv == 0 or len(distribution) == 0: #handles if empty cardinalty and card=1 when max was removed
                ME_Av.append(0) 
                weights.append(cardinality_sv/sizeS) # sv/s
            else:
                ME_Av.append(sum(distribution)/cardinality_sv) # Majority error for value in attribute
                weights.append(cardinality_sv/sizeS) # sv/s
        gain_Av = ME_S-(np.array(weights) @ np.array(ME_Av))
        ME_totals.append((i, gain_Av))
    return max(ME_totals, key=lambda k: k[1])[0]


def GiniIndexGain(S, A, L):
    sizeS = len(S)
    #-- Majority Error of S --
    p = 0
    for l in L:
        p += (len([ex for ex in S if ex.label == l])/sizeS)**2
    GI_S = 1-p
    GI_totals = []
    for i in A.keys():
        GI_Av = []
        weights = []
        #-- Gini Index for each Sv --
        for Av in A[i]: 
            distribution, cardinality_sv = Probabilies(S,Av,L,i)
            d = [(v/cardinality_sv)**2 for v in distribution if cardinality_sv !=0] 
            GI_Av.append(1-sum(d)) 
            weights.append(cardinality_sv/sizeS) # sv/s
        gain_Av = GI_S-(np.array(weights) @ np.array(GI_Av))
        GI_totals.append((i, gain_Av))
    return max(GI_totals, key=lambda k: k[1])[0]


#-------------TREE CONSTRUCTION--------------
def ID3(S, A, L, depth, _gain=InformationGain):
    """Create a decision tree based off of the input data 
        S=examples, A=attributes, L=labels
        depth = tree depth, _gain = splitting heuristic
    """
    purity = {ex.label for ex in S}
    if len(purity) == 0 or depth == 0 or len(A.keys())==0:
        freq = []
        for l in L:
            freq.append((l, len([ex for ex in S if ex.label == l])))
        most_common = max(freq, key=lambda k: k[1])[0]
        return node(_value=most_common, _terminal=True) # return most common label
    if len(purity) == 1:    
        return node(_value=purity.pop(),  _terminal=True) # return the pure lable

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


#---------ITERATING OVER TREE----------
def MakeDecision(root, entry):
    if root.terminal:
        return root.value
    for v in root.children:
        if v.value == entry.attr[root.value]:
            return MakeDecision(v.nextNode, entry)
    return None # only occurs if issue with tree construction

#--------TESTING EXAMPLES---------
def TestDecisionTree(tree, test_set):
    correct = []
    incorrect = []
    for entry in test_set:
        decision = MakeDecision(tree, entry)
        if decision == entry.label:
            correct.append((entry.label, decision))
        else:
            incorrect.append((entry.label, decision))
    return correct, incorrect


#-----OUTPUT FILES IF FOR DEBUG-----
def WriteResults(filename, correct, incorrect):
    with open(filename, "w+") as f:
        f.write("Number Correct: {}\n".format(len(correct)))
        f.write("Number Incorrect: {}\n\n".format(len(incorrect)))
        f.write("------Correct Instances-------\n")
        for c in correct:
            f.write("{}\n".format(c))
        f.write("\n------Incorrect Instances------\n")
        for i in incorrect:
            f.write("{}\n".format(i))


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


def Use_Numeric_Median(trainFile, testFile):
    """Q2 Specific"""
    S = []
    Attr = dict()
    Labels = set()
    numerics = set()
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
                    if i in Attr.keys():
                        Attr[i].add(term[i])
                    else:
                        Attr[i] = {term[i]}
            Labels.add(term[-1]) 
            S.append(entry(term[:-1], term[-1])) 

    for i in numerics: # index into Attr to replace list of numbers with media
        temp = sorted(Attr[i])
        if len(temp) % 2 == 0: # no direct median. Average the two middle values
            MEDIAN_MAP[i] = (temp[int(len(S)/2)] + temp[int((len(S)-1)/2)])/2
            Attr[i] = [lambda v,m: v <= m, lambda v,m: v > m] #FIXME!!!!!
        else:
            MEDIAN_MAP[i] = temp[len(temp)/2]
            Attr[i] = [lambda v,m: v <= m, lambda v,m: v > m] 

    tests = []
    with open(testFile, 'r') as f:
            for line in f:
                term = line.strip().split(',')
                for i in numerics:
                    term[i] = int(term[i])
                tests.append(entry(term[:-1], term[-1]))

    return S, Attr, Labels, tests



#-------------ENTRY POINT--------------
if __name__ == "__main__":
    training = sys.argv[1]
    testing = sys.argv[2]
    depth = int(sys.argv[3])

    S = []
    Attr = dict()
    Labels = set()
    tests = []

    if len(sys.argv) == 4:
        S, Attr, Labels, tests = Use_data_As_Is(training, testing)
    elif sys.argv[4] == '-num': 
        S, Attr, Labels, tests = Use_Numeric_Median(training, testing)
    elif sys.argv[4] == '-unkn':
        print("unknown param")
        exit()
    else:
        print("Unkown command sequence")
        exit()

    # build tree's with the different heuristics
    Info_tree = ID3(S, Attr, Labels, depth, InformationGain)
    ME_tree = ID3(S, Attr, Labels, depth, MajorityErrorGain)
    GI_tree = ID3(S, Attr, Labels, depth, GiniIndexGain)

    #Run tests and write results
    print("Runing testing on file:", testing)
    c,i = TestDecisionTree(Info_tree, tests)
    print("InformationGain", " correct:", len(c), " Incorrect:", len(i), " error:", len(i)/len(tests))
    #WriteResults("infoGain.txt", c, i)

    c,i = TestDecisionTree(ME_tree, tests)
    print("Majority Error", "  correct:", len(c), " Incorrect:", len(i), " error:", len(i)/len(tests))
    #WriteResults("meGain.txt", c, i)

    c,i = TestDecisionTree(GI_tree, tests)
    print("Gini Index", "\t correct:", len(c), " Incorrect:", len(i), " error:", len(i)/len(tests))
    #WriteResults("giGain.txt", c, i)