import numpy as np
import math as m
import sys

#-----HELPER CLASSES AND FUNCTIONS------
class node:
    def __init__(self, _value, _children, _nextNode=None,  _terminal=False):
        self.value = _value
        self.children = _children
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


#------HEURISTICS-------
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
    for i in A.keys(): # each attribute
        H_Av = []
        weights = []
        for Av in A[i]: # each value in that attribute
            distribution = []
            cardinality_sv = 0
            #--get each Entropy(Sv)--
            for l in L: # accumulate entries in attribute Av that have label l
                temp = [s for s in S if s.attr[i] == Av and s.label == l]
                distribution.append(len(temp)) # occurences of Av & l 
                cardinality_sv += len(temp)
            # marginal distributions
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
    #-- Majority Error for each Sv --
    ME_totals = []
    for i in A.keys(): # each attribute
        ME_Av = []
        weights = []
        for Av in A[i]: # each value in that attribute
            distribution = []
            cardinality_sv = 0
            for l in L: # accumulate entries in attribute Av that have label l
                temp = [s for s in S if s.attr[i] == Av and s.label == l]
                distribution.append(len(temp)) # occurences of Av & l 
                cardinality_sv += len(temp)
            max_p = max(distribution) #find the max value 
            d = distribution.remove(max_p) # remove max value so minorities are left
            if d == None: #handles empty set 
                ME_Av.append(0) # Majority error for value in attribute
                weights.append(cardinality_sv/sizeS) # sv/s
            else:
                ME_Av.append(sum(d)/cardinality_sv) # Majority error for value in attribute
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
    #-- Gini Index for each Sv --
    GI_totals = []
    for i in A.keys(): # each attribute
        GI_Av = []
        weights = []
        for Av in A[i]: # each value in that attribute
            distribution = []
            cardinality_sv = 0
            for l in L: # accumulate entries in attribute Av that have label l
                temp = [s for s in S if s.attr[i] == Av and s.label == l]
                distribution.append(len(temp)) # occurences of Av & l 
                cardinality_sv += len(temp)
            # squared marginal distributions
            d = [(v/cardinality_sv)**2 for v in distribution if cardinality_sv !=0 ] 
            GI_Av.append(1-sum(d)) # Majority error for value in attribute
            weights.append(cardinality_sv/sizeS) # sv/s
        gain_Av = GI_S-(np.array(weights) @ np.array(GI_Av))
        GI_totals.append((i, gain_Av))
    return max(GI_totals, key=lambda k: k[1])[0]


#------TREE CONSTRUCTION-------
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
        return node(_value=most_common, _children=[], _terminal=True) # return most common label
    if len(purity) == 1:    
        return node(_value=purity.pop(), _children=[], _terminal=True) # return the pure lable

    A_split = _gain(S,A,L) 

    root = node(_value=A_split, _children=[]) # new root node
    for v in A[A_split]:
        subset = [ex for ex in S if v in ex.attr[A_split]] # all examples in s that have value v
        if subset.count == 0:
            freq = []
            for l in L:
                freq.append((l, len([ex for ex in S if ex.label == l])))
            most_common = max(freq, key=lambda k: k[1])[0]
            root.addChild(node(_value=v,_children=[node(_value=most_common,_children=[],_terminal=True)])) # branch is terminal
        else: 
            next_A = { a:A[a] for a in A.keys() if a != A_split }
            next_node = ID3(subset, next_A, L, depth-1, _gain)
            n = node(_value=v, _nextNode=next_node, _children=[])
            root.addChild(n) # v=attribute value, nextnode is subtree
    return root


#------ITERATING OVER TREE-------
def MakeDecision(root, entry):
    if root.terminal:
        return root.value
    for v in root.children:
        if v.value == entry.attr[root.value]:
            return MakeDecision(v.nextNode, entry)
    return None # only occurs if issue with tree construction

#----TESTING EXAMPLES----
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


#-----OUTPUT FILE-----
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


#----ENTRY POINT----
if __name__ == "__main__":
    S = []
    Attr = dict()
    Labels = set() # TODO: does order matter for this problem?
    training = sys.argv[1]
    testing = sys.argv[2]
    depth = int(sys.argv[3])

    # second arg is training, third is testing
    with open(training, 'r') as f:
        for line in f:
            term = line.strip().split(',')
            for i in range(len(term)-1): # build dictionary of indexable attributes
                if i in Attr.keys():
                    Attr[i].add(term[i])
                else:
                    Attr[i] = {term[i]}
            Labels.add(term[-1]) # build set of labels
            S.append(entry(term[:-1], term[-1])) # build list of examples using entry objects
    # build tree's with the different heuristics
    E_tree = ID3(S, Attr, Labels, depth, InformationGain)
    ME_tree = ID3(S, Attr, Labels, depth, MajorityErrorGain)
    GI_tree = ID3(S, Attr, Labels, depth, GiniIndexGain)
    testing = []
    with open(training, 'r') as f:
            for line in f:
                term = line.strip().split(',')
                testing.append(entry(term[:-1], term[-1]))
    
    #Run tests and write results
    c,i = TestDecisionTree(E_tree, testing)
    #WriteResults("infoGain.txt", c, i)
    print("InformationGain", " correct:", len(c), " Incorrect:", len(i))

    c,i = TestDecisionTree(ME_tree, testing)
    #WriteResults("meGain.txt", c, i)
    print("Majority Error", " correct:", len(c), " Incorrect:", len(i))

    c,i = TestDecisionTree(GI_tree, testing)
    #WriteResults("giGain.txt", c, i)
    print("Gini Index", " correct:", len(c), " Incorrect:", len(i))