import pandas as pd
from collections import Counter
import math

tennis = pd.read_csv('PlayTennisV2.csv')
print("\n Given Play Tennis Data Set:\n\n", tennis)


def entropy(alist):
    c = Counter(x for x in alist)
    instances = len(alist)
    prob = [x / instances for x in c.values()]
    return sum([-p * math.log(p, 2) for p in prob])


def information_gain(__tennis, split, target):
    splitting = __tennis.groupby(split)
    n = len(__tennis.index)
    # print("splitting..", splitting)
    agent = splitting.agg({target: [entropy, lambda x: len(x) / n]})[target]  # aggregating
    # print("agent ", agent)
    agent.columns = ['Entropy', 'observations']
    newentropy = sum(agent['Entropy'] * agent['observations'])
    # print(agent['Entropy']  ,"...\n\n")
    oldentropy = entropy(__tennis[target])
    return oldentropy - newentropy


def id3(_tennis, target, _names):
    count = Counter(x for x in _tennis[target])  # class of YES/NO
    #print("total count: ", count, "count_len: ", len(count))
    if len(count) == 1:
        return next(iter(count))  # next input data set, or raises StopIteration when EOF is hit
    else:
        gain = [information_gain(_tennis, attr, target) for attr in _names]
        print("Gain= ", gain)
        maximum = gain.index(max(gain))  # get index from max gain
        best = _names[maximum]  # use that index to get name
        print("Best Attribute:", best)
        tree = {best: {}}
        remaining = [i for i in _names if i != best]
        # print("remaining ", remaining, " and ", "tree : ", tree)
        for val, subset in _tennis.groupby(best):
            # print("subset: ", subset , ".... ", val)
            subtree = id3(subset, target, remaining)
            # print("subtree:  ", subtree)
            tree[best][val] = subtree
            # print("tree : ", tree)
        return tree


names = list(tennis.columns)
print("List of Attributes:", names)
names.remove('PlayTennis')
print("Predicting Attributes:", names)
tree = id3(tennis, 'PlayTennis', names)
print("\n\nThe Resultant Decision Tree is:\n")
print(tree)
