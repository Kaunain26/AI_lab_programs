import numpy as np
import pandas as pd

data = pd.read_csv('enjoysport.csv')
concepts = np.array(data.iloc[:, 0:-1])
print("\nInstances are :\n", concepts)
target = np.array(data.iloc[:, -1])
print("\nTarget Values are: ", target)


def learn(concepts, target):
    # print("specific_h: ", specific_h)
    print("\nInitialization of specific_h and general_h")
    specific_h = concepts[0].copy()
    print("\nSpecific Boundary: ", specific_h)
    general_h = [["?" for _ in range(len(specific_h))] for _ in range(len(specific_h))]
    print("\nGeneric Boundary: ", general_h)

    for i, h in enumerate(concepts):
        print("\nInstances", i + 1, "is ", h)
        if target[i] == "yes":
            print("Instances is Positive")
            for x in range(len(specific_h)):
                if h[x] != specific_h[x]:
                    specific_h[x] = '?'
                    general_h[x][x] = '?'

        if target[i] == "no":
            print("Instance is Negative")
            for x in range(len(specific_h)):
                if h[x] != specific_h[x]:
                    general_h[x][x] = specific_h[x]
                else:
                    general_h[x][x] = '?'

        # print(specific_h + " a dad")
        print("Specific Boundary after ", i + 1, "Instances is ", specific_h)
        print("Generic Boundary after ", i + 1, "Instances is ", general_h)
        print("\n")

    indices = [i for i, val in enumerate(general_h) if val == ['?', '?', '?', '?', '?', '?']]
    for _ in indices:
        general_h.remove(['?', '?', '?', '?', '?', '?'])

    return specific_h, general_h


s_final, g_final = learn(concepts, target)

print("Final Specif_h: ", s_final)
print("Final General_h ", g_final)

# .....................................output.......................................................
#  Instances are :
#  [['sunny' 'warm' 'normal' 'strong' 'warm' 'same']
#  ['sunny' 'warm' 'high' 'strong' 'warm' 'same']
#  ['rainy' 'cold' 'high' 'strong' 'warm' 'change']
#  ['sunny' 'warm' 'high' 'strong' 'cool' 'change']]
#
# Target Values are:  ['yes' 'yes' 'no' 'yes']
#
# Initialization of specific_h and general_h
#
# Specific Boundary:  ['sunny' 'warm' 'normal' 'strong' 'warm' 'same']
#
# Generic Boundary:  [['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?']]
#
# Instances 1 is  ['sunny' 'warm' 'normal' 'strong' 'warm' 'same']
# Instances is Positive
# Specific Boundary after  1 Instances is  ['sunny' 'warm' 'normal' 'strong' 'warm' 'same']
# Generic Boundary after  1 Instances is  [['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?']]
#
#
#
# Instances 2 is  ['sunny' 'warm' 'high' 'strong' 'warm' 'same']
# Instances is Positive
# Specific Boundary after  2 Instances is  ['sunny' 'warm' '?' 'strong' 'warm' 'same']
# Generic Boundary after  2 Instances is  [['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?']]
#
#
#
# Instances 3 is  ['rainy' 'cold' 'high' 'strong' 'warm' 'change']
# Instance is Negative
# Specific Boundary after  3 Instances is  ['sunny' 'warm' '?' 'strong' 'warm' 'same']
# Generic Boundary after  3 Instances is  [['sunny', '?', '?', '?', '?', '?'], ['?', 'warm', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', 'same']]
#
#
#
# Instances 4 is  ['sunny' 'warm' 'high' 'strong' 'cool' 'change']
# Instances is Positive
# Specific Boundary after  4 Instances is  ['sunny' 'warm' '?' 'strong' '?' '?']
# Generic Boundary after  4 Instances is  [['sunny', '?', '?', '?', '?', '?'], ['?', 'warm', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?']]
#
#
# Final Specif_h:
# ['sunny' 'warm' '?' 'strong' '?' '?']
# Final General_h
# [['sunny', '?', '?', '?', '?', '?'], ['?', 'warm', '?', '?', '?', '?']]
