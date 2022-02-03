import csv
import numpy as np

with open('EnjoySport.csv', 'r') as f:
    reads = csv.reader(f)
    tmp_lst = np.array(list(reads))
concept = np.array(tmp_lst[:, :-1])
target = np.array(tmp_lst[:, -1])
specific_h = concept[0].copy()

for i in range(len(target)):
    if target[i] == 'Yes':
        specific_h = concept[i]
        break
h = []
general_h = [['?' for k in range(len(specific_h))] for i in range(len(specific_h))]
print(type(general_h))

for i in range(len(target)):
    if target[i] == 'Yes':
        for j in range(len(specific_h)):
            if specific_h[j] != concept[i][j]:
                specific_h[j] = '?'
                general_h[j][j] = '?'
    else:
        for j in range(len(specific_h)):
            if specific_h[j] != concept[i][j]:
                general_h[j][j] = specific_h[j]
            else:
                general_h[j][j] = '?'
    print("Step ", i + 1)
    print("The most generic is : ", general_h)
    print("The most specific is : ", specific_h)
