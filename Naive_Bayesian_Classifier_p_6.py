# Write a program to implement the naïve Bayesian classifier for a sample training data set stored
# as a .CSV file. Compute the accuracy of the classifier, considering few test data sets.

import csv

import numpy as np


def read_data(filename):
    with open(filename, 'r') as csvfile:
        datareader = csv.reader(csvfile)
        metadata = next(datareader)
        traindata = []
        for row in datareader:
            traindata.append(row)
    return metadata, traindata


def splitDataset(dataset, splitRatio):
    trainSize = int(len(dataset) * splitRatio)

    trainSet = []
    testset = list(dataset)
    i = 0
    while len(trainSet) < trainSize:
        trainSet.append(testset.pop(i))
    return [trainSet, testset]


def classify(trainData, testData):
    total_size = trainData.shape[0]
    print("\n")
    print("training data size= ", total_size)
    print("test data size= ", testData.shape[0])

    countYes = 0
    countNo = 0
    print("\n")
    print("target count probability")

    # counting total yes and no in training data
    for x in range(trainData.shape[0]):
        if trainData[x, trainData.shape[1] - 1] == 'Yes':
            countYes += 1
        if trainData[x, trainData.shape[1] - 1] == 'No':
            countNo += 1

    # calculate probability of yes and no
    probYes = countYes / total_size
    probNo = countNo / total_size

    print('Yes', "\t", countYes, "\t", probYes)
    print('No', "\t", countNo, "\t", probNo)

    prob0 = np.zeros((testData.shape[1] - 1))
    prob1 = np.zeros(testData.shape[1] - 1)
    print("prob0 nd prob1", prob0, prob1)
    accuracy = 0
    print("\n")
    print("instance prediction target")

    for t in range(testData.shape[0]):  # 6
        for k in range(testData.shape[1] - 1):  # 4
            count1 = count0 = 0
            for j in range(trainData.shape[0]):  # 8
                # how many times appeared with no
                # Ex: test[0,0] = Sunny , data[0,0] = Sunny
                if testData[t, k] == trainData[j, k] and trainData[j, trainData.shape[1] - 1] == 'No':
                    count0 += 1
                # how many times appeared with yes
                if testData[t, k] == trainData[j, k] and trainData[j, trainData.shape[1] - 1] == 'Yes':
                    count1 += 1

            if countNo != 0:
                prob0[k] = count0 / countNo
            if countYes != 0:
                prob1[k] = count1 / countYes

        probno = probNo
        probyes = probYes

        for i in range(testData.shape[1] - 1):  # 4
            probno = probno * prob0[i]  # 0.5 * [0.5,0.2,0.8,0.6] multiplied with each one
            probyes = probyes * prob1[i]
        if probno > probyes:
            predict = 'No'
        else:
            predict = 'Yes'

        print(t + 1, "\t", predict, "\t  ", testData[t, testData.shape[1] - 1])
        if predict == testData[t, testData.shape[1] - 1]:
            accuracy += 1

    final_accuracy = (accuracy / testData.shape[0]) * 100
    print("accuracy", final_accuracy, "%")
    return


metadata, traindata = read_data("PlayTennis.csv")
splitRatio = 0.6
trainingset, testset = splitDataset(traindata, splitRatio)
training = np.array(trainingset)
print("\n The Training data set are:")
for x in trainingset:
    print(x)

testing = np.array(testset)
print("\n The Test data set are:")
for x in testing:
    print(x)
classify(training, testing)

# Output...

#
# The Training data srt are:
# ['Sunny', 'Hot', 'High', 'Weak', 'No']
# ['Sunny', 'Hot', 'High', 'Strong', 'No']
# ['Overcast', 'Hot', 'High', 'Weak', 'Yes']
# ['Rain', 'Mild', 'High', 'Weak', 'Yes']
# ['Rain', 'Cool', 'Normal', 'Weak', 'Yes']
# ['Rain', 'Cool', 'Normal', 'Strong', 'No']
# ['Overcast', 'Cool', 'Normal', 'Strong', 'Yes']
# ['Sunny', 'Mild', 'High', 'Weak', 'No']
#
#  The Test data set are:
# ['Sunny' 'Cool' 'Normal' 'Weak' 'Yes']
# ['Rain' 'Mild' 'Normal' 'Weak' 'Yes']
# ['Sunny' 'Mild' 'Normal' 'Strong' 'Yes']
# ['Overcast' 'Mild' 'High' 'Strong' 'Yes']
# ['Overcast' 'Hot' 'Normal' 'Weak' 'Yes']
# ['Rain' 'Mild' 'High' 'Strong' 'No']
#
#
# training data size=  8
# test data size=  6
#
#
# target count probability
# Yes 	 4 	 0.5
# No 	 4 	 0.5
#
#
# instance prediction target
# 1 	 No 	   Yes
# 2 	 Yes 	   Yes
# 3 	 No 	   Yes
# 4 	 Yes 	   Yes
# 5 	 Yes 	   Yes
# 6 	 No 	   No
# accuracy 66.66666666666666 %
#
