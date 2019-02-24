#AUTHOR: Pulkit Pattnaik
#Date: 24 Feb 2019
#KNN raw code without libraries for data here: https://archive.ics.uci.edu/ml/datasets/Vertebral+Column

import matplotlib.pyplot as plt
import numpy as np

#FUNCTIONS
def find_euc_dist(vec1: np.array, vec2: np.array) -> float:
    return np.sqrt(np.sum(np.square(np.subtract(vec1, vec2))))

#PARAMETERS
k_max_val = 208

#DATA PREP
datapos = []
dataneg = []
data = []
with open('column_2C.dat', 'r') as f:
    d = f.readlines()
    for i in d:
        k = i.rstrip().split(" ")
        if k[-1] == 'AB':
            k[-1] = 1
            datapos.append(k)
        elif k[-1] == 'NO':
            k[-1] = 0
            dataneg.append(k)
        data.append(k)

datapos = np.array(datapos).astype(np.float32)
dataneg = np.array(dataneg).astype(np.float32)
data = np.array(data).astype(np.float32)
#VISUALIZATION
# plt.figure()
# plt.scatter(dataneg[1:50,1], dataneg[1:50, 2], c = 'r')
# plt.scatter(datapos[1:50,1], datapos[1:50, 2], c = 'b')
# plt.show()
# for i in range(data.shape[1] - 1):
#     plt.figure()
#     plt.boxplot(datapos[:,i])
#     plt.boxplot(dataneg[:,i])

#TRAIN TEST SPLIT
train_set = np.concatenate((dataneg[0:70,:],datapos[0:140,:]))
test_set = np.concatenate((dataneg[70:,], datapos[140:,]))

#METRICS AND TRAINING
accuracies = []
all_true_pos = []
all_true_neg = []
all_false_pos = []
all_false_neg = []
for k in range(k_max_val, 1, -3):
    accuracy = 0
    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0
    for test_example in test_set:
        dists = {}
        classes = []
        for train_vec in train_set:
            dists[find_euc_dist(test_example, train_vec)] = train_vec
        k_nearest = sorted(dists.items())[0:k]
        for i in k_nearest:
            classes.append(i[1][-1])
        pred_class = max(set(classes), key=classes.count)
        if pred_class == test_example[-1]:
            accuracy += 1
        if pred_class == 1 and test_example[-1] == 1:
            true_pos += 1
        elif pred_class == 1 and test_example[-1] == 0:
            false_pos += 1
        elif pred_class == 0 and test_example[-1] == 1:
            false_neg += 1
        elif pred_class == 0 and test_example[-1] == 0:
            true_neg += 1
    accuracy /= len(test_set)
    accuracies.append(accuracy)
    all_false_neg.append(false_neg)
    all_false_pos.append(false_pos)
    all_true_neg.append(true_neg)
    all_true_pos.append(true_pos)

#SUBSET OF TRAIN DATA
for n in range(10, 210, 10):
    train_set = np.concatenate((dataneg[0:int(n / 3),:],datapos[0:n - int(n / 3),:]))
    test_set = np.concatenate((dataneg[int(n / 3):,:],datapos[n - int(n / 3):,:]))
    n_acc = []
    for k in range(1, n, 5):
        accuracy = 0
        for test_example in test_set:
            dists = {}
            classes = []
            for train_vec in train_set:
                dists[find_euc_dist(test_example, train_vec)] = train_vec
            k_nearest = sorted(dists.items())[0:k]
            for i in k_nearest:
                classes.append(i[1][-1])
            pred_class = max(set(classes), key=classes.count)
            if pred_class == test_example[-1]:
                accuracy += 1
        accuracy /= len(test_set)
        accuracies.append(accuracy)
    n_acc.append(accuracies)