import numpy as np
import scipy.io as scio
from sklearn import svm
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler

# Read data 2
data_path = "D:/301/301_2/SVM_USELESS/DATA_1440.mat"
target_path = "D:/301/301_2/SVM_USELESS/Label.mat"
data = scio.loadmat(data_path)['DATA_1440']
print(data.shape)
target = scio.loadmat(target_path)['Label']
target = np.ravel(target)
print(target)
print(target.shape)

# Parameter setting
rbf_C = 2.07
rbf_gamma = 0.0001

# LeaveOneOut
print('LeaveOneOut')
clf = svm.SVC(C = rbf_C, probability = True, gamma = rbf_gamma)
print(clf)

loo = LeaveOneOut()
accuracy = []
predictLable = []
proba = []
proba1 = []
longth = []

for train_index, test_index in loo.split(data):

    train_data, test_data = data[train_index], data[test_index]
    train_target, test_target = target[train_index], target[test_index]

    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data)
    clf.fit(train_data,train_target)
    accuracy.append(clf.score(test_data,test_target))

    aa = clf.predict_proba(test_data)
    bb = clf.predict(test_data)
    cc = clf.decision_function(test_data)
    proba.append(aa[0, 0])
    proba1.append(aa[0, 1])
    predictLable.append(bb[0])
    #print(cc)
    longth.append(cc)

accuracy = np.array(accuracy)
print(accuracy)
print(accuracy.mean())
print(proba)
print(predictLable)
print(longth)


# KFold-CrossValidation
print('KFold-CrossValidation')
clf1 = svm.SVC(C = rbf_C, probability = True, gamma = rbf_gamma)
print(clf1)

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
accuracy1 = []
for train_index, test_index in skf.split(data,target):

    train_data, test_data = data[train_index], data[test_index]
    train_target, test_target = target[train_index], target[test_index]
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data)
    clf1.fit(train_data,train_target)
    accuracy1.append(clf1.score(test_data,test_target))

accuracy1 = np.array(accuracy1)
print(accuracy1)
print(accuracy1.mean())