import numpy as np
import scipy.io as scio
from sklearn import svm
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition

# Read data 2
data_path = "D:/301/301_2/SVM_USELESS/DATA_1440.mat"
target_path = "D:/301/301_2/SVM_USELESS/Label.mat"
data = scio.loadmat(data_path)['DATA_1440']
print(data.shape)
target = scio.loadmat(target_path)['Label']
target = np.ravel(target)
print(target)
print(target.shape)

# Read data 3
data_path_3 = "D:/301/301_3/SVM_USELESS/DATA_1440_NC_AD.mat"
target_path_3 = "D:/301/301_3/SVM_USELESS/Label_NC_AD.mat"
data_3 = scio.loadmat(data_path_3)['DATA_1440_NC_AD']
print(data_3.shape)
target_3 = scio.loadmat(target_path_3)['Label_NC_AD']
target_3 = np.ravel(target_3)
print(target_3)
print(target_3.shape)

# PCA 降维
Dim = 80
pca = decomposition.PCA(n_components=Dim,svd_solver='randomized',copy=True, whiten=True)
data_pca = pca.fit_transform(data)
data_3_pca = pca.transform(data_3)
print(data_pca.shape)
print(data_3_pca.shape)


# Parameter setting
rbf_C = 3.6
rbf_gamma = 0.0025

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


for train_index, test_index in loo.split(data_3_pca):

    train_data, test_data = data_pca, data_3_pca[test_index]
    train_target, test_target = target, target_3[test_index]

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
