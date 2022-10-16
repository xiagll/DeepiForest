import pandas as pd
import time
import numpy as np

from sklearn.metrics import roc_auc_score

from reshapeforest import CascadeLSHForest

data = pd.read_csv('dat/glass.csv', header=None)
X = data.values[:, :-1]
ground_truth = data.values[:, -1]

AUC = []
Train_time = []
Test_time = []
for j in range(1):
    model = CascadeLSHForest()
    start_time = time.time()
    model.fit(X)
    train_time = time.time() - start_time
    Train_time.append(train_time)
    y_pred = model.predict(X)
    test_time = time.time() - start_time - train_time
    Test_time.append(test_time)
    auc = roc_auc_score(ground_truth, -1.0 * y_pred) * 100
    AUC.append(auc)

mean_auc = np.mean(AUC)
std_auc = np.std(AUC)
mean_traintime = np.mean(Train_time)
mean_testtime = np.mean(Test_time)

print("AUC_mean:	", mean_auc)
print("AUC_std:	", std_auc)
print("Training time:	", mean_traintime)
print("Testing time:	", mean_testtime)