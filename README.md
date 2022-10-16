# DeepiForest

## The source code for the paper "DeepiForest: A Deep Anomaly Detection Framework with Hashing Based Isolation Forest"


### An example for running the code:
    import pandas as pd  
    import time  
    import numpy as np  
    from sklearn.metrics import roc_auc_score  
    from reshapeforest import CascadeLSHForest  

    data = pd.read_csv('dat/glass.csv', header=None)  
    X = data.values[:, :-1]  
    ground_truth = data.values[:, -1]  

    model = CascadeLSHForest()  
    start_time = time.time()  
    model.fit(X)   
    y_pred = model.predict(X) 
    all_time = time.time() - start_time 
    auc = roc_auc_score(ground_truth, -1.0 * y_pred) * 100  
    print("\nTesting Accuracy:	{:.3f} %".format(auc))  
    print("Execution time:	", all_time)  
    




