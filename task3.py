import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
from sklearn.model_selection  import RandomizedSearchCV
import numpy as np
from imblearn.over_sampling import ADASYN
from sklearn.preprocessing import StandardScaler

from sklearn import metrics
import time


class SVC_NS():
    def __init__(self, dataFilename, resample=False):
        self.X_train, self.X_test, self.y_train, self.y_test = SVC_NS.parseData(dataFilename, resample)
        
    def parseData(dataFilename, resample):
        # Read data
        df = pd.read_csv(dataFilename)
        
        # Separate features from labels
        X = df.iloc[:,1:-1].to_numpy()
        y = df.loc[:,'Class'].to_numpy()
        
        # Assign "false" samples as -1 rather than 0
        y[y == 0] = -1
        
        if (resample):
           X, y = SVC_NS.makeOverSamplesADASYN(X, y)
        
        # # Split dataset into training and test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
        ss = StandardScaler()
        X_train = ss.fit_transform(X_train)
        X_test = ss.fit_transform(X_test)
        
        return X_train, X_test, y_train, y_test
    
    def train(self):
        C = 1
        class_weight = None
        random_state = 42
        dual = True
        penalty = 'l2'
        max_iter = 2000
        loss = 'hinge'
        
        
        self.svc = LinearSVC(C=C, class_weight=class_weight,random_state=random_state,dual=dual,penalty=penalty,max_iter=max_iter,loss=loss)
        self.svc.fit(self.X_train, self.y_train)
        print(self.svc.get_params())
        
    def trainWithRandomizedSearch(self, random_grid):    
        svc = LinearSVC()
        self.svc = RandomizedSearchCV(estimator = svc, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
        
        # Fit the random search model
        self.svc.fit(self.X_train, self.y_train)
        print("Best params: ", self.svc.best_params_)
        return self.svc.best_estimator_
        
    def evaluate(self):
        predictions = SVC_NS.predict(self.svc, self.X_test)
        
        cm = metrics.confusion_matrix(self.y_test, predictions)
        print("Confusion Matrix\n", cm)
        
        print("f1: ", f1_score(self.y_test, predictions))
    
    def predict(svc, X_test):
        return svc.predict(X_test)
        
    def makeOverSamplesADASYN(X, y):
        ada = ADASYN(random_state=0, n_neighbors=8)
        XADA, yADA = ada.fit_resample(X, y)
        return(XADA, yADA)

     
if __name__ == "__main__":
    filepath = "E:\\UCL\\ML\\CW\\creditcard.csv"
    
    # C = [1,10, 50, 100]
    # class_weight = [None, 'balanced']
    # random_state = [None, 42]
    # dual = [False, True]
    # penalty = ['l1', 'l2']
    # max_iter = [500, 1000, 2000, 4000]
    # loss = ['hinge', 'squared_hinge']
    
    C = [1]
    class_weight = [None]
    random_state = [42]
    dual = [True]
    penalty = ['l2']
    max_iter = [2000]
    loss = ['hinge']
    
    random_grid = {'C': C, 
                   'class_weight': class_weight,
                   'random_state': random_state,
                   'dual': dual,
                   'penalty': penalty,
                   'max_iter': max_iter,
                   'loss': loss
                   }
    
    #randomModel = SVC_NS(filepath)
    #randomModel.trainWithRandomizedSearch(random_grid)
    #random_auc = randomModel.evaluate()
    
    start = time.time()
    
    baseModel = SVC_NS(filepath, True)
    baseModel.train()
    base_auc = baseModel.evaluate()
    
    end = time.time()
    print("Time: ", end - start)
