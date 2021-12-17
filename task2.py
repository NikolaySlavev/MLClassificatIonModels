import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection  import RandomizedSearchCV
from sklearn import metrics

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import ADASYN
import time

class RandomForestClassifierNS():
    
    def __init__(self, dataFilename, resample=False):
        self.X_train, self.X_test, self.y_train, self.y_test = RandomForestClassifierNS.parseData(dataFilename, resample)
        
    def parseData(dataFilename, resample):
        # Read data
        df = pd.read_csv(dataFilename)
        
        # Separate features from labels
        X = df.iloc[:,1:-1].to_numpy()
        y = df.loc[:,'Class'].to_numpy()
        
        if (resample):
           X, y = RandomForestClassifierNS.makeOverSamplesADASYN(X, y)

        # Split dataset into training and test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        ss = StandardScaler()
        X_train = ss.fit_transform(X_train)
        X_test = ss.fit_transform(X_test)
        
        return X_train, X_test, y_train, y_test
        
    def train(self):
        n_estimators = 150
        max_features = 'sqrt'
        max_depth = 300
        min_samples_split = 2
        min_samples_leaf = 1
        bootstrap = False
        
        self.rf = RandomForestClassifier(n_estimators=n_estimators,max_features=max_features,max_depth=max_depth,min_samples_leaf=min_samples_leaf,min_samples_split=min_samples_split,bootstrap=bootstrap)
        self.rf.get_params()
        self.rf.fit(self.X_train, self.y_train)
        
    def trainWithRandomizedSearch(self, random_grid):    
        rf = RandomForestClassifier()
        self.rf = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
        
        # Fit the random search model
        self.rf.fit(self.X_train, self.y_train)
        print("Best params: ", self.rf.best_params_)
        return self.rf.best_estimator_
        
    def evaluate(self):
        predictions, predictions_proba = RandomForestClassifierNS.predict(self.rf, self.X_test)
        
        cm = metrics.confusion_matrix(self.y_test, predictions)
        print("Confusion Matrix\n", cm)

        ns_probs = [0 for _ in range(len(self.y_test))]
        ns_fpr, ns_tpr, _ = roc_curve(self.y_test, ns_probs)
        lr_fpr, lr_tpr, _ = roc_curve(self.y_test, predictions_proba.reshape(-1, 1))

        plt.plot(lr_fpr, lr_tpr)
        plt.plot(ns_fpr, ns_tpr)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.show()

        rf_auc = roc_auc_score(self.y_test, predictions)
        print("rf_auc: ", rf_auc) 
        print("f1: ", f1_score(self.y_test, predictions))

        return f1_score(self.y_test, predictions)
    
    def predict(rf, X_test):
        predictions_proba = rf.predict_proba(X_test)[:, 1]
        predictions = rf.predict(X_test)
        return predictions, predictions_proba
    
    def makeOverSamplesADASYN(X, y):
        ada = ADASYN(random_state=0, n_neighbors=8)
        XADA, yADA = ada.fit_resample(X, y)
        return(XADA, yADA)



if __name__ == "__main__":
    filepath = "E:\\UCL\\ML\\CW\\creditcard.csv"
    
    # n_estimators = [10, 100, 150, 200, 400, 500]
    # max_features = ['auto, ''sqrt']
    # max_depth = [None, 10, 100, 200, 300, 500]
    # min_samples_split = [2, 5, 10]
    # min_samples_leaf = [1, 2, 4]
    # bootstrap = [True, False]
    
    n_estimators = [150]
    max_features = ['sqrt']
    max_depth = [300]
    min_samples_split = [2]
    min_samples_leaf = [1]
    bootstrap = [False]
    
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    #randomModel = RandomForestClassifierNS(filepath)
    #randomModel.trainWithRandomizedSearch(random_grid)
    #random_auc = randomModel.evaluate()
    
    start = time.time()

    baseModel = RandomForestClassifierNS(filepath, True)
    baseModel.train()
    base_auc = baseModel.evaluate()
    
    end = time.time()
    print(end - start)

    #print('Improvement of {:0.3f}%.'.format( 100 * (random_auc - base_auc) / base_auc))
    

# https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
# https://github.com/WillKoehrsen/Machine-Learning-Projects/blob/master/random_forest_explained/Improving%20Random%20Forest%20Part%202.ipynb