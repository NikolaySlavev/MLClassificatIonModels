import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import ADASYN
import time

# Classifier model to predict credit card loan defaults
# CSV file contains 284808 observations and 29 features
# Use the features to predict the "Class" variable

# Contruct Logistic Regression model for default prediction
# Explain workings of the model and evaluate your model performance
class LogisticRegressionNS:
    logisticRegr = None    
    
    def __init__(self, dataFilename, resample=False):
        self.X_train, self.X_test, self.y_train, self.y_test = LogisticRegressionNS.parseData(dataFilename, resample)
        
    def parseData(dataFilename, resample):
        # Read data
        df = pd.read_csv(dataFilename)
        
        # Separate features from labels
        X = df.iloc[:,1:-1].to_numpy()
        y = df.loc[:,'Class'].to_numpy()
        
        if (resample):
           X, y = LogisticRegressionNS.makeOverSamplesADASYN(X, y)

        # Split dataset into training and test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        ss = StandardScaler()
        X_train = ss.fit_transform(X_train)
        X_test = ss.fit_transform(X_test)
        
        #plt.plot(X,y)
        #plt.show()
        
        return X_train, X_test, y_train, y_test

    def train(self):
        self.logisticRegr = LogisticRegression()
        self.logisticRegr.fit(self.X_train, self.y_train)
        
    def evaluate(self):
        predictions, predictions_proba = LogisticRegressionNS.predict(self.logisticRegr, self.X_test)
        
        score = self.logisticRegr.score(self.X_test, self.y_test)
        print("Score: ", score)
        
        cm = metrics.confusion_matrix(self.y_test, predictions)
        print("Confusion Matrix: \n", cm)

        ns_probs = [0 for _ in range(len(self.y_test))]
        ns_fpr, ns_tpr, _ = roc_curve(self.y_test, ns_probs)
        lr_fpr, lr_tpr, _ = roc_curve(self.y_test, predictions_proba.reshape(-1, 1))

        plt.plot(lr_fpr, lr_tpr)
        plt.plot(ns_fpr, ns_tpr)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.show()

        lr_auc = roc_auc_score(self.y_test, predictions)
        print("lr_auc: ", lr_auc)
        print("f1: ", f1_score(self.y_test, predictions))
        
        print("Accuracy:",metrics.accuracy_score(self.y_test, predictions))
        
    def predict(logisticRegr, X_test):
        predictions_proba = logisticRegr.predict_proba(X_test)[:, 1]
        predictions = logisticRegr.predict(X_test)
        return predictions, predictions_proba
    
    def makeOverSamplesADASYN(X, y):
        ada = ADASYN(random_state=0, n_neighbors=8)
        XADA, yADA = ada.fit_resample(X, y)
        return(XADA, yADA)


if __name__ == "__main__":
    filepath = "E:\\UCL\\ML\\CW\\creditcard.csv"
    #lr = LogisticRegressionNS(filepath, True)
    #lr.train()
    #lr.evaluate()
    
    start = time.time()

    lr = LogisticRegressionNS(filepath, True)
    lr.train()
    lr.evaluate()
    
    end = time.time()
    print(end - start)
    