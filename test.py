MY_UNIQUE_ID = "MatteoGaliazzo"

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
import pickle
import pandas as pd
import numpy as np
import os

# Output: unique ID of the team
def getName():
    return MY_UNIQUE_ID


'''
Metodi da implementare in test.py
Il campo clfName è una stringa con i seguenti valori ammissibili:
o 'knn' → identifica classificatore K-Nearest Neighbour
o 'rf' → identifica classificatore Random Forest
o 'svm' → identifica classificatore Support Vector Machine
o 'ff' → identificare classificatore con reti neurali, architettura Feed Forward
o 'tb' → identificare classificatore con reti neurali, architettura TabNet
o 'tf' → identificare classificatore con reti neurali, architettura TabTransformer
'''


# Input: Dataset dictionary and classifier name
# Output: PreProcessed Dataset dictionary
def preprocess(dataset, clfName):
  dataset_processed = {}
  scaler = None
  if clfName == "knn":
      scaler = pickle.load(open("knn_scaler.save", 'rb'))
  elif 
  elif 
  if scaler is not None:
      dataset_processed['data'] = scaler.transform(X)
      dataset_processed['target'] = y
    
  return dataset 


# Input: Classifier name ("lr": Logistic Regression, "svc": Support Vector Classifier)
# Output: Classifier object
def load(clfName):
    clf = None
    
    if (clfName == "knn"):
        clf = pickle.load(open("knn.save", 'rb'))
        
    elif 

    elif 

    return clf


# Input: PreProcessed Dataset dictionary, Classifier Name, Classifier Object 
# Output: Performance dictionary
def predict(dataset, clf):
    X = dataset['data']
    y = dataset['target']
    
    ypred = clf.predict(X)

    acc = accuracy_score(y, ypred)
    bacc = balanced_accuracy_score(y, ypred)
    f1 = f1_score(y, ypred, average="weighted")
    
    perf = {"acc": acc, "bacc": bacc, "f1": f1}
    
    return perf
    
