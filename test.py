MY_UNIQUE_ID = "MatteoGaliazzo"

from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score
import pickle
import pandas as pd
import torch
from pytorch_tabular import TabularModel
from utils import *


def getName():
  return MY_UNIQUE_ID


def preprocess(dataset: pd.DataFrame, clfName: str) -> dict:
  y = dataset["grade"].map({"A": 6, "B": 5, "C": 4, "D": 3, "E": 2, "F": 1, "G": 0})
  dataset_processed = {"name": clfName, "grade": y}

  if clfName in ["knn", "rf", "svm"]:
    file_path = f"{clfName}_preprocessor.save"
    scaler = load_pickle(file_path)
    
    X = dataset.drop(columns=["grade"])
    
    dataset_processed['data'] = scaler.transform(X)

  elif clfName == "ff":
    num_scaler = load_pickle("ff_numerical_preprocessor.save")
    zip_scaler = load_pickle("ff_zip_preprocessor.save")

    embed_cols = ["borrower_address_zip"]
    
    X_num = dataset.drop(columns=["grade"] + embed_cols)
    X_zip = dataset[embed_cols]

    dataset_processed['data_num'] = num_scaler.transform(X_num)
    dataset_processed['data_zip'] = zip_scaler.transform(X_zip).flatten() 

  elif clfName in ["tb", "tf"]:
    one_hot_encoding_cols = ["borrower_housing_ownership_status", "borrower_income_verification_status",
                          "loan_status_current_code", "loan_purpose_category"]
    embed_cols = ["borrower_address_zip"] + one_hot_encoding_cols

    num_scaler = load_pickle(f"{clfName}_numerical_preprocessor.save")
    X_num = num_scaler.transform(dataset.drop(columns=["grade"]+ embed_cols))
    dataset_processed["data"] = pd.concat([X_num, dataset[embed_cols]], axis=1)

  else:
    raise ValueError(f"Unknown classifier name: {clfName}")

  return dataset_processed


def load(clfName):
  device = get_device()
  clf = None
  if (clfName == "knn"):
    clf = pickle.load(open("knn_model.save", 'rb'))
  elif (clfName == "rf"):
    clf = pickle.load(open("rf_model.save", 'rb'))
  elif (clfName == "svm"):
    clf = pickle.load(open("svm_model.save", 'rb'))
  elif (clfName == "ff"):
    checkpoint = torch.load('best_ff_model_weights.pth', map_location=device)
    clf = FeedForwardModel(cont_dim=120, hidden_dims=[128, 64, 32]).to(device)
    clf.load_state_dict(checkpoint)
  elif (clfName =="tb"):
    clf = TabularModel.load_model("saved_tabnet_model")
  elif (clfName =="tf"):
    clf = TabularModel.load_model("saved_tabtransformer_model")
  return clf


def predict(dataset, clf):
  device = get_device()
  y = dataset['grade']
  if dataset['name'] in ["knn", "rf", "svm"]:
    X = dataset['data']
    ypred = clf.predict(X)
  elif dataset['name'] == "ff":
    X_num = torch.tensor(dataset['data_num'], dtype=torch.float32).to(device)
    X_zip = torch.tensor(dataset['data_zip'], dtype=torch.long).to(device)
    # y = torch.tensor(dataset['grade'], dtype=torch.long)
    clf.eval()
    with torch.no_grad():
      logits = clf(X_num, X_zip)
      ypred = torch.argmax(logits, dim=1).cpu().numpy()
  elif dataset['name'] in ["tb", "tf"]:
    pred_df = clf.predict(dataset["data"])
    ypred = pred_df['grade_prediction'].values

  acc = accuracy_score(y, ypred)
  bacc = balanced_accuracy_score(y, ypred)
  f1 = f1_score(y, ypred, average="weighted")
  
  perf = {"acc": acc, "bacc": bacc, "f1": f1}
  
  return perf
    
