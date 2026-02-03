MY_UNIQUE_ID = "MatteoGaliazzo"

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score
import pickle
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
device = torch.device("mps")

from pytorch_tabular import TabularModel


def getName():
  return MY_UNIQUE_ID


class NumericExtractor(BaseEstimator, TransformerMixin):
    """Extracts integers from strings using regex"""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col in X.columns:
            X[col] = X[col].astype(str).str.extract(r"(\d+)").astype(float)
        return X

class CyclicalDateEncoder(BaseEstimator, TransformerMixin):
    """Converts mm-yyyy to year + sine/cosine month encoding."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col in X.columns:
            # errors="coerce" turns unparseable data/NaNs into NaT
            date_series = pd.to_datetime(X[col], format="%b-%Y", errors="coerce")
            # If date is NaT, these become NaN, which we handle in the pipeline later
            angle = 2 * np.pi * date_series.dt.month / 12

            X[f"{col}_year"] = date_series.dt.year
            X[f"{col}_month_sin"] = np.sin(angle)
            X[f"{col}_month_cos"] = np.cos(angle)
            
            X.drop(columns=[col], inplace=True)
        return X
    
class BinaryModeEncoder(BaseEstimator, TransformerMixin):
    """"Encodes 0 if value is mode, 1 if not"""
    def __init__(self):
        self.modes_ = {}

    def fit(self, X, y=None):
        # Calculate mode for each column and store it
        for col in X.columns:
            self.modes_[col] = X[col].mode()[0]
        return self

    def transform(self, X):
        X_copy = X.copy()
        for col, mode in self.modes_.items():
            # Apply: 1 if NOT the mode (least frequent), 0 if mode
            X_copy[col] = (X_copy[col] != mode).astype(int)
        return X_copy
    
class HighMissingDropper(BaseEstimator, TransformerMixin):
    """Drops columns with high missing percentage. Fits only on training data."""
    
    def __init__(self, threshold=20):
        self.threshold = threshold
        self.cols_to_drop_ = []

    def fit(self, X, y=None):
        # Calculate missing percentages only on training data
        missing_percentages = X.isna().mean() * 100
        self.cols_to_drop_ = missing_percentages[missing_percentages > self.threshold].index.tolist()
        return self

    def transform(self, X):
        X = X.copy()
        return X.drop(columns=self.cols_to_drop_)


class FeedForwardModel(nn.Module):
    def __init__(self, cont_dim, hidden_dims=[128, 64, 32], output_dim=7):
        super().__init__()
        zip_embed_dim = 64
        num_zip_codes = 883 # borrower_address_zip 882 different values + 1 missing
        self.emb = nn.Embedding(num_zip_codes, zip_embed_dim)
        
        self.input_dim = cont_dim + zip_embed_dim

        layers = []
        in_dim = self.input_dim
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(0.2))
            in_dim = h_dim
        
        self.mlp = nn.Sequential(*layers)
        
        # Projects the last hidden layer to the number of classes (7)
        self.head = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, X_cont, X_zip):
        # Embed zip codes
        # Result shape: (Batch_Size, zip_embed_dim)
        zip_embedded = self.emb(X_zip)
        
        # Concatenate continuous features + embeddings
        # Result shape: (Batch_Size, cont_dim + zip_embed_dim)
        x = torch.cat([X_cont, zip_embedded], dim=1)
        
        # Pass through MLP features
        x = self.mlp(x)
        
        # Final classification
        return self.head(x)


def load_pickle(filename):
  with open(filename, "rb") as f:
    return pickle.load(f)

def preprocess(dataset: pd.DataFrame, clfName: str) -> dict:
  import __main__
  if not hasattr(__main__, "HighMissingDropper"):
    __main__.HighMissingDropper = HighMissingDropper
  if not hasattr(__main__, "NumericExtractor"):
    __main__.NumericExtractor = NumericExtractor
  if not hasattr(__main__, "CyclicalDateEncoder"):
    __main__.CyclicalDateEncoder = CyclicalDateEncoder
  if not hasattr(__main__, "BinaryModeEncoder"):
    __main__.BinaryModeEncoder = BinaryModeEncoder

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
    
