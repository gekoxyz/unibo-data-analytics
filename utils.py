import pickle
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import torch
import torch.nn as nn

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
        
        x = self.mlp(x)
        
        return self.head(x)


def load_pickle(filename):
  with open(filename, "rb") as f:
    return pickle.load(f)
  

def get_device():
  if torch.backends.mps.is_available():
    device = torch.device("mps")
  elif torch.cuda.is_available():
    device = torch.device("cuda")
  else:
    device = torch.device("cpu")
  return device