MY_UNIQUE_ID = "MatteoGaliazzo"

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
import pickle
import pandas as pd
import numpy as np
import os

import torch
import torch.nn as nn

# Output: unique ID of the team
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


# knn, rf, svm, ff, tb, tf

# Input: Dataset dictionary and classifier name
# Output: PreProcessed Dataset dictionary
def preprocess(dataset, clfName):
  print("preprocessing")
  # Map custom classes to __main__
  import __main__
  # Check if the class is already in __main__, if not, add it
  if not hasattr(__main__, "HighMissingDropper"):
      __main__.HighMissingDropper = HighMissingDropper
  if not hasattr(__main__, "NumericExtractor"):
      __main__.NumericExtractor = NumericExtractor
  if not hasattr(__main__, "CyclicalDateEncoder"):
      __main__.CyclicalDateEncoder = CyclicalDateEncoder
  if not hasattr(__main__, "BinaryModeEncoder"):
      __main__.BinaryModeEncoder = BinaryModeEncoder


  if clfName in ["knn", "rf", "svm"]:
    X = dataset.drop(columns=["grade"])
    y = dataset["grade"].map({"A": 6, "B": 5, "C": 4, "D": 3, "E": 2, "F": 1, "G": 0})
  if clfName == "ff":
    numerical_column = ['loan_contract_approved_amount', 'loan_portfolio_total_funded', 'investor_side_funded_amount', 'loan_contract_term_months', 'loan_contract_interest_rate', 'loan_payment_installments_count', 'borrower_profile_employment_length', 'borrower_housing_ownership_status', 'borrower_income_annual', 'borrower_income_verification_status', 'loan_issue_date', 'loan_status_current_code', 'loan_payment_plan_flag', 'loan_purpose_category', 'loan_title', 'borrower_address_state', 'borrower_dti_ratio', 'credit_delinquencies_2yrs', 'credit_history_earliest_line', 'fico_score_low_bound', 'fico_score_high_bound', 'credit_inquiries_6m', 'months_since_last_delinquency', 'months_since_last_public_record', 'credit_open_accounts', 'credit_public_records', 'revolving_balance', 'revolving_utilization', 'credit_total_accounts', 'listing_initial_status', 'outstanding_principal_balance', 'outstanding_principal_investor_side', 'total_payment_received', 'total_payment_investor_side', 'total_received_principal', 'total_received_interest', 'total_received_late_fees', 'recoveries_cash', 'collection_recovery_fee', 'last_payment_date', 'last_payment', 'next_payment_date', 'last_credit_pull_date', 'last_fico_score_high_bound', 'last_fico_score_low_bound', 'collections_12m_ex_med', 'months_since_last_major_derog', 'platform_policy_code_id', 'application_type_label', 'joint_income_annual', 'joint_dti_ratio', 'joint_income_verification_status', 'accounts_now_delinquent', 'total_collection_amount', 'total_current_balance', 'open_accounts_6m', 'open_active_installment_loans', 'open_installment_loans_12m', 'open_installment_loans_24m', 'months_since_recent_installment_loan', 'total_balance_installment_loans', 'installment_utilization', 'open_revolving_accounts_12m', 'open_revolving_accounts_24m', 'bankcard_max_balance', 'overall_utilization', 'total_revolving_high_credit_limit', 'finance_inquiries', 'credit_union_trades_total', 'credit_inquiries_12m', 'accounts_open_past_24m', 'average_current_balance', 'bankcard_open_to_buy', 'bankcard_utilization', 'chargeoffs_within_12m', 'delinquency_amount', 'months_since_oldest_installment_acct', 'months_since_oldest_revolving_acct', 'months_since_recent_revolving_acct', 'months_since_recent_trade_line', 'mortgage_accounts', 'months_since_recent_bankcard', 'months_since_recent_bankcard_delinquency', 'months_since_recent_inquiry', 'months_since_recent_revolving_delinquency', 'accounts_ever_120dpd', 'active_bankcard_tradelines', 'active_revolving_tradelines', 'bankcard_satisfactory_accounts', 'bankcard_tradelines', 'installment_tradelines', 'open_revolving_tradelines', 'revolving_accounts', 'revolving_tradelines_balance_gt_0', 'satisfactory_accounts', 'tradelines_120dpd_2m', 'tradelines_30dpd', 'tradelines_90dpd_24m', 'tradelines_open_past_12m', 'tradelines_never_delinquent_ratio', 'bankcard_util_gt_75_ratio', 'public_record_bankruptcies', 'tax_liens_total', 'total_high_credit_limit', 'total_balance_ex_mortgage', 'total_bankcard_credit_limit', 'total_installment_high_credit_limit', 'joint_revolving_balance', 'secondary_applicant_fico_low', 'secondary_applicant_fico_high', 'secondary_applicant_earliest_credit_line', 'secondary_applicant_inquiries_6m', 'secondary_applicant_mortgage_accounts', 'secondary_applicant_open_accounts', 'secondary_applicant_revolving_utilization', 'secondary_applicant_open_active_installment_loans', 'secondary_applicant_revolving_accounts', 'secondary_applicant_chargeoffs_12m', 'secondary_applicant_collections_12m_ex_med', 'secondary_applicant_months_since_last_major_derog', 'hardship_flag_indicator', 'hardship_type_label', 'hardship_reason_label', 'hardship_status_label', 'hardship_deferral_term_months', 'hardship_amount_total', 'hardship_start_date', 'hardship_end_date', 'hardship_payment_plan_start_date', 'hardship_duration_days', 'hardship_days_past_due', 'hardship_loan_status_label', 'original_projected_additional_accrued_interest', 'hardship_payoff_balance', 'hardship_last_payment_amount_total', 'disbursement_method_type', 'debt_settlement_flag_indicator', 'debt_settlement_flag_date', 'settlement_status_label', 'settlement_date', 'settlement_amount_total', 'settlement_percentage', 'settlement_term_months']
    embed_column = ["borrower_address_zip"] 

    X_num = X[numerical_column]
    X_zip = X[embed_column]
    y = dataset["grade"].map({"A": 6, "B": 5, "C": 4, "D": 3, "E": 2, "F": 1, "G": 0})
  
  dataset_processed = {"name": clfName}
  scaler = None
  if clfName == "knn":
    scaler = pickle.load(open("knn_preprocessor.save", "rb"))
  elif clfName == "rf":
    scaler = pickle.load(open("rf_preprocessor.save", "rb"))
  elif clfName == "svm":
    scaler = pickle.load(open("svm_preprocessor.save", "rb"))
  elif clfName == "ff":
    num_scaler = pickle.load(open("ff_svm_preprocessor.save", "rb"))
    zip_scaler = pickle.load(open("ff_zip_preprocessor.save", "rb"))
  
  if scaler is not None and clfName in ["knn", "rf", "svm"]:
    dataset_processed['data'] = scaler.transform(X)
    dataset_processed['grade'] = y
  if scaler is not None and clfName == "ff":
    dataset_processed['data_num'] = num_scaler.transform(X_num)
    dataset_processed['data_zip'] = zip_scaler.transform(X_zip).flatten()
    dataset_processed['grade'] = y
    
  return dataset_processed


# Input: Classifier name ("lr": Logistic Regression, "svc": Support Vector Classifier)
# Output: Classifier object
def load(clfName):
  print("loading classifier")
  clf = None
  
  if (clfName == "knn"):
    clf = pickle.load(open("knn_model.save", 'rb'))
  elif (clfName == "rf"):
    clf = pickle.load(open("rf_model.save", 'rb'))
  elif (clfName == "svm"):
    clf = pickle.load(open("svm_model.save", 'rb'))
  elif (clfName == "ff"):
    checkpoint = torch.load('model_checkpoint.pth', map_location="mps")
    clf = FeedForwardModel(cont_dim=checkpoint['cont_dim'], hidden_dims=checkpoint["hidden_dims"]).to("mps")


  return clf


# Input: PreProcessed Dataset dictionary, Classifier Name, Classifier Object 
# Output: Performance dictionary
def predict(dataset, clf):
  print("predict")
  if dataset['name'] in ["knn", "rf", "svm"]:
    X = dataset['data']
    y = dataset['grade']
    ypred = clf.predict(X)
  elif dataset['name'] == "ff":
    X_num = dataset['data_num']
    X_zip = dataset['data_zip']
    y = dataset['grade']
    
    clf.eval()
    with torch.no_grad():
      logits = clf(X_num, X_zip)
      ypred = torch.argmax(logits, dim=1).cpu().numpy()
  

  acc = accuracy_score(y, ypred)
  bacc = balanced_accuracy_score(y, ypred)
  f1 = f1_score(y, ypred, average="weighted")
  
  perf = {"acc": acc, "bacc": bacc, "f1": f1}
  
  return perf
    
