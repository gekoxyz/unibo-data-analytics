import pandas as pd
import numpy as np
import random
import torch

from test import getName, preprocess, load, predict

# Use the GPU
if torch.backends.mps.is_available():
  print("MPS device is available.")
  device = torch.device("mps")
elif torch.cuda.is_available():
  print("CUDA device is available.")
  device = torch.device("cuda")
else:
  print("No GPU acceleration available.")
  device = torch.device("cpu")

# Fix the seed to have deterministic behaviour
def fix_random(seed: int) -> None:
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True  # slower

SEED = 1337
fix_random(SEED)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

DATASET_PATH = "dataset_train/dataset.csv"
dataset = pd.read_csv(DATASET_PATH, delimiter=",")

print(f"Shape of the dataset: {dataset.shape}")
duplicates = dataset[dataset.duplicated()]
print(f"Number of duplicates in the dataset: {duplicates.shape[0]}")

name = getName()
print(name)
dataset_processed = preprocess(dataset, 'rf')
clf = load('rf')
perf = predict(dataset_processed, clf)