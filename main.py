import pandas as pd
import numpy as np
import random
import torch

device = torch.device("mps")

from test import *

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

DATASET_PATH = "dataset_train/dataset.csv"
dataset = pd.read_csv(DATASET_PATH, delimiter=",")

name = getName()
print(name)

classifiers = ["knn", "rf", "svm", "ff", "tb", "tf"]
# classifiers = ["tb", "tf"]

for clfName in classifiers:
  print(f"--- {clfName} ---")
  dataset_processed = preprocess(dataset, clfName)
  clf = load(clfName)
  perf = predict(dataset_processed, clf)
  print(perf)