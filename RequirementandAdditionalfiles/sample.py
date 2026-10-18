import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import torch
import transformers
import datasets
import sentence_transformers
import faiss
from rank_bm25 import BM25Okapi
print("All libraries imported successfully!")
print("NumPy version:",np.__version__)
print("Pandas version:",pd.__version__)
print("PyTorch version:",torch.__version__)
print("CUDA available:",torch.cuda.is_available())