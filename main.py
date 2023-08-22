import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import os

train_df = pd.read_csv("train.csv")
print(train_df.head())
train_row, train_col = train_df.shape
print(f"\nThere are {train_row} rows and {train_col} columns in train dataframe.")

test_df = pd.read_csv("test.csv")
print(test_df.head())
test_row, test_col = test_df.shape
print(f"\nThere are {train_row} rows and {train_col} columns in test dataframe")
