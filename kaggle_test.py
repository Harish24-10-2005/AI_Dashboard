import pandas as pd
import zipfile
import os
from kaggle.api.kaggle_api_extended import KaggleApi

# Initialize the Kaggle API
api = KaggleApi()
api.authenticate()

# Define the dataset
dataset = 'heptapod/titanic'

# Define a temporary directory to download the dataset
temp_dir = 'temp_titanic'

# Create the temporary directory if it doesn't exist
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)

# Download the dataset files to the temporary directory
api.dataset_download_files(dataset, path=temp_dir, unzip=True)

# Load the CSV files into pandas DataFrames
train_df = pd.read_csv(os.path.join(temp_dir, 'train.csv'))
test_df = pd.read_csv(os.path.join(temp_dir, 'test.csv'))

# Display the first few rows of the training data
print(train_df.head())

# Clean up: remove the temporary directory and its contents
import shutil
shutil.rmtree(temp_dir)