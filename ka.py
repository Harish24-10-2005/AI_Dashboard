import os
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

# Initialize and authenticate the Kaggle API using environment variables
api = KaggleApi()
api.authenticate()

# Alternatively, you can authenticate using environment variables
# Uncomment the following lines if you want to set username and key directly

os.environ['KAGGLE_USERNAME'] = "harishravikumar24"
os.environ['KAGGLE_KEY'] = "5e258912c63a4b77486d04bb8fa7ebaf"

# Specify the dataset you want to fetch
dataset = 'heptapod/titanic'  # Replace with the actual dataset path

# Create a temporary directory to store the dataset files
temp_dir = 'temp_kaggle_dataset'
os.makedirs(temp_dir, exist_ok=True)

# Download the dataset files (it will be unzipped)
api.dataset_download_files(dataset, path=temp_dir, unzip=True)

# Load the dataset into a DataFrame
# Replace 'file.csv' with the actual filename of the dataset you want to load
df = pd.read_csv(os.path.join(temp_dir, 'file.csv'))  # Replace 'file.csv' with the actual file name

# Display the first few rows of the DataFrame
print(df.head())

# Optionally, clean up the temporary directory after loading the data
import shutil
shutil.rmtree(temp_dir)