import pandas as pd
import numpy as np
import glob
import os
from typing import Union, List, Dict, Any
from pathlib import Path
import requests
from io import StringIO, BytesIO
import sqlite3
import json
import warnings
import logging
import pickle
import h5py
import zipfile
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
import time
import tempfile
import yaml
from bs4 import BeautifulSoup
import gdown 
import os
from dotenv import load_dotenv
from kaggle.api.kaggle_api_extended import KaggleApi
from github import Github
import boto3

class EnhancedDatasetLoader:
    def __init__(self, verbose: bool = True):
        load_dotenv()
        self.verbose = verbose
        self._setup_logging()
        self.supported_formats = {
            'csv': self._load_csv,
            'excel': self._load_excel,
            'json': self._load_json,
            'sql': self._load_sql,
            'api': self._load_api,
            'pickle': self._load_pickle,
            'parquet': self._load_parquet,
            'hdf5': self._load_hdf5,
            'xml': self._load_xml,
            'txt': self._load_txt,
            'clipboard': self._load_clipboard,
            'gsheet': self._load_google_sheet,
            'zip': self._load_zip,
            'kaggle': self._load_kaggle,
            'github': self._load_github,
            'gdrive': self._load_google_drive,
            'url': self._load_url,
            'web_table': self._load_web_table,
            's3': self._load_s3
        }
        self.dataframes = {} 
        self._init_external_apis()

    def _init_external_apis(self):
        try:
            self.kaggle_api = KaggleApi()
            self.kaggle_api.authenticate()
            print("✅ Kaggle API Initialized Successfully")
            
        except Exception as e:
            print(f"❌ Kaggle API Initialization Failed: {e}")
            self.kaggle_api = None

        github_token = os.getenv('GITHUB_TOKEN')
        try:
            if github_token:
                self.github_api = Github(github_token)
                self.github_api.get_user()
                print("✅ GitHub API Initialized Successfully")
            else:
                print("❌ No GitHub Token Found")
                self.github_api = None
        except Exception as e:
            print(f"❌ GitHub API Initialization Failed: {e}")
            self.github_api = None

        try:
            aws_access_key = os.getenv('AWS_ACCESS_KEY')
            aws_secret_key = os.getenv('AWS_SECRET_KEY')
            if aws_access_key and aws_secret_key:
                self.s3_client = boto3.client(
                    's3', 
                    aws_access_key_id=aws_access_key,
                    aws_secret_access_key=aws_secret_key
                )
                print("✅ AWS S3 Client Initialized Successfully")
            else:
                print("❌ AWS Credentials Not Found")
                self.s3_client = None
        except Exception as e:
            print(f"❌ AWS S3 Initialization Failed: {e}")
            self.s3_client = None

    def _load_kaggle(self, dataset_name: str, **kwargs) -> pd.DataFrame:
        """Load dataset from Kaggle."""
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                self.kaggle_api.dataset_download_files(dataset_name, path=tmp_dir, unzip=True)
                files = os.listdir(tmp_dir)
                data_file = next(f for f in files if f.endswith(('.csv', '.xlsx', '.json')))
                file_path = os.path.join(tmp_dir, data_file)
                return self._load_single_source(file_path, **kwargs)
        except Exception as e:
            self.logger.error(f"Error loading Kaggle dataset: {str(e)}")
            raise

    def _load_github(self, repo_path: str, file_path: str, **kwargs) -> pd.DataFrame:
        """Load dataset from GitHub repository."""
        try:
            raw_url = f"https://raw.githubusercontent.com/{repo_path}/master/{file_path}"
            response = requests.get(raw_url)
            response.raise_for_status()
            if file_path.endswith('.csv'):
                return pd.read_csv(StringIO(response.text), **kwargs)
            elif file_path.endswith('.json'):
                return pd.read_json(StringIO(response.text), **kwargs)
            elif file_path.endswith(('.xlsx', '.xls')):
                return pd.read_excel(BytesIO(response.content), **kwargs)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
        except Exception as e:
            self.logger.error(f"Error loading GitHub dataset: {str(e)}")
            raise

    def _load_google_drive(self, file_id: str, **kwargs) -> pd.DataFrame:
        """Load dataset from Google Drive."""
        try:
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                gdown.download(f"https://drive.google.com/uc?id={file_id}", tmp_file.name, quiet=False)
                return self._load_single_source(tmp_file.name, **kwargs)
        except Exception as e:
            self.logger.error(f"Error loading Google Drive dataset: {str(e)}")
            raise
        finally:
            if 'tmp_file' in locals():
                os.unlink(tmp_file.name)

    def _load_url(self, url: str, **kwargs) -> pd.DataFrame:
        """Load dataset from a URL."""
        try:
            response = requests.get(url)
            response.raise_for_status()
            if url.endswith('.csv'):
                return pd.read_csv(StringIO(response.text), **kwargs)
            elif url.endswith('.json'):
                return pd.read_json(StringIO(response.text), **kwargs)
            elif url.endswith(('.xlsx', '.xls')):
                return pd.read_excel(BytesIO(response.content), **kwargs)
            else:
                raise ValueError(f"Unsupported file format: {url}")
        except Exception as e:
            self.logger.error(f"Error loading URL dataset: {str(e)}")
            raise

    def _load_web_table(self, url: str, table_index: int = 0, **kwargs) -> pd.DataFrame:
        """Load HTML table from web page."""
        try:
            tables = pd.read_html(url, **kwargs)
            if tables and len(tables) > table_index:
                return tables[table_index]
            raise ValueError(f"No table found at index {table_index}")
        except Exception as e:
            self.logger.error(f"Error loading web table: {str(e)}")
            raise

    def _load_s3(self, bucket: str, key: str, **kwargs) -> pd.DataFrame:
        """Load dataset from AWS S3."""
        try:
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                self.s3_client.download_file(bucket, key, tmp_file.name)
                return self._load_single_source(tmp_file.name, **kwargs)
        except Exception as e:
            self.logger.error(f"Error loading S3 dataset: {str(e)}")
            raise
        finally:
            if 'tmp_file' in locals():
                os.unlink(tmp_file.name)

    def load_from_config_file(self, config_path: str):
        """Load multiple datasets from a YAML configuration file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            for dataset_config in config['datasets']:
                name = dataset_config.pop('name')
                source_type = dataset_config.pop('type')
                source = dataset_config.pop('source')
                self.load_dataset(source, name, format_type=source_type, **dataset_config)
        except Exception as e:
            self.logger.error(f"Error loading from config file: {str(e)}")
            raise
        
    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO if self.verbose else logging.WARNING,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def load_dataset(self, 
                    source: Union[str, Dict[str, Any]],
                    name: str,
                    format_type: str = None,
                    **kwargs) -> pd.DataFrame:
        """
        Load a single dataset and store it with a name.
        
        Args:
            source: Data source path or configuration
            name: Name to identify the DataFrame
            format_type: Type of data format
            **kwargs: Additional loading arguments
        
        Returns:
            Loaded DataFrame
        """
        try:
            start_time = time.time()
            
            if isinstance(source, dict):
                df = self._load_from_config(source, **kwargs)
            else:
                df = self._load_single_source(source, format_type, **kwargs)
            
            self.dataframes[name] = df
            
            elapsed_time = time.time() - start_time
            self.logger.info(f"Loaded dataset '{name}' in {elapsed_time:.2f} seconds")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading dataset '{name}': {str(e)}")
            raise

    def get_dataset(self, name: str) -> pd.DataFrame:
        """Retrieve a stored DataFrame by name."""
        if name in self.dataframes:
            return self.dataframes[name]
        raise KeyError(f"Dataset '{name}' not found")

    def list_datasets(self) -> List[str]:
        """List all available dataset names."""
        return list(self.dataframes.keys())

    def remove_dataset(self, name: str):
        """Remove a dataset from storage."""
        if name in self.dataframes:
            del self.dataframes[name]
            self.logger.info(f"Removed dataset '{name}'")

    # Additional loading methods for new formats
    def _load_pickle(self, path: str, **kwargs) -> pd.DataFrame:
        """Load DataFrame from pickle file."""
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            self.logger.error(f"Error loading pickle file {path}: {str(e)}")
            raise

    def _load_parquet(self, path: str, **kwargs) -> pd.DataFrame:
        """Load DataFrame from parquet file."""
        try:
            return pd.read_parquet(path, **kwargs)
        except Exception as e:
            self.logger.error(f"Error loading parquet file {path}: {str(e)}")
            raise

    def _load_hdf5(self, path: str, key: str = 'data', **kwargs) -> pd.DataFrame:
        """Load DataFrame from HDF5 file."""
        try:
            return pd.read_hdf(path, key=key, **kwargs)
        except Exception as e:
            self.logger.error(f"Error loading HDF5 file {path}: {str(e)}")
            raise

    def _load_xml(self, path: str, **kwargs) -> pd.DataFrame:
        """Load DataFrame from XML file."""
        try:
            tree = ET.parse(path)
            root = tree.getroot()
            # Convert XML to DataFrame (implement based on your XML structure)
            # This is a simple example:
            data = []
            for child in root:
                data.append({elem.tag: elem.text for elem in child})
            return pd.DataFrame(data)
        except Exception as e:
            self.logger.error(f"Error loading XML file {path}: {str(e)}")
            raise

    def _load_txt(self, path: str, **kwargs) -> pd.DataFrame:
        """Load DataFrame from text file."""
        try:
            return pd.read_csv(path, delimiter=kwargs.get('delimiter', '\t'), **kwargs)
        except Exception as e:
            self.logger.error(f"Error loading text file {path}: {str(e)}")
            raise

    def _load_clipboard(self, **kwargs) -> pd.DataFrame:
        """Load DataFrame from clipboard."""
        try:
            return pd.read_clipboard(**kwargs)
        except Exception as e:
            self.logger.error(f"Error loading from clipboard: {str(e)}")
            raise

    def _load_google_sheet(self, url: str, **kwargs) -> pd.DataFrame:
        """Load DataFrame from Google Sheet."""
        try:
            # Requires gspread package
            # Implement Google Sheets API authentication and loading
            pass
        except Exception as e:
            self.logger.error(f"Error loading Google Sheet: {str(e)}")
            raise

    def _load_zip(self, path: str, file_name: str = None, **kwargs) -> pd.DataFrame:
        """Load DataFrame from a file within a ZIP archive."""
        try:
            with zipfile.ZipFile(path, 'r') as z:
                if file_name is None:
                    file_name = z.namelist()[0]
                with z.open(file_name) as f:
                    # Detect format from file_name and load accordingly
                    if file_name.endswith('.csv'):
                        return pd.read_csv(f, **kwargs)
                    elif file_name.endswith(('.xlsx', '.xls')):
                        return pd.read_excel(f, **kwargs)
                    # Add more format handlers as needed
        except Exception as e:
            self.logger.error(f"Error loading from ZIP {path}: {str(e)}")
            raise

# Example usage:
if __name__ == "__main__":
    loader = EnhancedDatasetLoader(verbose=True)
    
    # Load multiple datasets separately
    loader.load_dataset("data1.csv", "sales_data")
    # loader.load_dataset("data2.xlsx", "customer_data", format_type="excel")
    # loader.load_dataset("data3.pickle", "model_data", format_type="pickle")
    
    # # Access individual datasets
    # sales_df = loader.get_dataset("sales_data")
    # customer_df = loader.get_dataset("customer_data")
    
    # # List available datasets
    # print("Available datasets:", loader.list_datasets())
    
    # # Load from clipboard
    # loader.load_dataset(None, "clipboard_data", format_type="clipboard")
    
    # # Load from ZIP archive
    # loader.load_dataset("archive.zip", "archived_data", format_type="zip", 
    #                    file_name="data.csv")
    