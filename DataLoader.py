import pandas as pd
import os
import requests
import logging
import tempfile
import yaml
import sqlite3  # Import for SQL functionality
import gdown  # Import for Google Drive functionality
from dotenv import load_dotenv
from kaggle.api.kaggle_api_extended import KaggleApi
from github import Github
import boto3
import zipfile
from io import StringIO, BytesIO
import pickle
import h5py
import xml.etree.ElementTree as ET
import time  # Import time for timing dataset loading

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
            'pickle': self._load_pickle,
            'parquet': self._load_parquet,
            'hdf5': self._load_hdf5,
            'xml': self._load_xml,
            'txt': self._load_txt,
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

    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO if self.verbose else logging.WARNING,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def _load_single_source(self, source: str, format_type: str = None, **kwargs) -> pd.DataFrame:
        """
        Determine and load data source based on file extension or specified format.
        
        Args:
            source: Path or source of the data
            format_type: Explicitly specified format
            **kwargs: Additional loading arguments
        
        Returns:
            Loaded DataFrame
        """
        try:
            # If format_type is explicitly provided, use it
            if format_type and format_type.lower() in self.supported_formats:
                return self.supported_formats[format_type.lower()](source, **kwargs)
            
            # Handle URL directly
            if source.startswith(('http://', 'https://')):
                return self._load_url(source, **kwargs)

            # Handle special source types
            if source.lower().startswith(('kaggle:', 'github:', 'gdrive:', 's3:')):
                prefix, path = source.split(':', 1)
                if prefix.lower() == 'kaggle':
                    return self._load_kaggle(path, **kwargs)
                elif prefix.lower() == 'github':
                    repo_path, file_path = path.split('/', 1)
                    return self._load_github(repo_path, file_path, **kwargs)
                elif prefix.lower() == 'gdrive':
                    return self._load_google_drive(path, **kwargs)
                elif prefix.lower() == 's3':
                    bucket, key = path.split('/', 1)
                    return self._load_s3(bucket, key, **kwargs)

            # Ensure source is a string path
            source = str(source)
            
            # Normalize path to handle different OS path separators
            source = os.path.normpath(source)
            
            # Infer format from file extension
            file_ext = os.path.splitext(source)[1].lower().lstrip('.')
            if file_ext in self.supported_formats:
                return self.supported_formats[file_ext](source, **kwargs)

            raise ValueError("Unsupported format or source type.")
        
        except Exception as e:
            self.logger.error(f"Error loading source {source}: {str(e)}")
            raise

    def _load_csv(self, path: str, **kwargs) -> pd.DataFrame:
        try:
            return pd.read_csv(path, **kwargs)
        except Exception as e:
            self.logger.error(f"Error loading CSV file {path}: {str(e)}")
            raise

    def _load_excel(self, path: str, **kwargs) -> pd.DataFrame:
        try:
            return pd.read_excel(path, **kwargs)
        except Exception as e:
            self.logger.error(f"Error loading Excel file {path}: {str(e)}")
            raise

    def _load_json(self, path: str, **kwargs) -> pd.DataFrame:
        try:
            return pd.read_json(path, **kwargs)
        except Exception as e:
            self.logger.error(f"Error loading JSON file {path}: {str(e)}")
            raise

    def _load_sql(self, query: str, **kwargs) -> pd.DataFrame:
        try:
            conn = sqlite3.connect(kwargs.get('database'))
            return pd.read_sql_query(query, conn)
        except Exception as e:
            self.logger.error(f"Error loading SQL data: {str(e)}")
            raise
        finally:
            conn.close()

    def _load_kaggle(self, dataset_name: str, **kwargs) -> pd.DataFrame:
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
        try:
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                gdown.download(f"https://drive.google.com/uc?id={file_id}", tmp_file.name, quiet=False)
                return self._load_single_source(tmp_file .name, **kwargs)
        except Exception as e:
            self.logger.error(f"Error loading Google Drive dataset: {str(e)}")
            raise
        finally:
            if 'tmp_file' in locals():
                os.unlink(tmp_file.name)

    def _load_url(self, url: str, **kwargs) -> pd.DataFrame:
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
        except requests.exceptions.RequestException as e:
            self.logger.error(f"HTTP error while loading URL: {str(e)}")
            raise

    def _load_s3(self, bucket: str, key: str, **kwargs) -> pd.DataFrame:
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

    def _load_zip(self, path: str, file_name: str = None, **kwargs) -> pd.DataFrame:
        try:
            with zipfile.ZipFile(path, 'r') as z:
                if file_name is None:
                    file_name = z.namelist()[0]
                with z.open(file_name) as f:
                    if file_name.endswith('.csv'):
                        return pd.read_csv(f, **kwargs)
                    elif file_name.endswith(('.xlsx', '.xls')):
                        return pd.read_excel(f, **kwargs)
                    else:
                        raise ValueError(f"Unsupported file format in ZIP: {file_name}")
        except Exception as e:
            self.logger.error(f"Error loading from ZIP {path}: {str(e)}")
            raise

    def _load_pickle(self, path: str, **kwargs) -> pd.DataFrame:
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            self.logger.error(f"Error loading pickle file {path}: {str(e)}")
            raise

    def _load_parquet(self, path: str, **kwargs) -> pd.DataFrame:
        try:
            return pd.read_parquet(path, **kwargs)
        except Exception as e:
            self.logger.error(f"Error loading parquet file {path}: {str(e)}")
            raise

    def _load_hdf5(self, path: str, key: str = 'data', **kwargs) -> pd.DataFrame:
        try:
            return pd.read_hdf(path, key=key, **kwargs)
        except Exception as e:
            self.logger.error(f"Error loading HDF5 file {path}: {str(e)}")
            raise

    def _load_xml(self, path: str, **kwargs) -> pd.DataFrame:
        try:
            tree = ET.parse(path)
            root = tree.getroot()
            data = []
            for child in root:
                data.append({elem.tag: elem.text for elem in child})
            return pd.DataFrame(data)
        except Exception as e:
            self.logger.error(f"Error loading XML file {path}: {str(e)}")
            raise

    def _load_txt(self, path: str, **kwargs) -> pd.DataFrame:
        try:
            return pd.read_csv(path, delimiter=kwargs.get('delimiter', '\t'), **kwargs)
        except Exception as e:
            self.logger.error(f"Error loading text file {path}: {str(e)}")
            raise

    def _load_web_table(self, url: str, table_index: int = 0, **kwargs) -> pd.DataFrame:
        try:
            tables = pd.read_html(url, **kwargs)
            if tables and len(tables) > table_index:
                return tables[table_index]
            raise ValueError(f"No table found at index {table_index}")
        except Exception as e:
            self.logger.error(f"Error loading web table: {str(e)}")
            raise

    def load_dataset(self, source: str, name: str, format_type: str = None, **kwargs) -> pd.DataFrame:
        try:
            start_time = time.time()
            df = self._load_single_source(source, format_type, **kwargs)
            self.dataframes[name] = df
            elapsed_time = time.time()
            self.logger.info(f"Loaded dataset '{name}' in {elapsed_time:.2f} seconds")
            return df
        except Exception as e:
            self.logger.error(f"Error loading dataset '{name}': {str(e)}")
            raise

    def get_dataset(self, name: str) -> pd.DataFrame:
        if name in self.dataframes:
            return self.dataframes[name]
        raise KeyError(f"Dataset '{name}' not found")

    def list_datasets(self) -> list:
        return list(self.dataframes.keys())

    def remove_dataset(self, name: str):
        if name in self.dataframes:
            del self.dataframes[name]
            self.logger.info(f"Removed dataset '{name}'")

if __name__ == "__main__":
    try:
        loader = EnhancedDatasetLoader(verbose=True)
        loader.load_dataset("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv", "titanic_data")
        titanic_df = loader.get_dataset("titanic_data")
        print(titanic_df.head())
        print("Available datasets:", loader.list_datasets())
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()