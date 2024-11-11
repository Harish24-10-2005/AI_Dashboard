import pandas as pd
import numpy as np
from dotenv import load_dotenv
from DataLoader import EnhancedDatasetLoader


load_dotenv()

loader = EnhancedDatasetLoader(verbose=True)

loader.load_dataset("test_data\iris.json", "titanic_data")

df = loader.get_dataset("titanic_data")





