import pandas as pd
import streamlit as st
import logging
import cohere
import json
import logging
from DataLoader import EnhancedDatasetLoader
import traceback
logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def input_prompt_summary(data: pd.DataFrame,dataset_analysis_report):
        input_text = f"""
        You are an expert AI data scientist and strategist specializing in advanced data analysis, relational database exploration, and generating innovative data science project ideas. Your task is to deeply analyze a dataset summary, understand relationships between datasets, and suggest actionable insights, innovative data science applications, and potential improvements.

        Below is the dataset analysis report:

        {json.dumps(dataset_analysis_report, indent=4)}
        {data.head().to_string()}
        Summary:
        - Provide a concise overview of the datasets, their structure, and the detected relationships.
        - Highlight the strengths or unique characteristics of the data.

        Insights and Opportunities:
        - Suggest key patterns, trends, or insights that could be extracted from the data.
        - Propose hypotheses or questions the data could answer.

        Data Science Applications:
        - Recommend innovative and impactful data science projects based on the datasets and their relationships.
        - For each suggestion, briefly explain its value and potential real-world applications.

        Enhancements:
        - Identify any missing data, potential biases, or limitations in the datasets.
        - Suggest additional data or relationships that could improve the quality of analysis.

        Technical Plan:
        - Propose specific machine learning models, algorithms, or techniques that can be used for the suggested projects.
        - Include tools or frameworks suitable for implementing the ideas.

        Phase 1: Preparation (Before Building the Dashboard)
        Step 1: Define Dashboard Objectives
        - Activity: Clearly outline what insights the dashboard should provide.
        - Output: Written objectives (e.g., "Track employee productivity", "Analyze project allocation by department").

        Step 2: Identify Target Audience
        - Activity: Determine who will use the dashboard.
        - Output: List of user roles (e.g., managers, analysts, HR personnel) with their expected interactions.

        Step 3: Data Preparation
        - Activity: Perform exploratory data analysis (EDA) on the datasets. Clean and preprocess the data (handle missing values, outliers, etc.). Merge datasets based on the identified relationships. Ensure data is in a suitable format for analysis.
        - Tools: Python (Pandas, NumPy), R, SQL, or data preprocessing tools like OpenRefine.
        - Output: Prepared, unified dataset(s).
        """
        return input_text

def load_datasets(datasets):
    """
    Safely load datasets with error handling
    """
    dataFrame = {}
    loader = EnhancedDatasetLoader(verbose=True)
    
    for key, value in datasets.items():
        try:
            loader.load_dataset(value, key)
            df = loader.get_dataset(key)
            dataFrame[key] = df
        except Exception as e:
            st.error(f"Error loading dataset {key}: {e}")
            logger.error(f"Dataset loading error: {traceback.format_exc()}")
    
    return dataFrame

def get_datasets_input():
    """
    Collect dataset information from user in a dynamic and flexible manner
    """
    # Number of datasets to input
    num_datasets = st.sidebar.number_input(
        "How many datasets do you want to analyze?", 
        min_value=1, 
        max_value=10, 
        value=2,
        key="num_datasets_input"
    )
    
    # Dictionary to store dataset information
    datasets = {}
    
    # Input fields for each dataset
    for i in range(num_datasets):
        st.sidebar.subheader(f"Dataset {i+1}")
        
        dataset_name = st.sidebar.text_input(
            f"Enter name for Dataset {i+1}", 
            placeholder="e.g., employees",
            key=f"dataset_name_input_{i}"
        )
        dataset_path = st.sidebar.text_input(
            f"Enter full path for {dataset_name or f'Dataset{i+1}'}", 
            placeholder="C:/data/employees.csv",
            key=f"dataset_path_direct_input_{i}"
        )
        if dataset_name and dataset_path:
            if '\\\\'in dataset_path and 'https:' not in dataset_path and 'http:' not in dataset_path:
                dataset_path = dataset_path.replace('\\\\', '\\')
        datasets[dataset_name] = dataset_path
    
    return datasets