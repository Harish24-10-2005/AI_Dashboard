import pandas as pd
import numpy as np
import relation
from dotenv import load_dotenv
from DataLoader import EnhancedDatasetLoader
from relation import SmartDatasetAnalyzer
import json
from Ai_decision import AIDecisionMaker

load_dotenv()

loader = EnhancedDatasetLoader(verbose=True)
Relation_Analyzer = SmartDatasetAnalyzer()
LLM = AIDecisionMaker('cohere_api_key')
dataFrame = {}
datasets = {
    "employee_projects": "relation_data\employee_projects.json",
    "employees": "relation_data\employees.csv",
    "departments": "relation_data\departments.csv",
    "datasets": "relation_data\datasets.xlsx"
    }

if len(datasets) > 1:
    for key,value in datasets.items():
        loader.load_dataset(value, key)
        df = loader.get_dataset(key)
        print(df.head())
        dataFrame[key] = df

    results,report = relation.analyze_and_merge_datasets(
        dataFrame,
        output_dir='analysis_results',
        use_llm=True,
        min_similarity=0.75,
        save_format='csv',
        cache_dir='analysis_cache'
    )

    print(json.dumps(report, indent=2)) 
    if results:
        print("\nAnalysis completed successfully!")
        print("\nFinal Datasets:")
        for name, df in results.items():
            print(f"\n{name}:")
            print(df.head())
            print(f"Shape: {df.shape}")
            print("Columns:", df.columns.tolist())
    LLM.summary_of_data(results,report)

else:
    for key,value in datasets.items():
        loader.load_dataset(value, key)
        df = loader.get_dataset(key)
        print(df.head())
        dataFrame[key] = df

# loader.load_dataset("steve1215rogg/student-lifestyle-dataset", "titanic_data")

# df = loader.get_dataset("titanic_data")

# print(df.head())





