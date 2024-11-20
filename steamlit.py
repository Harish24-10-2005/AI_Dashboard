import streamlit as st
import pandas as pd
import numpy as np
import relation
from dotenv import load_dotenv
from DataLoader import EnhancedDatasetLoader
from relation import SmartDatasetAnalyzer
import json
from Ai_decision import AIDecisionMaker
import logging
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

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

def main():
    st.title("Dataset Merger and Analyzer")

    # Sidebar for dataset selection
    st.sidebar.header("Dataset Configuration")
    
    # Default dataset paths
    default_datasets = {
        "employee_projects": "relation_data\employee_projects.json",
        "employees": "relation_data\employees.csv",
        "departments": "relation_data\departments.csv",
        "datasets": "relation_data\datasets.xlsx"
    }
    
    datasets = get_datasets_input()
    use_llm = st.sidebar.checkbox("Use AI Analysis", value=True)
    min_similarity = st.sidebar.slider(
        "Minimum Similarity Threshold", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.75
    )

    # Perform analysis button
    if st.sidebar.button("Analyze Datasets"):
        try:
            # Load datasets
            dataFrame = load_datasets(datasets)
            if not dataFrame:
                st.error("No datasets loaded. Please check your files.")
                return

            st.subheader("Input Datasets")
            for name, df in dataFrame.items():
                st.write(f"Dataset: {name}")
                st.dataframe(df.head())
            st.write(dataFrame)
            results,report = relation.analyze_and_merge_datasets(
                dataFrame,
                output_dir='analysis_results',
                use_llm=True,
                min_similarity=0.75,
                save_format='csv',
                cache_dir='analysis_cache'
            )

            if results:
                st.success("Analysis Completed Successfully!")
                st.subheader("Merged Datasets")
                for name, df in results.items():
                    st.write(f"Dataset: {name}")
                    st.dataframe(df)
                    st.write(f"Shape: {df.shape}")

                # Display analysis report
                st.subheader("Analysis Report")
                st.title(report["title"])
                st.write(f"**Generated On:** {report['generated_on']}")

                # Original Datasets Section
                st.header("Original Datasets")
                for dataset in report["original_datasets"]:
                    st.subheader(dataset["name"])
                    st.write(f"**Rows:** {dataset['rows']}, **Columns:** {dataset['columns']}")
                    st.write("**Column Types:**")
                    for col, col_type in dataset["column_types"].items():
                        st.write(f"- `{col}`: {col_type}")

                # Detected Relationships Section
                st.header("Detected Relationships")
                relationship_data = []
                for rel in report["detected_relationships"]:
                    relationship_data.append([
                        rel["relationship"],
                        rel["similarity"],
                        rel["confidence"],
                        rel["cardinality"]
                    ])
                
                st.write(pd.DataFrame(relationship_data, columns=["Relationship", "Similarity", "Confidence", "Cardinality"]))

                # Merged Results Section
                st.header("Merged Results")
                for merged in report["merged_results"]:
                    st.subheader(merged["name"])
                    st.write(f"**Rows:** {merged['rows']}, **Columns:** {merged['columns']}")

                # Visualization options
                st.subheader("Analysis Visualization")
                
                # Relationship graph
                if st.checkbox("Show Relationship Graph"):
                    try:
                        import networkx as nx
                        import matplotlib.pyplot as plt
                        
                        G = nx.DiGraph()
                        for rel in report.get('detected_relationships', []):
                            source, target = rel['relationship'].split('<-->')
                            G.add_edge(source.strip(), target.strip())
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        nx.draw(G, with_labels=True, node_color='lightblue', 
                                node_size=1500, font_size=10, 
                                font_weight='bold', ax=ax)
                        st.pyplot(fig)
                    except Exception as viz_error:
                        st.error(f"Visualization error: {viz_error}")

            else:
                st.warning("No merged datasets produced.")

        except Exception as e:
            st.error(f"Analysis Error: {e}")
            st.error(traceback.format_exc())
            logger.error(f"Analysis error: {traceback.format_exc()}")

if __name__ == "__main__":
    load_dotenv()
    main()