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
from LLM.Summary import Summary_overview
from Streamlit.Utils import input_prompt_summary, load_datasets, get_datasets_input
import cohere
from dotenv import load_dotenv
import time
import os
load_dotenv()

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

Cohere_API_KEY = os.getenv('cohere_api_key')
co = cohere.ClientV2(Cohere_API_KEY)

def summary_of_data(data, report):
    status_container = st.empty()
    text_container = st.empty()
    
    full_response = ""
    input_text = input_prompt_summary(data,report)
    try:
        status_container.info("Starting analysis...")
        
        response = co.chat_stream(
            model="command-r-plus-08-2024",
            messages=[{"role": "user", "content": input_text}]
        )
        for event in response:
            if event:
                if event.type == "content-delta":
                    print(f"Received delta: {event.delta.message.content.text}")
                    
                    full_response += event.delta.message.content.text
                    
                    text_container.markdown(full_response)
                    status_container.info("Generating analysis...")
                
                elif event.type == "stream-end":
                    text_container.markdown(full_response)
                    status_container.success("Analysis complete!")
                    return full_response

    except Exception as e:
        error_msg = f"An error occurred: {str(e)}"
        status_container.error(error_msg)
        logging.error(f"Error in summary_of_data: {str(e)}")
        return None


def get_data_overview(data,dataset_analysis_report):
    st.title("Data Analysis Summary")
    
    with st.expander("Debug Info"):
        st.write("Data Preview:")
        st.dataframe(data)
        st.write("Analysis Report:")
        st.json(dataset_analysis_report)

    st.session_state.analysis_result = summary_of_data(data, dataset_analysis_report)
    if st.session_state.analysis_result:
        with st.expander("Raw Response"):
            st.text(st.session_state.analysis_result)

def main():
    st.set_page_config(page_title="Dataset Merger and Analyzer", layout="wide")  

    st.sidebar.header("Dataset Configuration")
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

                st.subheader("Analysis Report")
                st.title(report["title"])
                st.write(f"**Generated On:** {report['generated_on']}")

                st.header("Original Datasets")
                for dataset in report["original_datasets"]:
                    st.subheader(dataset["name"])
                    st.write(f"**Rows:** {dataset['rows']}, **Columns:** {dataset['columns']}")
                    st.write("**Column Types:**")
                    for col, col_type in dataset["column_types"].items():
                        st.write(f"- `{col}`: {col_type}")

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
                st.header("Merged Results")
                for merged in report["merged_results"]:
                    st.subheader(merged["name"])
                    st.write(f"**Rows:** {merged['rows']}, **Columns:** {merged['columns']}")

                st.subheader("Dataset Overview")
                for name, df in results.items():
                    st.write(f"Dataset: {name}")
                    result = get_data_overview(df,report)
                    if result:
                        st.write("Analysis completed successfully!")
                                

            else:
                st.warning("No merged datasets produced.")

        except Exception as e:
            st.error(f"Analysis Error: {e}")
            st.error(traceback.format_exc())
            logger.error(f"Analysis error: {traceback.format_exc()}")

if __name__ == "__main__":
    load_dotenv()
    main()