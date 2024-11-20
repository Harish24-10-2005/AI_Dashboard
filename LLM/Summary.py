import pandas as pd
import streamlit as st
import logging
import cohere
import json
class Summary_overview:
    def __init__(self):
        import os
        api_key = "1TyaPaqTNlXozRCJYWb1RSw30nkPwPqbG8ApPLFr"
        self.co = cohere.ClientV2(api_key)
    def summary_of_data(self,data: pd.DataFrame,dataset_analysis_report):
        status_container = st.empty()
        text_container = st.empty()
        
        # Initialize an empty string to accumulate the response
        full_response = ""
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

        try:
            # Show initial status
            status_container.info("Starting analysis...")
            
            # Make the API call
            response = self.co.chat_stream(
                model="command-r-plus-08-2024",
                messages=[{"role": "user", "content": input_text}]
            )

            # Process the streaming response
            for event in response:
                if event:
                    if event.type == "content-delta":
                        # Debug print
                        print(f"Received delta: {event.delta.message.content.text}")
                        
                        # Append new content to the full response
                        full_response += event.delta.message.content.text
                        
                        # Update the text container
                        text_container.markdown(full_response)
                        
                        # Update status
                        status_container.info("Generating analysis...")
                    
                    elif event.type == "stream-end":
                        # Final update
                        text_container.markdown(full_response)
                        status_container.success("Analysis complete!")
                        return full_response

        except Exception as e:
            error_msg = f"An error occurred: {str(e)}"
            status_container.error(error_msg)
            logging.error(f"Error in summary_of_data: {str(e)}")
            return None

def main():
    st.set_page_config(
        page_title="Data Analysis Summary",
        layout="wide"
    )
    
    st.title("Data Analysis Summary")
    
    # Initialize the Summary_overview class
    summarizer = Summary_overview()
    
    # Example data and analysis report
    data = pd.DataFrame({
        'column1': [1, 2, 3],
        'column2': ['a', 'b', 'c']
    })
    
    dataset_analysis_report = {
        "dataset_info": {
            "total_rows": 3,
            "total_columns": 2,
            "column_types": {
                "column1": "numeric",
                "column2": "categorical"
            }
        }
    }
    
    # Add a debug section
    with st.expander("Debug Info"):
        st.write("Data Preview:")
        st.dataframe(data)
        st.write("Analysis Report:")
        st.json(dataset_analysis_report)
    
    if st.button("Generate Analysis"):
        st.session_state.analysis_result = summarizer.summary_of_data(data, dataset_analysis_report)
        
        # Debug output
        if st.session_state.analysis_result:
            with st.expander("Raw Response"):
                st.text(st.session_state.analysis_result)

if __name__ == "__main__":
    main()