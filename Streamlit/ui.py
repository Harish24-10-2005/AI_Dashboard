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
import time
import os

# Logging Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Load Environment Variables
load_dotenv()

# API Configuration
Cohere_API_KEY = os.getenv('cohere_api_key')
co = None
if Cohere_API_KEY:
    try:
        co = cohere.ClientV2(Cohere_API_KEY)
    except Exception as e:
        logger.warning(f"Failed to initialize Cohere client: {e}")

# Styling Constants
PRIMARY_COLOR = "#1A73E8"      # Vibrant Blue
SECONDARY_COLOR = "#34A853"    # Fresh Green
BACKGROUND_COLOR = "#F1F3F4"   # Light Gray-Blue

def apply_custom_styling():
    """Apply custom CSS styling to Streamlit app"""
    st.markdown(f"""
    <style>
    .reportview-container {{
        background-color: {BACKGROUND_COLOR};
    }}
    .sidebar .sidebar-content {{
        background-color: {PRIMARY_COLOR};
        color: white;
    }}
    .stButton>button {{
        background-color: {SECONDARY_COLOR};
        color: white;
        border-radius: 10px;
    }}
    .stExpander {{
        border-radius: 10px;
        border: 1px solid {SECONDARY_COLOR};
    }}
    h1, h2, h3, h4 {{
        color: {PRIMARY_COLOR};
    }}
    </style>
    """, unsafe_allow_html=True)

def stream_ai_analysis(data, report):
    """Stream AI-powered data analysis"""
    status_container = st.empty()
    text_container = st.empty()
    
    full_response = ""
    input_text = input_prompt_summary(data, report)
    
    try:
        if co is None:
            status_container.warning("Cohere API key not configured. Skipping AI summary.")
            return None
        status_container.info("Starting AI-powered analysis...")
        
        response = co.chat_stream(
            model="command-r-plus-08-2024",
            messages=[{"role": "user", "content": input_text}]
        )
        
        status_container.info("🔍 Generating intelligent insights...")
        
        for event in response:
            if event:
                if event.type == "content-delta":
                    full_response += event.delta.message.content.text
                    text_container.markdown(f"**AI Insights:** {full_response}")
                    status_container.info("Generating analysis...")
                
                elif event.type == "stream-end":
                    text_container.markdown(full_response)
                    status_container.success("✅ Analysis Complete!")
                    return full_response

    except Exception as e:
        error_msg = f"❌ Analysis Error: {str(e)}"
        status_container.error(error_msg)
        logging.error(f"Error in AI analysis: {str(e)}")
        return None

def get_dataset_overview(data, dataset_analysis_report):
    """Generate comprehensive dataset overview"""
    st.title("Dataset Analysis Overview")
    
    # Debug Information Expander
    with st.expander("Dataset Details"):
        st.write("Data Preview:")
        st.dataframe(data.head())
        
        st.write("Dataset Statistics:")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Rows", data.shape[0])
            st.metric("Total Columns", data.shape[1])
        
        with col2:
            st.metric("Memory Usage", f"{data.memory_usage().sum() / 1024**2:.2f} MB")
            st.metric("Data Types", ", ".join(data.dtypes.unique().astype(str)))
    
    # AI-Powered Insights
    st.subheader("🤖 AI-Powered Insights")
    ai_analysis = stream_ai_analysis(data, dataset_analysis_report)
    
    # Statistical Summary
    st.subheader("Statistical Summary")
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    
    if len(numeric_columns) > 0:
        summary_stats = data[numeric_columns].describe()
        st.dataframe(summary_stats)
    
    return ai_analysis

def main():
    # Streamlit Page Configuration
    st.set_page_config(
        page_title="Smart Dataset Analyzer", 
        page_icon="📊", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply Custom Styling
    apply_custom_styling()

    # Initialize Session State
    if 'analysis_results' not in st.session_state:
        st.session_state.update({
            'analysis_results': None,
            'analysis_report': None,
            'selected_dataset': None,
            'dataset_overviews': {}
        })

    # Main Title
    st.title("🚀 Smart Dataset Merger and Analyzer")
    st.markdown("Intelligently merge, analyze, and gain insights from multiple datasets.")

    # Sidebar Configuration
    datasets = get_datasets_input()
    use_llm = st.sidebar.checkbox("Use AI Analysis", value=True)
    min_similarity = st.sidebar.slider(
        "Minimum Similarity Threshold", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.75
    )

    # Analysis Button
    if st.button("Analyze Datasets", type="primary"):
        try:
            # Load Datasets
            dataFrame = load_datasets(datasets)
            if not dataFrame:
                st.error("No datasets loaded. Please check your files.")
                return

            # Display Input Datasets
            st.subheader("Input Datasets")
            for name, df in dataFrame.items():
                st.write(f"Dataset: {name}")
                st.dataframe(df.head())
            
            # Perform Dataset Analysis
            results, report = relation.analyze_and_merge_datasets(
                dataFrame,
                output_dir='analysis_results',
                use_llm=use_llm,
                min_similarity=min_similarity,
                save_format='csv',
                cache_dir='analysis_cache'
            )

            # Store Results in Session State
            st.session_state.analysis_results = results
            st.session_state.analysis_report = report

        except Exception as e:
            st.error(f"Analysis Error: {e}")
            st.error(traceback.format_exc())
            logger.error(f"Analysis error: {traceback.format_exc()}")

    # Display Analysis Results
    if st.session_state.analysis_results:
        results = st.session_state.analysis_results
        report = st.session_state.analysis_report

        st.subheader("Merged Datasets")
        for name, df in results.items():
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.write(f"Dataset: {name}")
                st.dataframe(df.head())
                st.write(f"Shape: {df.shape}")
            
            with col2:
                if st.button(f"Analyze {name}", key=f"analyze_button_{name}"):
                    # Generate or retrieve dataset overview
                    if name not in st.session_state.dataset_overviews:
                        overview = get_dataset_overview(df, report)
                        st.session_state.dataset_overviews[name] = overview
                    else:
                        st.write(st.session_state.dataset_overviews[name])

    st.markdown("---")
    st.markdown("Powered by AI and Data Science 🧠💡")

if __name__ == "__main__":
    load_dotenv()
    main()