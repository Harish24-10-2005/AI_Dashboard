import streamlit as st
import pandas as pd
import numpy as np
import relation
import plotly.express as px
import plotly.graph_objs as go
from dotenv import load_dotenv
import logging
import traceback
import cohere
import os

# Configuration
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Initialize Cohere Client
Cohere_API_KEY = os.getenv('cohere_api_key')
co = cohere.ClientV2(Cohere_API_KEY)

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

def home_page():
    """Landing page for the application"""
    st.title("🚀 Smart Dataset Merger and Analyzer")
    st.markdown("""
    ## Welcome to Your Data Intelligence Platform
    
    This application helps you:
    - 📊 Upload and merge multiple datasets
    - 🔍 Analyze dataset relationships
    - 💡 Generate AI-powered insights
    
    Get started by navigating through the pages in the sidebar!
    """)
    
    st.image("/api/placeholder/800/400", caption="Data Analysis Dashboard")

def dataset_upload_page():
    """Page for dataset configuration and upload"""
    st.title("📤 Dataset Configuration")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload Datasets", 
        accept_multiple_files=True, 
        type=['csv', 'xlsx', 'json']
    )
    
    # Dataset preview section
    if uploaded_files:
        st.subheader("Uploaded Datasets Preview")
        for file in uploaded_files:
            try:
                # Determine file type and read accordingly
                if file.name.endswith('.csv'):
                    df = pd.read_csv(file)
                elif file.name.endswith('.xlsx'):
                    df = pd.read_excel(file)
                elif file.name.endswith('.json'):
                    df = pd.read_json(file)
                
                st.write(f"Dataset: {file.name}")
                st.dataframe(df.head())
                st.write(f"Shape: {df.shape}")
            except Exception as e:
                st.error(f"Error processing {file.name}: {e}")
    
    # Advanced settings
    st.sidebar.header("Analysis Settings")
    min_similarity = st.sidebar.slider(
        "Minimum Relationship Similarity", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.75,
        help="Threshold for detecting dataset relationships"
    )
    use_ai_analysis = st.sidebar.checkbox("Enable AI-Powered Analysis", value=True)
    
    # Analyze button
    if st.button("Prepare Datasets for Analysis"):
        # Store datasets in session state for next page
        st.session_state.uploaded_files = uploaded_files
        st.session_state.min_similarity = min_similarity
        st.session_state.use_ai_analysis = use_ai_analysis
        st.success("Datasets prepared! Navigate to Analysis Page.")

def dataset_analysis_page():
    """Page for dataset analysis and merging"""
    st.title("🔬 Dataset Analysis")
    
    # Check if datasets are available
    if 'uploaded_files' not in st.session_state or not st.session_state.uploaded_files:
        st.warning("Please upload datasets first in the Dataset Configuration page.")
        return
    
    # Load datasets
    dataframes = {}
    for file in st.session_state.uploaded_files:
        try:
            if file.name.endswith('.csv'):
                df = pd.read_csv(file)
            elif file.name.endswith('.xlsx'):
                df = pd.read_excel(file)
            elif file.name.endswith('.json'):
                df = pd.read_json(file)
            
            dataframes[file.name] = df
        except Exception as e:
            st.error(f"Error loading {file.name}: {e}")
    
    # Perform dataset analysis
    if st.button("Merge and Analyze Datasets"):
        with st.spinner("Analyzing datasets..."):
            try:
                results, report = relation.analyze_and_merge_datasets(
                    dataframes,
                    use_llm=st.session_state.use_ai_analysis,
                    min_similarity=st.session_state.min_similarity
                )
                
                # Display analysis results
                st.success("Analysis Completed Successfully!")
                
                # Merged Datasets
                st.subheader("Merged Datasets")
                for name, df in results.items():
                    st.write(f"Dataset: {name}")
                    st.dataframe(df)
                    st.write(f"Shape: {df.shape}")
                
                # Store results for insights page
                st.session_state.analysis_results = results
                st.session_state.analysis_report = report
                
            except Exception as e:
                st.error(f"Analysis Error: {e}")
                logger.error(traceback.format_exc())

def insights_page():
    """Page for generating AI-powered insights"""
    st.title("💡 Data Insights")
    
    # Check if analysis results are available
    if 'analysis_results' not in st.session_state:
        st.warning("Please complete dataset analysis first.")
        return
    
    # Display relationship visualization
    st.subheader("Dataset Relationship Visualization")
    relationships = st.session_state.analysis_report.get("detected_relationships", [])
    relationship_df = pd.DataFrame(relationships)
    
    fig = px.bar(
        relationship_df, 
        x="relationship", 
        y="similarity", 
        title="Dataset Relationship Similarities",
        color="confidence"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # AI-Generated Insights
    st.subheader("AI-Powered Analysis")
    if st.button("Generate Insights"):
        with st.spinner("Generating insights..."):
            try:
                # Use Cohere for generating insights
                response = co.chat(
                    model="command-r-plus-08-2024",
                    message=f"Provide deep insights for these datasets: {st.session_state.analysis_report}"
                )
                st.markdown(response.text)
            except Exception as e:
                st.error(f"Insight Generation Error: {e}")

def main():
    st.set_page_config(
        page_title="Smart Dataset Analyzer", 
        page_icon="📊", 
        layout="wide"
    )
    
    apply_custom_styling()
    
    # Page selection
    page = st.sidebar.radio("Navigate", [
        "Home", 
        "Dataset Configuration", 
        "Dataset Analysis", 
        "Insights"
    ])
    
    # Page routing
    if page == "Home":
        home_page()
    elif page == "Dataset Configuration":
        dataset_upload_page()
    elif page == "Dataset Analysis":
        dataset_analysis_page()
    elif page == "Insights":
        insights_page()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("Powered by AI and Data Science 🧠💡")

if __name__ == "__main__":
    main()