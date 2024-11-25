import pandas as pd
import streamlit as st
import logging
import cohere
import json
import logging
from DataLoader import EnhancedDatasetLoader
import traceback
from LiDA.utils import lida_Model
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from streamlit_card import card
import matplotlib.pyplot as plt
from PIL import Image
import io
import base64
from lida.datamodel import Goal
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64
from datetime import datetime
import json
import math
import tempfile
logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

PRIMARY_COLOR = "#1A73E8"      # Vibrant Blue
SECONDARY_COLOR = "#34A853"    # Fresh Green
BACKGROUND_COLOR = "#F1F3F4"   # Light Gray-Blue

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

def generate_dashboard(data, goals, lida_model, data_summary, selected_library="plotly"):
    """
    Generate a dynamic dashboard with visualization export options and progress animation
    """
    st.write("### Dynamic Dashboard")
    
    # Create a progress container
    progress_container = st.empty()
    status_container = st.empty()
    viz_container = st.empty()
    
    # Initialize progress bar
    progress_bar = progress_container.progress(0)
    status_text = status_container.text("Initializing dashboard...")
    
    try:
        total_goals = len(goals)
        progress_per_goal = 90 / total_goals 
        
        # Generate all visualizations with progress tracking
        all_visualizations = []
        
        for idx, goal in enumerate(goals):
            # Update status
            status_text.text(f"Generating visualizations for goal {idx + 1}/{total_goals}: {goal.question}")
            
            # Generate visualizations for current goal
            try:
                visualizations = lida_model.visualize(
                    summary=data_summary,
                    goal=goal,
                    library=selected_library,
                )
                if visualizations:
                    all_visualizations.extend(visualizations)
                    # Update progress
                    current_progress = min(int((idx + 1) * progress_per_goal), 90)
                    progress_bar.progress(current_progress)
                    time.sleep(0.1)  # Small delay for smooth animation
                    
            except Exception as e:
                st.warning(f"Warning: Failed to generate visualization for goal {idx + 1}: {str(e)}")
                continue
        
        # Final setup phase
        status_text.text("Finalizing dashboard layout...")
        progress_bar.progress(95)
        
        if not all_visualizations:
            progress_container.empty()
            status_container.error("No visualizations could be generated.")
            return
        
        # Export options in sidebar
        with st.sidebar:
            st.write("### Export Options")
            export_col1, export_col2 = st.columns(2)
            with export_col1:
                if st.button("Download All (ZIP)", type="primary"):
                    create_zip_download(all_visualizations)
            with export_col2:
                if st.button("Share Dashboard"):
                    st.text("Dashboard URL copied!")
                    
        # Complete the progress
        progress_bar.progress(100)
        time.sleep(0.5)  # Show completed state briefly
        
        # Clear progress indicators
        progress_container.empty()
        status_container.empty()
        
        # Create dashboard layout
        with viz_container:
            st.write(f"### Showing {len(all_visualizations)} Visualizations")
            
            # Display visualizations in a responsive grid with caching
            @st.cache_data(ttl=3600)  # Cache for 1 hour
            def get_cached_visualization(viz_data, idx):
                return display_visualization(viz_data, idx)
            
            # Create tabs for different view modes
            tab1, tab2 = st.tabs(["Grid View", "Full Width View"])
            
            with tab1:
                # Grid view - 2 columns
                for i in range(0, len(all_visualizations), 2):
                    cols = st.columns(2)
                    
                    # First visualization in row
                    with cols[0]:
                        get_cached_visualization(all_visualizations[i], i)
                    
                    # Second visualization in row (if exists)
                    if i + 1 < len(all_visualizations):
                        with cols[1]:
                            get_cached_visualization(all_visualizations[i + 1], i + 1)
            
            with tab2:
                # Full width view - single column
                for i, viz in enumerate(all_visualizations):
                    get_cached_visualization(viz, i)
                    st.divider()
                    
    except Exception as e:
        st.error(f"Error generating dashboard: {str(e)}")
        if progress_container is not None:
            progress_container.empty()
        if status_container is not None:
            status_container.empty()

def display_visualization(viz, idx):
    """
    Display a single visualization with enhanced controls and error handling
    """
    try:
        # Create container for visualization
        viz_container = st.container()
        
        with viz_container:
            # Title and description
            st.write(f"#### {viz.title if hasattr(viz, 'title') else f'Visualization {idx + 1}'}")
            if hasattr(viz, 'description'):
                st.caption(viz.description)
            
            # Display visualization
            if viz.raster:
                try:
                    # Decode and display image
                    imgdata = base64.b64decode(viz.raster)
                    img = Image.open(io.BytesIO(imgdata))
                    
                    # Add image controls
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
                    with col1:
                        st.image(img, use_column_width=True)
                    
                    # Control buttons
                    with col2:
                        if st.button("View Code", key=f"code_{idx}"):
                            st.code(viz.code, language="python")
                            
                        if st.button("Fullscreen", key=f"fullscreen_{idx}"):
                            st.image(img, use_column_width=True)
                    
                    with col3:
                        # Download options
                        download_format = st.selectbox(
                            "Format",
                            ["PNG", "JPEG", "SVG"],
                            key=f"format_{idx}"
                        )
                        
                        st.download_button(
                            label="Download",
                            data=imgdata,
                            file_name=f"chart_{idx + 1}.{download_format.lower()}",
                            mime=f"image/{download_format.lower()}",
                            key=f"download_{idx}"
                        )
                    
                except Exception as e:
                    st.error(f"Error displaying visualization {idx + 1}: {str(e)}")
            else:
                st.warning("No visualization data available")
    
    except Exception as e:
        st.error(f"Error in visualization component {idx + 1}: {str(e)}")

def create_zip_download(visualizations):
    """
    Create a ZIP file containing all visualizations with enhanced error handling
    """
    try:
        # Show progress for ZIP creation
        with st.status("Creating ZIP file...", expanded=True) as status:
            zip_io = io.BytesIO()
            with zipfile.ZipFile(zip_io, mode='w', compression=zipfile.ZIP_DEFLATED) as zip_file:
                for idx, viz in enumerate(visualizations):
                    status.update(f"Adding visualization {idx + 1}/{len(visualizations)}")
                    
                    if viz.raster:
                        try:
                            # Add visualization image
                            img_data = base64.b64decode(viz.raster)
                            zip_file.writestr(f"visualizations/images/chart_{idx + 1}.png", img_data)
                            
                            # Add code file
                            zip_file.writestr(f"visualizations/code/code_{idx + 1}.py", viz.code)
                            
                            # Add metadata if available
                            if hasattr(viz, 'metadata'):
                                zip_file.writestr(
                                    f"visualizations/metadata/metadata_{idx + 1}.json",
                                    json.dumps(viz.metadata, indent=2)
                                )
                        except Exception as e:
                            st.warning(f"Skipped visualization {idx + 1} due to error: {str(e)}")
                            continue
                
                # Add README
                readme_content = """
                # Dashboard Visualizations Export
                
                This ZIP file contains:
                - /visualizations/images/: PNG images of all visualizations
                - /visualizations/code/: Python source code for each visualization
                - /visualizations/metadata/: Additional metadata and configuration
                
                Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                """.strip()
                
                zip_file.writestr("README.md", readme_content)
            
            # Offer ZIP file for download
            st.download_button(
                label="📥 Download ZIP",
                data=zip_io.getvalue(),
                file_name="dashboard_export.zip",
                mime="application/zip",
                key="download_all"
            )
            
            status.update(label="ZIP file created successfully!", state="complete")
            
    except Exception as e:
        st.error(f"Error creating ZIP file: {str(e)}")

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

def AI_Visualization(data: pd.DataFrame,dataset_analysis_report):
    lida_model = lida_Model()


    # Add title
    st.subheader("AI-Driven Data Analysis Summary", divider=True)

    summary = lida_model.summarize(data)
    goals = lida_model.goals(summary)

    dataset_summary = summary

    for field in dataset_summary["fields"]:
        field_name = field["column"]
        field_properties = field["properties"]
        
        st.subheader(f"Field: {field_name}", divider=True)

        st.write(f"**Data Type:** {field_properties['dtype']}")

        st.write(f"**Number of Unique Values:** {field_properties['num_unique_values']}")

        if field_properties["dtype"] == "number":
            unique_values = field_properties.get("samples", [])
            st.write(f"**Sample Values:** {unique_values}")

            fig = go.Figure([go.Bar(
                x=list(range(len(unique_values))),
                y=unique_values,
                marker=dict(color='blue')
            )])
            fig.update_layout(title=f"{field_name} - Sample Values", xaxis_title="Index", yaxis_title="Value")
            st.plotly_chart(fig)

        elif field_properties["dtype"] == "string":
            unique_values = field_properties.get("samples", [])
            st.write(f"**Sample Values:** {', '.join(unique_values)}")

            value_counts = pd.Series(unique_values).value_counts()
            fig = px.pie(value_counts, values=value_counts.values, names=value_counts.index,
                        title=f"{field_name} - Distribution of Sample Values")
            st.plotly_chart(fig)

    if summary:
        st.sidebar.write("### Goal Selection")

        num_goals = st.sidebar.slider(
            "Number of goals to generate",
            min_value=1,
            max_value=10,
            value=4
        )
        own_goal = st.sidebar.checkbox("Add Your Own Goal")
        st.subheader(f" Key Goals from Analysis ({len(goals)}) Goals", divider=True)
        default_goal = goals[0].question
        goal_questions = [goal.question for goal in goals]

        if own_goal:
            user_goal = st.sidebar.text_input("Describe Your Goal")

            if user_goal:
                new_goal = Goal(question=user_goal, visualization=user_goal, rationale="")
                goals.append(new_goal)
                goal_questions.append(new_goal.question)

        selected_goal = st.selectbox('Choose a generated goal', options=goal_questions, index=0)

        selected_goal_index = goal_questions.index(selected_goal)
        st.subheader(":red[Visualization Details]:")
        st.write(goals[selected_goal_index].visualization)
        st.subheader(":red[Rationale] :")
        st.write(goals[selected_goal_index].rationale)
        selected_goal_object = goals[selected_goal_index]


        if selected_goal_object:
            st.sidebar.write("## Visualization Library")
            visualization_libraries = ["seaborn", "matplotlib", "plotly"]

            selected_library = st.sidebar.selectbox(
                'Choose a visualization library',
                options=visualization_libraries,
                index=0
            )

            st.subheader(" Visualizations", divider=True)

            num_visualizations = st.sidebar.slider(
                "Number of visualizations to generate",
                min_value=1,
                max_value=10,
                value=2)

            visualizations = lida_model.visualize(summary=summary,goal=selected_goal_object,library=selected_library)
            
            if visualizations:
                viz_titles = [f'Visualization {i+1}' for i in range(len(visualizations))]
                
                if None in viz_titles:  # Check for None entries
                    viz_titles = [title for title in viz_titles if title is not None]

                selected_viz_title = st.selectbox('Choose a visualization', options=viz_titles, index=0)

                if selected_viz_title is not None:
                    selected_viz = visualizations[viz_titles.index(selected_viz_title)]

                    if selected_viz.raster:
                        imgdata = base64.b64decode(selected_viz.raster)
                        img = Image.open(io.BytesIO(imgdata))
                        st.image(img, caption=selected_viz_title, use_column_width=True)
                    if st.button("Show Code", type="secondary"):
                        st.write("### Visualization Code")
                        st.code(selected_viz.code)
            else:
                st.error("No visualizations found. Please ensure that your data and goal have valid results.")

    st.subheader("Interactive Data Visualizations", divider=True)
    generate_dashboard(data, goals, lida_model, summary, selected_library="plotly")
    

