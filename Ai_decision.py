import os
import json
import logging
from typing import Dict

import pandas as pd
import streamlit as st
import torch
import cohere
from transformers import AutoTokenizer, AutoModel, RobertaTokenizer, RobertaModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from scipy import stats
import category_encoders as ce
from pandas.api.types import is_numeric_dtype, is_string_dtype, is_categorical_dtype
class AIDecisionMaker:
    def __init__(self, cohere_api_key: str | None = None):
        # Prefer env var if explicit key not provided
        api_key = cohere_api_key or os.getenv("cohere_api_key")
        self.co = None
        if api_key:
            try:
                self.co = cohere.ClientV2(api_key)
            except Exception as e:
                logging.warning(f"Failed to initialize Cohere client: {e}")

    def analyze_error(self, error_message: str, code_context: str) -> Dict[str, str]:
        """Analyze errors using CodeBERT and Cohere for intelligent error resolution."""
        try:
            # Get CodeBERT embeddings for error context if model/tokenizer are available
            code_embeddings = None
            if hasattr(self, "tokenizer") and hasattr(self, "model") and self.tokenizer and self.model:
                inputs = self.tokenizer(code_context, return_tensors="pt", truncation=True, max_length=512)
                with torch.no_grad():
                    code_embeddings = self.model(**inputs).last_hidden_state.mean(dim=1)
            
            # Generate error analysis using Cohere
            prompt = f"""
            Analyze the following error in the data preprocessing context:
            Error: {error_message}
            Code Context: {code_context}
            
            Provide:
            1. Root cause analysis
            2. Suggested fixes
            3. Prevention strategies
            """
            
            if self.co is None:
                raise RuntimeError("Cohere API key not configured")
            response = self.co.generate(
                prompt=prompt,
                max_tokens=500,
                temperature=0.7,
                k=0,
                model='command'
            )
            
            analysis = response.generations[0].text if hasattr(response, "generations") else ""
            
            return {
                'error_type': self._classify_error(error_message),
                'analysis': analysis,
                'embeddings': code_embeddings.numpy().tolist() if code_embeddings is not None else []
            }
        except Exception as e:
            logging.error(f"Error in AI analysis: {str(e)}")
            return {'error_type': 'unknown', 'analysis': str(e)}
            
    def suggest_preprocessing(self, data_sample: pd.DataFrame) -> Dict[str, any]:
        try:
            profile = self._generate_data_profile(data_sample)
    
            prompt = f"""
            Analyze the following dataset profile and suggest optimal preprocessing steps:
            {json.dumps(profile, indent=2)}
            
            Consider:
            1. Data types and distributions
            2. Missing values patterns
            3. Outlier detection strategy
            4. Feature engineering opportunities
            """
            
            if self.co is None:
                raise RuntimeError("Cohere API key not configured")
            response = self.co.generate(
                prompt=prompt,
                max_tokens=800,
                temperature=0.7,
                model='command'
            )
            
            recommendations = response.generations[0].text
            
            return {
                'profile': profile,
                'recommendations': recommendations,
                'suggested_config': self._generate_config(profile, recommendations)
            }
        except Exception as e:
            logging.error(f"Error in preprocessing suggestions: {str(e)}")
            return {}
        
    def _generate_data_profile(self, data: pd.DataFrame) -> Dict[str, any]:
        """Generate comprehensive data profile for AI analysis."""
        profile = {
            'shape': data.shape,
            'dtypes': data.dtypes.astype(str).to_dict(),
            'missing_stats': data.isnull().sum().to_dict(),
            'numeric_stats': {},
            'categorical_stats': {},
            'correlation_matrix': None
        }
        
        for col in data.columns:
            if is_numeric_dtype(data[col]):
                profile['numeric_stats'][col] = {
                    'mean': float(data[col].mean()),
                    'std': float(data[col].std()),
                    'skew': float(stats.skew(data[col].dropna())),
                    'unique_count': int(data[col].nunique())
                }
            elif is_categorical_dtype(data[col]) or is_string_dtype(data[col]):
                profile['categorical_stats'][col] = {
                    'unique_count': int(data[col].nunique()),
                    'top_categories': data[col].value_counts().head().to_dict()
                }
                
        numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_cols) > 0:
            profile['correlation_matrix'] = data[numeric_cols].corr().to_dict()
            
        return profile
    
    def create_code(self, data: pd.DataFrame,dataset_analysis_report):
        max_attempts = 5
        current_attempt = 0
        full_generated_code = ""

        input_text = f"""
        You are an expert Data Science AI specialized in Exploratory Data Analysis (EDA). Your task is to generate a comprehensive Python script for performing end-to-end EDA on a given DataFrame and provide full code.
        Your previous code generation was {'incomplete' if current_attempt > 0 else 'started'}.
        Below is the dataset analysis report:
        Dataset Analysis Report:{json.dumps(dataset_analysis_report, indent=4)}
        Sample Data:
        {data.to_string()} dont perform for this sample data, it just head(5) so, dont consider this as full dataset
        {'PREVIOUS CODE GENERATED SO FAR:' + full_generated_code if full_generated_code else ''} + you stictly continue from here code and dont start from initial, strictly continue the code flow.
        DETAILED EDA SCRIPT REQUIREMENTS:
        - Follow the structured approach of the previous code.
        - Continue from the last point generated, ensuring to maintain logical flow and consistency in the analysis.
        - give me a code not only logic.
        info:
        add before line and after code finish add "code start" and "code end"
        CORE INSTRUCTIONS:
        1. Use libraries: pandas, numpy, matplotlib, seaborn, scipy
        2. Include comprehensive error handling
        3. Provide detailed comments explaining each analysis step
        4. Create visualizations that are informative and aesthetically pleasing
        5.The code should be concise, specific to this dataset, and not designed for reusability or scalability but should work perfectly with the given full dataset and report. Output the merged dataset and its summary (number of rows and columns).

        DETAILED EDA SCRIPT REQUIREMENTS:

        I. DATA UNDERSTANDING PHASE:
        - Create function to analyze DataFrame structure
        - Provide detailed statistical summary
        - Generate comprehensive missing value report
        - Detect and characterize data types
        - Compute memory usage analysis

        II. DATA TRANSFORMATION PHASE:
        - Implement missing value handling strategies
        - Develop feature engineering techniques
        - Create categorical and numerical encoding methods
        - Design feature scaling approaches

        III. EXPLORATORY DATA ANALYSIS:
        A. Univariate Analysis:
        - Generate statistical distributions
        - Create visualization for each feature
        - Detect and handle outliers
        - Compute descriptive statistics

        B. Bivariate Analysis:
        - Explore feature relationships
        - Generate correlation matrices
        - Create advanced visualization techniques
        - Perform statistical significance testing

        C. Multivariate Analysis:
        - Implement dimensionality reduction techniques
        - Generate advanced visualization methods
        - Explore complex feature interactions

        IV. ADVANCED ANALYSIS:
        - Feature importance ranking
        - Anomaly detection mechanisms
        - Statistical hypothesis testing

        V. REPORTING:
        - Generate comprehensive HTML/markdown report
        - Create interactive visualizations
        - Summarize key insights and recommendations

        ADDITIONAL REQUIREMENTS:
        - Modular and object-oriented design
        - Support for different DataFrame types
        - Configurable analysis depth
        - Performance optimization
        - Comprehensive logging mechanism

        OUTPUT FORMAT:
        - Complete Python script
        - Inline documentation
        - Type hints
        - Error handling
        - Configurable parameters

        EXAMPLE INPUT CONTEXT:
        - Provide sample DataFrame details
        - Specify analysis objectives
        - Mention any domain-specific constraints

        CONSTRAINTS:
        - Ensure code is PEP 8 compliant
        - Minimize computational complexity
        - Maximize interpretability
        - Support different data types and structures

        DESIRED OUTPUT:
        A fully functional, production-ready EDA script that can automatically analyze any pandas DataFrame, providing deep insights, visualizations, and actionable recommendations.

        FINAL INSTRUCTION:
        Generate the most comprehensive, flexible, and intelligent EDA script possible, demonstrating advanced data science expertise and thoughtful implementation.
        """
        finish_reason = "MAX_TOKENS"
        response_events = []
        while finish_reason != "COMPLETE" and current_attempt < max_attempts:
            if self.co is None:
                raise RuntimeError("Cohere API key not configured")
            response = self.co.chat_stream(
                model="command-r-plus-08-2024",
                messages=[{"role": "user", "content": input_text}]
            )
            
            full_response = ""
            is_complete = False

            for event in response:
                response_events.append(event)

                if event.type == "content-delta":
                    delta_text = event.delta.message.content.text
                    full_response += delta_text
                    print(delta_text, end="", flush=True)

                if event.type == "message-end":
                    is_complete = True
                    finish_reason = event.delta.finish_reason if hasattr(event.delta, 'finish_reason') else "UNKNOWN"

            # Completion status analysis
            print("\n\n--- Streaming Completion Analysis ---")
            print(f"Sequence Complete: {is_complete}")
            print(f"Finish Reason: {finish_reason}")

            # Detailed finish reason interpretation
            finish_reason_map = {
                "COMPLETE": "✅ Full response generated successfully",
                "MAX_TOKENS": "⚠️ Response stopped due to token limit",
                "STOP_SEQUENCE": "⏹️ Response stopped by predefined stop sequence",
                "TOOL_CALL": "🛠️ Response stopped for tool call",
                "ERROR": "❌ Generation encountered an error"
            }

            print(finish_reason_map.get(finish_reason, "❓ Unknown completion status"))

            full_generated_code += full_response or ""
            current_attempt += 1
        if response_events:
            usage_stats = {
                "input_tokens": response_events[0].delta.usage.get('input_tokens', 0) if hasattr(response_events[0].delta, 'usage') else 0,
                "output_tokens": response_events[0].delta.usage.get('output_tokens', 0) if hasattr(response_events[0].delta, 'usage') else 0
            }
            print("\nUsage Statistics:")
            print(f"Input Tokens: {usage_stats['input_tokens']}")
            print(f"Output Tokens: {usage_stats['output_tokens']}")

        return {
            "full_response": full_generated_code,
            "is_complete": is_complete,
            "finish_reason": finish_reason,
            "usage_stats": usage_stats,
            "raw_events": response_events
        }



    def summary_of_data(self, data: pd.DataFrame,dataset_analysis_report):
        df = pd.DataFrame(data)
        input_text = f"""
        You are an expert AI data scientist and strategist specializing in advanced data analysis, relational database exploration, and generating innovative data science project ideas. Your task is to deeply analyze a dataset summary, understand relationships between datasets, and suggest actionable insights, innovative data science applications, and potential improvements.

        Below is the dataset analysis report:

        {json.dumps(dataset_analysis_report, indent=4)}
        {df.head().to_string()}
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

        if self.co is None:
            logging.warning("Cohere API key not configured; skipping AI summary")
            return None
        response = self.co.chat_stream(
            model="command-r-plus-08-2024",
            messages=[{"role": "user", "content": input_text}]
        )

        for event in response:
            if event:
                if event.type == "content-delta":
                    print(event.delta.message.content.text, end="")
                elif event.type == "stream-end":
                    print("\nStream ended.")

                