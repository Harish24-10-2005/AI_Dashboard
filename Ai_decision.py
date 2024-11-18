import pandas as pd
from typing import List, Dict, Union, Optional, Tuple
import logging
import json
import torch
from transformers import AutoTokenizer, AutoModel, RobertaTokenizer, RobertaModel
import cohere
from scipy import stats
import category_encoders as ce
from pandas.api.types import is_numeric_dtype, is_string_dtype, is_categorical_dtype
from transformers import AutoTokenizer, AutoModelForCausalLM

class AIDecisionMaker:
    def __init__(self, cohere_api_key: str):
        self.cohere_client = cohere.Client(cohere_api_key)
        self.tokenizer = AutoTokenizer.from_pretrained("bigcode/starcoder")
        self.model = AutoModelForCausalLM.from_pretrained("bigcode/starcoder")
        
    def analyze_error(self, error_message: str, code_context: str) -> Dict[str, str]:
        """Analyze errors using CodeBERT and Cohere for intelligent error resolution."""
        try:
            # Get CodeBERT embeddings for error context
            inputs = self.codebert_tokenizer(code_context, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                code_embeddings = self.codebert_model(**inputs).last_hidden_state.mean(dim=1)
            
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
            
            response = self.cohere_client.generate(
                prompt=prompt,
                max_tokens=500,
                temperature=0.7,
                k=0,
                model='command'
            )
            
            analysis = response.generations[0].text
            
            return {
                'error_type': self._classify_error(error_message),
                'analysis': analysis,
                'embeddings': code_embeddings.numpy().tolist()
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
            
            response = self.cohere_client.generate(
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
                
        # Add correlation matrix for numeric columns
        numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_cols) > 0:
            profile['correlation_matrix'] = data[numeric_cols].corr().to_dict()
            
        return profile

    def summary_of_data(self, data: pd.DataFrame,dataset_analysis_report):
        input_text = """
        <fim_prefix>
        You are an expert AI data scientist and strategist specializing in advanced data analysis, relational database exploration, and generating innovative data science project ideas. Your task is to deeply analyze a dataset summary, understand relationships between datasets, and suggest actionable insights, innovative data science applications, and potential improvements.

        Below is the dataset analysis report:

        {json.dumps(dataset_analysis_report, indent=4)}

        Summary:

        Provide a concise overview of the datasets, their structure, and the detected relationships.
        Highlight the strengths or unique characteristics of the data.
        Insights and Opportunities:

        Suggest key patterns, trends, or insights that could be extracted from the data.
        Propose hypotheses or questions the data could answer.
        Data Science Applications:

        Recommend innovative and impactful data science projects based on the datasets and their relationships.
        For each suggestion, briefly explain its value and potential real-world applications.
        Enhancements:

        Identify any missing data, potential biases, or limitations in the datasets.
        Suggest additional data or relationships that could improve the quality of analysis.
        Technical Plan:

        Propose specific machine learning models, algorithms, or techniques that can be used for the suggested projects.
        Include tools or frameworks suitable for implementing the ideas.

        <fim_suffix>
        Provide the analysis below:
        <fim_middle>
        """

        inputs = self.tokenizer.encode(input_text, return_tensors="pt").to(device)

        outputs = self.model.generate(
            inputs,
            max_length=1024,  # Adjust max length as needed
            temperature=0.7,  # Set temperature for creativity level
            top_p=0.9,  # Nucleus sampling for diverse output
            num_return_sequences=1  # Generate a single response
        )

        # Decode and print the output
        decoded_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(decoded_output)