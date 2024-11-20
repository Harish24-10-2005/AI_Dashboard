import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple, Set, Optional, Union
import logging
from tqdm import tqdm
import networkx as nx
from datetime import datetime
import os
from pathlib import Path
import multiprocessing
from difflib import SequenceMatcher
import jellyfish
from collections import defaultdict
from transformers import AutoTokenizer, AutoModel
import torch
import json
import warnings
import shutil
warnings.filterwarnings('ignore')

class SmartDatasetAnalyzer:
    def __init__(
        self,
        min_similarity: float = 0.75,
        use_llm: bool = True,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        cache_dir: Optional[str] = None
    ):
        """
        Initialize Smart Dataset Analyzer with advanced features.
        
        Args:
            min_similarity: Minimum similarity threshold for relationships
            use_llm: Whether to use LLM for semantic analysis
            model_name: Name of the transformer model to use
            cache_dir: Directory to cache analysis results
        """
        self.min_similarity = min_similarity
        self.use_llm = use_llm
        self.cache_dir = cache_dir
        self.logger = self._setup_logger()
        self.n_workers = max(1, int(multiprocessing.cpu_count() * 0.75))
        
        if self.use_llm:
            self._setup_llm(model_name)
            
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
    
    def _setup_llm(self, model_name: str):
        """Setup LLM model for semantic analysis."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            if torch.cuda.is_available():
                self.model = self.model.cuda()
            self.model.eval()
        except Exception as e:
            self.logger.warning(f"Failed to load LLM model: {str(e)}")
            self.use_llm = False
    
    def _setup_logger(self) -> logging.Logger:
        """Setup enhanced logging configuration."""
        logger = logging.getLogger('SmartDatasetAnalyzer')
        logger.setLevel(logging.INFO)
        
        c_handler = logging.StreamHandler()
        f_handler = logging.FileHandler('dataset_analyzer.log')
        
        c_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        f_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        c_handler.setFormatter(c_formatter)
        f_handler.setFormatter(f_formatter)
        
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)
        
        return logger
    
    def _get_column_embeddings(self, series: pd.Series) -> np.ndarray:
        """Get semantic embeddings for column values using LLM."""
        if not self.use_llm:
            return None
            
        try:
            sample_size = min(100, len(series))
            sample_values = series.dropna().sample(n=sample_size, random_state=42)
            
            text = " ".join(str(x) for x in sample_values)
            
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512,
                padding=True
            )
            
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            embeddings = outputs.last_hidden_state.mean(dim=1)
            return embeddings.cpu().numpy()
            
        except Exception as e:
            self.logger.warning(f"Error in embedding generation: {str(e)}")
            return None
    
    def _compute_statistical_similarity(
        self, 
        col1: pd.Series, 
        col2: pd.Series
    ) -> float:
        """Compute statistical similarity between columns."""
        try:
            similarities = []
            
            stats1 = col1.describe()
            stats2 = col2.describe()
            
            if pd.api.types.is_numeric_dtype(col1) and pd.api.types.is_numeric_dtype(col2):
                hist1, _ = np.histogram(col1, bins=20, density=True)
                hist2, _ = np.histogram(col2, bins=20, density=True)
                hist_sim = 1 - np.mean(np.abs(hist1 - hist2))
                similarities.append(hist_sim)
                
                range_sim = 1 - abs(
                    (col1.max() - col1.min()) - (col2.max() - col2.min())
                ) / max(col1.max() - col1.min(), col2.max() - col2.min())
                similarities.append(range_sim)
                
            null_sim = 1 - abs(
                col1.isnull().mean() - col2.isnull().mean()
            )
            similarities.append(null_sim)
            
            unique_sim = 1 - abs(
                (col1.nunique() / len(col1)) - (col2.nunique() / len(col2))
            )
            similarities.append(unique_sim)
            
            return np.mean(similarities)
            
        except Exception as e:
            self.logger.warning(f"Error in statistical similarity: {str(e)}")
            return 0.0
    
    def _compute_semantic_similarity(
        self, 
        col1: pd.Series, 
        col2: pd.Series
    ) -> float:
        """Compute semantic similarity using LLM embeddings."""
        if not self.use_llm:
            return 0.0
            
        try:
            emb1 = self._get_column_embeddings(col1)
            emb2 = self._get_column_embeddings(col2)
            
            if emb1 is None or emb2 is None:
                return 0.0
            
            similarity = np.dot(emb1[0], emb2[0]) / (
                np.linalg.norm(emb1[0]) * np.linalg.norm(emb2[0])
            )
            
            return float(similarity)
            
        except Exception as e:
            self.logger.warning(f"Error in semantic similarity: {str(e)}")
            return 0.0
    
    def _detect_relationship_type(
        self, 
        col1: pd.Series, 
        col2: pd.Series
    ) -> Dict[str, Union[str, float]]:
        """Detect detailed relationship type between columns."""
        relationship = {
            'type': 'unknown',
            'confidence': 0.0,
            'cardinality': 'N:M'
        }
        
        try:
            if col1.equals(col2):
                relationship.update({
                    'type': 'exact_match',
                    'confidence': 1.0,
                    'cardinality': '1:1'
                })
                return relationship
            
            unique1 = set(col1.dropna())
            unique2 = set(col2.dropna())
            
            overlap = len(unique1 & unique2)
            
            if overlap > 0:
                if len(unique1) == len(unique2) and overlap == len(unique1):
                    cardinality = '1:1'
                elif len(unique1) > len(unique2):
                    cardinality = 'N:1'
                else:
                    cardinality = '1:N'
                
                confidence = overlap / min(len(unique1), len(unique2))
                
                relationship.update({
                    'type': 'value_match',
                    'confidence': confidence,
                    'cardinality': cardinality
                })
            
            return relationship
            
        except Exception as e:
            self.logger.warning(f"Error in relationship detection: {str(e)}")
            return relationship
    
    def analyze_datasets(
        self, 
        datasets: Dict[str, pd.DataFrame]
    ) -> Dict[str, List[Dict]]:
        """
        Analyze relationships between multiple datasets.
        
        Args:
            datasets: Dictionary of dataset names and their DataFrames
            
        Returns:
            Dictionary of relationships between datasets
        """
        self.logger.info("Starting dataset analysis...")
        
        relationships = defaultdict(list)
        # cache_file = None
        
        # if self.cache_dir:
        #     cache_file = Path(self.cache_dir) / "relationships_cache.json"
        #     if cache_file.exists():
        #         try:
        #             with open(cache_file, 'r') as f:
        #                 cached_relationships = json.load(f)
        #             self.logger.info("Loaded relationships from cache")
        #             for key, rels in cached_relationships.items():
        #                 table1, table2 = key.split('-')
        #                 if table1 in datasets and table2 in datasets:
        #                     relationships[key] = [
        #                         rel for rel in rels
        #                         if rel['column1'] in datasets[table1].columns and rel['column2'] in datasets[table2].columns
        #                     ]
        #             self.logger.info("Filtered cached relationships based on current datasets")
        #             return relationships
        #         except Exception as e:
        #             self.logger.warning(f"Failed to load cache: {str(e)}")
        
        def process_column_pair(args: Tuple) -> Optional[Dict]:
            df1_name, df2_name, col1, col2 = args
            df1, df2 = datasets[df1_name], datasets[df2_name]
            
            if df1_name == df2_name and col1 == col2:
                return None
            
            stat_sim = self._compute_statistical_similarity(df1[col1], df2[col2])
            sem_sim = self._compute_semantic_similarity(df1[col1], df2[col2]) if self.use_llm else 0.0
            
            similarity = 0.7 * stat_sim + 0.3 * sem_sim
            
            if similarity >= self.min_similarity:
                relationship = self._detect_relationship_type(df1[col1], df2[col2])
                
                if relationship['confidence'] > 0:
                    return {
                        'table1': df1_name,
                        'column1': col1,
                        'table2': df2_name,
                        'column2': col2,
                        'similarity': similarity,
                        'relationship_type': relationship['type'],
                        'confidence': relationship['confidence'],
                        'cardinality': relationship['cardinality']
                    }
            
            return None
        
        column_pairs = [
            (df1_name, df2_name, col1, col2)
            for df1_name, df1 in datasets.items()
            for df2_name, df2 in datasets.items()
            if df1_name <= df2_name
            for col1 in df1.columns
            for col2 in df2.columns
        ]
        
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            results = list(tqdm(
                executor.map(process_column_pair, column_pairs),
                total=len(column_pairs),
                desc="Analyzing relationships"
            ))
        
        for result in results:
            if result:
                key = f"{result['table1']}-{result['table2']}"
                relationships[key].append(result)
        # if cache_file:
        #     try:
        #         with open(cache_file, 'w') as f:
        #             json.dump(dict(relationships), f)
        #         self.logger.info("Cached relationships")
        #     except Exception as e:
        #         self.logger.warning(f"Failed to cache relationships: {str(e)}")
        
        return relationships
    
    def merge_datasets(
        self, 
        datasets: Dict[str, pd.DataFrame],
        relationships: Dict[str, List[Dict]]
    ) -> Dict[str, pd.DataFrame]:
        """
        Merge datasets based on detected relationships.
        
        Args:
            datasets: Dictionary of dataset names and their DataFrames
            relationships: Dictionary of relationships between datasets
            
        Returns:
            Dictionary of merged and standalone datasets
        """
        self.logger.info("Starting dataset merging...")
        
        G = nx.Graph()
        
        for rels in relationships.values():
            for rel in rels:
                if rel['confidence'] >= 0.8: 
                    G.add_edge(
                        rel['table1'],
                        rel['table2'],
                        weight=1-rel['similarity'],
                        columns=(rel['column1'], rel['column2']),
                        cardinality=rel['cardinality']
                    )
        
        components = list(nx.connected_components(G))
        
        result_datasets = {}
        
        for i, component in enumerate(components, 1):
            if len(component) > 1:
                mst = nx.minimum_spanning_tree(G.subgraph(component))
                start_table = max(G.degree, key=lambda x: x[1])[0]
                
                merged_df = datasets[start_table].copy()
                processed_tables = {start_table}
                merged_columns = set(merged_df.columns)
                
                for edge in nx.edge_dfs(mst, start_table):
                    table1, table2 = edge[0], edge[1]
                    if table2 in processed_tables:
                        continue
                    
                    join_cols = G[table1][table2]['columns']
                    cardinality = G[table1][table2]['cardinality']
                    
                    how = {
                        '1:1': 'inner',
                        '1:N': 'left',
                        'N:1': 'right'
                    }.get(cardinality, 'outer')
                    
                    df2 = datasets[table2].copy()
                    rename_dict = {
                        col: f"{col}_{table2}"
                        for col in df2.columns
                        if col in merged_columns and col != join_cols[1]
                    }
                    df2 = df2.rename(columns=rename_dict)
                    
                    merged_df = pd.merge(
                        merged_df,
                        df2,
                        left_on=join_cols[0],
                        right_on=join_cols[1],
                        how=how,
                        suffixes=(f'_{table1}', f'_{table2}')
                    )
                    
                    processed_tables.add(table2)
                    merged_columns.update(merged_df.columns)
                
                result_datasets[f"merged_dataset_{i}"] = merged_df
                self.logger.info(f"Created merged dataset {i} from tables: {', '.join(component)}")
        
        connected_tables = set().union(*components) if components else set()
        standalone_tables = set(datasets.keys()) - connected_tables
        
        for table in standalone_tables:
            result_datasets[f"standalone_{table}"] = datasets[table]
            self.logger.info(f"Kept {table} as standalone dataset (no strong relationships found)")
        
        return result_datasets
    
    def save_results(
        self,
        results: Dict[str, pd.DataFrame],
        output_dir: str,
        format: str = 'csv'
    ) -> Dict[str, str]:
        """
        Save merged and standalone datasets to files.
        
        Args:
            results: Dictionary of dataset names and their DataFrames
            output_dir: Directory to save results
            format: Output format ('csv' or 'parquet')
            
        Returns:
            Dictionary of dataset names and their file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        saved_paths = {}
        
        for name, df in results.items():
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{name}_{timestamp}"
                
                if format == 'csv':
                    filepath = os.path.join(output_dir, f"{filename}.csv")
                    df.to_csv(filepath, index=False)
                else:
                    filepath = os.path.join(output_dir, f"{filename}.parquet")
                    df.to_parquet(filepath, index=False)
                
                saved_paths[name] = filepath
                self.logger.info(f"Saved {name} to {filepath}")
                
            except Exception as e:
                self.logger.error(f"Error saving {name}: {str(e)}")
        
        return saved_paths
    
    def generate_report(
    self,
    datasets: Dict[str, pd.DataFrame],
    relationships: Dict[str, List[Dict]],
    results: Dict[str, pd.DataFrame]
) -> Dict[str, Union[str, List[Dict]]]:
        """
        Generate analysis report and return it as a structured dictionary.
        
        Args:
            datasets: Original datasets
            relationships: Detected relationships
            results: Merged results
            
        Returns:
            Dictionary containing the report structured by sections
        """
        try:
            report = {
                "title": "Dataset Analysis Report",
                "generated_on": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "original_datasets": [],
                "detected_relationships": [],
                "merged_results": []
            }
            
            for name, df in datasets.items():
                dataset_info = {
                    "name": name,
                    "rows": len(df),
                    "columns": len(df.columns),
                    "column_types": {col: str(df[col].dtype) for col in df.columns}
                }
                report["original_datasets"].append(dataset_info)
            
            for key, rels in relationships.items():
                for rel in rels:
                    relationship_info = {
                        "relationship": f"{rel['table1']}.{rel['column1']} <--> {rel['table2']}.{rel['column2']}",
                        "similarity": rel['similarity'],
                        "confidence": rel['confidence'],
                        "cardinality": rel['cardinality']
                    }
                    report["detected_relationships"].append(relationship_info)
            
            # Results summary
            for name, df in results.items():
                result_info = {
                    "name": name,
                    "rows": len(df),
                    "columns": len(df.columns)
                }
                report["merged_results"].append(result_info)
            
            self.logger.info("Generated analysis report")
            return report
                
        except Exception as e:
            self.logger.error(f"Error generating report: {str(e)}")
            return {}

def analyze_and_merge_datasets(
    datasets: Dict[str, pd.DataFrame],
    output_dir: str = "output",
    use_llm: bool = True,
    min_similarity: float = 0.75,
    save_format: str = 'csv',
    cache_dir: Optional[str] = None
) -> Dict[str, pd.DataFrame]:
    
    if os.path.exists(output_dir):
        try:
            # Remove entire output directory contents
            shutil.rmtree(output_dir)
            print(f"Cleaned previous analysis data from {output_dir}")
        except Exception as e:
            print(f"Error cleaning output directory: {e}")

    try:
        analyzer = SmartDatasetAnalyzer(
            min_similarity=min_similarity,
            use_llm=use_llm,
            cache_dir=cache_dir
        )
        print("Analyzing datasets...")
        relationships = analyzer.analyze_datasets(datasets) 
        print("Found relationships:", relationships)     
        print("Merging datasets...")
        results = analyzer.merge_datasets(datasets, relationships)
        print("Saving results...")
        analyzer.save_results(results, output_dir, format=save_format)
        
        report = analyzer.generate_report(datasets, relationships, results)
        
        return results,report
        
    except Exception as e:
        print(f"Error in analysis: {str(e)}")
        return None

if __name__ == "__main__":
    students = pd.DataFrame({
        'student_id': [1, 2, 3, 4],
        'name': ['Alice', 'Bob', 'Charlie', 'David']
    })

    courses = pd.DataFrame({
        'course_id': [101, 102, 103, 104],
        'course_name': ['Math', 'Science', 'History', 'Art'],
        'student_id': [1, 2, 3, 4]  
    })

    print("Students DataFrame:")
    print(students)
    print("\nCourses DataFrame:")
    print(courses)


    datasets = {
        'students': students,
        'courses': courses
    }

    results,report = analyze_and_merge_datasets(
        datasets,
        output_dir='analysis_results',
        use_llm=True,
        min_similarity=0.75,
        save_format='csv',
        cache_dir='analysis_cache'
    )
    
    
    if results:
        print("\nAnalysis completed successfully!")
        print("\nFinal Datasets:")
        for name, df in results.items():
            print(f"\n{name}:")
            print(df.head())
            print(f"Shape: {df.shape}")
            print("Columns:", df.columns.tolist())