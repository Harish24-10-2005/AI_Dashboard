import pandas as pd
import numpy as np
import relation
from dotenv import load_dotenv
import json
from Ai_decision import AIDecisionMaker
load_dotenv()

LLM = AIDecisionMaker('cohere_api_key')
data = {
    "employee_id": [1, 2, 2, 3, 4],
    "employee_name": ["Alice", "Bob", "Bob", "Charlie", "Diana"],
    "department_id": [101, 102, 102, 101, 103],
    "project_id": [1001, 1002, 1003, 1001, 1003],
    "department_name": ["Engineering", "Sales", "Sales", "Engineering", "HR"],
    "project_id_datasets": [1001, 1002, 1002, 1001, 1003],
    "project_name": ["Website Redesign", "Product Launch", "Product Launch", "Website Redesign", "Recruitment Drive"],
}
df = pd.DataFrame(data)
report = {
    "title": "Dataset Analysis Report",
    "generated_on": "2024-11-19 20:48:29",
    "original_datasets": [
        {
            "name": "employee_projects",
            "rows": 5,
            "columns": 2,
            "column_types": {
                "employee_id": "int64",
                "project_id": "int64"
            }
        },
        {
            "name": "employees",
            "rows": 4,
            "columns": 3,
            "column_types": {
                "employee_id": "int64",
                "employee_name": "object",
                "department_id": "int64"
            }
        },
        {
            "name": "departments",
            "rows": 3,
            "columns": 2,
            "column_types": {
                "department_id": "int64",
                "department_name": "object"
            }
        },
        {
            "name": "datasets",
            "rows": 3,
            "columns": 3,
            "column_types": {
                "project_id": "int64",
                "project_name": "object",
                "department_id": "int64"
            }
        }
    ],
    "detected_relationships": [
        {
            "relationship": "employee_projects.employee_id <--> employees.employee_id",
            "similarity": 0.9417375421524048,
            "confidence": 1.0,
            "cardinality": "1:1"
        },
        {
            "relationship": "departments.department_id <--> employees.department_id",
            "similarity": 0.9180659254391972,
            "confidence": 1.0,
            "cardinality": "1:1"
        },
        {
            "relationship": "datasets.project_id <--> employee_projects.project_id",
            "similarity": 0.8913069788614961,
            "confidence": 1.0,
            "cardinality": "1:1"
        },
        {
            "relationship": "datasets.department_id <--> employees.department_id",
            "similarity": 0.9180659254391972,
            "confidence": 1.0,
            "cardinality": "1:1"
        },
        {
            "relationship": "datasets.department_id <--> departments.department_id",
            "similarity": 0.9999999821186065,
            "confidence": 1.0,
            "cardinality": "1:1"
        }
    ],
    "merged_results": [
        {
            "name": "merged_dataset_1",
            "rows": 5,
            "columns": 7,
            "confidence": 1.0,
            "cardinality": "1:1"
        }
    ]
}

LLM.create_code(df,report)