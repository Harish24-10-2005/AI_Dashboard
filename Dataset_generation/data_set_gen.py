import streamlit as st
import cohere
import pandas as pd
import json
import re

# Initialize Cohere client
api_key = "1TyaPaqTNlXozRCJYWb1RSw30nkPwPqbG8ApPLFr"
co = cohere.Client(api_key)

def extract_json_safely(text):
    """
    Robust method to extract JSON from text with multiple strategies
    """
    # Try multiple extraction strategies
    extraction_patterns = [
        # Find JSON between first [ and last ]
        r'\[.*\]',
        # Find JSON between first { and last }
        r'\{.*\}',
        # More aggressive JSON extraction
        r'\[(?:[^[\]]*(?:\[(?:[^[\]]*(?:\[(?:[^[\]]*)\])?[^[\]]*)\])?[^[\]]*)\]'
    ]

    for pattern in extraction_patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            json_text = match.group(0)
            
            # Attempt to repair common JSON formatting issues
            json_text = re.sub(r',\s*}', '}', json_text)  # Remove trailing commas
            json_text = re.sub(r',\s*]', ']', json_text)  # Remove trailing commas
            
            try:
                # Attempt to parse the extracted text
                parsed_json = json.loads(json_text)
                
                # Validate it's a list of dictionaries
                if isinstance(parsed_json, list) and all(isinstance(item, dict) for item in parsed_json):
                    return parsed_json
            except json.JSONDecodeError as e:
                st.warning(f"JSON parsing attempt failed: {e}")
    
    return None

def generate_dataset(prompt, max_tokens=1500, num_attempts=3):
    """
    Generate dataset with multiple attempts and robust parsing
    """
    for attempt in range(num_attempts):
        try:
            response = co.generate(
                model='command-r-plus-08-2024',
                prompt=prompt,
                max_tokens=max_tokens,
                stop_sequences=["\n\n"],
            )
            
            # Extract JSON from the generated text
            dataset = extract_json_safely(response.generations[0].text)
            
            if dataset:
                return dataset
            
        except Exception as e:
            st.warning(f"Attempt {attempt + 1} failed: {e}")
            max_tokens += 500
    
    st.error("Failed to generate a valid dataset after multiple attempts")
    return None

def main():
    st.title("Advanced Dataset Generator")

    # Sidebar for configuration
    st.sidebar.header("Dataset Generation Settings")
    
    # Prompt input with more explicit JSON instructions
    prompt = st.text_area(
        "Enter your dataset generation prompt:",
        value="Generate a comprehensive dataset of Tamil movies. Provide the output strictly as a valid JSON array. Each movie entry must be a JSON object with consistent keys. Ensure no syntax errors.",
        height=150
    )

    # Configuration options
    col1, col2 = st.columns(2)
    with col1:
        num_rows = st.slider("Number of Rows", 5, 500, 50)
    with col2:
        max_tokens = st.slider("Max Generation Tokens", 500, 2000, 1500)

    # Column selection with more robust handling
    column_input =  st.text_input(
    "Enter Column Names (comma-separated):",
    value="movie_name, movie_rating, movie_hero"
)
    columns = [col.strip() for col in column_input.split(",") if col.strip()]
    # Generate button
    if st.button("Generate Dataset", type="primary"):
        with st.spinner("Generating dataset..."):
            # Refined prompt with explicit JSON formatting instructions
            # refined_prompt = (
            #     f"Generate a JSON array of exactly {num_rows} movie entries. "
            #     f"Each entry must be a JSON object with these keys: {', '.join(columns)}. "
            #     "Format the output as a VALID JSON array. "
            #     "Ensure each entry has all specified keys. "
            #     "Use proper JSON syntax with no trailing commas. "
            #     "Example: [{\"movie_name\":\"Movie1\",\"movie_rating\":8.5,\"movie_hero\":\"Hero1\"}]"
            # )
            refined_prompt = f'''You are an advanced dataset generator with the capability to create a diverse range of structured datasets based on user specifications. Your task is to generate datasets that are accurate, comprehensive, and well-structured. Follow the guidelines below to produce high-quality data.

                            1. **User  Input**: The user will provide a description of the data they need, including the type of dataset (e.g., movies, books, products, etc.), the number of entries required, and the specific columns or fields to include. The input will be provided as follows:
                            - **Prompt**: {prompt}
                            - **Columns**: {columns}
                            - **Number of Rows**: {num_rows}
                            2. **Data Structure**: Format the dataset as a valid JSON array. Each entry in the array must be a JSON object containing the specified fields. Ensure that each field has the appropriate data type (e.g., strings for names, numbers for ratings or prices, booleans for true/false values).

                            3. **Data Accuracy and Realism**: The generated data should be realistic and representative of the specified domain. Utilize common knowledge, current trends, and typical values relevant to the domain to populate the dataset. Ensure that the data reflects diversity and variability where applicable.

                            4. **Example Format**: Provide the output strictly as a valid JSON array. For example, if the user requests a dataset of movies, the output should resemble the following structure:
                            
                            6. **User  Instructions and Summary**: After generating the dataset, provide a summary of its characteristics, including:
                            - Total number of entries generated
                            - List of fields included in the dataset
                            - Notable trends, patterns, or observations based on the generated data, such as average ratings or common genres.

                            7. **Validation**: Before finalizing the output, ensure that the generated JSON is syntactically correct and free from errors. Perform checks to confirm that the JSON structure is valid and complete.

                            Now, based on these guidelines, please generate a dataset according to the user's specifications.
                            8. **Error Handling**: If you encounter any issues during the generation process, provide a clear error message indicating what went wrong and ensure that no partial or invalid JSON is returned.

                            Now, based on these guidelines, please generate a dataset according to the user's specifications.

                            ### Key Changes Made:

                            1. **Simplified Instructions**: The prompt has been simplified to focus solely on the essentials needed for generating valid JSON.

                            2. **Clear Example Structure**: The example is now clearly formatted as JSON, which helps reinforce the expected output format.

                            3. **Emphasis on Validation**: The validation step has been highlighted to ensure that the generated JSON is correct before any output is returned.

                            This streamlined prompt should reduce the chances of errors during JSON parsing and help ensure that valid datasets are generated successfully. If you continue to face issues, please provide specific details about the input you are using, and I can help further refine the approach.'''
            # Generate dataset
            dataset = generate_dataset(refined_prompt, max_tokens)

            if dataset:
                try:
                    # Filter and process dataset
                    filtered_data = [
                        {col: entry.get(col, "N/A") for col in columns} 
                        for entry in dataset 
                        if all(col in entry for col in columns)
                    ]

                    # Create DataFrame
                    df = pd.DataFrame(filtered_data)

                    # Display dataset
                    st.dataframe(df)

                    # Download options
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            label="Download CSV",
                            data=df.to_csv(index=False),
                            file_name="generated_dataset.csv",
                            mime="text/csv"
                        )
                    with col2:
                        st.download_button(
                            label="Download JSON",
                            data=json.dumps(filtered_data, indent=2),
                            file_name="generated_dataset.json",
                            mime="application/json"
                        )

                    # Show dataset statistics
                    st.write(f"Dataset Statistics:")
                    st.write(f"Total Rows: {len(df)}")
                    st.write(f"Columns: {', '.join(df.columns)}")

                except Exception as e:
                    st.error(f"Error processing dataset: {e}")
                    # Optionally, show the raw dataset for debugging
                    st.json(dataset)
            else:
                st.error("Could not generate a valid dataset. Please modify your prompt.")

if __name__ == "__main__":
    main()