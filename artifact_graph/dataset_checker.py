"""
Dataset checker - responsible for generating and executing dataset check scripts
"""

import re
from .base import ExperimentPhase




class DatasetCheckGenerator:
    """Dataset check script generator"""
    
    def __init__(self, coder):
        self.coder = coder
        self.client = coder.client
    
    def generate_script(self, dataset_name: str, dataset_readme: str = None) -> str:
        """Generate dataset check script with README information"""
        
        # Format dataset README for inclusion in script
        dataset_readme_section = ""
        if dataset_readme:
            # Clean and truncate README for inclusion
            clean_readme = dataset_readme.replace('"""', "'''").replace('\\', '\\\\')[:2000]
            dataset_readme_section = f'''

# Include dataset README information as documentation
DATASET_README = """
{clean_readme}
"""
'''
        
        prompt = f"""
Generate a simple dataset checking script for the HuggingFace dataset: {dataset_name}

IMPORTANT: refer to the following dataset README section for the dataset information:
{dataset_readme_section}

Requirements:
1. Load and verify the dataset can be accessed
2. Get dataset splits information (train/test/validation sizes)
3. Extract ONE actual example from each available split

ONLY save results to 'dataset_metadata.json' if dataset access is successful. Use this structure:
{{
    "dataset_name": "{dataset_name}",
    "splits": {{"train": 1000, "test": 100, "validation": 200}},
    "sample_examples": [
        {{
            "split": "train",
            "example": actual_data_example_from_train_split
        }},
        {{
            "split": "test", 
            "example": actual_data_example_from_test_split
        }}
    ]
}}

CRITICAL: Extract REAL examples from the dataset, not placeholder data.
If ANY error occurs (authentication failures, download errors, dataset access issues), 
DO NOT save the metadata.json file - just let the script exit with an error.
The metadata.json should ONLY exist when everything works successfully.

Generate a complete Python script named 'dataset_check.py' that accomplishes these tasks.
"""
        
        response = self.client.chat.completions.create(
            model=self.coder.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        
        script = response.choices[0].message.content
        
        # Extract code block if present
        if "```python" in script:
            script = script.split("```python")[1].split("```")[0]
        elif "```" in script:
            script = script.split("```")[1].split("```")[0]
        
        # If no code block found, return entire response
        return script.strip()
    
    def get_stage(self) -> ExperimentPhase:
        return ExperimentPhase.DATASET_CHECK
    
    def get_dataset_readme(self, dataset_name: str) -> str:
        """Get dataset README information"""
        try:
            import requests
            import json
            
            # Try to get dataset info via HuggingFace API
            api_url = f"https://huggingface.co/api/datasets/{dataset_name}"
            response = requests.get(api_url, timeout=10)
            
            if response.status_code == 200:
                dataset_info = response.json()
                return json.dumps(dataset_info, indent=2)
        except Exception as e:
            print(f"API request failed: {e}")
        
        # If API fails, try to get README directly
        try:
            import requests
            readme_url = f"https://huggingface.co/datasets/{dataset_name}/raw/main/README.md"
            response = requests.get(readme_url, timeout=10)
            
            if response.status_code == 200:
                return response.text[:2000]  # Limit length
        except Exception as e:
            print(f"README request failed: {e}")
        
        return f"Model: {dataset_name} (no additional info available)"
    

