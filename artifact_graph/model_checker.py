"""
Model checker - responsible for generating and executing model check scripts
"""

import json

import requests

from .base import ExperimentPhase


class ModelCheckGenerator:
    """Model check script generator"""

    def __init__(self, coder):
        self.coder = coder
        self.client = coder.client

    def generate_script(
        self,
        model_name: str,
        dataset_name: str,
        dataset_metadata: dict = None,
        model_readme: str = None,
    ) -> str:
        """Generate model check script with dataset compatibility testing"""

        # Format dataset metadata for the prompt
        dataset_info = ""
        if dataset_metadata:
            dataset_info = f"""

DATASET CONTEXT (for compatibility testing):
Dataset: {dataset_name}
Splits: {dataset_metadata.get('splits', {})}
Sample Examples: {dataset_metadata.get('sample_examples', [])}

Use these actual dataset examples to test model compatibility!
"""

        # Format model README for inclusion in script
        model_readme_section = ""
        if model_readme:
            # Clean and truncate README for inclusion
            clean_readme = model_readme.replace('"""', "'''").replace("\\", "\\\\")[:2000]
            model_readme_section = f'''

# Include model README information as documentation
MODEL_README = """
{clean_readme}
"""
'''

        prompt = f"""
Generate a model checking script for the HuggingFace model: {model_name}
{dataset_info}

IMPORTANT: refer to the following model README section for the dataset information:
{model_readme_section}

Requirements:
1. Load and verify the model can be instantiated
2. Run model inference on the provided dataset examples (any output is fine)
3. Ensure the code executes without errors

ONLY save results to 'model_metadata.json' if the model runs successfully. Use this structure:
{{
    "model_name": "{model_name}",
    "runnable": "yes"
}}

Focus on making the code runnable, not on correctness of results.
If ANY error occurs (missing model files, authentication issues, loading errors, runtime exceptions),
DO NOT save the metadata.json file - just let the script exit with an error.
The metadata.json should ONLY exist when everything works successfully.

Generate a complete Python script named 'model_check.py' that accomplishes these tasks.
"""

        response = self.client.chat.completions.create(
            model=self.coder.model, messages=[{"role": "user", "content": prompt}], temperature=0.1
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
        return ExperimentPhase.MODEL_CHECK

    def get_model_readme(self, model_name: str) -> str:
        """Get model README information"""
        try:
            # Try to get model info via HuggingFace API
            api_url = f"https://huggingface.co/api/models/{model_name}"
            response = requests.get(api_url, timeout=10)

            if response.status_code == 200:
                model_info = response.json()
                return json.dumps(model_info, indent=2)
        except Exception as e:
            print(f"API request failed: {e}")

        # If API fails, try to get README directly
        try:
            readme_url = f"https://huggingface.co/{model_name}/raw/main/README.md"
            response = requests.get(readme_url, timeout=10)

            if response.status_code == 200:
                return response.text[:2000]  # Limit length
        except Exception as e:
            print(f"README request failed: {e}")

        return f"Model: {model_name} (no additional info available)"
