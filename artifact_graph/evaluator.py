"""
Evaluator - Responsible for generating and executing evaluation scripts
"""

import os

from .utils.llm import get_response_from_llm


class EvaluationGenerator:
    """Evaluation script generator"""

    def __init__(self, coder):
        self.coder = coder

    def generate_evaluate_script_with_metadata(
        self,
        model_name: str,
        dataset_name: str,
        metric: str,
        model_readme: str,
        model_metadata: dict,
        dataset_metadata: dict,
    ) -> bool:
        model_check_content = self._read_script_content("model_check.py")
        dataset_check_content = self._read_script_content("dataset_check.py")

        # Extract dataset structure information
        dataset_splits = dataset_metadata.get("splits", {})
        sample_examples = dataset_metadata.get("sample_examples", [])

        # Format splits information
        splits_info = f"Available splits: {list(dataset_splits.keys())}"
        if dataset_splits:
            splits_detail = ", ".join(
                [f"{split}: {count} samples" for split, count in dataset_splits.items()]
            )
            splits_info += f" ({splits_detail})"

        # Format sample examples
        examples_info = ""
        if sample_examples:
            examples_info = "\nSample data structure from dataset_metadata:\n"
            for example in sample_examples[:2]:  # Show first 2 examples
                split = example.get("split", "unknown")
                example_data = example.get("example", {})
                examples_info += f"- {split} split example: {example_data}\n"

        prompt = f"""
Create a comprehensive Python evaluation script for:
- Model: {model_name}
- Dataset: {dataset_name}
- Metric: {metric}

DATASET STRUCTURE INFORMATION:
{splits_info}
{examples_info}

GENERATED MODEL CHECK SCRIPT:
```python
{model_check_content}
```

GENERATED DATASET CHECK SCRIPT:
```python
{dataset_check_content}
```

Use the above scripts as reference for understanding how to properly load and process the model and dataset.
Pay special attention to the dataset structure and example format shown above.

Based on the metadata above, the script should:

1. Import all necessary libraries (transformers, datasets, torch, evaluate, sklearn, json, etc.)
2. Load the pre-trained model and tokenizer with optimal configuration (you should check the GENERATED MODEL CHECK SCRIPT)
3. Load and preprocess the dataset with proper configuration:
   - Use the dataset splits information provided above
   - Handle the data format as shown in the sample examples
   - Follow the pattern from the GENERATED DATASET CHECK SCRIPT
4. Implement evaluation logic for {metric}:
   - Use the correct preprocessing pipeline for the model architecture
   - Process data according to the example format structure
   - Calculate {metric} accurately for the task type

5. Save comprehensive results to results.json with robust writing:
   - Include the calculated {metric}, model_name, dataset_name, total_samples, processing_time, device_used
   - CRITICAL: Use proper JSON format:
     results = {{
         "{metric}": calculated_metric_value,
         "model_name": "{model_name}",
         "dataset_name": "{dataset_name}",
         "total_samples": total_samples,
         "processing_time": processing_time,
         "device_used": device_used
     }}

The script should be optimized for the specific model-dataset combination based on the metadata analysis.

Return only the complete Python code for metric_check.py.
"""

        try:
            print("🤖 Generating evaluation script with metadata...")
            response, _ = get_response_from_llm(
                msg=prompt,
                client=self.coder.client,
                model=self.coder.actual_model,
                system_message="You are an expert ML engineer. Generate a robust, metadata-informed evaluation script.",
            )
            code = self._extract_code(response)

            file_path = os.path.join(self.coder.output_dir, "metric_check.py")
            with open(file_path, "w") as f:
                f.write(code)

            print(f"✅ Metadata-informed metric_check.py generated ({len(code)} characters)")
            return True

        except Exception as e:
            print(f"❌ Failed to generate metric_check.py with metadata: {e}")
            return False

    def _read_script_content(self, script_name: str) -> str:
        """Read generated script content"""
        try:
            script_path = os.path.join(self.coder.output_dir, script_name)
            if os.path.exists(script_path):
                with open(script_path, "r", encoding="utf-8") as f:
                    return f.read()
            else:
                return f"# {script_name} not found"
        except Exception as e:
            return f"# Error reading {script_name}: {e}"

    def _extract_code(self, response: str) -> str:
        """Extract code from response"""
        # Find code blocks
        patterns = [
            r"```python\n(.*?)\n```",
            r"```\n(.*?)\n```",
            r"```python(.*?)```",
            r"```(.*?)```",
        ]

        import re

        for pattern in patterns:
            match = re.search(pattern, response, re.DOTALL)
            if match:
                return match.group(1).strip()

        # If no code block found, return entire response
        return response.strip()
