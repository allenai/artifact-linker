import os
import networkx as nx
import openai
from tqdm import tqdm

from artifact_graph.collectors.dataset_collector import DatasetCollector
from artifact_graph.collectors.model_collector import ModelCollector


class LLMLinkPredictor:
    def __init__(self, model_name="gpt-3.5-turbo", api_key=None):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = openai.OpenAI(api_key=self.api_key)
        self.model_collector = ModelCollector(hf_token=os.getenv("HF_TOKEN"))
        self.dataset_collector = DatasetCollector(hf_token=os.getenv("HF_TOKEN"))

    def _build_prompt(
        self,
        model_name,
        dataset_name,
        model_card=None,
        dataset_card=None,
        model_neighbors=None,
        dataset_neighbors=None,
        mode="simple",
    ):
        if mode == "simple":
            prompt = f"Given a machine learning model named '{model_name}' and a dataset named '{dataset_name}'"
            if model_card:
                prompt += f"\nModel card: {model_card}"
            if dataset_card:
                prompt += f"\nDataset card: {dataset_card}"
            prompt += "\nPlease predict the expected accuracy (as a float between 0 and 1) that this model would achieve on this dataset. Only output a single float number between 0 and 1."
            return prompt
        elif mode == "neighborhood":
            prompt = f"Given a machine learning model named '{model_name}' and a dataset named '{dataset_name}'"
            if model_card:
                prompt += f"\nModel card: {model_card}"
            if dataset_card:
                prompt += f"\nDataset card: {dataset_card}"
            prompt += "\nThe model's performance on other datasets:\n"
            if model_neighbors:
                for ds, acc in model_neighbors:
                    prompt += f"- {ds}: {acc:.2f}\n"
            else:
                prompt += "- (no other datasets)\n"
            prompt += "The dataset's performance with other models:\n"
            if dataset_neighbors:
                for mdl, acc in dataset_neighbors:
                    prompt += f"- {mdl}: {acc:.2f}\n"
            else:
                prompt += "- (no other models)\n"
            prompt += (
                "Please predict the expected accuracy (as a float between 0 and 1) that this model would achieve on this dataset. "
                "Only output a single float number between 0 and 1."
            )
            return prompt
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def predict(
        self,
        edge_pairs,
        G,
        model_dir="output/models",
        dataset_dir="output/datasets",
        mode="simple",
    ):
        """
        Predicts accuracy for given model-dataset pairs.

        Args:
            edge_pairs (list of tuples): List of (model_name, dataset_name) pairs.
            G (nx.Graph): The artifact graph.
            model_dir (str): Directory for model artifacts.
            dataset_dir (str): Directory for dataset artifacts.
            mode (str): "simple" or "neighborhood".
        """
        results = []
        for model_name, dataset_name in tqdm(edge_pairs, desc="Predicting Links"):
            try:
                # Load metadata and readme using collectors
                model_card_bytes = ModelCollector.load_readme(
                    model_name, readme_dir=os.path.join(model_dir, "readmes")
                )
                if model_card_bytes:
                    model_card = model_card_bytes.decode("utf-8", errors="ignore")
                else:
                    model_card = "Model card not available."
                    print(f"Warning: Could not find README for model {model_name}")

                dataset_card_bytes = DatasetCollector.load_readme(
                    dataset_name, readme_dir=os.path.join(dataset_dir, "readmes")
                )
                if dataset_card_bytes:
                    dataset_card = dataset_card_bytes.decode("utf-8", errors="ignore")
                else:
                    dataset_card = "Dataset card not available."
                    print(f"Warning: Could not find README for dataset {dataset_name}")

                # Get neighborhood information
                model_neighbors = None
                dataset_neighbors = None
                if mode == "neighborhood":
                    model_neighbors = []
                    for neighbor in G.neighbors(model_name):
                        if neighbor != dataset_name and G.nodes[neighbor].get("type") == "dataset" and "accuracy" in G[model_name][neighbor]:
                            model_neighbors.append((neighbor, G[model_name][neighbor]["accuracy"]))
                    
                    dataset_neighbors = []
                    for neighbor in G.neighbors(dataset_name):
                        if neighbor != model_name and G.nodes[neighbor].get("type") == "model" and "accuracy" in G[neighbor][dataset_name]:
                            dataset_neighbors.append((neighbor, G[neighbor][dataset_name]["accuracy"]))

                # Build prompt
                prompt = self._build_prompt(
                    model_name,
                    dataset_name,
                    model_card,
                    dataset_card,
                    model_neighbors,
                    dataset_neighbors,
                    mode=mode,
                )

                # Get prediction from OpenAI
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=8,
                    temperature=0.0,
                )
                answer = response.choices[0].message.content.strip()
                
                # Parse result
                try:
                    prob = float(answer.split()[0])
                    prob = max(0.0, min(1.0, prob))
                except Exception:
                    prob = None
                
                results.append(prob)

            except Exception as e:
                print(f"Error predicting for ({model_name}, {dataset_name}): {e}")
                results.append(None)
        
        return results
