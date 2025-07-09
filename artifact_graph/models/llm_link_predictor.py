import os

import openai
import torch


class OpenAIGPTLinkPredictor:
    def __init__(self, model_name="gpt-3.5-turbo", api_key=None):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = openai.OpenAI(api_key=self.api_key)

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
        node_names,
        G=None,
        model_cards=None,
        dataset_cards=None,
        batch_size=5,
        mode="simple",
    ):
        """
        edge_pairs: torch.LongTensor, [2, num_edges]
        node_names: list[str], node names in order (index aligned with node index)
        G: networkx.Graph, optional, required for neighborhood mode
        model_cards: dict, {model_name: card_info}
        dataset_cards: dict, {dataset_name: card_info}
        mode: "simple" or "neighborhood"
        """
        results = []
        num_edges = edge_pairs.size(1)
        for i in range(0, num_edges, batch_size):
            batch_prompts = []
            for j in range(i, min(i + batch_size, num_edges)):
                src, dst = edge_pairs[0, j].item(), edge_pairs[1, j].item()
                model_name = node_names[dst]
                dataset_name = node_names[src]

                # Get card information
                model_card = model_cards.get(model_name, None) if model_cards else None
                dataset_card = dataset_cards.get(dataset_name, None) if dataset_cards else None

                model_neighbors = None
                dataset_neighbors = None
                if mode == "neighborhood":
                    if G is None:
                        raise ValueError(
                            "G (networkx graph) must be provided for neighborhood mode."
                        )
                    # model_neighbors: this model's accuracy on other datasets
                    model_neighbors = []
                    for neighbor in G.neighbors(model_name):
                        if neighbor != dataset_name and "accuracy" in G[model_name][neighbor]:
                            model_neighbors.append((neighbor, G[model_name][neighbor]["accuracy"]))
                    # dataset_neighbors: other models' accuracy on this dataset
                    dataset_neighbors = []
                    for neighbor in G.neighbors(dataset_name):
                        if neighbor != model_name and "accuracy" in G[neighbor][dataset_name]:
                            dataset_neighbors.append(
                                (neighbor, G[neighbor][dataset_name]["accuracy"])
                            )
                prompt = self._build_prompt(
                    model_name,
                    dataset_name,
                    model_card,
                    dataset_card,
                    model_neighbors,
                    dataset_neighbors,
                    mode=mode,
                )
                batch_prompts.append({"role": "user", "content": prompt})

            for prompt in batch_prompts:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[prompt],
                    max_tokens=8,
                    temperature=0.0,
                )
                answer = response.choices[0].message.content.strip()
                try:
                    prob = float(answer.split()[0])
                    prob = max(0.0, min(1.0, prob))
                except Exception:
                    prob = None
                results.append(prob)
        return results


if __name__ == "__main__":
    edge_pairs = torch.randint(0, 100, (2, 10))  # 10 edges
    node_names = [f"node_{i}" for i in range(100)]
    predictor = OpenAIGPTLinkPredictor(model_name="gpt-4o")
    results = predictor.predict(edge_pairs, node_names)
    print(results)
