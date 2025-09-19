import argparse

from tiny_scientist import TinyScientist


def main():
    """
    This script uses TinyScientist to automate the process of reproducing
    a model evaluation on a given dataset for a specific task.
    """
    parser = argparse.ArgumentParser(
        description="Reproduce a model evaluation using TinyScientist."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="The Hugging Face model name (e.g., 'dslim/bert-base-NER').",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="The Hugging Face dataset name (e.g., 'eriktks/conll2003').",
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="The task to perform (e.g., 'Named Entity Recognition', 'Text Classification', 'Summarization').",
    )
    parser.add_argument(
        "--gpt_model", type=str, default="gpt-4o", help="The GPT model to use for TinyScientist."
    )

    args = parser.parse_args()

    # Before running, ensure you have tiny_scientist installed:
    # pip install tiny-scientist

    # Initialize TinyScientist with the specified model
    print(f"Initializing TinyScientist with model: {args.gpt_model}")
    scientist = TinyScientist(model=args.gpt_model)

    # 1. Define the research intent based on user input.
    # This string is the core instruction for TinyScientist.
    intent = (
        f"I want to write a script to reproduce the evaluation of the Hugging Face model '{args.model}' "
        f"on the dataset '{args.dataset}'. The task is {args.task}. "
        f"The script should load the model and dataset, run the evaluation, "
        f"and report the standard evaluation metrics for this task."
    )

    print(f"🔬 Intent: {intent}")

    # Step 1: Generate a json-format research idea/plan
    print("\nStep 1: Thinking and generating a research plan...")
    idea = scientist.think(intent=intent)
    if not idea:
        print("❌ TinyScientist failed to generate an idea. Please check your intent or API key.")
        return
    print("✅ Research plan generated.")
    print(idea)

    # Step 2: Generate and run the experiment code
    print("\nStep 2: Generating and running experiment code...")
    status, experiment_dir = scientist.code(idea=idea)

    # If the experiments run successfully, proceed to writing the paper
    if status is True:
        print(f"✅ Experiments completed successfully. Results are in: {experiment_dir}")

        # Step 3: Write a research paper based on the findings
        print("\nStep 3: Writing a research paper...")
        pdf_path = scientist.write(idea=idea, experiment_dir=experiment_dir)
        if not pdf_path:
            print("❌ Failed to write the paper.")
            return
        print(f"✅ Paper written and saved to: {pdf_path}")

        # Step 4: Review the generated paper
        print("\nStep 4: Reviewing the paper...")
        review = scientist.review(pdf_path=pdf_path)
        print("✅ Review complete.")
        print("\n--- Paper Review ---")
        print(review)
        print("--------------------")
    else:
        print(f"❌ Experiments failed. Check the logs in the experiment directory: {experiment_dir}")


if __name__ == "__main__":
    main()
