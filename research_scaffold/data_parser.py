import os
import json
from litellm import completion
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

readme_dir = "model_readmes_download_ranks"
output_dir = "eval_datasets_json_download_ranks"
os.makedirs(output_dir, exist_ok=True)

# Thread-safe progress bar
pbar_lock = threading.Lock()

def ask_gpt4o_for_eval_dataset(readme_text):
    """Extract evaluation datasets from README text using GPT-4o."""
    system_prompt = (
        "You are a helpful assistant that extracts evaluation dataset names and corresponding metrics "
        "from Hugging Face model README files. "
        "Given the README content, return a JSON dictionary with dataset names as keys and list of evaluation metrics as values. "
        "For example, if the README mentions 'We evaluate on the SQuAD dataset using F1 and EM metrics', "
        "the output should be: "
        '{"SQuAD": {"F1": null, "EM": null]}.\n'
        "If the evaluation dataset is mentioned and the number is mentioned, include it as well, like: "
        '{"SQuAD": {"F1": 0.9, "EM": 0.8}.\n'
        "Only include datasets used for evaluation, not training. If no evaluation dataset is mentioned, return an empty dictionary: {}"
    )
    try:
        response = completion(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": readme_text[:12000]}
            ],
            temperature=0
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"✗ GPT-4o-mini error: {e}")
        return "{}"

def process_single_file(filename, pbar=None):
    """Process a single README file."""
    if not filename.endswith(".md"):
        return None
        
    model_id_safe = filename[:-3]  # remove .md
    output_path = os.path.join(output_dir, f"{model_id_safe}.json")
    
    # Skip if already processed
    if os.path.exists(output_path):
        if pbar:
            with pbar_lock:
                pbar.update(1)
                pbar.set_description(f"Skipped {model_id_safe}")
        return f"Skipped {model_id_safe} (already exists)"
    
    try:
        readme_path = os.path.join(readme_dir, filename)
        with open(readme_path, "r", encoding="utf-8") as f:
            readme_text = f.read()
        
        raw_response = ask_gpt4o_for_eval_dataset(readme_text)
        raw_response = raw_response.replace("```json", "").replace("```", "").strip()
        #print(raw_response)
        # Try parsing as JSON
        try:
            parsed = json.loads(raw_response)
            if isinstance(parsed, dict):
                with open(output_path, "w", encoding="utf-8") as f_out:
                    json.dump(parsed, f_out, indent=2, ensure_ascii=False)
                result = f"✅ Processed {model_id_safe}"
            else:
                raise ValueError("Response was not a dict")
                
        except Exception as e:
            print(f"✗ Invalid JSON for {model_id_safe}: {e}")
            with open(output_path, "w", encoding="utf-8") as f_out:
                json.dump({}, f_out)  # Save as empty JSON
            result = f"⚠️ Processed {model_id_safe} (invalid JSON, saved empty)"
            
    except Exception as e:
        result = f"✗ Failed to process {model_id_safe}: {e}"
    
    if pbar:
        with pbar_lock:
            pbar.update(1)
            pbar.set_description(result.split()[-1] if "✅" in result else "Error")
    
    return result

def main(max_workers=5):
    """Main function to process all README files in parallel."""
    # Get all README files
    all_files = [f for f in os.listdir(readme_dir) if f.endswith(".md")]
    
    # Check how many already exist
    existing_count = sum(1 for f in all_files 
                        if os.path.exists(os.path.join(output_dir, f[:-3] + ".json")))
    
    print(f"Found {len(all_files)} README files")
    print(f"Already processed: {existing_count}")
    print(f"To process: {len(all_files) - existing_count}")
    print(f"Using {max_workers} parallel workers")
    
    if len(all_files) == existing_count:
        print("All files already processed!")
        return
    
    # Process files in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        with tqdm(total=len(all_files), desc="Processing") as pbar:
            # Submit all tasks
            future_to_file = {
                executor.submit(process_single_file, filename, pbar): filename 
                for filename in all_files
            }
            
            # Collect results
            results = []
            for future in as_completed(future_to_file):
                filename = future_to_file[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as exc:
                    error_msg = f"✗ {filename} generated an exception: {exc}"
                    results.append(error_msg)
                    print(error_msg)
    
    # Print summary
    processed = sum(1 for r in results if "✅" in r)
    skipped = sum(1 for r in results if "Skipped" in r)
    errors = sum(1 for r in results if "✗" in r)
    warnings = sum(1 for r in results if "⚠️" in r)
    
    print(f"\n📊 Summary:")
    print(f"  ✅ Processed: {processed}")
    print(f"  ⏭️ Skipped: {skipped}")
    print(f"  ⚠️ Warnings: {warnings}")
    print(f"  ✗ Errors: {errors}")

if __name__ == "__main__":
    # You can adjust max_workers based on your API rate limits
    # For GPT-4o, start with 3-5 workers to avoid rate limiting
    main(max_workers=4)