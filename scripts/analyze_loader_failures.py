#!/usr/bin/env python3
"""
Analyze loader failures using LLM and categorize them.
"""

import os
import json
import sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import argparse

# Set up LiteLLM
import litellm
litellm.drop_params = True
os.environ["LITELLM_DROP_PARAMS"] = "true"

SCRIPTS_DIR = Path(__file__).parent

# Failure categories
FAILURE_CATEGORIES = {
    "gated_access": "Model/dataset requires gated access or authentication",
    "not_found": "Repository not found (404) or deleted",
    "network_error": "Network connection error, timeout, or rate limit",
    "missing_dependency": "Missing Python package or module",
    "incompatible_format": "Incompatible model/dataset format or architecture",
    "memory_error": "Out of memory (GPU or CPU)",
    "cuda_error": "CUDA/GPU related error",
    "code_bug": "Bug in the generated loader code (syntax, logic error)",
    "deprecated_api": "Using deprecated API or outdated library version",
    "data_processing": "Error in data processing or feature extraction",
    "model_loading": "Error loading model weights or config",
    "tokenizer_error": "Tokenizer initialization or encoding error",
    "unknown": "Unknown or unclear error",
}

ANALYSIS_PROMPT = """You are an expert ML engineer analyzing loader script failures.

Given the following information about a failed loader:
- Loader type: {loader_type}
- ID: {loader_id}
- Script content (if available):
```python
{script_content}
```

- Error log:
```
{error_log}
```

Analyze the failure and classify it into ONE of these categories:
{categories}

Respond in JSON format:
{{
    "category": "<category_key>",
    "reason": "<brief explanation of why it failed>",
    "fix_suggestion": "<how to potentially fix this>"
}}

Only output the JSON, nothing else.
"""


def get_failed_loaders(loader_dir: Path, loader_type: str) -> list:
    """Get list of failed loaders (no results.json or load_success=False)."""
    script_name = "load_model.py" if loader_type == "model" else "load_dataset.py"
    
    failed = []
    for item in sorted(loader_dir.iterdir()):
        if not item.is_dir() or item.name.startswith('.') or item.name == "summary.json":
            continue
        
        loader_id = item.name.replace("_", "/", 1)
        script_path = item / script_name
        results_path = item / "results.json"
        validation_log = item / "validation.log"
        
        # Skip if no script
        if not script_path.exists():
            continue
        
        # Check if failed
        is_failed = False
        error_source = None
        
        if not results_path.exists():
            is_failed = True
            error_source = "no_results"
        else:
            try:
                with open(results_path) as f:
                    results = json.load(f)
                if not results.get("load_success", False):
                    is_failed = True
                    error_source = "load_failed"
            except:
                is_failed = True
                error_source = "invalid_results"
        
        if is_failed:
            # Read script content
            script_content = ""
            try:
                script_content = script_path.read_text(encoding="utf-8")[:3000]
            except:
                pass
            
            # Read error log
            error_log = ""
            if validation_log.exists():
                try:
                    error_log = validation_log.read_text(encoding="utf-8")[-4000:]
                except:
                    pass
            
            # Try to get error from results.json if exists
            if results_path.exists() and not error_log:
                try:
                    with open(results_path) as f:
                        results = json.load(f)
                    if "error" in results:
                        error_log = str(results.get("error", ""))
                except:
                    pass
            
            failed.append({
                "id": loader_id,
                "path": str(item),
                "error_source": error_source,
                "script_content": script_content,
                "error_log": error_log,
            })
    
    return failed


def analyze_with_llm(loader_info: dict, loader_type: str, llm_model: str = "gpt-4o-mini") -> dict:
    """Use LLM to analyze a single failure."""
    categories_str = "\n".join([f"- {k}: {v}" for k, v in FAILURE_CATEGORIES.items()])
    
    prompt = ANALYSIS_PROMPT.format(
        loader_type=loader_type,
        loader_id=loader_info["id"],
        script_content=loader_info["script_content"][:2000] if loader_info["script_content"] else "(not available)",
        error_log=loader_info["error_log"][:3000] if loader_info["error_log"] else "(not available)",
        categories=categories_str,
    )
    
    try:
        response = litellm.completion(
            model=llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=500,
        )
        
        content = response.choices[0].message.content.strip()
        
        # Parse JSON response
        # Handle markdown code blocks
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1])
        
        result = json.loads(content)
        return {
            "category": result.get("category", "unknown"),
            "reason": result.get("reason", ""),
            "fix_suggestion": result.get("fix_suggestion", ""),
        }
    except Exception as e:
        return {
            "category": "unknown",
            "reason": f"LLM analysis failed: {str(e)}",
            "fix_suggestion": "",
        }


def analyze_without_llm(loader_info: dict) -> dict:
    """Quick rule-based analysis without LLM."""
    error_log = (loader_info.get("error_log", "") or "").lower()
    script = (loader_info.get("script_content", "") or "").lower()
    
    # Check common patterns
    if "gated repo" in error_log or "access to model" in error_log or "401" in error_log:
        return {"category": "gated_access", "reason": "Requires authentication", "fix_suggestion": "Add HF token with access"}
    
    if "repository not found" in error_log or "404" in error_log:
        return {"category": "not_found", "reason": "Repository not found", "fix_suggestion": "Check if model/dataset still exists"}
    
    if "connectionerror" in error_log or "timeout" in error_log or "429" in error_log or "rate limit" in error_log:
        return {"category": "network_error", "reason": "Network issue", "fix_suggestion": "Retry later"}
    
    if "modulenotfounderror" in error_log or "no module named" in error_log:
        return {"category": "missing_dependency", "reason": "Missing Python package", "fix_suggestion": "Install required package"}
    
    if "out of memory" in error_log or "oom" in error_log or "cuda out of memory" in error_log:
        return {"category": "memory_error", "reason": "Out of memory", "fix_suggestion": "Use smaller batch or lower precision"}
    
    if "cuda" in error_log and "error" in error_log:
        return {"category": "cuda_error", "reason": "GPU error", "fix_suggestion": "Check CUDA compatibility"}
    
    if "syntaxerror" in error_log:
        return {"category": "code_bug", "reason": "Syntax error in generated code", "fix_suggestion": "Regenerate loader"}
    
    if "typeerror" in error_log or "attributeerror" in error_log or "keyerror" in error_log:
        return {"category": "code_bug", "reason": "Runtime error in code", "fix_suggestion": "Fix code logic"}
    
    if "deprecated" in error_log or "removed" in error_log:
        return {"category": "deprecated_api", "reason": "Using deprecated API", "fix_suggestion": "Update to new API"}
    
    if "tokenizer" in error_log and "error" in error_log:
        return {"category": "tokenizer_error", "reason": "Tokenizer issue", "fix_suggestion": "Check tokenizer compatibility"}
    
    if not error_log:
        return {"category": "unknown", "reason": "No error log available", "fix_suggestion": "Run validation to get logs"}
    
    return {"category": "unknown", "reason": "Could not determine from log", "fix_suggestion": "Manual inspection needed"}


def main():
    parser = argparse.ArgumentParser(description="Analyze loader failures")
    parser.add_argument("--type", choices=["model", "dataset", "both"], default="both",
                        help="Type of loaders to analyze")
    parser.add_argument("--llm", action="store_true",
                        help="Use LLM for detailed analysis (slower, more accurate)")
    parser.add_argument("--llm-model", default="gpt-4o-mini",
                        help="LLM model to use")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit number of failures to analyze (0 = all)")
    parser.add_argument("--output", default="loader_failure_analysis.json",
                        help="Output file path")
    args = parser.parse_args()
    
    all_results = {}
    
    for loader_type in ["dataset", "model"]:
        if args.type != "both" and args.type != loader_type:
            continue
        
        loader_dir = SCRIPTS_DIR / f"{loader_type}_loaders"
        if not loader_dir.exists():
            continue
        
        print(f"\n{'='*60}")
        print(f"Analyzing {loader_type} loader failures...")
        print(f"{'='*60}")
        
        failed = get_failed_loaders(loader_dir, loader_type)
        print(f"Found {len(failed)} failed loaders")
        
        if args.limit > 0:
            failed = failed[:args.limit]
            print(f"Limiting to first {args.limit}")
        
        results = {
            "total_failed": len(failed),
            "by_category": defaultdict(list),
            "details": [],
        }
        
        for i, loader_info in enumerate(failed):
            print(f"  [{i+1}/{len(failed)}] Analyzing {loader_info['id']}...")
            
            if args.llm:
                analysis = analyze_with_llm(loader_info, loader_type, args.llm_model)
            else:
                analysis = analyze_without_llm(loader_info)
            
            result = {
                "id": loader_info["id"],
                "path": loader_info["path"],
                "error_source": loader_info["error_source"],
                **analysis,
            }
            
            results["details"].append(result)
            results["by_category"][analysis["category"]].append({
                "id": loader_info["id"],
                "reason": analysis["reason"],
            })
        
        # Convert defaultdict to dict
        results["by_category"] = dict(results["by_category"])
        all_results[loader_type] = results
        
        # Print summary
        print(f"\n📊 {loader_type.upper()} FAILURE SUMMARY")
        print("-" * 40)
        for category, items in sorted(results["by_category"].items(), key=lambda x: -len(x[1])):
            cat_desc = FAILURE_CATEGORIES.get(category, category)
            print(f"  {category}: {len(items)} ({len(items)/results['total_failed']*100:.1f}%)")
            # Show first 3 examples
            for item in items[:3]:
                print(f"    - {item['id']}: {item['reason'][:50]}...")
            if len(items) > 3:
                print(f"    ... and {len(items) - 3} more")
    
    # Save results
    output_path = SCRIPTS_DIR / args.output
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n📄 Results saved to: {output_path}")
    
    # Print overall summary
    print(f"\n{'='*60}")
    print("OVERALL SUMMARY")
    print(f"{'='*60}")
    
    total_by_category = defaultdict(int)
    for loader_type, results in all_results.items():
        for category, items in results.get("by_category", {}).items():
            total_by_category[category] += len(items)
    
    total = sum(total_by_category.values())
    print(f"\nTotal failures analyzed: {total}")
    print(f"\nBy category:")
    for category, count in sorted(total_by_category.items(), key=lambda x: -x[1]):
        cat_desc = FAILURE_CATEGORIES.get(category, category)
        print(f"  {category}: {count} ({count/total*100:.1f}%) - {cat_desc}")


if __name__ == "__main__":
    main()
