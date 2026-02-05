#!/usr/bin/env python3
"""
Validate all loaders in dataset_loaders and model_loaders directories.
Run each loader in Docker and categorize errors.
"""

import os
import json
import subprocess
import sys
import re
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import argparse

SCRIPTS_DIR = Path(__file__).parent
GPU_ID = 0
TIMEOUT = 300  # 5 minutes per loader
DOCKER_IMAGE = "artifact-linker-verification:latest"


def categorize_error(output: str, exit_code: int) -> tuple:
    """Categorize error based on output and exit code.
    
    Returns (category, subcategory, details)
    """
    output_lower = output.lower()
    
    # Network/Download errors
    if any(x in output_lower for x in ["connectionerror", "connection error", "timeout", "networkerror"]):
        return ("network", "connection", "Network connection failed")
    if any(x in output_lower for x in ["rate limit", "429", "too many requests"]):
        return ("network", "rate_limit", "API rate limit exceeded")
    if "gated repo" in output_lower or "access to model" in output_lower:
        return ("access", "gated", "Model/dataset is gated, requires access")
    if "repository not found" in output_lower or "404" in output_lower:
        return ("access", "not_found", "Repository not found (404)")
    
    # Import/Module errors
    if "modulenotfounderror" in output_lower or "no module named" in output_lower:
        match = re.search(r"no module named ['\"]?([^'\"]+)['\"]?", output_lower)
        module = match.group(1) if match else "unknown"
        return ("import", "missing_module", f"Missing module: {module}")
    if "importerror" in output_lower:
        return ("import", "import_error", "Import error")
    
    # Memory errors
    if "out of memory" in output_lower or "oom" in output_lower or "cuda out of memory" in output_lower:
        return ("memory", "oom", "Out of memory (GPU/CPU)")
    if "killed" in output_lower and exit_code == 137:
        return ("memory", "killed", "Process killed (likely OOM)")
    
    # CUDA errors
    if "cuda" in output_lower and "error" in output_lower:
        if "device" in output_lower:
            return ("cuda", "device", "CUDA device error")
        return ("cuda", "general", "CUDA error")
    
    # Model loading errors
    if "config" in output_lower and "error" in output_lower:
        return ("model", "config", "Model config error")
    if "weight" in output_lower and ("mismatch" in output_lower or "error" in output_lower):
        return ("model", "weights", "Model weights mismatch")
    if "tokenizer" in output_lower and "error" in output_lower:
        return ("model", "tokenizer", "Tokenizer error")
    
    # Dataset errors
    if "dataset" in output_lower and "error" in output_lower:
        return ("dataset", "load", "Dataset loading error")
    if "split" in output_lower and ("not found" in output_lower or "error" in output_lower):
        return ("dataset", "split", "Dataset split not found")
    
    # Syntax/Code errors
    if "syntaxerror" in output_lower:
        return ("code", "syntax", "Syntax error in generated code")
    if "typeerror" in output_lower:
        return ("code", "type", "Type error")
    if "attributeerror" in output_lower:
        return ("code", "attribute", "Attribute error")
    if "keyerror" in output_lower:
        return ("code", "key", "Key error")
    if "valueerror" in output_lower:
        return ("code", "value", "Value error")
    
    # Timeout
    if exit_code == -1 or "timeout" in output_lower:
        return ("timeout", "execution", "Execution timeout")
    
    # Unknown
    return ("unknown", "other", f"Exit code: {exit_code}")


def run_loader_in_docker(loader_path: Path, loader_type: str) -> dict:
    """Run a loader script in Docker and return results."""
    output_dir = loader_path.parent
    script_name = "load_model.py" if loader_type == "model" else "load_dataset.py"
    
    hf_token = os.getenv("HF_TOKEN", "")
    
    docker_cmd = [
        "docker", "run", "--rm",
        "-v", f"{output_dir.absolute()}:/workspace",
        "-w", "/workspace",
        "-m", "32g",
        "--gpus", f"device={GPU_ID}",
        "-e", f"HF_TOKEN={hf_token}",
        "-e", "PYTHONPATH=/workspace",
        DOCKER_IMAGE,
        "bash", "-c",
        f"python {script_name}"
    ]
    
    try:
        result = subprocess.run(
            docker_cmd,
            capture_output=True,
            text=True,
            timeout=TIMEOUT
        )
        output = result.stdout + result.stderr
        exit_code = result.returncode
        success = exit_code == 0
        
        # Check for results.json
        results_file = output_dir / "results.json"
        if results_file.exists():
            try:
                with open(results_file) as f:
                    results_data = json.load(f)
                success = results_data.get("load_success", False)
            except:
                pass
        
        # Save log file
        log_file = output_dir / "validation.log"
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Script: {script_name}\n")
            f.write(f"Exit code: {exit_code}\n")
            f.write(f"Success: {success}\n")
            f.write(f"{'='*60}\n")
            f.write(output)
        
        return {
            "success": success,
            "exit_code": exit_code,
            "output": output[-5000:],  # Last 5000 chars
        }
    except subprocess.TimeoutExpired as e:
        # Save timeout log
        log_file = output_dir / "validation.log"
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Script: {script_name}\n")
            f.write(f"Exit code: -1 (timeout)\n")
            f.write(f"Success: False\n")
            f.write(f"{'='*60}\n")
            f.write(f"Execution timeout after {TIMEOUT} seconds\n")
            if hasattr(e, 'stdout') and e.stdout:
                f.write(f"\nPartial output:\n{e.stdout}\n")
            if hasattr(e, 'stderr') and e.stderr:
                f.write(f"\nPartial stderr:\n{e.stderr}\n")
        
        return {
            "success": False,
            "exit_code": -1,
            "output": "Execution timeout",
        }
    except Exception as e:
        # Save error log
        log_file = output_dir / "validation.log"
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Script: {script_name}\n")
            f.write(f"Exit code: -2 (exception)\n")
            f.write(f"Success: False\n")
            f.write(f"{'='*60}\n")
            f.write(f"Exception: {str(e)}\n")
        
        return {
            "success": False,
            "exit_code": -2,
            "output": str(e),
        }


def validate_loaders(loader_dir: Path, loader_type: str, run_docker: bool = False) -> dict:
    """Validate all loaders in a directory.
    
    Args:
        loader_dir: Path to loader directory (dataset_loaders or model_loaders)
        loader_type: "model" or "dataset"
        run_docker: If True, actually run loaders in Docker. If False, just check existing results.json
    
    Returns:
        Dict with validation results
    """
    script_name = "load_model.py" if loader_type == "model" else "load_dataset.py"
    
    stats = {
        "total": 0,
        "success": 0,
        "failed": 0,
        "no_script": 0,
        "no_results": 0,
        "errors_by_category": defaultdict(lambda: defaultdict(list)),
    }
    
    for item in sorted(loader_dir.iterdir()):
        if not item.is_dir() or item.name.startswith('.') or item.name == "summary.json":
            continue
        
        stats["total"] += 1
        loader_id = item.name.replace("_", "/", 1)
        
        script_path = item / script_name
        results_path = item / "results.json"
        
        if not script_path.exists():
            stats["no_script"] += 1
            continue
        
        if run_docker:
            # Actually run the loader
            print(f"  Running: {loader_id}...")
            result = run_loader_in_docker(script_path, loader_type)
            
            if result["success"]:
                stats["success"] += 1
            else:
                stats["failed"] += 1
                category, subcategory, details = categorize_error(result["output"], result["exit_code"])
                stats["errors_by_category"][category][subcategory].append({
                    "id": loader_id,
                    "details": details,
                    "output_snippet": result["output"][-500:],
                })
        else:
            # Just check existing results.json
            if not results_path.exists():
                stats["no_results"] += 1
                continue
            
            try:
                with open(results_path) as f:
                    results = json.load(f)
                
                if results.get("load_success", False):
                    stats["success"] += 1
                else:
                    stats["failed"] += 1
                    # Try to get error info from results
                    error_msg = results.get("error", "") or ""
                    category, subcategory, details = categorize_error(error_msg, 1)
                    stats["errors_by_category"][category][subcategory].append({
                        "id": loader_id,
                        "details": details,
                    })
            except Exception as e:
                stats["no_results"] += 1
    
    return stats


def print_report(stats: dict, loader_type: str):
    """Print a formatted report of validation results."""
    print(f"\n{'='*80}")
    print(f"📊 {loader_type.upper()} LOADERS VALIDATION REPORT")
    print(f"{'='*80}")
    
    print(f"\nTotal: {stats['total']}")
    print(f"✅ Success: {stats['success']} ({stats['success']/stats['total']*100:.1f}%)" if stats['total'] > 0 else "")
    print(f"❌ Failed: {stats['failed']} ({stats['failed']/stats['total']*100:.1f}%)" if stats['total'] > 0 else "")
    print(f"📄 No script: {stats['no_script']}")
    print(f"📋 No results.json: {stats['no_results']}")
    
    if stats["errors_by_category"]:
        print(f"\n{'='*80}")
        print("🔍 ERROR CATEGORIES")
        print(f"{'='*80}")
        
        for category, subcategories in sorted(stats["errors_by_category"].items()):
            total_in_category = sum(len(items) for items in subcategories.values())
            print(f"\n📁 {category.upper()} ({total_in_category} errors)")
            
            for subcategory, items in sorted(subcategories.items()):
                print(f"  └── {subcategory}: {len(items)}")
                # Show first 3 examples
                for item in items[:3]:
                    print(f"      - {item['id']}: {item['details']}")
                if len(items) > 3:
                    print(f"      ... and {len(items) - 3} more")


def main():
    parser = argparse.ArgumentParser(description="Validate dataset and model loaders")
    parser.add_argument("--type", choices=["model", "dataset", "both"], default="both",
                        help="Type of loaders to validate")
    parser.add_argument("--run", action="store_true",
                        help="Actually run loaders in Docker (slow)")
    parser.add_argument("--gpu-id", type=int, default=0,
                        help="GPU ID to use for Docker")
    parser.add_argument("--json", action="store_true",
                        help="Output results as JSON")
    args = parser.parse_args()
    
    global GPU_ID
    GPU_ID = args.gpu_id
    
    all_stats = {}
    
    if args.type in ["dataset", "both"]:
        print("\n🔍 Validating dataset loaders...")
        dataset_dir = SCRIPTS_DIR / "dataset_loaders"
        if dataset_dir.exists():
            stats = validate_loaders(dataset_dir, "dataset", run_docker=args.run)
            all_stats["dataset"] = stats
            if not args.json:
                print_report(stats, "dataset")
    
    if args.type in ["model", "both"]:
        print("\n🔍 Validating model loaders...")
        model_dir = SCRIPTS_DIR / "model_loaders"
        if model_dir.exists():
            stats = validate_loaders(model_dir, "model", run_docker=args.run)
            all_stats["model"] = stats
            if not args.json:
                print_report(stats, "model")
    
    if args.json:
        # Convert defaultdict to regular dict for JSON serialization
        for loader_type in all_stats:
            errors = all_stats[loader_type]["errors_by_category"]
            all_stats[loader_type]["errors_by_category"] = {
                cat: dict(subcats) for cat, subcats in errors.items()
            }
        print(json.dumps(all_stats, indent=2))
    
    # Save report
    report_path = SCRIPTS_DIR / "loader_validation_report.json"
    with open(report_path, "w") as f:
        # Convert for JSON
        output = {}
        for loader_type in all_stats:
            stats = all_stats[loader_type].copy()
            errors = stats["errors_by_category"]
            stats["errors_by_category"] = {
                cat: dict(subcats) for cat, subcats in errors.items()
            }
            output[loader_type] = stats
        json.dump(output, f, indent=2)
    
    print(f"\n📄 Report saved to: {report_path}")


if __name__ == "__main__":
    main()
