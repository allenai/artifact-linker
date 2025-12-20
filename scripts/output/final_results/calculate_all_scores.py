#!/usr/bin/env python3
"""
Master script to calculate all evaluation scores for LLM experiments.
Runs link prediction, link ranking, attribute prediction, and attribute ranking evaluations.
"""

import subprocess
import sys
from pathlib import Path


def run_script(script_name):
    """Run an evaluation script and capture its output."""
    script_path = Path(__file__).parent / script_name
    
    if not script_path.exists():
        print(f"❌ Script not found: {script_path}")
        return False
    
    print(f"\n{'='*60}")
    print(f"🔄 Running {script_name}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=script_path.parent,
            capture_output=False,  # Show output in real-time
            text=True
        )
        
        if result.returncode == 0:
            print(f"✅ {script_name} completed successfully")
            return True
        else:
            print(f"❌ {script_name} failed with return code {result.returncode}")
            return False
            
    except Exception as e:
        print(f"❌ Error running {script_name}: {e}")
        return False


def main():
    """Run all evaluation scripts."""
    
    print("🚀 Starting comprehensive evaluation of LLM experiment results...")
    
    # List of evaluation scripts to run
    scripts = [
        "calculate_link_prediction_score.py",
        "calculate_link_ranking_score.py", 
        "calculate_attribute_prediction_score.py",
        "calculate_attribute_ranking_score.py"
    ]
    
    results = {}
    
    for script in scripts:
        success = run_script(script)
        results[script] = success
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 EVALUATION SUMMARY")
    print(f"{'='*60}")
    
    for script, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"  - {script:<35} {status}")
    
    total_success = sum(results.values())
    total_scripts = len(scripts)
    
    print(f"\nOverall: {total_success}/{total_scripts} scripts completed successfully")
    
    if total_success == total_scripts:
        print("🎉 All evaluations completed successfully!")
    else:
        print("⚠️  Some evaluations failed. Check the output above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
