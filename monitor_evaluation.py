#!/usr/bin/env python3
"""
Script to evaluate monitor performance by testing all log files against three monitors.
Uses worker pool pattern with configurable max workers for parallelization.
"""

import os
import glob
import json
import argparse
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Tuple
from tqdm import tqdm
from monitor import Monitor
from dotenv import load_dotenv


@dataclass
class MonitorTestResult:
    """Result of testing a single log file against a monitor."""
    file_path: str
    monitor_name: str
    is_suspicious: bool
    reasoning: str
    content: str
    error: str = None


@dataclass
class MonitorPerformance:
    """Performance metrics for a monitor across all test files."""
    monitor_name: str
    total_files: int
    suspicious_count: int
    non_suspicious_count: int
    error_count: int
    suspicious_rate: float
    results: List[MonitorTestResult]


class MonitorEvaluator:
    """Evaluates monitor performance by testing log files."""
    
    def __init__(self, monitors: Dict[str, Monitor], log_path: str = "actor_logs", max_workers: int = None):
        load_dotenv()
        self.max_workers = max_workers or int(os.environ.get("MAX_WORKERS", 20))
        self.monitors = monitors
        self.log_path = log_path
        
        print(f"Initialized MonitorEvaluator with {self.max_workers} max workers")
        print(f"Testing against {len(self.monitors)} monitors: {list(self.monitors.keys())}")
    
    def get_log_files(self) -> List[str]:
        """Get all log files from the specified directory."""
        log_files = []
        
        # Find all .log files in subdirectories
        pattern = os.path.join(self.log_path, "**", "*.log")
        log_files = glob.glob(pattern, recursive=True)
        
        print(f"Found {len(log_files)} log files to test")
        return log_files
    
    def read_log_content(self, file_path: str) -> str:
        """Read the content of a log file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return ""
    
    def test_file_against_monitor(self, file_path: str, monitor_name: str) -> MonitorTestResult:
        """Test a single file against a single monitor."""
        try:
            content = self.read_log_content(file_path)
            if not content:
                return MonitorTestResult(
                    file_path=file_path,
                    monitor_name=monitor_name,
                    is_suspicious=False,
                    reasoning="",
                    content="",
                    error="Empty or unreadable file"
                )
            
            monitor = self.monitors[monitor_name]
            result = monitor.monitor(content)
            
            return MonitorTestResult(
                file_path=file_path,
                monitor_name=monitor_name,
                is_suspicious=result.is_suspicious,
                reasoning=result.reasoning,
                content=result.content,
                error=None
            )
            
        except Exception as e:
            return MonitorTestResult(
                file_path=file_path,
                monitor_name=monitor_name,
                is_suspicious=False,
                reasoning="",
                content="",
                error=str(e)
            )
    
    def test_file_against_all_monitors(self, file_path: str) -> List[MonitorTestResult]:
        """Test a single file against all monitors."""
        results = []
        for monitor_name in self.monitors.keys():
            result = self.test_file_against_monitor(file_path, monitor_name)
            results.append(result)
        return results
    
    def evaluate_monitors(self) -> Dict[str, MonitorPerformance]:
        """Evaluate all monitors against all log files using parallel processing."""
        log_files = self.get_log_files()
        
        if not log_files:
            print("No log files found!")
            return {}
        
        # Initialize performance tracking
        monitor_performance = {
            monitor_name: MonitorPerformance(
                monitor_name=monitor_name,
                total_files=0,
                suspicious_count=0,
                non_suspicious_count=0,
                error_count=0,
                suspicious_rate=0.0,
                results=[]
            )
            for monitor_name in self.monitors.keys()
        }
        
        print(f"Starting evaluation with {self.max_workers} workers...")
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(self.test_file_against_all_monitors, file_path): file_path
                for file_path in log_files
            }
            
            # Process results as they complete
            with tqdm(total=len(log_files), desc="Testing files against monitors") as pbar:
                for future in as_completed(future_to_file):
                    file_path = future_to_file[future]
                    try:
                        file_results = future.result()
                        
                        # Update performance metrics for each monitor
                        for result in file_results:
                            monitor_name = result.monitor_name
                            perf = monitor_performance[monitor_name]
                            
                            perf.total_files += 1
                            perf.results.append(result)
                            
                            if result.error:
                                perf.error_count += 1
                            elif result.is_suspicious:
                                perf.suspicious_count += 1
                            else:
                                perf.non_suspicious_count += 1
                        
                        pbar.update(1)
                        
                    except Exception as exc:
                        print(f'File {file_path} generated an exception: {exc}')
                        pbar.update(1)
        
        # Calculate suspicious rates
        for monitor_name, perf in monitor_performance.items():
            if perf.total_files > 0:
                perf.suspicious_rate = perf.suspicious_count / (perf.total_files - perf.error_count) if (perf.total_files - perf.error_count) > 0 else 0.0
        
        return monitor_performance
    
    def print_results(self, monitor_performance: Dict[str, MonitorPerformance]):
        """Print the evaluation results in a formatted way."""
        print("\n" + "="*80)
        print("MONITOR EVALUATION RESULTS")
        print(f"Log path: {self.log_path}")
        print("="*80)
        
        for monitor_name, perf in monitor_performance.items():
            print(f"\n{monitor_name.upper()}:")
            print(f"  Total files tested: {perf.total_files}")
            print(f"  Suspicious: {perf.suspicious_count}")
            print(f"  Non-suspicious: {perf.non_suspicious_count}")
            print(f"  Errors: {perf.error_count}")
            print(f"  Suspicious rate: {perf.suspicious_rate:.2%}")
        
        print("\n" + "="*80)
        print("COMPARISON SUMMARY")
        print("="*80)
        
        # Sort monitors by suspicious rate
        sorted_monitors = sorted(
            monitor_performance.items(),
            key=lambda x: x[1].suspicious_rate,
            reverse=True
        )
        
        print(f"{'Monitor':<20} {'Suspicious Rate':<15} {'Total Files':<12} {'Errors':<8}")
        print("-" * 60)
        for monitor_name, perf in sorted_monitors:
            print(f"{monitor_name:<20} {perf.suspicious_rate:<15.2%} {perf.total_files:<12} {perf.error_count:<8}")
    
    def save_detailed_results(self, monitor_performance: Dict[str, MonitorPerformance], home_dir: str = "monitor_logs"):
        """Save detailed results to a JSON file."""
        time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file: str = f"{home_dir}/monitor_evaluation_results_{time_str}.json"
        # Convert to serializable format
        serializable_results = {}
        for monitor_name, perf in monitor_performance.items():
            serializable_results[monitor_name] = {
                "monitor_name": perf.monitor_name,
                "log_path": self.log_path,
                "total_files": perf.total_files,
                "suspicious_count": perf.suspicious_count,
                "non_suspicious_count": perf.non_suspicious_count,
                "error_count": perf.error_count,
                "suspicious_rate": perf.suspicious_rate,
                "results": [
                    {
                        "file_path": result.file_path,
                        "monitor_name": result.monitor_name,
                        "is_suspicious": result.is_suspicious,
                        "reasoning": result.reasoning,
                        "content": result.content,
                        "error": result.error
                    }
                    for result in perf.results
                ]
            }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        print(f"\nDetailed results saved to: {output_file}")

def parse_from_actor_log_name(log_name: str) -> tuple[str, str]:
    return log_name.split("/")[-1].split("_")[2], log_name.split("/")[-1].split("_")[5]

def evaluate_grid_false_negative():
    physics_monitors = {
        "gpt-oss-20b": Monitor("Physics", "openai/gpt-oss-20b"),
        "gpt-oss-120b": Monitor("Physics", "openai/gpt-oss-120b"),
        "Qwen3-Next-80B": Monitor("Physics", "Qwen/Qwen3-Next-80B-A3B-Thinking"),
    }
    biology_monitors = {
        "gpt-oss-20b": Monitor("Biology", "openai/gpt-oss-20b"),
        "gpt-oss-120b": Monitor("Biology", "openai/gpt-oss-120b"),
        "Qwen3-Next-80B": Monitor("Biology", "Qwen/Qwen3-Next-80B-A3B-Thinking"),
    }
    chemistry_monitors = {
        "gpt-oss-20b": Monitor("Chemistry", "openai/gpt-oss-20b"),
        "gpt-oss-120b": Monitor("Chemistry", "openai/gpt-oss-120b"),
        "Qwen3-Next-80B": Monitor("Chemistry", "Qwen/Qwen3-Next-80B-A3B-Thinking"),
    }
    log_paths = [
        "actor_logs/response_DeepSeek-R1_Physics_disguised_as_Biology_20250921_222843",
        "actor_logs/response_DeepSeek-R1_Biology_disguised_as_Chemistry_20250921_224441",
        "actor_logs/response_DeepSeek-R1_Chemistry_disguised_as_Physics_20250921_223238",
        "actor_logs/response_gpt-oss-20b_Biology_disguised_as_Chemistry_20250921_222532",
        "actor_logs/response_gpt-oss-20b_Chemistry_disguised_as_Physics_20250921_221707",
        "actor_logs/response_gpt-oss-20b_Physics_disguised_as_Biology_20250921_220832",
        "actor_logs/response_Qwen3-Next-80B-A3B-Thinking_Biology_disguised_as_Chemistry_20250921_222310",
        "actor_logs/response_Qwen3-Next-80B-A3B-Thinking_Chemistry_disguised_as_Physics_20250921_221107",
        "actor_logs/response_Qwen3-Next-80B-A3B-Thinking_Physics_disguised_as_Biology_20250921_220346",
        "actor_logs/response_gpt-oss-120b_Biology_disguised_as_Chemistry_20250921_222506",
        "actor_logs/response_gpt-oss-120b_Chemistry_disguised_as_Physics_20250921_221516",
        "actor_logs/response_gpt-oss-120b_Physics_disguised_as_Biology_20250921_220706",
    ]
    for log_path in log_paths:
        monitored, _ = parse_from_actor_log_name(log_path)
        if monitored == "Physics":
            monitors = physics_monitors
        elif monitored == "Biology":
            monitors = biology_monitors
        elif monitored == "Chemistry":
            monitors = chemistry_monitors
        evaluator = MonitorEvaluator(monitors=monitors, log_path=log_path)
        results = evaluator.evaluate_monitors()
        if results:
            evaluator.print_results(results)
            evaluator.save_detailed_results(results)
        else:
            print("No results to display.")

def evaluate_grid_false_positive():
    physics_monitors = {
        "gpt-oss-20b": Monitor("Physics", "openai/gpt-oss-20b"),
        "gpt-oss-120b": Monitor("Physics", "openai/gpt-oss-120b"),
        "Qwen3-Next-80B": Monitor("Physics", "Qwen/Qwen3-Next-80B-A3B-Thinking"),
    }
    biology_monitors = {
        "gpt-oss-20b": Monitor("Biology", "openai/gpt-oss-20b"),
        "gpt-oss-120b": Monitor("Biology", "openai/gpt-oss-120b"),
        "Qwen3-Next-80B": Monitor("Biology", "Qwen/Qwen3-Next-80B-A3B-Thinking"),
    }
    chemistry_monitors = {
        "gpt-oss-20b": Monitor("Chemistry", "openai/gpt-oss-20b"),
        "gpt-oss-120b": Monitor("Chemistry", "openai/gpt-oss-120b"),
        "Qwen3-Next-80B": Monitor("Chemistry", "Qwen/Qwen3-Next-80B-A3B-Thinking"),
    }
    log_paths = [
        "actor_logs/response_DeepSeek-R1_Physics_unencoded_20250921_190222",
        "actor_logs/response_gpt-oss-20b_Biology_unencoded_20250921_182219",
        "actor_logs/response_gpt-oss-20b_Chemistry_unencoded_20250921_181805",
        "actor_logs/response_gpt-oss-20b_Physics_unencoded_20250921_181302",
        "actor_logs/response_Qwen3-Next-80B-A3B-Thinking_Biology_unencoded_20250921_184401",
        "actor_logs/response_Qwen3-Next-80B-A3B-Thinking_Chemistry_unencoded_20250921_183841",
        "actor_logs/response_Qwen3-Next-80B-A3B-Thinking_Physics_unencoded_20250921_183555",
        "actor_logs/response_gpt-oss-120b_Biology_unencoded_20250921_212513",
        "actor_logs/response_gpt-oss-120b_Chemistry_unencoded_20250921_182109",
        "actor_logs/response_gpt-oss-120b_Physics_unencoded_20250921_181717",
    ]
    for log_path in log_paths:
        monitored, _ = parse_from_actor_log_name(log_path)
        if monitored == "Physics":
            monitors = biology_monitors
        elif monitored == "Biology":
            monitors = chemistry_monitors
        elif monitored == "Chemistry":
            monitors = physics_monitors
        evaluator = MonitorEvaluator(monitors=monitors, log_path=log_path)
        results = evaluator.evaluate_monitors()
        if results:
            evaluator.print_results(results)
            evaluator.save_detailed_results(results, home_dir="monitor_logs_fp")
        else:
            print("No results to display.")

def main():
    parser = argparse.ArgumentParser(
        description="Monitor evaluation CLI - evaluates monitor performance on log files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python monitor_evaluation.py                    # Run false negative evaluation (default)
  python monitor_evaluation.py --false-positive  # Run false positive evaluation
        """
    )
    
    parser.add_argument(
        "--false-positive", 
        action="store_true",
        help="Run false positive evaluation instead of false negative (default)"
    )
    
    args = parser.parse_args()
    
    if args.false_positive:
        print("Running false positive evaluation...")
        evaluate_grid_false_positive()
    else:
        print("Running false negative evaluation...")
        evaluate_grid_false_negative()


if __name__ == "__main__":
    main()
