import os
import datetime
from monitor import MonitorResult


def write_actor_response_to_log(full_content: str, log_dir: str):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{log_dir}/response_{timestamp}.log"
    with open(log_filename, "w", encoding="utf-8") as log_file:
        log_file.write(full_content)
    return log_filename

def write_monitor_result_to_log(monitor_result: MonitorResult, log_dir: str):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{log_dir}/monitor_result_{timestamp}.log"
    with open(log_filename, "w", encoding="utf-8") as log_file:
        log_file.write(monitor_result.__repr__())
    return log_filename

def write_result_to_log(results: str, prefix: str):
    os.makedirs("result_logs", exist_ok=True)
    results_log = "\nDETAILED RESULTS:"
    for i, result in enumerate(results):
        results_log += f"\n-----------Result {i+1}-----------\n"
        results_log += f"  Question: {result['Question'][:100]}...\n"
        results_log += f"  Monitor result: {result['Monitor result']}"
        results_log += f"  Monitor log: {result['Monitor log']}\n"
        results_log += f"  Correct answer: {result['Correct answer']}\n"
        results_log += f"  Parser-parsed LLM answer: {result['Parser-parsed LLM answer']}\n"
        results_log += f"  Judge result: {result['Judge result']}\n"
        results_log += f"  Actor log: {result['Actor log']}\n"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"result_logs/result_{prefix}_{timestamp}.log"
    with open(log_filename, "w", encoding="utf-8") as log_file:
        log_file.write(results_log)
    print(f"Results written to log file: {log_filename}")
    return log_filename