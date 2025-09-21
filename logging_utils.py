import os
import datetime
from monitor import MonitorResult


def write_actor_response_to_log(full_content: str):
    os.makedirs("actor_logs", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"actor_logs/response_{timestamp}.log"
    with open(log_filename, "w", encoding="utf-8") as log_file:
        log_file.write(full_content)
    return log_filename

def write_monitor_result_to_log(monitor_result: MonitorResult):
    os.makedirs("monitor_logs", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"monitor_logs/monitor_result_{timestamp}.log"
    with open(log_filename, "w", encoding="utf-8") as log_file:
        log_file.write(monitor_result.__repr__())
    return log_filename

def write_result_to_log(results: str):
    os.makedirs("result_logs", exist_ok=True)
    results_log = "\nDETAILED RESULTS:"
    for i, result in enumerate(results):
        results_log += f"\nResult {i+1}:"
        results_log += f"  Question: {result['Question'][:100]}..."
        results_log += f"  Monitor result: {result['Monitor result']}"
        results_log += f"  Monitor log: {result['Monitor log']}"
        results_log += f"  Correct answer: {result['Correct answer']}"
        results_log += f"  Parser-parsed LLM answer: {result['Parser-parsed LLM answer']}"
        results_log += f"  Judge result: {result['Judge result']}"
        results_log += f"  Actor log: {result['Actor log']}"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"result_logs/result_{timestamp}.log"
    with open(log_filename, "w", encoding="utf-8") as log_file:
        log_file.write(results_log)
    print(f"Results written to log file: {log_filename}")
    return log_filename