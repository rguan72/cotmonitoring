from together import Together
from dotenv import load_dotenv
import datetime
import os

from actor import Response
from actor import Actor
from parser import Parser
from judge import Judge
from monitor import Monitor, MonitorResult
import pandas as pd

def write_response_to_log(full_content: str):
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

def main():    
    df = pd.read_csv("data/gpqa_main.csv")
    question_topic = "Physics"
    actor = Actor(disguise_topic="Biology", question_topic=question_topic)
    monitor = Monitor(question_topic=question_topic)
    physics_df = df[df["High-level domain"] == "Physics"]
    physics_df = physics_df.sample(n=2, random_state=42)
    parser = Parser()
    judge = Judge()
    for index, row in physics_df.iterrows():
        question = row["Pre-Revision Question"]
        response = actor.act(question=question)
        full_content = "=== RESPONSE CONTENT ===\n" + response.content + "\n\n=== RESPONSE REASONING ===\n" + response.reasoning
        print(f"Response written to log file: {write_response_to_log(full_content)}")
        monitor_result = monitor.monitor(full_content)
        print(f"Monitor result written to log file: {write_monitor_result_to_log(monitor_result)}")
        parse_result = parser.parse(response.content)
        print(f"Correct answer: {row['Pre-Revision Correct Answer']}")
        print(f"Parser-parsed LLM answer: {parse_result}")
        judge_result = judge.judge(parse_result, row["Pre-Revision Correct Answer"])
        print(f"Judge result: {judge_result}")
if __name__ == "__main__":
    main()
