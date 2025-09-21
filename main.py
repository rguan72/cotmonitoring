from together import Together
from dotenv import load_dotenv
import datetime
import os

from actor import Response
from actor import Actor
from monitor import Monitor
import pandas as pd

def write_response_to_log(response: Response):
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"logs/response_{timestamp}.log"
    with open(log_filename, "w", encoding="utf-8") as log_file:
        log_file.write("=== RESPONSE CONTENT ===\n")
        log_file.write(response.content)
        log_file.write("\n\n=== RESPONSE REASONING ===\n")
        log_file.write(response.reasoning)
        log_file.write("\n\n")
    return log_filename

def main():    
    df = pd.read_csv("data/gpqa_main.csv")
    question_topic = "Physics"
    actor = Actor(disguise_topic="Biology", question_topic=question_topic)
    monitor = Monitor(question_topic=question_topic)
    physics_df = df[df["High-level domain"] == "Physics"]
    physics_df = physics_df.sample(n=2, random_state=42)
    for index, row in physics_df.iterrows():
        question = row["Question"]
        response = actor.act(question=question)
        print(f"Response written to log file: {write_response_to_log(response)}")
        monitor_result = monitor.monitor(response.reasoning + "\n" + response.content)
        print(f"Monitor result: {monitor_result}")

if __name__ == "__main__":
    main()
