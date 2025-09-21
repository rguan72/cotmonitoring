from together import Together
from dotenv import load_dotenv
import datetime
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from actor import Response
from actor import EncodedActor
from parser import Parser
from judge import Judge
from monitor import Monitor, MonitorResult
import pandas as pd
from logging_utils import write_actor_response_to_log, write_monitor_result_to_log, write_result_to_log
from pipeline import GPQAPipeline

def main():    
    df = pd.read_csv("data/gpqa_main.csv")
    question_topic = "Physics"
    disguise_topic = "Biology"
    pipeline = GPQAPipeline(gpqa_df=df, question_topic=question_topic, disguise_topic=disguise_topic, model="openai/gpt-oss-20b", sample_size=100)
    results = pipeline.parallel_pipeline()
    categories = pipeline.create_evaluation_summary(results)
    print(categories)
    results_log = write_result_to_log(results)
    print(f"Results written to log file: {results_log}")
if __name__ == "__main__":
    main()
