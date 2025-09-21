import pandas as pd
from logging_utils import write_result_to_log
from pipeline import GPQAPipeline

def control(question_topic: str, sample_size: int | None = None):
    df = pd.read_csv("data/gpqa_main.csv")
    pipeline = GPQAPipeline(gpqa_df=df, question_topic=question_topic, model="openai/gpt-oss-20b", sample_size=sample_size)
    results = pipeline.unencoded_parallel_pipeline()
    categories = pipeline.create_evaluation_summary(results)
    print(categories)
    results_log = write_result_to_log(results)
    print(f"Results written to log file: {results_log}")

def disguised(question_topic: str, disguise_topic: str, sample_size: int | None = None):
    df = pd.read_csv("data/gpqa_main.csv")
    pipeline = GPQAPipeline(gpqa_df=df, question_topic=question_topic, disguise_topic=disguise_topic, model="openai/gpt-oss-20b", sample_size=sample_size)
    results = pipeline.parallel_pipeline()
    categories = pipeline.create_evaluation_summary(results)
    print(categories)
    results_log = write_result_to_log(results)
    print(f"Results written to log file: {results_log}")

def main():    
    control(question_topic="Physics", sample_size=10)

if __name__ == "__main__":
    main()
