import pandas as pd
from actor import EncodedActor, NonEncodedActor
from logging_utils import write_result_to_log
from pipeline import GPQAPipeline

def control(question_topic: str, sample_size: int | None = None):
    df = pd.read_csv("data/gpqa_main.csv")
    model = "openai/gpt-oss-20b"
    pipeline = GPQAPipeline(gpqa_df=df, question_topic=question_topic, sample_size=sample_size)
    unencoded_actor = NonEncodedActor(model=model)
    results = pipeline.unencoded_parallel_pipeline(unencoded_actor=unencoded_actor)
    categories = pipeline.create_evaluation_summary(results)
    print(categories)
    write_result_to_log(results, prefix=f"{question_topic}_{model}_control")

def disguised(question_topic: str, disguise_topic: str, sample_size: int | None = None):
    df = pd.read_csv("data/gpqa_main.csv")
    model = "openai/gpt-oss-20b"
    pipeline = GPQAPipeline(gpqa_df=df, question_topic=question_topic, sample_size=sample_size)
    encoded_actor = EncodedActor(disguise_topic=disguise_topic, question_topic=question_topic, model=model)
    results = pipeline.parallel_pipeline(encoded_actor=encoded_actor)
    categories = pipeline.create_evaluation_summary(results)
    print(categories)
    model_str = model.split("/")[-1]
    write_result_to_log(results, prefix=f"{question_topic}_{model_str}_disguised_as_{disguise_topic}")

def main():
    disguised(question_topic="Physics", disguise_topic="Biology", sample_size=5)

if __name__ == "__main__":
    main()
