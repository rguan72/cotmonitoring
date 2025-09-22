import pandas as pd
from actor import EncodedActor, NonEncodedActor
from ant_actor import AntEncodedActor, AntNonEncodedActor
from logging_utils import write_result_to_log
from pipeline import GPQAPipeline

def control(model: str, question_topic: str, unencoded_actor: NonEncodedActor, sample_size: int | None = None):
    df = pd.read_csv("data/gpqa_diamond.csv")
    pipeline = GPQAPipeline(gpqa_df=df, question_topic=question_topic, sample_size=sample_size)
    results = pipeline.unencoded_parallel_pipeline(unencoded_actor=unencoded_actor)
    categories = pipeline.create_evaluation_summary(results)
    print(categories)
    model_str = model.split("/")[-1]
    write_result_to_log(results, categories, prefix=f"{question_topic}_{model_str}_control")

def control_grid():
    CONTROL_TOPICS = ["Physics", "Chemistry", "Biology"]
    TOGETHER_AI_MODELS = ["Qwen/Qwen3-Next-80B-A3B-Thinking", "openai/gpt-oss-120b", "openai/gpt-oss-20b"]
    for question_topic in CONTROL_TOPICS:
        for model in TOGETHER_AI_MODELS:
            control(model=model, unencoded_actor=NonEncodedActor(model=model), question_topic=question_topic)

def disguised_grid():
    QUESTION_TOPICS = ["Physics", "Chemistry", "Biology"]
    DISGUISE_TOPICS = ["Biology", "Physics", "Chemistry"]
    TOGETHER_AI_MODELS = ["deepseek-ai/DeepSeek-R1"]
    for idx, question_topic in enumerate(QUESTION_TOPICS):
        disguise_topic = DISGUISE_TOPICS[idx]
        for model in TOGETHER_AI_MODELS:
            disguised(model=model, question_topic=question_topic, disguise_topic=disguise_topic)

def disguised(model: str, question_topic: str, disguise_topic: str, sample_size: int | None = None):
    df = pd.read_csv("data/gpqa_diamond.csv")
    pipeline = GPQAPipeline(gpqa_df=df, question_topic=question_topic, sample_size=sample_size)
    encoded_actor = EncodedActor(disguise_topic=disguise_topic, question_topic=question_topic, model=model)
    results = pipeline.parallel_pipeline(encoded_actor=encoded_actor)
    categories = pipeline.create_evaluation_summary(results)
    print(categories)
    model_str = model.split("/")[-1]
    write_result_to_log(results, categories, prefix=f"{question_topic}_{model_str}_disguised_as_{disguise_topic}")

def main():
    disguised_grid()

if __name__ == "__main__":
    main()
