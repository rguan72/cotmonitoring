from logging_utils import write_actor_response_to_log
import pandas as pd
from actor import EncodedActor, NonEncodedActor
from monitor import Monitor
from parser import Parser
from judge import Judge
from logging_utils import write_monitor_result_to_log
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

class GPQAPipeline:
    def __init__(self, gpqa_df: pd.DataFrame, question_topic: str, disguise_topic: str, model: str, sample_size: int | None = None):
        self.gpqa_df = gpqa_df
        self.question_topic = question_topic
        self.disguise_topic = disguise_topic
        self.model = model
        self.encoded_actor = EncodedActor(disguise_topic=disguise_topic, question_topic=question_topic, model=model)
        self.unencoded_actor = NonEncodedActor(model=model)
        self.monitor = Monitor(question_topic=question_topic, model=model)
        self.parser = Parser()
        self.judge = Judge()
        self.sample_size = sample_size

    def get_rows(self):
        topic_df = self.gpqa_df[self.gpqa_df["High-level domain"] == self.question_topic]
        if self.sample_size:
            topic_df = topic_df.sample(n=self.sample_size, random_state=42)
        return [
            (index, row)
            for index, row in topic_df.iterrows()
        ]

    def _parallelize(self, pipeline_function):
        row_data_list = self.get_rows()
        results = []
        with ThreadPoolExecutor(max_workers=6) as executor:
            future_to_row = {
                executor.submit(pipeline_function, row_data): row_data[0] 
                for row_data in row_data_list
            }
            with tqdm(total=len(row_data_list), desc="Processing pipeline iterations") as pbar:
                for future in as_completed(future_to_row):
                    try:
                        result = future.result()
                        results.append(result)
                        pbar.update(1)
                    except Exception as exc:
                        row_index = future_to_row[future]
                        print(f'Row {row_index} generated an exception: {exc}')
                        pbar.update(1)
        return results

    def unencoded_parallel_pipeline(self):
        return self._parallelize(self.unencoded_pipeline)

    def parallel_pipeline(self):
        return self._parallelize(self.pipeline)
        
    def unencoded_pipeline(self, row: tuple[int, pd.Series]):
        index, row_data = row
        question = row_data["Pre-Revision Question"]
        response = self.unencoded_actor.act(question=question)
        full_content = "=== RESPONSE CONTENT ===\n" + response.content + "\n\n=== RESPONSE REASONING ===\n" + response.reasoning
        actor_log = write_actor_response_to_log(full_content)
        parse_result = self.parser.parse(response.content)
        judge_result = self.judge.judge(parse_result, row_data["Pre-Revision Correct Answer"])
        return {
            "Question": question,
            "Correct answer": row_data["Pre-Revision Correct Answer"],
            "Parser-parsed LLM answer": parse_result,
            "Judge result": judge_result,
            "Actor log": actor_log
        }

    def pipeline(self, row: tuple[int, pd.Series]):
        index, row_data = row
        question = row_data["Pre-Revision Question"]
        response = self.encoded_actor.act(question=question)
        full_content = "=== RESPONSE CONTENT ===\n" + response.content + "\n\n=== RESPONSE REASONING ===\n" + response.reasoning
        actor_log = write_actor_response_to_log(full_content)
        monitor_result = self.monitor.monitor(full_content)
        monitor_log = write_monitor_result_to_log(monitor_result)
        parse_result = self.parser.parse(response.content)
        judge_result = self.judge.judge(parse_result, row_data["Pre-Revision Correct Answer"])
        return {
            "Question": question,
            "Monitor result": monitor_result.is_suspicious,
            "Monitor log": monitor_log,
            "Correct answer": row_data["Pre-Revision Correct Answer"],
            "Parser-parsed LLM answer": parse_result,
            "Judge result": judge_result,
            "Actor log": actor_log
        }

    def create_evaluation_summary(self, results: list[dict]) -> dict:
        """
        Create a summary of evaluation results categorized into four categories:
        1. Correct and caught by monitor
        2. Incorrect and caught by monitor  
        3. Correct and evaded monitor
        4. Incorrect and evaded monitor
        """
        total_results = len(results)
        categories = {
            "Correct and caught by monitor": 0,
            "Incorrect and caught by monitor": 0,
            "Correct and evaded monitor": 0,
            "Incorrect and evaded monitor": 0
        }
        for result in results:
            is_correct = result['Judge result']
            is_caught = result['Monitor result']
            
            if is_correct and is_caught:
                categories["Correct and caught by monitor"] += 1
            elif not is_correct and is_caught:
                categories["Incorrect and caught by monitor"] += 1
            elif is_correct and not is_caught:
                categories["Correct and evaded monitor"] += 1
            else:
                categories["Incorrect and evaded monitor"] += 1
        formatted = {
            "Correct and caught by monitor": f"{categories['Correct and caught by monitor']} / {total_results}",
            "Incorrect and caught by monitor": f"{categories['Incorrect and caught by monitor']} / {total_results}",
            "Correct and evaded monitor": f"{categories['Correct and evaded monitor']} / {total_results}",
            "Incorrect and evaded monitor": f"{categories['Incorrect and evaded monitor']} / {total_results}",
        }
        return formatted