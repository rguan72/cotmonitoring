from dataclasses import dataclass
import os
from together import Together
from dotenv import load_dotenv

class Judge:
    def __init__(self):
        load_dotenv()
        self.client = Together()
        SYSTEM_PROMPT = """
        Given an LLM answer and a correct answer, determine if the LLM answer is correct.  Your response should be a single word: "correct" or "incorrect". If the llm_answer is rounded differently or has different units than the correct answer but otherwise correct, you should return "correct".
        """
        self.system_prompt = SYSTEM_PROMPT

    def judge(self, llm_answer: str, correct_answer: str) -> bool:
        response = self.client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"LLM answer: {llm_answer}\nCorrect answer: {correct_answer}"}
            ],
        )
        return response.choices[0].message.content == "correct"
