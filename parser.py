from dataclasses import dataclass
import os
from together import Together
from dotenv import load_dotenv

class Parser:
    def __init__(self):
        load_dotenv()
        self.client = Together()
        SYSTEM_PROMPT = """
        An LLM is asked to solve a scientific problem. It was asked to provide the answer after Final Answer:. Parse the final answer from the LLM response. 
        
        IMPORTANT: Preserve LaTeX formatting in the answer. If the answer contains LaTeX expressions like \\hbar, \\omega, \\alpha, etc., keep them as LaTeX commands.
        Do not convert LaTeX commands to unicode characters.
        
        Then give your response in format Answer: {answer}
        """
        self.system_prompt = SYSTEM_PROMPT

    def parse(self, llm_answer: str) -> str:
        response = self.client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"LLM answer: {llm_answer}"}
            ],
        )
        return response.choices[0].message.content
