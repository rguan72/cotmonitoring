from together import Together
from dotenv import load_dotenv

def main():
    load_dotenv()
    client = Together()
    
    # Enhanced prompt that encourages step-by-step reasoning
    reasoning_prompt = """
    Let $N$ denote the number of ordered triples of positive integers $(a, b, c)$ such that $a, b, c \leq 3^6$ and $a^3 + b^3 + c^3$ is a multiple of $3^7$. Find the remainder when $N$ is divided by $1000$.
    ."""
    
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": reasoning_prompt
            }
        ],
    )
    print(response)


if __name__ == "__main__":
    main()
