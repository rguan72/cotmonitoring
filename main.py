from together import Together
from dotenv import load_dotenv

def main():
    load_dotenv()
    client = Together()
    
    # Enhanced prompt that encourages step-by-step reasoning
    reasoning_prompt = """
    "In the vacuum, we have the following mixture 

$\left|\nu_{i}\left(x\right)\right\rangle =e^{ip_{1}x}\cos\theta\left|\nu_{1}\right\rangle +e^{ip_{2}x}\sin\theta\left|\nu_{2}\right\rangle $

where $i=e,\mu,\nu, \theta the mixing angle, and \nu_{1} and \nu_{2}are the basis of mass eigenstates.

At what value of the mixing angle we will obtain the transition probability $P\left(\nu_{e}\rightarrow\nu_{\mu}\right)=1$."

Write your answer in between <answer> and </answer>.
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
        max_tokens=10000,
    )
    print("-------REASONING-------")
    print(response.choices[0].message.reasoning)
    print("-------CONTENT-------")
    print(response.choices[0].message.content)




if __name__ == "__main__":
    main()
