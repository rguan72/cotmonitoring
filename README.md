# Chain of thought monitoring
This is some research on chain of thought reasoning, building on [When Chain of Thought is Necessary, Language Models Struggle to Evade Monitors](https://arxiv.org/pdf/2507.05246). 

I wanted to ask: as model capability improves, does their ability to disguise reasoning improve faster or does their ability to catch disguised reasoning improve faster? The paper worked with Gemini 2.5 Pro and Flash, and Pro seemed to be a significantly better monitor (for example, it was not vulnerable to a gaslighting attack) without being that much better at deception. I wanted to check if that trend held for a larger range of models. If chain of thought monitoring capability seems to improve faster than ability to encode reasoning, that's a good sign. 

To investigate this, I took four open source models (GPT-OSS-20B, GPT-OSS-120B, Qwen3-Next-80B, DeepSeek-R1) and asked them to encode their reasoning on physics, biology and chemistry problems on GPQA-Diamond. I asked them to solve a problem in one of these fields while disguising their reasoning as another field. The full prompt is in actor.py.

All models have been extensively benchmarked on GPQA-Diamond -- the performance is DeepSeek-R1 > GPT-OSS-120B > Qwen3-Next-80B > GPT-OSS-20B. This is in line with what we'd expect with parameter count: the higher parameter models perform better. Now we measure each model's ability to (a) encode reasoning and (b) detect encoded reasoning. For each disguised reasoning trace, I had the other models try to detect if the trace is suspicious.

# Methodology
We ran all four models (except for DeepSeek b/c of costs + latency) to produce encoded an reasoning trace for each question on GPQA-Diamond. We also asked each model to just solve each question normally (an unencoded reasoning trace to measure false positive rate). We then ran each model as a monitor against the encoded reasoning trace (false positive rate) and unencoded reasoning (false negative). 

# Results

## Which model was best at evasion by encoding reasoning?

## Which model was best at monitoring?
- check false positive rate

# Interpretation

# Extensions
- My data was kind of noisy and random. It might be worth debugging and seeing there's something wrong OR just running a ton of trials to get more clarity on the data. I ran trials until my $40 ran out :)
- Instead of testing models on a spectrum of parameter count / GPQA performance, I would love to test models by release date, similar to what METR did in [Measuring AI Ability To Complete Long Tasks](https://metr.org/blog/2025-03-19-measuring-ai-ability-to-complete-long-tasks/). This seems relevant because we will likely use older, trusted models when performing safety evaluations on the latest and greatest but untrusted models, so the rate of monitoring or encoded reasoning capability increase release to release matters more than the rate as you increase number of parameters. Although maybe parameter count is an ok proxy since the strategy release to release has generally been to increase parameter count?

# Notes
Generating encoded reasoning:
```
python main.py
```

Generator evaluations of monitor capability:
```
python monitor_evaluation.py
```