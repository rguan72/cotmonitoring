# Chain of thought monitoring
This is some research on chain of thought reasoning, building on [When Chain of Thought is Necessary, Language Models Struggle to Evade Monitors](https://arxiv.org/pdf/2507.05246). 

I wanted to ask: as model capability improves, does their ability to disguise reasoning improve faster or does their ability to catch disguised reasoning improve faster? The paper worked with Gemini 2.5 Pro and Flash, and Pro seemed to be a significantly better monitor (for example, it was not vulnerable to a gaslighting attack) without being that much better at deception. I wanted to check if that trend held for a larger range of models. If chain of thought monitoring capability seems to improve faster than ability to encode reasoning, that's a good sign. 

To investigate this, I took four open source models (GPT-OSS-20B, GPT-OSS-120B, Qwen3-Next-80B, DeepSeek-R1) and asked them to encode their reasoning on physics, biology and chemistry problems on GPQA-Diamond. I asked them to solve a problem in one of these fields while disguising their reasoning as another field. The full prompt is in actor.py.

All models have been extensively benchmarked on GPQA-Diamond -- the performance is DeepSeek-R1 > GPT-OSS-120B > Qwen3-Next-80B > GPT-OSS-20B. Now we measure each model's ability to (a) encode reasoning and (b) detect encoded reasoning. For each disguised reasoning trace, I had the other models try to detect if the trace is suspicious.

Extensions:
- Instead of just providing suspicious reasoning traces to monitors and seeing if they detect it (false negative rate), I should also ask monitors to evaluate non-suspicious reasoning traces (false positive rate). That will give a better overall picture of monitor performance.
- Get better overall longitudial data on ability to encode reasoning and monitor encoded reasoning. Instead of just using open source models with varying performance on GPQA-Diamond, I would love to test a wide range of models including deprecated ones, similar to what METR did in [Measuring AI Ability To Complete Long Tasks](https://metr.org/blog/2025-03-19-measuring-ai-ability-to-complete-long-tasks/)