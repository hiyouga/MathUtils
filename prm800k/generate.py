"""
This code is partially borrowed from: https://github.com/QwenLM/Qwen2.5-Math/blob/main/evaluation/examples.py
"""

import json
import os
from typing import List

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer
from vllm import LLM, SamplingParams


DEMOS = [
    (
        "Kevin Kangaroo begins hopping on a number line at 0. He wants to get to 1, but he can hop only $\\frac{1}{3}$ of the distance. Each hop tires him out so that he continues to hop $\\frac{1}{3}$ of the remaining distance. How far has he hopped after five hops? Express your answer as a common fraction.",
        "Let's think step by step\nKevin hops $1/3$ of the remaining distance with every hop.\nHis first hop takes $1/3$ closer.\nFor his second hop, he has $2/3$ left to travel, so he hops forward $(2/3)(1/3)$.\nFor his third hop, he has $(2/3)^2$ left to travel, so he hops forward $(2/3)^2(1/3)$.\nIn general, Kevin hops forward $(2/3)^{k-1}(1/3)$ on his $k$th hop.\nWe want to find how far he has hopped after five hops.\nThis is a finite geometric series with first term $1/3$, common ratio $2/3$, and five terms.\nThus, Kevin has hopped $\\frac{\\frac{1}{3}\\left(1-\\left(\\frac{2}{3}\\right)^5\\right)}{1-\\frac{2}{3}} = \\boxed{\\frac{211}{243}}$.\nThe answer is \\frac{211}{243}}",
    ),
    (
        "What is the area of the region defined by the equation $x^2+y^2 - 7 = 4y-14x+3$?",
        "Let's think step by step\nWe rewrite the equation as $x^2 + 14x + y^2 - 4y = 10$ and then complete the square,\nresulting in  $(x+7)^2-49 + (y-2)^2-4=10$,\nor $(x+7)^2+(y-2)^2=63$.\nThis is the equation of a circle with center $(-7, 2)$ and radius $\\sqrt{63},$\nso the area of this region is $\\pi r^2 = \\boxed{63\\pi}$.\nThe answer is 63\\pi",
    ),
    (
        "If $x^2+y^2=1$, what is the largest possible value of $|x|+|y|$?",
        "Let's think step by step\nIf $(x,y)$ lies on the circle,\nso does $(x,-y),$ $(-x,-y),$ and $(-x,-y),$ (which all give the same value of $|x| + |y|$),\nso we can assume that $x \\ge 0$ and $y \\ge 0.$\nThen $|x| + |y| = x + y.$  Squaring, we get\n\\[(x + y)^2 = x^2 + 2xy + y^2 = 1 + 2xy.\\]\nNote that $(x - y)^2 \\ge 0.$\nExpanding, we get $x^2 - 2xy + y^2 \\ge 0,$ so $2xy \\le x^2 + y^2 = 1.$\nHence,\\[1 + 2xy \\le 2,\\]which means $x + y \\le \\sqrt{2}.$\nEquality occurs when $x = y = \\frac{1}{\\sqrt{2}},$\nso the maximum value of $|x| + |y|$ is $\\boxed{\\sqrt{2}}.$\nThe answer is \\sqrt{2}",
    ),
    (
        "If $f(x)=\\frac{ax+b}{cx+d}, abcd\\not=0$ and $f(f(x))=x$ for all $x$ in the domain of $f$, what is the value of $a+d$?",
        "Let's think step by step\nThe condition $f(f(x))$ means that $f$ is the inverse of itself,\nso its graph is symmetrical about the line $y = x$.\nWith a rational function of this form, we will have two asymptotes:\na vertical one at $x=-d/c$ if $cx+d$ does not divide $ax+b$,\nand a horizontal one at $y=a/c$,\nif we take the limit of $f(x)$ as $x$ goes to $\\pm\\infty$.\nIn order for $f$ to be its own inverse, the intersection of the asymptotes must lie on the line $y=x$\nso that it and its asymptotes reflect onto themselves.\nThis means that $-d/c=a/c$,\nand therefore $-d=a$ and $a+d=\\boxed{0}$.\nThe answer is 0",
    ),
    (
        "Expand $(2z^2 + 5z - 6)(3z^3 - 2z + 1)$.",
        "Let's think step by step\n$$\\begin{array}{crrrrrrr}\n& & & 3z^3 & & -2z & + 1 & \\\\\n\\times & & & & 2z^2 & +5z & -6 \\\\\n\\cline{1-7}\\rule{0pt}{0.17in}\n& & & -18z^3 & & +12z & -6 & \\\\\n& & +15z^4 & & -10z^2 & +5z & & \\\\\n+ & 6z^5 & & -4z^3 & +2z^2 & & & \\\\\n\\cline{1-7}\\rule{0pt}{0.17in}\n& 6z^5 & +15z^4 & -22z^3 & - 8z^2 &+17z & -6 &\n\\end{array}$$\nThe answer is 6z^5+15z^4-22z^3-8z^2+17z-6",
    ),
]


def _encode_sample(tokenizer: "PreTrainedTokenizer", problem: str, n_shot: int, system: str) -> List[int]:
    if n_shot > 5:
        raise ValueError("`n_shot` must be less than or equal to 5.")

    messages = []
    if system:
        messages.append({"role": "system", "content": system})

    for i in range(n_shot):
        problem = f"{DEMOS[i][0]}\n{DEMOS[i][1]}\n\n{problem}"

    messages.append({"role": "user", "content": problem})
    return tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)


def vllm_generate(
    model: str,
    json_path: str = "math_splits/test.jsonl",
    save_path: str = "predicts/test.jsonl",
    n_shot: int = 0,
    system: str = r"Please reason step by step, and put your final answer within \boxed{}.",
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 4096,
):
    tokenizer = AutoTokenizer.from_pretrained(model)
    inputs = []
    samples = []
    with open(json_path, encoding="utf-8") as f:
        for line in tqdm(f, desc="Processing samples"):
            sample = json.loads(line)
            samples.append(sample)
            input_ids = _encode_sample(tokenizer, sample["problem"], n_shot, system)
            inputs.append({"prompt_token_ids": input_ids})

    print("Input example:")
    print("=" * 50)
    print(tokenizer.decode(inputs[0]["prompt_token_ids"]))
    print("=" * 50)

    vllm_engine = LLM(model=model, tensor_parallel_size=torch.cuda.device_count())
    sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens)
    results = vllm_engine.generate(inputs, sampling_params)
    preds = [result.outputs[0].text for result in results]

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        for pred, sample in zip(preds, samples):
            sample["solution"] = pred
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print("=" * 50)
    print(f"Generated results have been saved at `{save_path}`.")
    print(f"Use `math_utils eval {save_path}` to evaluate them.")
    print("=" * 50)
