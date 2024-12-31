# Math Utils

*A tool for evaluating LLMs on the [MATH](https://github.com/hendrycks/math) dataset.*

## Installation

We use [vLLM](https://github.com/vllm-project/vllm) to accelerate the generation.

```bash
git clone https://github.com/hiyouga/MathUtils.git
cd MathUtils
pip install .
```

## Generate

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 math_utils gen Qwen/Qwen2.5-Math-7B-Instruct
```

Example output:

> Processed prompts: 100%|██████| 500/500 [00:36<00:00, 13.75it/s, est. speed input: 15765.84 toks/s, output: 5299.80 toks/s]

## Evaluate

```bash
math_utils eval predicts/test.jsonl
```

Example output:

> Processing sample: 500it [00:00, 926.32it/s]
>
> Accuracy: 413/500 = 82.60%.

## Experimental Results

|                  Command                        | Measured Acc | Reported Acc |
| ----------------------------------------------- | ------------ | ------------ |
| math_utils gen meta-llama/Meta-Llama-3-8B       | 29.2%        | 29.1%*       |
| math_utils gen meta-llama/Llama-3.1-8B-Instruct | 50.8%        | 51.9%*       |
| math_utils gen meta-llama/Llama-3.2-3B-Instruct | 48.4%        | 48.0%**      |
| math_utils gen Qwen/Qwen2.5-Math-7B-Instruct    | 82.6%        | 83.6%***     |

- *: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
- **: https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
- ***: https://qwenlm.github.io/blog/qwen2.5-math/

## Example Use

```python
from prm800k.grader import extract_boxed_content, grade_answer

grade_answer(given_answer: str, ground_truth: str)
grade_answer(extract_boxed_content(generated_result: str), answer: str)
```

## Acknowledgement

- [openai/prm800k](https://github.com/openai/prm800k)
- [QwenLM/Qwen2.5-Math](https://github.com/QwenLM/Qwen2.5-Math)
- [vllm-project/vllm](https://github.com/vllm-project/vllm)

## Citation

```bibtex
@Misc{mathutils,
  title = {MathUtils},
  author = {hiyouga},
  howpublished = {\url{https://github.com/hiyouga/MathUtils}},
  year = {2025}
}
```
