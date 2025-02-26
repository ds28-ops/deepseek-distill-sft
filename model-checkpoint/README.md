---
base_model: deepseek-ai/DeepSeek-R1-Distill-Llama-8B
library_name: transformers
model_name: deepseek-distill-llama-cot-sft
tags:
- generated_from_trainer
- trl
- sft
licence: license
---

# Model Card for deepseek-distill-llama-cot-sft

This model is a fine-tuned version of [deepseek-ai/DeepSeek-R1-Distill-Llama-8B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B).
It has been trained using [TRL](https://github.com/huggingface/trl).

## Quick start

```python
from transformers import pipeline

question = "If you had a time machine, but could only go to the past or the future once and never return, which would you choose and why?"
generator = pipeline("text-generation", model="ds28/deepseek-distill-llama-cot-sft", device="cuda")
output = generator([{"role": "user", "content": question}], max_new_tokens=128, return_full_text=False)[0]
print(output["generated_text"])
```

## Training procedure

[<img src="https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-28.svg" alt="Visualize in Weights & Biases" width="150" height="24"/>](https://wandb.ai/ds7395-new-york-university/deepseek_distilled_llama_sft/runs/iqon1bme) 


This model was trained with SFT.

### Framework versions

- TRL: 0.13.0
- Transformers: 4.48.0
- Pytorch: 2.4.1
- Datasets: 3.0.1
- Tokenizers: 0.21.0

## Citations



Cite TRL as:
    
```bibtex
@misc{vonwerra2022trl,
	title        = {{TRL: Transformer Reinforcement Learning}},
	author       = {Leandro von Werra and Younes Belkada and Lewis Tunstall and Edward Beeching and Tristan Thrush and Nathan Lambert and Shengyi Huang and Kashif Rasul and Quentin Gallouédec},
	year         = 2020,
	journal      = {GitHub repository},
	publisher    = {GitHub},
	howpublished = {\url{https://github.com/huggingface/trl}}
}
```