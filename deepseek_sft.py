import os
#os.environ['CUDA_VISIBLE_DEVICES'] = "1,2,3"
import random
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import LoraConfig
import wandb
from huggingface_hub import login
# Replace 'your_access_token' with your actual access token
access_token = "hf_XXXXXXX"
# # Log in to Hugging Face
login(token=access_token)
# Function to set the random seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(50)

from datasets import load_dataset

# Load the dataset (only has "train")
dataset = load_dataset("Magpie-Align/Magpie-Reasoning-V1-150K-CoT-Deepseek-R1-Llama-70B", split="train", trust_remote_code=True)

# Split dataset into 90% train, 10% test
split_dataset = dataset.train_test_split(test_size=0.1, seed=42)

# Access the new splits
train_ds = split_dataset["train"]
test_ds = split_dataset["test"]

# Print sizes
print(f"Train size: {len(train_ds)}")
print(f"Test size: {len(test_ds)}")


base_model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
llama_tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
llama_tokenizer.pad_token = llama_tokenizer.eos_token
llama_tokenizer.padding_side = "right"


quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,

    bnb_4bit_use_double_quant=False
) 
base_model = AutoModelForCausalLM.from_pretrained(
    
    base_model_name,
    quantization_config=quant_config,
    device_map="auto"
)
base_model.config.use_cache = False
base_model.config.pretraining_tp = 1

peft_parameters = LoraConfig(
    lora_alpha=64,
    lora_dropout=0.05,
    r=32,
    bias="none",
    task_type="CAUSAL_LM"
)
base_model.add_adapter(peft_parameters)
def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

trainable_parameters = count_trainable_parameters(base_model)
print(f"Number of trainable parameters: {trainable_parameters}")


def format(ex):
    system_message=f"""
    You are a master at {ex['task_category']}. Based on the question asked, answer the question to the best of your abilities
    """
    prompt = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"Answer this question:\n{ex['instruction']}"}
            ]
    prompt=llama_tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True, add_special_tokens=False)
    full= [
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"Answer this question:\n{ex['instruction']}"},
                {"role":"assistant", "content": ex['response']}
            ]
    
    resp=ex['response']
    return {'full': full, 'query': prompt, 'response': resp}
    
    
    

train_ds=train_ds.map(format)


train_dataset=train_ds.remove_columns(['uuid', 'instruction', 'conversations', 'gen_input_configs', 'gen_response_configs', 'intent', 'knowledge', 'difficulty', 'difficulty_generator', 'input_quality', 'quality_explanation', 'quality_generator', 'task_category', 'other_task_category', 'task_category_generator', 'language', 'full'])

split_dataset = train_dataset.train_test_split(test_size=0.1, seed=50)

instruction_template = "<｜User｜>"
response_template = "<｜Assistant｜>"
collator = DataCollatorForCompletionOnlyLM(instruction_template=instruction_template,response_template=response_template, tokenizer=llama_tokenizer)

# Define training arguments
training_args = SFTConfig(
    output_dir="/scratch/ds7395/deepseek_distill_llama/",
    evaluation_strategy="steps",
    save_strategy="steps",
    save_steps=50,
    eval_steps=500,
    logging_steps=50,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    fp16=True,
    save_total_limit=2,
    dataset_text_field="query",
    push_to_hub=True,  # Enable pushing to Hugging Face Hub
    hub_model_id="ds28/deepseek-distill-llama-cot-sft",
    report_to="wandb",  # Optional: Log to Weights & Biases
)

# Initialize SFTTrainer WITHOUT formatting_func
trainer = SFTTrainer(
    model=base_model,
    train_dataset=split_dataset['train'],
    eval_dataset=split_dataset['test'],
    args=training_args,
    data_collator=collator,  # Your custom data collator
    tokenizer=llama_tokenizer,
)
import wandb
wandb.login(key="")
wandb.init(project="deepseek_distilled_llama_sft", name='run1')
# Start training
trainer.train()
trainer.push_to_hub()
llama_tokenizer.push_to_hub("ds28/deepseek-distill-llama-cot-sft")
