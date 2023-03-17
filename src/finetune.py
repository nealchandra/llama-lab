import os
import argparse

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
import bitsandbytes as bnb
from datasets import load_dataset
import transformers
from transformers import AutoTokenizer, AutoConfig, LlamaForCausalLM, LlamaTokenizer
from peft import prepare_model_for_int8_training, LoraConfig, get_peft_model, TaskType, PeftModel, tuners


MICRO_BATCH_SIZE = 4  # this could actually be 5 but i like powers of 2
BATCH_SIZE = 256
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
#EPOCHS = 3  # we don't need 3 tbh
EPOCHS = 1
LEARNING_RATE = 3e-4  # the Karpathy constant
CUTOFF_LEN = 256  # 256 accounts for about 96% of the data
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

## POTENTIAL ARGS
# LORA_ALPHA = 8
# LORA_DROPOUT = 0.1

def merge(model_path, peft_path, output_path):
    print('Merging models.')
    model = transformers.LlamaForCausalLM.from_pretrained(model_path)
    model = PeftModel.from_pretrained(model, peft_path, device_map={'': 0})
    model.eval()

    print('Copying modules.')
    key_list = [key for key, _ in model.base_model.model.named_modules() if "lora" not in key]
    for key in key_list:
        parent, target, target_name = model.base_model._get_submodules(key)
        if isinstance(target, tuners.lora.Linear):
            bias = target.bias is not None
            new_module = torch.nn.Linear(target.in_features, target.out_features, bias=bias)
            model.base_model._replace_module(parent, target_name, new_module, target)

    print('Saving.')
    model = model.base_model.model
    model.save_pretrained(output_path, use_temp_dir=False)

def finetune(model_path, tokenizer_path, output_path):
    model = LlamaForCausalLM.from_pretrained(
        model_path,
        load_in_8bit=True,
        device_map="auto",
    )
    tokenizer = LlamaTokenizer.from_pretrained(
       tokenizer_path, add_eos_token=True
    )

    model = prepare_model_for_int8_training(model)

    config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, config)
    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
    data = load_dataset("json", data_files="data/alpaca_data.json")

    from src.prompts.alpaca import prompt

    def tokenize(data):
        result = tokenizer(
            prompt.generate_prompt(data),
            truncation=True,
            max_length=CUTOFF_LEN + 1,
            padding="max_length",
        )
        return {
            "input_ids": result["input_ids"][:-1],
            "attention_mask": result["attention_mask"][:-1],
        }

    data = data.shuffle().map(lambda x: tokenize(x))

    trainer = transformers.Trainer(
        model=model,
        train_dataset=data["train"],
        args=transformers.TrainingArguments(
            per_device_train_batch_size=MICRO_BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            warmup_steps=100,
            num_train_epochs=EPOCHS,
            learning_rate=LEARNING_RATE,
            fp16=True,
            logging_steps=10,
            output_dir=output_path,
            save_total_limit=3,
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    model.config.use_cache = False
    trainer.train(resume_from_checkpoint=False)

    model.save_pretrained(output_path)