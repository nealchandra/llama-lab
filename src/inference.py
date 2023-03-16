import sys
import os
import curses
from pathlib import Path

import torch
import transformers

from peft import get_peft_model, PeftModel

sys.path.insert(0, os.path.join(os.path.dirname( __file__ ), str(Path("../repositories/GPTQ-for-LLaMa"))))
import llama

TRAINING_TEXT_NO_INPUT= """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""

class Stream(transformers.StoppingCriteria):
    def __init__(self, tokenizer, console):
        self.tokenizer = tokenizer
        self.console = console

    def __call__(self, input_ids, scores) -> bool:
        self.console.erase()
        self.console.addstr(self.tokenizer.decode(input_ids[0], skip_special_tokens=True))
        self.console.refresh()
        return False


def inference(model_path, tokenizer_path, peft_path, prompt, max_length):
    model = transformers.LLaMAForCausalLM.from_pretrained(model_path, load_in_8bit=True, device_map='auto')
    model = PeftModel.from_pretrained(model, peft_path, device_map={'': 0})

    tokenizer = transformers.LLaMATokenizer.from_pretrained(tokenizer_path)
    batch = tokenizer(TRAINING_TEXT_NO_INPUT.format(instruction=prompt), return_tensors="pt")

    generation_config = transformers.GenerationConfig(
        temperature=0.1,
        top_p=0.75,
        num_beams=4,
    )

    console = curses.initscr()
    try:
        with torch.no_grad():
            out = model.generate(
                generation_config=generation_config,
                input_ids=batch["input_ids"].cuda(),
                attention_mask=torch.ones_like(batch["input_ids"]).cuda(),
                max_length=max_length,
                stopping_criteria=[Stream(tokenizer=tokenizer, console=console)]
            )
    except e:
        curses.endwin()
        raise e
    curses.endwin()
    print(tokenizer.decode(out[0]))

def inference_4bit_llama(model_path, tokenizer_path, pt_path, prompt, max_length):
    model = load_quantized(model_path, pt_path)

    tokenizer = transformers.LLaMATokenizer.from_pretrained(tokenizer_path)
    batch = tokenizer(TRAINING_TEXT_NO_INPUT.format(instruction=prompt), return_tensors="pt")

    generation_config = transformers.GenerationConfig(
        temperature=0.1,
        top_p=0.75,
        num_beams=4,
    )

    with torch.no_grad():
        out = model.generate(
            generation_config=generation_config,
            input_ids=batch["input_ids"].cuda(),
            attention_mask=torch.ones_like(batch["input_ids"]).cuda(),
            max_length=max_length
        )
    print(tokenizer.decode(out[0]))

def load_quantized(model_path, pt_path):
    load_quant = llama.load_quant
    model = load_quant(model_path, pt_path, 4)
    model = model.to(torch.device('cuda:0'))
    return model