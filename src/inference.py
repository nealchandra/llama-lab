import torch
import transformers

from peft import get_peft_model, PeftModel

TRAINING_TEXT_NO_INPUT= """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""

def inference(model_path, tokenizer_path, peft_path, prompt, max_length):
    model = transformers.LLaMAForCausalLM.from_pretrained(model_path, load_in_8bit=True, device_map='auto')
    model = PeftModel.from_pretrained(model, peft_path)

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