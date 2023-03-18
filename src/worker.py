import zmq
import inference
improt process

from inference import load_model

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

def load_model(model_path, tokenizer_path, peft_path):
    if int4:
        model = load_quantized(model_path, peft_path)
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(model_path, load_in_8bit=True, device_map='auto')
        model = PeftModel.from_pretrained(model, peft_path, device_map={'': 0})
    tokenizer = transformers.LlamaTokenizer.from_pretrained(tokenizer_path)
    return model, tokenizer

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")

def run_worker(model_path, tokenizer_path, peft_path, int4=False):
    model, tokenizer = load_model(model_path, tokenizer_path, peft_path, int4)
    generation_config = transformers.GenerationConfig(
        temperature=0.1,
        top_p=0.75,
        num_beams=4,
    )

    print("Model loaded. Ready to receive prompts.")
    while True:
        #  Wait for next prompt from client
        prompt = socket.recv()

        # exit process without error if prompt is "<exit>"
        if prompt == b"<exit>":
            process.exit(0)

        # Tokenize prompt and generate against the model
        batch = tokenizer(TRAINING_TEXT_NO_INPUT.format(instruction=prompt), return_tensors="pt")

        # Generate response from model using stopping criteria to stream the output
        with torch.no_grad():
            out = model.generate(
                generation_config=generation_config,
                input_ids=batch["input_ids"].cuda(),
                attention_mask=torch.ones_like(batch["input_ids"]).cuda(),
                max_length=500,
                # stopping_criteria=[Stream(tokenizer=tokenizer, console=console)]
            )

            #  Send reply back to client
            socket.send(tokenizer.decode(out[0]))