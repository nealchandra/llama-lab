import sys
import os
import curses
import zmq
from pathlib import Path

import torch
import transformers

from peft import get_peft_model, PeftModel

sys.path.insert(0, os.path.join(os.path.dirname( __file__ ), str(Path("../repositories/GPTQ-for-LLaMa"))))
import llama


def inference(prompt):
    # connect to zmq worker
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:5555")

    # console = curses.initscr()
    # socket.send(prompt)

    # try:
       
    # except e:
    #     curses.endwin()
    #     raise e
    # curses.endwin()
    
    # send prompt to zmq worker
    socket.send(prompt)

    # print the response from zmq worker
    response = socket.recv()
    print(response)

def load_quantized(model_path, pt_path):
    load_quant = llama.load_quant
    model = load_quant(model_path, pt_path, 4)
    model = model.to(torch.device('cuda:0'))
    return model