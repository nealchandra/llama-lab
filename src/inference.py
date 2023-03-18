import curses
import zmq

import torch
import transformers

from peft import get_peft_model, PeftModel


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
    socket.send(prompt.encode('utf-8'))

    # print the response from zmq worker
    response = socket.recv()
    print(response)