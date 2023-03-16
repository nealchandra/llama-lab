# Llama Lab
My current goals:
* Create a reliable fine-tuning pipeline for Llama that runs in 8bit and utilizes the PEFT technique using LoRA
* Generalize the stanford alpaca dataset generation technique to be able to create new finetunes for a particular objective
* Run inferences against finetuned models using 4bit quantization
* Create a python worker process which loads the model and can be used to repeatedly prompt it for e.g. a conversational style API
* Consider exposing this as an actual web API?

## Setup

Clone the repo and ([install conda](https://docs.conda.io/en/latest/miniconda.html).

Run the following commands:
```
conda create -n llama
conda activate llama
conda install torchvision torchaudio pytorch-cuda=11.7 git -c pytorch -c nvidia
pip install -r requirements.txt
```

The third line assumes that you have an NVIDIA GPU. 

* If you have an AMD GPU, replace the third command with this one:

```
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/rocm5.2
```
  	  
* If you are running it in CPU mode, replace the third command with this one:

```
conda install pytorch torchvision torchaudio git -c pytorch
```

## Run

python ./cli.py --help