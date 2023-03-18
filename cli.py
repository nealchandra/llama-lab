import click

from src import finetune as finetune_module, inference as inference_module, worker as worker_module

@click.group()
def cli():
    pass

@click.command()
@click.option('-p', '--model_path', type=click.Path(exists=True), required=True, help='Path to the base LLaMA model',)
@click.option('-t', '--tokenizer_path', type=click.Path(exists=True),  required=True, help='Path to the LLaMA tokenizer.model')
@click.option('-o', '--output_path', type=click.Path(), required=True, help='Output dir for finetuned model')
def finetune(model_path, tokenizer_path, output_path):
    return finetune_module.finetune(model_path, tokenizer_path, output_path)

@click.command()
@click.option('-p', '--model_path', type=click.Path(exists=True), required=True, help='Path to the base LLaMA model')
@click.option('-f', '--finetune_path', type=click.Path(exists=True), required=True, help='Path to the finetuned model')
@click.option('-o', '--output_path', type=click.Path(), required=True, help='Output dir for finetuned model')
def merge(model_path, finetune_path, output_path):
    return finetune_module.merge(model_path, finetune_path, output_path)

@click.command()
@click.option('--prompt', required=True, help='Prompt for the inference')
def inference(prompt):
    return inference_module.inference(prompt)

@click.command()
@click.option('-p', '--model_path', type=click.Path(exists=True), required=True, help='Path to the base LLaMA model')
@click.option('-t', '--tokenizer_path', type=click.Path(exists=True), required=True, help='Path to the LLaMA tokenizer.model')
@click.option('-f', '--finetune_path', type=click.Path(exists=True), required=True, help='Path to the finetuned model, or pt file in case of quantized model')
@click.option('--quantized', is_flag=True, help='Flag to indicate if the model is 4bit quantized')
def start_worker(model_path, tokenizer_path, finetune_path, quantized=False):
    return worker_module.run_worker(model_path, tokenizer_path, finetune_path, quantized)

cli.add_command(finetune)
cli.add_command(inference)
cli.add_command(merge)
cli.add_command(start_worker)

if __name__ == '__main__':
    cli()