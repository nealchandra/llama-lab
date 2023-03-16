import click

from src import finetune as finetune_module, inference as inference_module

@click.group()
def cli():
    pass

@click.command()
@click.option('-m', '--model_path', type=click.Path(exists=True), required=True, help='Path to the base LLaMA model',)
@click.option('-t', '--tokenizer_path', type=click.Path(exists=True),  required=True, help='Path to the LLaMA tokenizer.model')
@click.option('-o', '--output_path', type=click.Path(), required=True, help='Output dir for finetuned model')
def finetune(model_path, tokenizer_path, output_path):
    return finetune_module.finetune(model_path, tokenizer_path, output_path)


@click.command()
@click.option('-m', '--model_path', type=click.Path(exists=True), required=True, help='Path to the base LLaMA model')
@click.option('-t', '--tokenizer_path', type=click.Path(exists=True), required=True, help='Path to the LLaMA tokenizer.model')
@click.option('-f', '--finetune_path', type=click.Path(exists=True), required=True, help='Path to the finetuned model')
@click.option('--prompt', required=True, help='Prompt for the inference')
@click.option('--max_length', default=250, help='Max length for inference')
def inference(model_path, tokenizer_path, finetune_path, prompt, max_length):
    return inference_module.inference(model_path, tokenizer_path, finetune_path, prompt, max_length)

@click.command()
@click.option('-m', '--model_path', type=click.Path(exists=True), required=True, help='Path to the base LLaMA model')
@click.option('-t', '--tokenizer_path', type=click.Path(exists=True), required=True, help='Path to the LLaMA tokenizer.model')
@click.option('-p', '--pt_path', required=True, help='Prompt for the inference')
@click.option('--prompt', required=True, help='Prompt for the inference')
@click.option('--max_length', default=250, help='Max length for inference')
def inference_4bit_llama(model_path, tokenizer_path, pt_path, prompt, max_length):
    return inference_module.inference_4bit_llama(model_path, tokenizer_path, pt_path, prompt, max_length)

@click.command()
@click.option('-m', '--model_path', type=click.Path(exists=True), required=True, help='Path to the base LLaMA model')
# @click.option('-t', '--tokenizer_path', type=click.Path(exists=True), required=True, help='Path to the LLaMA tokenizer.model')
@click.option('-f', '--finetune_path', type=click.Path(exists=True), required=True, help='Path to the finetuned model')
@click.option('-o', '--output_path', type=click.Path(), required=True, help='Output dir for finetuned model')
def merge(model_path, finetune_path, output_path):
    return finetune_module.merge(model_path, finetune_path, output_path)

cli.add_command(finetune)
cli.add_command(inference)
cli.add_command(inference_4bit_llama)
cli.add_command(merge)

if __name__ == '__main__':
    cli()