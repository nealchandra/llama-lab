from dataclasses import dataclass

@dataclass
class Prompt():
    params: list[str]
    prompt_without_input: str
    prompt_with_input: str

    def generate_prompt(self, data):
        args = {k: data[k] for k in self.params}
        if data['input']:
            return self.prompt_with_input.format(input=data['input'], **args)
        else:
            return self.prompt_without_input.format(**args)