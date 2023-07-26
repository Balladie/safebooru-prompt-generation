import torch

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, GenerationConfig


def get_generation_config():
    generation_config = GenerationConfig(
        _from_model_config=True,
        bos_token_id=50256,
        eos_token_id=50256,
        transformers_version='4.30.2',
        num_return_sequences=5,
        max_new_tokens=50,
        do_sample=True,
        top_k=50,
        top_p=0.90
    )
    return generation_config


if __name__ == '__main__':
    model_dir = '/home/gangin/workspace/da-studio/diffusion/distilgpt2-prompt-gen/runs/run-20230714_185536-gwmq1w9o/checkpoint-12000'
    # model_dir = '/home/gangin/workspace/da-studio/diffusion/distilgpt2-prompt-gen/runs/run-20230721_160802-hxuibjun/checkpoint-6400'
    
    text_input = 'sks man, masterpiece, best quality, lineart, monochrome, solo, playing guitar'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generation_config = get_generation_config()

    tokenizer = AutoTokenizer.from_pretrained('distilgpt2')
    tokenized = tokenizer(text_input, return_tensors='pt')

    model = AutoModelForCausalLM.from_pretrained(model_dir)
    outputs = model.generate(**tokenized, generation_config=generation_config)

    for output in outputs:
        print(tokenizer.batch_decode([output], skip_special_tokens=True))
        print()