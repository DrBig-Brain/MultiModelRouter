import os
from services.load_model import load_model
import json
from transformers import AutoTokenizer
from dotenv import load_dotenv
from typing import Dict
MODEL_ID = os.getenv("MODEL_ID")


def inference(model, tokenizer, prompt):
    model.eval()

    encoded_prompt = tokenizer(
        prompt,
        return_tensors = 'pt',
        add_special_tokens = True,
        max_length = 512,
        padding = "max_length",
        truncation = True
    )

    result = model(encoded_prompt)
    return result

if __name__ == "__main__":
    model, tokenizer = load_model()
    prompt = "Who is the president of India ?"
    result = inference(model, tokenizer, prompt)
    print(result)
