from scripts.model import CustomModel
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
import json

def load_model():
    MODEL_ID = "nvidia/prompt-task-and-complexity-classifier"

    config_path = hf_hub_download(
        repo_id = MODEL_ID,
        filename = "config.json"
    )

    with open(config_path, "r") as f:
        config = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    MODEL_ID = "nvidia/prompt-task-and-complexity-classifier"
    config_path = hf_hub_download(
        repo_id = MODEL_ID,
        filename = "config.json"
    )

    with open(config_path, "r") as f:
        config = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    model = CustomModel(
        target_sizes=config["target_sizes"],
        task_type_map=config["task_type_map"],
        weights_map=config["weights_map"],
        divisor_map=config["divisor_map"],
    ).from_pretrained(MODEL_ID)

    model.eval()

    return model, tokenizer



if __name__ == "__main__":
    load_model()
    print("model loaded")
