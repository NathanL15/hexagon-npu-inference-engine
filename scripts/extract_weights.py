import json
import os

import torch


def extract_weights(model_path: str = "weights/model.pth", output_path: str = "weights/weights.json"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    state_dict = torch.load(model_path, map_location="cpu")
    output = {}

    for name, tensor in state_dict.items():
        output[name] = tensor.detach().cpu().numpy().tolist()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fp:
        json.dump(output, fp, indent=2)

    print(f"Extracted weights to {output_path}")


if __name__ == "__main__":
    extract_weights()
