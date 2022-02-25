from typing import Dict, List, Union
import requests

import numpy as np
from transformers import AutoTokenizer
from transformers import (
    AutoModelForTokenClassification,
    TFAutoModelForTokenClassification,
)
from utils.constants import ModelFramework


def huggingface_predict_ner(
    api_key: str,
    assets: Union[List[Dict], List[str]],
    model_framework: str,
    model_path: str,
    verbose: int = 0,
):

    if model_framework == ModelFramework.PyTorch:
        tokenizer = AutoTokenizer.from_pretrained(model_path, from_pt=True)
        model = AutoModelForTokenClassification.from_pretrained(model_path)
    elif model_framework == ModelFramework.Tensorflow:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = TFAutoModelForTokenClassification.from_pretrained(model_path)
    else:
        raise NotImplementedError(
            f"Predictions with model framework {model_framework} not implemented"
        )

    asset_predictions = []

    for ind, asset in enumerate(assets):
        response = requests.get(
            asset["content"],
            headers={
                "Authorization": f"X-API-Key: {api_key}",
            },
        )

        sequence = response.text[: model.config.max_position_embeddings]
        if model_framework == ModelFramework.PyTorch:
            tokens = tokenizer(
                sequence,
                return_tensors="pt",
                max_length=model.config.max_position_embeddings,
            )
        else:
            tokens = tokenizer(
                sequence,
                return_tensors="tf",
                max_length=model.config.max_position_embeddings,
            )

        output = model(**tokens)
        decoded_tokens = tokenizer.batch_decode(tokens["input_ids"])

        predictions = np.argmax(output["logits"].detach().numpy(), axis=-1).tolist()
        print(f"-------")
        print(f"example {ind}")
        print(f"decoded tokens: {decoded_tokens}")
        predicted_labels = [model.config.id2label[p] for p in predictions[0]]
        print(f"predictions: {predicted_labels}")
        asset_predictions.append(predictions)

    return asset_predictions
