import os
import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

JSON_CONTENT_TYPE = 'application/json'
MODEL_NAME = 'unitary/unbiased-toxic-roberta'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sigmoid = torch.nn.Sigmoid()

def model_fn(model_dir):
    """
    Here we define tokenizer and model
    """
    tokenizer_init = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, return_dict=False).eval().to(device)
    model_config = AutoConfig.from_pretrained(MODEL_NAME)
    labels = {k: v for k, v in sorted(model_config.label2id.items(), key=lambda item: item[1])}.keys()

    return model, tokenizer_init, labels


def input_fn(serialized_input_data, content_type=JSON_CONTENT_TYPE):
    """
    Read data from JSON input
    """
    if content_type == JSON_CONTENT_TYPE:
        input_data = json.loads(serialized_input_data)
        return input_data
    else:
        raise Exception('Requested unsupported ContentType in Accept: ' + content_type)
        return
    

def predict_fn(input_data, models):
    # Initialize models and tokenizer
    model_toxic, tokenizer, labels = models
    embeddings  = tokenizer(input_data[0],
                            max_length=512,
                            padding="max_length",
                            truncation=True,
                            add_special_tokens = True,
                            return_tensors="pt").to(device)
    inputs = tuple(embeddings.values())
    # Convert example inputs to a format that is compatible with TorchScript tracing
    with torch.no_grad():
        model_toxic.to(device)
        outputs = model_toxic(*inputs)[0][0]
    probas = sigmoid(outputs).cpu().detach().numpy().astype('str')
    return dict(zip(labels, probas))


def output_fn(prediction_output, accept=JSON_CONTENT_TYPE):
    """
    Output of the model turned to JSON
    """
    if accept == JSON_CONTENT_TYPE:
        return json.dumps(prediction_output), accept
    
    raise Exception('Requested unsupported ContentType in Accept: ' + accept)
    
