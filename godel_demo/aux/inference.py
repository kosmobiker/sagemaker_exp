import os
import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

JSON_CONTENT_TYPE = 'application/json'
MODEL_NAME = 'unitary/unbiased-toxic-roberta'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sigmoid = torch.nn.Sigmoid()

def input_fn(serialized_input_data, content_type=JSON_CONTENT_TYPE):
    if content_type == JSON_CONTENT_TYPE:
        input_data = json.loads(serialized_input_data)
        return input_data
    else:
        raise Exception('Requested unsupported ContentType in Accept: ' + content_type)
        return

def model_fn(model_dir):
    """
    Here we define tokenizer and model
    """
    tokenizer_init = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, return_dict=False).eval().to(device)
    model_config = AutoConfig.from_pretrained(MODEL_NAME)
    labels = {k: v for k, v in sorted(model_config.label2id.items(), key=lambda item: item[1])}.keys()

    return model, tokenizer_init, labels


def predict_fn(input_data, models):
    # Initialize models and tokenizer
    model, tokenizer, labels = models
    # Tokenize sentences
    sentences = input_data.pop("inputs", input_data)
    inputs = tokenizer(sentences, padding=True, max_length=512, truncation=True, return_tensors='pt').to(device)
    with torch.no_grad():
        model.to(device)
        outputs = model(**inputs)[0][0]
    probas = sigmoid(outputs).cpu().detach().numpy().astype('str')
    return dict(zip(labels, probas))


def output_fn(prediction_output, accept=JSON_CONTENT_TYPE):
    if accept == JSON_CONTENT_TYPE:
        return json.dumps(prediction_output), accept
    
    raise Exception('Requested unsupported ContentType in Accept: ' + accept)