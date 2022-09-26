import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

JSON_CONTENT_TYPE = 'application/json'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sigmoid = torch.nn.Sigmoid()

def model_fn(model_dir):

    tokenizer_init = AutoTokenizer.from_pretrained('unitary/toxic-bert')
    model = AutoModelForSequenceClassification.from_pretrained('unitary/toxic-bert').eval().to(device)
    
    return (model, tokenizer_init)


def input_fn(serialized_input_data, content_type=JSON_CONTENT_TYPE):
    if content_type == JSON_CONTENT_TYPE:
        input_data = json.loads(serialized_input_data)
        return input_data
    else:
        raise Exception('Requested unsupported ContentType in Accept: ' + content_type)
        return
    

def predict_fn(input_data, models):

    model_toxic, tokenizer = models
    sequence = input_data[0] 
    
    max_length = 512
    tokenized_sequence_pair = tokenizer.encode_plus(sequence,
                                                    max_length=max_length,
                                                    padding='max_length',
                                                    truncation=True,
                                                    return_tensors='pt').to(device)
    
    # Convert example inputs to a format that is compatible with TorchScript tracing
    example_inputs = tokenized_sequence_pair['input_ids'], tokenized_sequence_pair['attention_mask']
    
    with torch.no_grad():
        outputs = model(**inputs).logits
    probas = sigmoid(outputs).cpu().detach().numpy()[0]
    return {
        'toxicity': probas[0],
        'severe_toxic': probas[1],
        'obscene': probas[2],
        'threat': probas[3],
        'insult': probas[4],
        'identity_hate': probas[5]
    }


def output_fn(prediction_output, accept=JSON_CONTENT_TYPE):
    if accept == JSON_CONTENT_TYPE:
        return json.dumps(prediction_output), accept
    
    raise Exception('Requested unsupported ContentType in Accept: ' + accept)