import os
import json
import torch
import torch_neuron
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

JSON_CONTENT_TYPE = 'application/json'
os.environ["NEURON_RT_NUM_CORES"] = "1"
sigmoid = torch.nn.Sigmoid()


def model_fn(model_dir):
    
    model_dir = '/opt/ml/model/'
    dir_contents = os.listdir(model_dir)
    model_path = next(filter(lambda item: 'model' in item, dir_contents), None)
    tokenizer_init = AutoTokenizer.from_pretrained('unitary/toxic-bert')
    model = torch.jit.load(os.path.join(model_dir, model_path))
    
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
    model_toxic.eval()
    max_length = 512
    tokenized_sequence_pair = tokenizer(input_data[0],
                                        max_length=max_length,
                                        padding="max_length",
                                        truncation=True,
                                        add_special_tokens = True,
                                        return_tensors="pt")
    # Convert example inputs to a format that is compatible with TorchScript tracing
    neuron_inputs = tokenized_sequence_pair['input_ids'], tokenized_sequence_pair['attention_mask']
    # Make predictions
    with torch.no_grad():
        outputs = model_toxic(*neuron_inputs)
    probas = np.around(sigmoid(outputs).cpu().detach().numpy(), 4).astype('str')
    return {
        'toxicity': i[0],
        'severe_toxic': i[1],
        'obscene': i[2],
        'threat': i[3],
        'insult': i[4],
        'identity_hate': i[5]}

def output_fn(prediction_output, accept=JSON_CONTENT_TYPE):
    if accept == JSON_CONTENT_TYPE:
        return json.dumps(prediction_output), accept
    
    raise Exception('Requested unsupported ContentType in Accept: ' + accept)
    
