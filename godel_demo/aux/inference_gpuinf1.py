import os
import json
import torch
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
    tokenizer_init = AutoTokenizer.from_pretrained(MODEL_NAME)
    model_config = AutoConfig.from_pretrained(MODEL_NAME)
    labels = {k: v for k, v in sorted(model_config.label2id.items(), key=lambda item: item[1])}.keys()
    compiled_model = os.path.exists(f'{model_dir}/model_neuron.pt')
    if compiled_model:
        import torch_neuron
        os.environ["NEURONCORE_GROUP_SIZES"] = "1"
        model = torch.jit.load(f'{model_dir}/model_neuron.pt')
    else: 
        model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    return (model, tokenizer_init, labels)


def predict_fn(input_data, models):
    # Initialize models and tokenizer
    model, tokenizer, labels = models
    # Tokenize sentences
    sentences = input_data.pop("inputs", input_data)
    inputs = tokenizer(sentences, padding=True, max_length=512, truncation=True, return_tensors='pt').to(device)
    with torch.no_grad():
        model.to(device)
        outputs = model(**inputs).logits
        probas = sigmoid(outputs).cpu().detach().numpy().round(3).astype('str')
    return [dict(zip(labels, probas[i])) for i in range(len(probas))]


def output_fn(prediction_output, accept=JSON_CONTENT_TYPE):
    if accept == JSON_CONTENT_TYPE:
        return json.dumps(prediction_output), accept
    raise Exception('Requested unsupported ContentType in Accept: ' + accept)
