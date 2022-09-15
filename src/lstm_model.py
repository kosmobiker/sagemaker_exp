import pandas as pd
import boto3
import numpy as np
from collections import Counter
import os
import random
from datetime import datetime
import re, string
from typing import Dict

import torch, torchtext
import torchvision.models as models
from torchtext.data.utils import get_tokenizer
from torch.utils.data import TensorDataset, DataLoader
from torchtext.vocab import GloVe
from torch import nn, optim
from torch.nn import Module, Embedding, LSTM, RNN, GRU, Linear, Sequential, Dropout
from torch.nn.functional import sigmoid, relu, elu, tanh
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.nn.utils.rnn import PackedSequence

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, roc_auc_score
import time

from utils import setup_applevel_logger, get_logger, replace_typical_misspell, clean_text, clean_numbers
from utils import save_to_s3, get_from_s3
from quality_calculator import compute_bias_metrics_for_model, calculate_overall_auc, get_final_metric

SEED = 1234
TODAY = datetime.today().strftime("%Y%m%d")
BUCKET_NAME = 'sagemaker-godeltech'
TRAIN_PATH = f"s3://{BUCKET_NAME}/data/train/train.csv"
VAL_PATH = f"s3://{BUCKET_NAME}/data/validate/validate.csv"
TEST_PATH = f"s3://{BUCKET_NAME}/data/test/test.csv"
VOCAB_PATH = "lstm/vocab"
MODEL_PATH = "lstm/models"
PATH_LOGS = 'loggings'


def seed_everything(seed=SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
def build_vocab(text):
    log.info('start creating vocab')
    #tokenization
    tokenizer = get_tokenizer("basic_english")
    counter = Counter()
    for line in tqdm(train_text):
        counter.update(tokenizer(line))
    # Create a vocabulary with words seen at least 3 (min_freq) times
    vocab = torchtext.vocab.vocab(counter, min_freq=3)
    # Add the unknown token and use it by default for unknown words
    unk_token = '<unk>'
    # vocab.insert_token(unk_token, 0)
    vocab.set_default_index(0)
    # Add the pad token
    pad_token = '<pad>'
    vocab.insert_token(pad_token, 1)
    #save vocab
    torch.save(vocab, f'/tmp/vocab_obj_{TODAY}.pth')    
    save_to_s3(BUCKET_NAME, f'/tmp/vocab_obj_{TODAY}.pth', f'{VOCAB_PATH}/vocab_obj_{TODAY}.pth')
    log.debug("Vocab is ready")
    return vocab

def text_transform_pipeline(vocab, token) 
    return lambda x: [vocab[token] for token in tokenizer(x)]

def transform_text(text_list, max_len):
    # Transform the text
    transformed_data = [text_transform_pipeline(vocab, text)[:max_len] for text in text_list]

    # Pad zeros if the text is shoter than max_len
    for data in transformed_data:
        data[len(data) : max_len] = np.ones(max_len - len(data))

    return torch.tensor(transformed_data, dtype=torch.int64)

class SpatialDropout(nn.Dropout2d):
    def forward(self, x):
        x = x.unsqueeze(2)    # (N, T, 1, K)
        x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)
        x = super(SpatialDropout, self).forward(x)  # (N, K, 1, T), some features are masked
        x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)
        x = x.squeeze(2)  # (N, T, K)
        return x
    
class NeuralNet(nn.Module):
    def __init__(self, embedding_matrix, lstm_hiden_size, dense_hiden_size):
        super(NeuralNet, self).__init__()
        max_features = embedding_matrix.shape[0]
        embed_size = embedding_matrix.shape[1]
        
        self.embedding = nn.Embedding(max_features, embed_size)
        self.embedding.weight = nn.Parameter(embedding_matrix.clone().detach())
        self.embedding.weight.requires_grad = False
        self.embedding_dropout = SpatialDropout(0.6)
        self.lstm1 = nn.LSTM(embed_size, lstm_hiden_size, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(lstm_hiden_size * 2, lstm_hiden_size, bidirectional=True, batch_first=True)
        self.linear1 = nn.Linear(dense_hiden_size, dense_hiden_size)
        self.linear2 = nn.Linear(dense_hiden_size, dense_hiden_size)
        self.linear_out = nn.Linear(dense_hiden_size, 1)
        
    def forward(self, x):
        h_embedding = self.embedding(x)
        h_embedding = self.embedding_dropout(h_embedding)
        h_lstm1, _ = self.lstm1(h_embedding)
        h_lstm2, _ = self.lstm2(h_lstm1)
        # global average pooling
        avg_pool = torch.mean(h_lstm2, 1)
        max_pool, _ = torch.max(h_lstm2, 1)
        h_conc = torch.cat((max_pool, avg_pool), 1)
        h_conc_linear1  = relu(self.linear1(h_conc))
        h_conc_linear2  = relu(self.linear2(h_conc))
        hidden = h_conc + h_conc_linear1 + h_conc_linear2       
        result = self.linear_out(hidden)

        return result

def train(n_epoches, train_loader, val_loader, train_label, val_label)
    log.info("Training of model is starting...")
    #  We will use Adam optimizer
    trainer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # We will use Binary Cross-entropy loss
    cross_ent_loss = nn.BCEWithLogitsLoss(reduction='mean')
    train_losses = []
    val_losses = []
    train_scores = []
    val_scores = []
    model.to(device)
    for epoch in range(n_epoches):
        start = time.time()
        training_loss = 0
        val_loss = 0
        train_score = 0
        val_score = 0
        model.train()
        for data, target in tqdm(train_loader):
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            loss = cross_ent_loss(output.squeeze(1), target)
            trainer.zero_grad()
            loss.backward()
            trainer.step()
            training_loss += loss.item()
            try:
                train_score += roc_auc_score(target.cpu(), output.detach().cpu().numpy())
                log.debug(f"Training loss = {training_loss[-1]}, training score = {train_score[-1]}")
            except ValueError:
                pass

        model.eval()
        for data, target in tqdm(val_loader):
            data = data.to(device)
            target = target.to(device)
            val_predictions = torch.sigmoid(model(data)).squeeze(1)
            loss = cross_ent_loss(val_predictions, target)
            val_loss += loss.item()
            try:
                val_score += roc_auc_score(target.cpu(), val_predictions.detach().cpu().numpy())
                log.debug(f"Validation loss = {val_loss[-1]}, validation score = {val_score[-1]}")
            except ValueError:
                pass

        # Let's take the average losses
        training_loss = training_loss / len(train_label)
        val_loss = val_loss / len(val_label)
        train_score = train_score / len(train_label)
        val_score = val_score / len(val_label)
        train_losses.append(training_loss)
        val_losses.append(val_loss)
        train_scores.append(train_score)
        val_scores.append(val_score)
        end = time.time()
        log.info(
            f"Epoch {epoch}. Train_loss {training_loss}. Validation_loss {val_loss}. Seconds {end-start}"
        )
        torch.save(model.state_dict(), f"/tmp/lstm_model_{TODAY}.pth")
        save_to_s3(BUCKET_NAME, f"/tmp/lstm_model_{TODAY}.pth", f"{MODEL_PATH}/lstm_model_{TODAY}.pth")
        log.debug('Model was uploaded to S3')
        return model

def sigmoid(x):
        return 1 / (1 + np.exp(-x))

def predict(model, test_loader):
    model.eval()
    test_predictions = []
    for data, target in tqdm(test_loader):
        test_preds = model(data.to(device))
        test_predictions.extend(
            [sigmoid(test_pred[0]) for test_pred in test_preds.detach().cpu().numpy()]
        )
    return test_predictions
    
def main(**lstm_params):
    #set the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log.info((f"Today I'm going to use {device.type}"))
    #reading the files
    train = wr.s3.read_csv([TRAIN_PATH])
    val = wr.s3.read_csv([VAL_PATH])
    test = wr.s3.read_csv([TEST_PATH])
    #transform values...
    train_text, val_text, test_text = train['comment_text'], val['comment_text'], test['comment_text']
    train_label, val_label, test_label = train['toxicity'].astype('float32'), val['toxicity'].astype('float32'), test['toxicity'].astype('float32')
    #...and send them to device
    train_label = torch.tensor(train_label.values, dtype=torch.float32).to(device)
    val_label = torch.tensor(val_label.values, dtype=torch.float32).to(device) 
    test_label = torch.tensor(test_label.values, dtype=torch.float32).to(device)
    #vocab
    vocab = build_vocab(train_text)
    #PARAMETERS
    n_epoches = lstm_params['n_epoches']
    max_len = lstm_params['max_len']
    batch_size = lstm_params['batch_size']
    # Size of the state vectors
    lstm_hiden_size = lstm_params['lstm_hiden_size']
    dense_hiden_size = 4 * lstm_params['lstm_hiden_size']
    # General NN training parameters
    learning_rate = lstm_params['batch_size']    
    # Create data loaders
    train_dataset = TensorDataset(transform_text(train_text, max_len), train_label)
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_dataset = TensorDataset(transform_text(val_text, max_len), val_label)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_dataset = TensorDataset(transform_text(test_text, max_len), test_label)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    log.info('Loaders were created')
    glove = GloVe(name="6B", dim=300, vectors_cache='/tmp')
    embedding_matrix = glove.get_vecs_by_tokens(vocab.get_itos())
    model = NeuralNet(embedding_matrix, lstm_hiden_size, dense_hiden_size)
    #trainings and evaluation
    lstm_model = train(n_epoches, train_loader, val_loader)
    predictions = predict(lstm_model, test_loader)
    np.savetxt(f"/tmp/lstm_predictions_{TODAY}.csv", predictions, delimiter=",")
    save_to_s3(BUCKET_NAME, f"/tmp/lstm_predictions_{TODAY}.csv", f"{MODEL_PATH}/lstm_predictions_{TODAY}.csv")
    oof_name = 'predicted_target'
    identity_columns = ['male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish', 'muslim', 'black', 'white', 'psychiatric_or_mental_illness']
    test[oof_name] = predictions
    #evaluation
    bias_metrics_df = compute_bias_metrics_for_model(test, identity_columns, oof_name, 'toxicity')
    log.info(print(bias_metrics_df))
    FINAL_SCORE = get_final_metric(bias_metrics_df, calculate_overall_auc(test, oof_name))
    log.info(f"FINAL SCORE FOR LSTM MODEL IS {FINAL_SCORE}")    
    

if __name__ == "__main__":
    log = setup_applevel_logger(file_name = f"{PATH_LOGS}/logs_{TODAY}")            
    if not os.path.isdir('tmp'):
        os.mkdir('tmp')
    seed_everything()
    params = {
        "n_epoches" : 10,
        "max_len" : 100,
        "batch_size" : 128,
        "lstm_hiden_size" : 128,
        "learning_rate" : 10e-4   
    }
    main(params)
   