import os
import time
import awswrangler as wr
import pandas as pd
from gensim.utils import simple_preprocess
from gensim import corpora
from gensim.utils import simple_preprocess
from gensim import models
from gensim.corpora import Dictionary
from gensim.matutils import corpus2dense, corpus2csc
from tqdm import tqdm
from utils import setup_applevel_logger, get_logger, replace_typical_misspell, clean_text, clean_numbers
from utils import save_to_s3, get_from_s3
from datetime import datetime
from xgboost import XGBClassifier



TODAY = datetime.today().strftime("%Y%m%d")
BUCKET_NAME = 'sagemaker-godeltech'
TRAIN_PATH = f"s3://{BUCKET_NAME}/data/train/train.csv"
VAL_PAT = f"s3://{BUCKET_NAME}/data/validate/validate.csv"
DICTIONARY_PATH = "xgboost/dictionary"
MODEL_PATH = "xgboost/models"
PATH_LOGS = 'loggings'



class XGBoostTrainer():
    def __init__(self, xgb_params=None, predict_params=None):
        super(XGBoostTrainer, self).__init__()
        self.xgb_params = xgb_params
        self.predict_params = predict_params

    def create_dictionary(self):
        """
        It creates a BoW and Dictionary from train dataset
        Dictionary is saved to S3 bucket 
        """
        log.debug('Loading files...')
        train = wr.s3.read_csv([TRAIN_PATH])
        val = wr.s3.read_csv([VAL_PAT])
        train_text = train['comment_text']
        val_text = val['comment_text']
        self.train_label = train['toxicity']
        self.val_label = val['toxicity']
        log.debug('Starting to build a dictionary...')
        self.dictionary = corpora.Dictionary()
        log.info('New dictionary was initialized')
        self.bow_train = [self.dictionary.doc2bow(doc, allow_update=True) for doc in tqdm(self.simple_preproc(train_text))]     #allow_update=True - add new words to dictionary
        self.bow_val = [self.dictionary.doc2bow(doc, allow_update=False) for doc in tqdm(self.simple_preproc(val_text))]
        self.dictionary.save("tmp/dictionary_full")
        save_to_s3(BUCKET_NAME, "tmp/dictionary_full", f"{DICTIONARY_PATH}/dictionary_full_{TODAY}")
        log.info('Dictionary was uploaded to S3')
        self.num_docs = self.dictionary.num_docs
        self.num_terms = len(self.dictionary.keys())
        log.info(f"Number of docs is {self.num_docs}, there are {self.num_terms} words in dictionary")

    def create_tfidf_model(self):
        """
        Transform data (train and valid) to sparse matrices
        It uses Dictionary from previous step
        """
        get_from_s3(BUCKET_NAME, f"{DICTIONARY_PATH}/dictionary_full_{TODAY}", "tmp/my_dictionary_full")
        self.loaded_dict = corpora.Dictionary.load("tmp/my_dictionary_full")
        log.debug('Dictionary was downloaded...')
        self.tfidf = models.TfidfModel(self.bow_train, dictionary=self.loaded_dict)
        train_tfidf = self.tfidf[self.bow_train]
        val_tfidf = self.tfidf[self.bow_val]
        self.tfidf.save(f"tmp/tfidf_model_{TODAY}")
        save_to_s3(BUCKET_NAME, f"tmp/tfidf_model_{TODAY}", f"{MODEL_PATH}/tfidf_model_{TODAY}")
        log.debug('TfidfModel was saved to S3 bucket')
        self.train_tfidf_sparse = corpus2csc(train_tfidf, num_terms=self.num_terms, num_docs=self.num_docs).T
        self.val_tfidf_sparse = corpus2csc(val_tfidf, num_terms=self.num_terms, num_docs=self.num_docs).T
        log.debug('Ready for training...')
        
        
    def train(self):
        if xgb_params is None:
            raise ValueError('Training is not possible, specify training params')
        log.debug('Start training of XGBoost model')
        start = time.time()
        self.xgb_model = XGBClassifier(**self.xgb_params)
        self.xgb_model.fit(self.train_tfidf_sparse, self.train_label)
        end = time.time()
        self.xgb_model.save_model(f"tmp/xgb_model_{TODAY}.json")
        save_to_s3(BUCKET_NAME, f"tmp/xgb_model_{TODAY}.json", f"{MODEL_PATH}/xgb_model_{TODAY}.json")
        log.info(f'XGboost model was trained and saved to S3. Time spent is {end-start} seconds')
        
    def predict(self, input_text: pd.DataFrame):
        ##uploading of previously trained dict andm models
        if self.predict_params is None:
            dictionary = self.predict_params['uploaded_dictionary']
            tfidf_model = self.tfidf
            xgboost_model = self.xgb_model
        else:
            dictionary = self.dictionary
            tfidf_model = self.predict_params['tfidf_model']
            xgboost_model = self.predict_params['xgboost_model']
        ##predicions
        input_text = input_text['comment_text'].apply(replace_typical_misspell)
        input_text = input_text['comment_text'].apply(clean_text)
        input_text = input_text['comment_text'].apply(clean_numbers)
        bow_input_text = [dictionary.doc2bow(doc, allow_update=False) for doc in tqdm(self.simple_preproc(input_text))]
        input_text_tfidf = tfidf_model[bow_input_text]
        input_text_tfidf_sparse = corpus2csc(input_text_tfidf,
                                             num_terms=len(dictionary.keys()),
                                             num_docs=dictionary.num_docs).T
        predictions = xgboost_model.predict_proba(input_text_tfidf_sparse)[:, 1]
        return predictions
        
    
    @staticmethod
    def simple_preproc(text, deacc=False, min_len=2, max_len=10):
        """
        It is a generator to preprocess texts.
        This lowercases, tokenizes, de-accents (optional) 
        the output are final tokens = unicode strings, that won’t be processed any further.
        """
        for line in text:
            yield simple_preprocess(line)
                       
if __name__ == "__main__":     
    log = setup_applevel_logger(file_name = f"{PATH_LOGS}/logs_{TODAY}")            
    if not os.path.isdir('tmp'):
        os.mkdir('tmp')
    params = {
        'max_depth': 7,
        'eta' : 1e-05,
        'gamma' : 0.0008,
        'min_child_weight' : 3,
        'subsample': 0.8,
        'verbosity' : 3,
        'colsample_bytree' : 0.7,
        'objective' : 'binary:logistic'
            }
    xgb = XGBoostTrainer()
    xgb.create_dictionary()
    xgb.create_tfidf_model()
    # xgb.train()
    
    
    