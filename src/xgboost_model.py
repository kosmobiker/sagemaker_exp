import os
import time
import awswrangler as wr
import pandas as pd
from gensim.utils import simple_preprocess
from gensim import corpora
from gensim.utils import simple_preprocess
from gensim import models
from gensim.corpora import Dictionary, MmCorpus
from gensim.matutils import corpus2dense, corpus2csc
from tqdm import tqdm
from utils import setup_applevel_logger, get_logger, replace_typical_misspell, clean_text, clean_numbers
from utils import save_to_s3, get_from_s3
from datetime import datetime
from xgboost import XGBClassifier
from quality_calculator import compute_bias_metrics_for_model, calculate_overall_auc, get_final_metric


TODAY = datetime.today().strftime("%Y%m%d")
BUCKET_NAME = 'sagemaker-godeltech'
TRAIN_PATH = f"s3://{BUCKET_NAME}/data/train/train.csv"
VAL_PATH = f"s3://{BUCKET_NAME}/data/validate/validate.csv"
TEST_PATH = f"s3://{BUCKET_NAME}/data/test/test.csv"
DICTIONARY_PATH = "xgboost/dictionary"
MODEL_PATH = "xgboost/models"
PATH_LOGS = 'loggings'

def simple_preproc(text):
    """
    It is a generator to preprocess texts.
    This lowercases, tokenizes, de-accents (optional) 
    the output are final tokens = unicode strings, that won’t be processed any further.
    """
    for line in text:
        yield simple_preprocess(line)

def main(xgb_params):
        #reading the files
        train = wr.s3.read_csv([TRAIN_PATH])
        val = wr.s3.read_csv([VAL_PATH])
        test = wr.s3.read_csv([TEST_PATH])
        #transformations
        train_text = train['comment_text']
        val_text = val['comment_text']
        train_label = train['toxicity']
        val_label = val['toxicity']
        test_text = test['comment_text']
        log.debug('Starting to build a dictionary...')
        dictionary = corpora.Dictionary()
        log.info('New dictionary was initialized')
        #create a BoW
        bow_train = [dictionary.doc2bow(doc, allow_update=True) for doc in tqdm(simple_preproc(train_text))]
        bow_val = [dictionary.doc2bow(doc, allow_update=False) for doc in tqdm(simple_preproc(val_text))]
        bow_test = [dictionary.doc2bow(doc, allow_update=False) for doc in tqdm(simple_preproc(test_text))]
        dictionary.filter_extremes(no_below=10, no_above=0.5, keep_n=50000)
        dictionary.compactify()
        # MmCorpus.serialize(f'tmp/bow_corpus_{TODAY}.mm', bow_train)  # save corpus to disk 
        # save_to_s3(BUCKET_NAME, f'tmp/bow_corpus_{TODAY}.mm', f"{DICTIONARY_PATH}/bow_corpus_{TODAY}.mm")
        log.info('Corpus was uploaded to S3')
        # dictionary.save(f"tmp/dictionary_full_{TODAY}.dict")
        # save_to_s3(BUCKET_NAME, f"tmp/dictionary_full_{TODAY}.dict", f"{DICTIONARY_PATH}/dictionary_full_{TODAY}.dict")
        log.info('Dictionary was uploaded to S3')
        num_docs = dictionary.num_docs
        num_terms = len(dictionary.keys())
        log.info(f"Number of docs is {num_docs}, there are {num_terms} words in dictionary")
        #creating of TF-IDF matrices
        tfidf = models.TfidfModel(bow_train, dictionary=dictionary, smartirs='ntc')
        train_tfidf = tfidf[bow_train]
        val_tfidf = tfidf[bow_val]
        test_tfidf = tfidf[bow_test]
        train_tfidf_sparse = corpus2csc(train_tfidf, num_terms=num_terms, num_docs=num_docs).T
        val_tfidf_sparse = corpus2csc(val_tfidf, num_terms=num_terms).T
        test_tfidf_sparse = corpus2csc(test_tfidf, num_terms=num_terms).T
        log.debug('Ready for training...')
        log.debug('Start training of XGBoost model')
        #training
        start = time.time()
        xgb_model = XGBClassifier(**xgb_params)
        xgb_model.fit(train_tfidf_sparse, train_label)
        end = time.time()
        xgb_model.save_model(f"tmp/xgb_model_{TODAY}.json")
        save_to_s3(BUCKET_NAME, f"tmp/xgb_model_{TODAY}.json", f"{MODEL_PATH}/xgb_model_{TODAY}.json")
        log.info(f'XGboost model was trained and saved to S3. Time spent is {end-start} seconds')
        lof.info("Let's make predictions...")
        #predictions
        predictions = xgboost_model.predict_proba(test_tfidf_sparse)[:, 1]
        oof_name = 'predicted_target'
        identity_columns = ['male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish', 'muslim', 'black', 'white', 'psychiatric_or_mental_illness']
        test_df[oof_name] = preds
        #evaluation
        bias_metrics_df = compute_bias_metrics_for_model(test_df, identity_columns, oof_name, 'toxicity')
        log.info(print(bias_metrics_df))
        FINAL_SCORE = get_final_metric(bias_metrics_df, calculate_overall_auc(test_df, oof_name))
        log.info(f"FINAL SCORE FOR XGBOOST IS {FINAL_SCORE}")    
                       
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
    main(params)
    
    
    
    