"""
trying make friends with AWS SageMaker
"""

import os
import boto3
import pandas as pd
from utils import setup_applevel_logger, get_logger, create_bucket, check_data, replace_typical_misspell, clean_text, clean_numbers
from sklearn.model_selection import train_test_split
from datetime import datetime
import numpy as np
import awswrangler as wr

PATH_LOGS = 'loggings'
TODAY = datetime.today().strftime("%Y%m%d")
BUCKET_NAME = 'sagemaker-godeltech'
REGION = 'eu-west-1'
RAW_DATA = 'data/raw/toxic.csv'
PATH_DATA = f"s3://{BUCKET_NAME}/{RAW_DATA}"
SEED = 1234


def transform_raw_data(df: pd.DataFrame):
    """
    This function is used to transform the raw data
    """
    df['comment_text'] = df['comment_text'].fillna(" ")
    identity_columns = ['male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish', 'muslim', 'black', 'white', 'psychiatric_or_mental_illness']
    for col in identity_columns + ['toxicity']:
        df.loc[:, col] = np.where(df[col] >= 0.5, 1, 0)
    col_to_drop = [col for col in df.columns if col not in identity_columns + ['toxicity', 'comment_text', 'split']]
    df = df.drop(col_to_drop, axis=1)
    train_df = df[df['split'] == 'train']
    test_df = df[df['split'] != 'train']
    train_text, val_text = train_test_split(train_df[['comment_text', 'toxicity']], test_size=0.2, random_state=SEED)
    test_df = test_df.reset_index(drop=True)
    train_text = train_text.reset_index(drop=True)
    val_text = val_text.reset_index(drop=True)
    log.info(f"Train shape is {train_text.shape}, Val shape is {val_text.shape}, Test shape is {test_df.shape}")
    # clean misspellings
    train_text['comment_text'] = train_text['comment_text'].apply(replace_typical_misspell)
    val_text['comment_text'] = val_text['comment_text'].apply(replace_typical_misspell)
    # clean the text
    train_text['comment_text'] = train_text['comment_text'].apply(clean_text)
    val_text['comment_text'] = val_text['comment_text'].apply(clean_text)
    # clean numbers
    train_text['comment_text'] = train_text['comment_text'].apply(clean_numbers)
    val_text['comment_text'] = val_text['comment_text'].apply(clean_numbers)
    try:
        wr.s3.to_csv(train_text, f"s3://{BUCKET_NAME}/data/train/train.csv", index=False)
        wr.s3.to_csv(val_text, f"s3://{BUCKET_NAME}/data/validate/validate.csv", index=False)
        wr.s3.to_csv(test_df, f"s3://{BUCKET_NAME}/data/test/test.csv", index=False)
        log.info(f"Data were successfully landed to {BUCKET_NAME} S3 bucket")
    except Exception as e:
        log.error(e)


if __name__ == "__main__":
    ###### some preperations ########
    #check if folder for logs exists
    if not os.path.isdir(PATH_LOGS):
        os.mkdir(PATH_LOGS)
    log = setup_applevel_logger(file_name = f"{PATH_LOGS}/logs_{TODAY}")
    #create bucket if not exists
    create_bucket(BUCKET_NAME, region=REGION)
    assert check_data(BUCKET_NAME, RAW_DATA) == True
    
    ##### prepare the data ##########
    chunks= wr.s3.read_csv(path=PATH_DATA, chunksize=10000)
    df = pd.concat(chunks)
    transform_raw_data(df)
    

    