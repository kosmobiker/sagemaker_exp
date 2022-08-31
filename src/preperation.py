import os
import boto3

if not os.path.isfile('/data/toxic_data.csv'):
    if not os.path.exists("./data"):
        os.makedirs("./data")
    s3 = boto3.client('s3')
    s3.download_file('godelsagemaker', 'data/toxic_data.csv', 'data/toxic_data.csv')


