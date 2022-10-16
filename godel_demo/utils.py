"""
Standard logger 
it is used to get Loggings
"""
import logging
import sys
import boto3
from botocore.exceptions import ClientError
import re
import string



APP_LOGGER_NAME = 'godel-sagemaker'

def setup_applevel_logger(logger_name = APP_LOGGER_NAME, file_name=None):
    """
    This function is used for logging. It saves the logs
    into the log files in appropriate folder.
    Default log level is *DEBUG*
    Parameters:
        logger_name (str): name of the app used in logging
        file_name (str): path for logfiles
    """ 
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.handlers.clear()
    logger.addHandler(sh)
    if file_name:
        fh = logging.FileHandler(file_name)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger

def get_logger(module_name):
    """
    It is used when there is a need to embed the logging into the function
    """    
    return logging.getLogger(APP_LOGGER_NAME).getChild(module_name)

log = get_logger(__name__)

def create_bucket(bucket_name: str, s3_client=None, region=None):
    """
    Create an S3 bucket in a specified region if it is not exist
    If a region is not specified, the bucket is created in the S3 default
    region (us-east-1).
    Params
        bucket_name: Bucket to create
        s3_client: client for S3
        region: String region to create bucket in, e.g., 'us-west-2'
        
    Return
        True if bucket was created, otherwise False
    """
    if not s3_client:
        s3_client = boto3.client('s3')
    # Create bucket
    try:
        if region is None:
            response = s3_client.list_buckets()
            if bucket_name not in [bucket['Name'] for bucket in response['Buckets']]:
                s3_client.create_bucket(Bucket=bucket_name)
                log.debug(f'Bucket {bucket_name} was created')
        else:
            response = s3_client.list_buckets()
            if bucket_name not in [bucket['Name'] for bucket in response['Buckets']]:
                location = {'LocationConstraint': region}
                s3_client.create_bucket(Bucket=bucket_name,
                                        CreateBucketConfiguration=location)
                log.debug(f'Bucket {bucket_name} was created')
    except ClientError as e:
        log.error(e)
        return False
    return True

def check_data(bucket: str, key: str, s3_client=None):
    """
    Check if data is availible
    Params
        bucket_name: Bucket to create
        key: name of the file
        s3_client: client for S3
    Return
        True if data is availible
        False otherwise
    """
    if not s3_client:
        s3_client = boto3.client('s3')
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        log.info('Data is availible. Move on!')
        return True
    except ClientError as e:
        log.error(e)
        return False
    
def save_to_s3(bucket: str, filename: str, key: str, s3_client=None):
    """
    Upload file to s3 bucket
    """
    if not s3_client:
        s3_client = boto3.resource('s3')
    try:
        s3_client.meta.client.upload_file(Filename = filename, Bucket= bucket, Key = key)
    except Exception as e:
        log.error(e)

def get_from_s3(bucket: str, key: str, object_name: str, s3_client=None):
    if not s3_client:
        s3_client = boto3.resource('s3')
    try:
        s3_client.Bucket(bucket).download_file(key, object_name)
    except ClientError as e:
        if e.response['Error']['Code'] == "404":
            log.error("The object does not exist.")
        else:
            raise
 
##### BLOCK OF FUNCTIONS TO CLEAN THE TEXT
misspell_dict = {"aren't": "are not", "can't": "cannot", "couldn't": "could not",
                 "didn't": "did not", "doesn't": "does not", "don't": "do not",
                 "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
                 "he'd": "he would", "he'll": "he will", "he's": "he is",
                 "i'd": "I had", "i'll": "I will", "i'm": "I am", "isn't": "is not",
                 "it's": "it is", "it'll": "it will", "i've": "I have", "let's": "let us",
                 "mightn't": "might not", "mustn't": "must not", "shan't": "shall not",
                 "she'd": "she would", "she'll": "she will", "she's": "she is",
                 "shouldn't": "should not", "that's": "that is", "there's": "there is",
                 "they'd": "they would", "they'll": "they will", "they're": "they are",
                 "they've": "they have", "we'd": "we would", "we're": "we are",
                 "weren't": "were not", "we've": "we have", "what'll": "what will",
                 "what're": "what are", "what's": "what is", "what've": "what have",
                 "where's": "where is", "who'd": "who would", "who'll": "who will",
                 "who're": "who are", "who's": "who is", "who've": "who have",
                 "won't": "will not", "wouldn't": "would not", "you'd": "you would",
                 "you'll": "you will", "you're": "you are", "you've": "you have",
                 "'re": " are", "wasn't": "was not", "we'll": " will", "tryin'": "trying"}

def _get_misspell(misspell_dict):
    misspell_re = re.compile('(%s)' % '|'.join(misspell_dict.keys()))
    return misspell_dict, misspell_re

def replace_typical_misspell(text):
    misspellings, misspellings_re = _get_misspell(misspell_dict)

    def replace(match):
        return misspellings[match.group(0)]

    return misspellings_re.sub(replace, text)
    
puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']',
          '>', '%', '=', '#', '*', '+', '\\', '•', '~', '@', '£', '·', '_', '{', '}', '©', '^',
          '®', '`', '<', '→', '°', '€', '™', '›', '♥', '←', '×', '§', '″', '′', 'Â', '█',
          '½', 'à', '…', '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶',
          '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', '▒', '：', '¼',
          '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲',
          'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', '∙', '）', '↓', '、', '│', '（', '»', '，', '♪',
          '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', '\n']

def clean_text(x):
    x = str(x)
    for punct in puncts + list(string.punctuation):
        if punct in x:
            x = x.replace(punct, f' {punct} ')
    return x

def clean_numbers(x):
    return re.sub('\d+', ' ', x)

