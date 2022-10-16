# AWS Sagemaker: Toxic Comment Detection

## Goals

1. Detect toxic comments and reduce the unintended bias of the model using ML techniques
2. Learn and explore possibilities of AWS SageMaker

## Toxicity comment detection:

>A main area of focus is machine learning models that can identify toxicity in online conversations, where toxicity is defined as anything rude, disrespectful or otherwise likely to make someone leave a discussion.

## About the dataset

### Background
At the end of 2017 the Civil Comments platform shut down and chose make their ~2m public comments from their platform available in a lasting open archive so that researchers could understand and improve civility in online conversations for years to come. Jigsaw sponsored this effort and extended annotation of this data by human raters for various toxic conversational attributes.

In the data supplied for this competition, the text of the individual comment is found in the comment_text column. Each comment in Train has a toxicity label (target), and models should predict the target toxicity for the Test data. This attribute (and all others) are fractional values which represent the fraction of human raters who believed the attribute applied to the given comment. For evaluation, test set examples with target >= 0.5 will be considered to be in the positive class (toxic).

General info about dataset:
+  ~2M public comments
+ CSV file (873.6 MB) 
+ labeled by humans
+ mostly American English
+ train/validation/test ~ 0.72/0.18/0.10
+ target >= 0.5 will be in the positive class (toxic)
+ subgroups

The data also has several additional toxicity subtype attributes. Models do not need to predict these attributes for the competition, they are included as an additional avenue for research. Subtype attributes are:

- male
- female
- homosexual_gay_or_lesbian
- christian 
- jewish 
- muslim
- black
- white
- psychiatric_or_mental_illness


Additionally, a subset of comments have been labeled with a variety of identity attributes, representing the identities that are mentioned in the comment. The columns corresponding to identity attributes are listed below. Only identities with more than 500 examples in the test set (combined public and private) will be included in the evaluation calculation. These identities are shown in bold.

Latest version of dataset is available ***[here](https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification)***.

### Labelling Schema
To obtain the toxicity labels, each comment was shown to up to 10 annotators. Annotators were asked to: *"Rate the toxicity of this comment"*


>- Very Toxic (a very hateful, aggressive, or disrespectful comment that is very likely to make you leave a discussion or give up on sharing your perspective)
>- Toxic (a rude, disrespectful, or unreasonable comment that is somewhat likely to make you leave a discussion or give up on sharing your perspective)
>- Hard to Say
>- Not Toxic

Based on that we need to build a model for toxic comments findings

___

## Plan what to do

+ Build a model
  * Text preprocessing
      + Text cleaning (removing HTML/XML tags, figures, formula, etc.)
      + Removing stop words
      + Tokenization (*tf-idf, word2vec, BertTokenizer)
  * Model training
      + Baseline with tfidf and xgboost (score ~ 89%)
      + GloVe and LSTM (score ~ 90%)
      + transformers using `unitary/unbiased-toxic-roberta` model (score ~ 93%)
+ Deploy model
    * Real-time endpoint [x]
    * AWS Inf1 endpoint [x]
    * Asynch endpooint [x]
    * Serverless endpoint [ ]
    
## Future improvements

+ Multilanguage model
+ `API Gateway -> Lambda -> Sagemaker endpoint`
+ add full MLOps
___

## Reference:

- What Is Amazon SageMaker? - Amazon SageMaker. (2022). Retrieved October 14, 2022, from https://docs.aws.amazon.com/sagemaker/latest/dg/whatis.html 
- Amazon SageMaker Python SDK — sagemaker 2.112.2 documentation. (2022). Retrieved October 14, 2022, from https://sagemaker.readthedocs.io/en/stable/index.html 
- Jigsaw Unintended Bias in Toxicity Classification | Kaggle. Retrieved October 14, 2022, from https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification 
- Jordan, J. (2021, January 3). Organizing machine learning projects: project management guidelines. Jeremy Jordan. Retrieved October 14, 2022, from https://www.jeremyjordan.me/ml-projects-guide/
- Schmid, P. (2021, October 6). Scalable, Secure Hugging Face Transformer Endpoints with Amazon SageMaker, AWS Lambda, and CDK. Philschmid Blog. Retrieved October 14, 2022, from https://www.philschmid.de/huggingface-transformers-cdk-sagemaker-lambda
- Card, D., Gabriel, S., Choi, Y., & Smith, N. A. (2019). The Risk of Racial Bias in Hate Speech Detection. Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics. https://doi.org/10.18653/v1/p19-1163


