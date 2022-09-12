# Experiments with Sagemaker
My attempt to make friends with AWS Sagemaker

In this small experiment I will try to compare the estimators:

    1. Traditional approach. TF-IDF + XGBoost
    2. Embedding with GloVe + LSTM
    3. Transformers     
    
Plan:

    * download the data
    * divide it into the parts: train, validation, test. Test is by the `split` column, the rest is for train/validation 80/20
    
