import os
import io
import boto3
import json
import csv

# grab environment variables
ENDPOINT_NAME = os.environ['ENDPOINT_NAME']
runtime= boto3.client('runtime.sagemaker')

def lambda_handler(event, context):
    runtime= boto3.client('runtime.sagemaker')
    response = runtime.invoke_endpoint(EndpointName = 'toxicbert-custom-endpoint',
                                       ContentType = 'application/json',
                                       Body = json.dumps(event)
                                  )
    output = json.loads(response['Body'].read().decode('utf-8'))
    
    return {
        'statusCode' : 200,
        'headers' : { 'Content-Type' : 'text/plain', 'Access-Control-Allow-Origin' : '*' },
        'body' : output
    }