import csv
import json
import awswrangler as wr
from utils import get_from_s3, save_to_s3

BUCKET_NAME = "sagemaker-godeltech"
TEST_PATH = f"s3://{BUCKET_NAME}/data/test/test.csv"
DATASET_CSV_FILE ="tmp/test_data.csv"
DATASET_JSONL_FILE="tmp/test_data.jsonl"

#read data
test = wr.s3.read_csv([TEST_PATH])
test.rename(columns = {'comment_text':'input'}, inplace = True)
test['input'].to_csv(DATASET_CSV_FILE, index=False)

#transform it in specific wformat for SageMaker
with open(DATASET_CSV_FILE, "r+") as infile, open(DATASET_JSONL_FILE, "w+") as outfile:
    reader = csv.DictReader(infile)
    for row in reader:
        json.dump(row, outfile)
        outfile.write('\n')
        
#uplaod to s3
save_to_s3(BUCKET_NAME, DATASET_JSONL_FILE, 'data/test/test_data.jsonl')





