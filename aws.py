import os
import boto3

aws_lambda = boto3.client(
    'lambda',
    region_name='us-west-2',
    aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
)