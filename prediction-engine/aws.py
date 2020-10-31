import os
import boto3

s3 = boto3.resource(
    's3',
    region_name='us-west-2'
)

bucket = s3.Bucket('kevins-project-demo-bucket')