import os
from os.path import splitext
import sys
import zipfile
from aws import bucket

def has_ext(file):
    fn, ext = splitext(file)
    if ext == '':
        return False
    else:
        return True

def download_dependency(dependency):
    pkgdir = '/tmp/requirements'
    if not has_ext(dependency):
        s3_key = dependency + '.zip'
    else:
        s3_key = dependency

    print(1)
    tmp_obj = '/tmp/' + s3_key
    bucket.download_file(s3_key, tmp_obj)
    print(2)
    if tmp_obj.endswith('.zip'):
        zipfile.ZipFile(tmp_obj, 'r').extractall(pkgdir)
    os.remove(tmp_obj)
    print(3)
    
    if pkgdir not in sys.path:
        sys.path.append(pkgdir)