import os

DATASET_TAGS = ['medium']
CLEARML_DATASET_NAME = 'stem_continuation_dataset'
CLEARML_DATASET_VERSION = '1.0.0'
DEFAULT_STEM_NAME = 'drum'
STORAGE_BUCKET_NAME = 'stem-continuation-dataset'
DASK_CLUSTER_NAME = 'stem-continuation-dataset-generator-cluster'


def get_original_files_path():
    return os.path.join(STORAGE_BUCKET_NAME, 'original')


def get_merged_files_path(stem_name: str = DEFAULT_STEM_NAME):
    return os.path.join(STORAGE_BUCKET_NAME, stem_name, 'merged')


def get_augmented_files_path(stem_name: str = DEFAULT_STEM_NAME):
    return os.path.join(STORAGE_BUCKET_NAME, stem_name, 'augmented')


def get_distorted_files_path(stem_name: str = DEFAULT_STEM_NAME):
    return os.path.join(STORAGE_BUCKET_NAME, stem_name, 'distorted')


def get_encoded_files_path(stem_name: str = DEFAULT_STEM_NAME):
    return os.path.join(STORAGE_BUCKET_NAME, stem_name, 'encoded')


def get_split_files_path(stem_name: str = DEFAULT_STEM_NAME):
    return os.path.join(STORAGE_BUCKET_NAME, stem_name, 'split')

