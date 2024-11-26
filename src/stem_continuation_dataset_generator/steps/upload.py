import tempfile
from typing import List, cast, Tuple
import os
from s3fs.core import S3FileSystem
from tqdm import tqdm
import multiprocessing
import multiprocessing.pool

from stem_continuation_dataset_generator.constants import CLEARML_DATASET_VERSION, DATASET_TAGS, get_split_files_path
from stem_continuation_dataset_generator.utils.utils import upload_dataset


def get_input_dirs(split_files_path: str) -> List[str]:
    return [
        os.path.join(split_files_path, 'train'),
        os.path.join(split_files_path, 'validation'),
        os.path.join(split_files_path, 'test'),
    ]


def get_files(fs: S3FileSystem, dir: str) -> List[str]:
    return [path for path in cast(List[str], fs.glob(os.path.join(dir, '**/*.pkl')))]


def download_file(params: Tuple[S3FileSystem, str, str, str]):
    (fs, file, source_directory, output_directory) = params
    local_file_path = os.path.join(output_directory, os.path.relpath(file, source_directory))
    if not os.path.exists(local_file_path):
        try:
            fs.get(file, local_file_path, recursive=True)
        except Exception as e:
            print('Error getting the file, retrying')
            print(e)
            try:
                fs.get(file, local_file_path, recursive=True)
            except Exception as e2:
                print('Error getting the file, retrying')
                print(e2)
                try:
                    fs.get(file, local_file_path, recursive=True)
                except Exception as e3:
                    print('Error getting the file, abandoning')
                    print(e3)


def upload(split_files_path: str, tags: List[str]):
    
    fs = S3FileSystem(use_listings_cache=False)
    input_dirs = get_input_dirs(split_files_path)

    for split_dir in input_dirs:
        set = os.path.split(split_dir)[1]
        
        with tempfile.TemporaryDirectory() as local_directory:
            print(f'Downloading {set} dataset (folder {split_dir}) from S3 into {local_directory}')
            
            files = get_files(fs, split_dir)
            inputs = [(fs, file, split_dir, local_directory) for file in files]

            with multiprocessing.pool.ThreadPool(multiprocessing.cpu_count()) as pool:
                list(tqdm(pool.imap(download_file, inputs), total=len(inputs)))

            print(f'Uploading {set} dataset to ClearML')
            upload_dataset(path=local_directory, version=CLEARML_DATASET_VERSION, tags=tags + ['final'], dataset_set=set)


if __name__ == '__main__':
    upload(get_split_files_path(), DATASET_TAGS)