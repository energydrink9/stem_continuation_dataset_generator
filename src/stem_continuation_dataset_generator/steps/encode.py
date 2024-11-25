import os
import pickle
import traceback
from typing import List, Tuple, cast
from distributed import Client, progress
from s3fs.core import S3FileSystem

from stem_continuation_dataset_generator.cluster import get_client
from stem_continuation_dataset_generator.codec import encode_file
from stem_continuation_dataset_generator.constants import STEM_NAME
from stem_continuation_dataset_generator.utils.device import get_device

BUCKET_NAME = 'stem-continuation-dataset'
PROTOCOL = 's3://'
BUCKET = f'{PROTOCOL}{BUCKET_NAME}'
SOURCE_FILES_DIR = f'{STEM_NAME}/distorted'
OUTPUT_FILES_DIR = f'{STEM_NAME}/encoded'

# Set this flag to True to run locally (i.e. not on Coiled)
RUN_LOCALLY = False


def get_ogg_files(fs: S3FileSystem, dir: str) -> List[str]:
    return [PROTOCOL + path for path in cast(List[str], fs.glob(os.path.join(dir, '**/*.ogg')))]


def encode(params: Tuple[S3FileSystem, str, str, str]):
    fs, file_path, source_directory, output_directory = params
    device = get_device()

    try:
        file_dir = os.path.dirname(file_path)
        relative_path = os.path.relpath(file_dir, source_directory)
        file_output_directory = os.path.join(output_directory, relative_path)
        fs.makedirs(file_output_directory, exist_ok=True)
        
        first_chunk_file_name = os.path.basename(file_path).split('.')[0] + '-c0.pkl'
        first_chunk_file_path = os.path.join(file_output_directory, first_chunk_file_name)
        
        if not fs.exists(first_chunk_file_path):
            with fs.open(file_path, 'rb') as file:
                encoded_audio, frame_rate = encode_file(file, device)                

                output_filename = os.path.basename(file_path).split('.')[0] + '.pkl'
                output_file_path = os.path.join(file_output_directory, output_filename)
                if not fs.exists(output_file_path):
                    with fs.open(output_file_path, 'wb') as output_file:
                        pickle.dump(encoded_audio.detach().to('cpu'), output_file)

    except Exception:
        print(f'Error while encoding file {file_path}')
        print(traceback.format_exc())
    

def encode_all(source_directory: str, output_directory: str):
    fs = S3FileSystem(use_listings_cache=False)
    files = get_ogg_files(fs, os.path.join(BUCKET, source_directory))
    
    params_list: List[Tuple[S3FileSystem, str, str, str]] = [(fs, file_path, os.path.join(BUCKET, source_directory), os.path.join(BUCKET, output_directory)) for file_path in files]

    client = cast(Client, get_client(
        RUN_LOCALLY,
        n_workers=[1, 1],
        # worker_vm_types=['c6a.xlarge'],
        worker_vm_types=['g4dn.xlarge'],
        scheduler_vm_types=['t3.medium'],
        spot_policy='spot',
        use_best_zone=True,
    ))
    
    print('Encoding audio tracks')
    
    # for i in range(len(params_list) - 1, 0, -1):
    #     print(f'Processing {i} of {len(params_list)} {round(cast(float, i) / len(params_list) * 100)}')
    #     encode(params_list[i])

    futures = client.map(encode, params_list)
    progress(futures)

    return output_directory


if __name__ == '__main__':
    encode_all(SOURCE_FILES_DIR, OUTPUT_FILES_DIR)