import os
import random
from typing import List, Set, Tuple, cast
from s3fs.core import S3FileSystem
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from multiprocessing.pool import Pool 
import multiprocessing

from stem_continuation_dataset_generator.constants import get_encoded_files_path, get_split_files_path
from stem_continuation_dataset_generator.utils.constants import get_random_seed

SPLIT_NAMES = ['train', 'validation', 'test']
VALIDATION_SIZE = 0.12
TEST_SIZE = 0.06


def get_directories_containing_pkl_files(fs: S3FileSystem, dir: str) -> Set[str]:
    files = cast(List[str], fs.glob(os.path.join(dir, '**/*.pkl')))
    directories = {os.path.dirname(file) for file in files}
    
    return directories


def split_by_artist(artists, validation_size, test_size, seed=get_random_seed()) -> Tuple[List[str], List[str], List[str]]:
    train_artists, rest_artists = train_test_split(artists, test_size=validation_size + test_size, random_state=seed)
    validation_artists, test_artists = train_test_split(rest_artists, test_size=test_size / (validation_size + test_size), random_state=seed)

    return train_artists, validation_artists, test_artists


def copy_artist(params: Tuple[S3FileSystem, str, str, str]) -> None:
    fs, source_directory, output_directory, artist = params
    artist_path = os.path.join(source_directory, artist)
    relative_path = os.path.relpath(artist_path, source_directory)
    file_output_dir = os.path.join(output_directory, relative_path)
    fs.copy(artist_path, file_output_dir, recursive=True)


def split_all(source_directory: str, output_directory: str) -> List[str]:
    
    fs = S3FileSystem(use_listings_cache=False)

    file_paths = get_directories_containing_pkl_files(fs, source_directory)
    file_paths_artists = [os.path.split(os.path.split(file_path)[0])[-1] for file_path in file_paths]

    artists = list(set(file_paths_artists))
    splits = split_by_artist(artists, validation_size=VALIDATION_SIZE, test_size=TEST_SIZE)
    output_directories = []

    for i, split in enumerate(splits):
        print(f'Creating split {SPLIT_NAMES[i]}')

        split_directory = os.path.join(output_directory, SPLIT_NAMES[i])
        params: List[Tuple[S3FileSystem, str, str, str]] = [(fs, source_directory, split_directory, artist) for artist in split]

        with Pool(multiprocessing.cpu_count()) as pool:
            list(tqdm(pool.imap(copy_artist, params), total=len(params)))
            output_directories.append(split_directory)

    return output_directories


if __name__ == '__main__':
    random.seed(get_random_seed())
    split_all(get_encoded_files_path(), get_split_files_path())