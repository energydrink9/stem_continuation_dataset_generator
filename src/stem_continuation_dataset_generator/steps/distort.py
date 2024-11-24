import io
import os
import traceback
from typing import List, Tuple, cast
from fsspec import AbstractFileSystem
import numpy as np
from pydub import AudioSegment
from audiomentations import Compose, AddGaussianSNR, BandStopFilter, RoomSimulator, SevenBandParametricEQ, SomeOf
from s3fs.core import S3FileSystem
from dask.distributed import Client
from distributed import progress

from stem_continuation_dataset_generator.cluster import get_client
from stem_continuation_dataset_generator.constants import STEM_NAME
from stem_continuation_dataset_generator.utils.utils import clamp_audio_data, convert_audio_to_float_32, convert_audio_to_int_16

BUCKET_NAME = 'stem-continuation-dataset'
PROTOCOL = 's3://'
BUCKET = f'{PROTOCOL}{BUCKET_NAME}'
SOURCE_FILES_DIR = f'{STEM_NAME}/augmented'
OUTPUT_FILES_DIR = f'{STEM_NAME}distorted'

# Set this flag to True to run locally (i.e. not on Coiled)
RUN_LOCALLY = False


def get_full_track_files(fs: S3FileSystem, dir: str) -> List[str]:
    return [PROTOCOL + path for path in cast(List[str], fs.glob(os.path.join(dir, '**/all.ogg')))]


def get_stem_file(dir: str):
    return os.path.join(dir, 'stem.ogg')


def get_files_pairs(fs: S3FileSystem, dir: str) -> List[Tuple[str, str]]:
    full_track_files = get_full_track_files(fs, dir)
    pairs = [(full_track_file, get_stem_file(os.path.dirname(full_track_file))) for full_track_file in full_track_files]
    return pairs


def distort_audio(original_audio: AudioSegment) -> AudioSegment:
    sample_rate = original_audio.frame_rate
    channels = original_audio.channels
    audio = convert_audio_to_float_32(np.array(original_audio.get_array_of_samples()))
    transform = Compose(
        transforms=[
            # ApplyImpulseResponse(),
            # BitCrush(p=1, min_bit_depth=5, max_bit_depth=10),
            # BandPassFilter(min_center_freq=200., max_center_freq=4000., p=1.0),
            SomeOf(
                1,
                [
                    BandStopFilter(p=1, min_center_freq=500., max_center_freq=4000.),
                    RoomSimulator(p=1, leave_length_unchanged=True),
                    SevenBandParametricEQ(p=1, min_gain_db=-3.5, max_gain_db=3.5),
                ],
            ),
            AddGaussianSNR(min_snr_db=20., max_snr_db=35., p=0.8),
        ],
        p=0.5,
        shuffle=False,
    )
    augmented_audio = transform(audio, sample_rate=sample_rate)
    data = convert_audio_to_int_16(clamp_audio_data(augmented_audio))
    data = data.reshape((-1, 2))
    return AudioSegment(data=data, sample_width=2, frame_rate=sample_rate, channels=channels)  # type: ignore


def distort_file(fs: AbstractFileSystem, file_path: str, output_file_path: str):
    with fs.open(file_path, 'rb') as audio_file:
        data = io.BytesIO(audio_file.read())  # type: ignore
        audio = AudioSegment.from_ogg(data)  # type: ignore
        augmented = distort_audio(audio)

        # Export the final merged track to a single .ogg file
        with fs.open(output_file_path, 'wb') as file:
            bytes_io = io.BytesIO()
            augmented.export(bytes_io, format='ogg', codec='libopus')  # type: ignore
            file.write(bytes_io.getvalue())  # type: ignore


def distort(params: Tuple[S3FileSystem, Tuple[str, str], str, str]) -> None:

    fs, (full_track_file_path, stem_file_path), source_directory, output_directory = params

    try:
        file_dir = os.path.dirname(full_track_file_path)
        full_track_relative_path = os.path.relpath(file_dir, source_directory)
        actual_output_dir = os.path.join(output_directory, full_track_relative_path)
        fs.makedirs(actual_output_dir, exist_ok=True)
        full_track_output_file_path = os.path.join(actual_output_dir, os.path.basename(full_track_file_path))

        if not fs.exists(full_track_output_file_path):
            distort_file(fs, full_track_file_path, full_track_output_file_path)

        stem_relative_path = os.path.relpath(stem_file_path, source_directory)
        stem_output_file_path = os.path.join(output_directory, stem_relative_path)
        if not fs.exists(stem_output_file_path):
            fs.copy(stem_file_path, stem_output_file_path)
    
    except Exception as e:
        print(f'Error processing {full_track_file_path} or {stem_file_path}: {e}')
        print(traceback.format_exc())


def distort_all(source_directory: str, output_directory: str):
    fs = S3FileSystem(use_listings_cache=False)
    files: List[Tuple[str, str]] = get_files_pairs(fs, os.path.join(BUCKET, source_directory))
    
    params_list: List[Tuple[S3FileSystem, Tuple[str, str], str, str]] = [(fs, file_pair, os.path.join(BUCKET, source_directory), os.path.join(BUCKET, output_directory)) for file_pair in files]

    client = cast(Client, get_client(
        RUN_LOCALLY,
        n_workers=[1, 10],
    ))
    
    print('Distorting audio tracks')
    futures = client.map(distort, params_list)
    progress(futures)

    return output_directory


if __name__ == '__main__':
    distort_all(SOURCE_FILES_DIR, OUTPUT_FILES_DIR)