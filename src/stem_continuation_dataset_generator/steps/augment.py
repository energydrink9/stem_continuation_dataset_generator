import io
import os
import traceback
from typing import Any, List, Tuple, cast
from dask.distributed import Client
from distributed import progress
import numpy as np
from pydub import AudioSegment
from audiomentations import Compose, PitchShift, TimeStretch, Gain
import soundfile
from s3fs.core import S3FileSystem

from stem_continuation_dataset_generator.cluster import get_client
from stem_continuation_dataset_generator.constants import STEM_NAME
from stem_continuation_dataset_generator.utils.utils import clamp_audio_data, convert_audio_to_int_16

BUCKET = 'stem-continuation-dataset'
SOURCE_FILES_DIR = os.path.join(BUCKET, STEM_NAME, 'merged')
OUTPUT_FILES_DIR = os.path.join(BUCKET, STEM_NAME, 'augmented')
AUGMENTATIONS_COUNT = 4
AUGMENT_PITCH = False

# Set this flag to True to run locally (i.e. not on Coiled)
RUN_LOCALLY = False


def get_full_track_files(fs: S3FileSystem, dir: str) -> List[str]:
    return cast(List[str], fs.glob(os.path.join(dir, '**/all.ogg')))


def augment_files(fs: S3FileSystem, file_paths: List[Tuple[str, str]], transform: Compose) -> None:

    for file_path, output_file_path in file_paths:
        with fs.open(file_path, 'rb') as input_file:
            data = io.BytesIO(input_file.read())  # type: ignore
            audio, sr = soundfile.read(data, dtype='float32')
            audio = cast(np.ndarray[Any, np.dtype[np.float32]], audio)
            channels = audio.shape[1]
            audio = np.transpose(audio)
            augmented_audio = audio
            augmented_audio = transform(audio, sample_rate=sr)
            transform.freeze_parameters()
            length = augmented_audio.shape[1]
            correct_length = length - (length % (augmented_audio.dtype.itemsize * sr))
            augmented_audio = augmented_audio[:, :correct_length]
            augmented_audio = np.transpose(augmented_audio).reshape(-1)
            augmented_audio = convert_audio_to_int_16(clamp_audio_data(augmented_audio))
            # Using AudioSegment to save to file as soundfile presents a bug with saving in OGG format
            segment = AudioSegment(data=augmented_audio, sample_width=augmented_audio.dtype.itemsize, frame_rate=sr, channels=channels)

            # Export the final merged track to a single .ogg file
            with fs.open(output_file_path, 'wb') as output_file:
                bytes_io = io.BytesIO()
                segment.export(bytes_io, format='ogg', codec='libopus')  # type: ignore
                output_file.write(bytes_io.getvalue())  # type: ignore


def augment_pitch_and_tempo(fs, file_paths: List[Tuple[str, str]]) -> None:
    transform = Compose(
        transforms=[
            PitchShift(p=1 if AUGMENT_PITCH is True else 0, min_semitones=-2, max_semitones=2),
            TimeStretch(p=1, leave_length_unchanged=False),
            Gain(p=1, min_gain_db=-6, max_gain_db=-5)
        ],
        p=1,
    )

    augment_files(fs, file_paths, transform)


def augment(params: Tuple[S3FileSystem, str, str, str]) -> None:
    
    fs, file_path, source_directory, output_directory = params

    try:
        file_dir = os.path.dirname(file_path)
        stem_file_path = os.path.join(file_dir, 'stem.ogg')
        file_dir = os.path.dirname(file_path)
        relative_path = os.path.relpath(file_dir, source_directory)
        output_file_path = os.path.join(output_directory, relative_path + '-original')

        full_track_output_file_path = os.path.join(output_file_path, os.path.basename(file_path))
        
        if not fs.exists(full_track_output_file_path):
            fs.makedirs(os.path.dirname(full_track_output_file_path), exist_ok=True)
            if fs.exists(file_path):
                fs.copy(file_path, full_track_output_file_path)

        stem_output_file_path = os.path.join(output_file_path, os.path.basename(stem_file_path))

        if not fs.exists(stem_output_file_path):
            fs.makedirs(os.path.dirname(full_track_output_file_path), exist_ok=True)
            if fs.exists(stem_file_path):
                fs.copy(stem_file_path, stem_output_file_path)

        for i in range(AUGMENTATIONS_COUNT):
            file_dir = os.path.dirname(file_path)
            relative_path = os.path.relpath(file_dir, source_directory)
            output_file_path = os.path.join(output_directory, relative_path + f'-augmented{i}')
            full_track_output_file_path = os.path.join(output_file_path, os.path.basename(file_path))
            stem_output_file_path = os.path.join(output_file_path, os.path.basename(stem_file_path))

            if not fs.exists(full_track_output_file_path) or not fs.exists(stem_output_file_path):
                fs.makedirs(output_file_path, exist_ok=True)
                augment_pitch_and_tempo(
                    fs,
                    [
                        (file_path, full_track_output_file_path),
                        (stem_file_path, stem_output_file_path)
                    ]
                )

    except Exception as e:
        print(f'Error augmenting file {file_path}: {e}')
        print(traceback.format_exc())


def augment_all(source_directory: str, output_directory: str):

    fs = S3FileSystem()
    files = get_full_track_files(fs, source_directory)

    client = cast(
        Client,
        get_client(
            RUN_LOCALLY,
        ),
    )
    
    params_list: List[Tuple[S3FileSystem, str, str, str]] = [(fs, file_path, source_directory, output_directory) for file_path in files]

    print('Augmenting audio tracks')
    futures = client.map(augment, params_list)
    progress(futures)

    return output_directory


if __name__ == '__main__':
    augment_all(SOURCE_FILES_DIR, OUTPUT_FILES_DIR)