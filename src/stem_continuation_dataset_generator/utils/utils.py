import io
from clearml import Dataset
import numpy as np
import librosa
from typing import Union

from stem_continuation_dataset_generator.constants import CLEARML_DATASET_NAME
from stem_continuation_dataset_generator.utils.constants import get_clearml_project_name


def upload_dataset(path: str, version: str, tags: list[str] = [], dataset_set=None):
    print(f'Creating dataset (set: {dataset_set}, tags: {tags})')
    tags = [f'{dataset_set}-set'] + tags if dataset_set is not None else tags
    dataset = Dataset.create(
        dataset_project=get_clearml_project_name(), 
        dataset_name=CLEARML_DATASET_NAME,
        dataset_version=version,
        dataset_tags=tags,
    )
    print('Adding files')
    dataset.add_files(path=path)
    print('Uploading')
    dataset.upload(show_progress=True, preview=False)
    print('Finalizing')
    dataset.finalize()


def clamp_audio_data(audio_data: np.ndarray) -> np.ndarray:
    return np.clip(audio_data, -1, 1)


def convert_audio_to_int_16(audio_data: np.ndarray) -> np.ndarray:
    max_16bit = 2**15 - 1
    assert audio_data.max() <= 1 and audio_data.min() >= -1, f'Overflow error during audio conversion: {audio_data.max()} vs {1}'
    raw_data = audio_data * max_16bit
    assert raw_data.max() <= max_16bit, f'Overflow error during audio conversion: {raw_data.max()} vs {max_16bit}'
    return raw_data.astype(np.int16)


def convert_audio_to_float_32(audio_data: np.ndarray) -> np.ndarray:
    max_32bit = 2**31 - 1
    assert audio_data.max() <= max_32bit, f'Overflow error during audio conversion: {audio_data.max()} vs {max_32bit}'
    raw_data = audio_data / max_32bit
    return raw_data.astype(np.float32)


def is_mostly_silent(file: Union[io.TextIOWrapper, io.BufferedReader], percentage_non_silent: float) -> bool:        
    audio, sr = librosa.load(file)  # type: ignore
    no_of_samples = audio.shape[-1]
    splits = librosa.effects.split(audio, top_db=60)
    non_silent_samples = sum([end - start for (start, end) in splits])
    return non_silent_samples / no_of_samples < percentage_non_silent