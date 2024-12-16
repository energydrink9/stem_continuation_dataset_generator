import glob
import os
import shlex
import tempfile
from typing import List, Tuple, cast

from distributed import Client, progress
import demucs.separate
from s3fs.core import S3FileSystem

from stem_continuation_dataset_generator.cluster import get_client
from stem_continuation_dataset_generator.constants import get_original_files_path, get_whole_tracks_files_path
from stem_continuation_dataset_generator.steps.convert_to_ogg import convert_to_ogg
from stem_continuation_dataset_generator.utils.utils import is_mostly_silent


RUN_LOCALLY = False
PERCENTAGE_OF_NON_SILENT_AUDIO_FILE = 0.25
EXCLUDED_STEMS = ['piano', 'vocals']  # Piano and vocals stems produced by Demucs are low quality 


def get_whole_track_files(fs: S3FileSystem, dir: str) -> List[str]:
    return cast(List[str], fs.glob(os.path.join(dir, '**/*.mp3')))


def stem_file(output_directory: str, file_path: str) -> tuple[str, list[tuple[str, str]]]:
    """
    Separates an audio file into its individual tracks using the Demucs model.

    This function takes an audio file as input, separates it into its individual tracks using the Demucs model,
    and returns the directory where the separated tracks are stored along with a list of tuples containing the
    instrument name of each track and its corresponding file path.

    Args:
        filename (str): The path to the audio file to be separated.

    Returns:
        tuple[str, list[tuple[str, str]]]: A tuple containing the directory path where the separated tracks are stored,
        and a list of tuples where each tuple contains the instrument name of a track and its file path.
    """
    demucs.separate.main(shlex.split(f'-n htdemucs_6s --clip-mode clamp --out "{output_directory}" "{file_path}"'))
    return (output_directory, [(os.path.splitext(os.path.basename(filename))[0], filename) for filename in glob.glob(os.path.join(output_directory, '**', '*.wav'), recursive=True)])


def stem(params: Tuple[S3FileSystem, str, str, str, str]):
    fs, file_path, artist, source_directory, base_output_directory = params

    basename = os.path.basename(file_path)
    song_name = basename.replace('.mp3', '')
    output_directory = os.path.join(base_output_directory, artist, song_name)

    with tempfile.TemporaryDirectory() as local_directory:
        local_path = os.path.join(local_directory, basename)
        fs.download(file_path, local_path)
        stem_file(local_directory, local_path)
        os.remove(local_path)
        convert_to_ogg(local_directory)
        ogg_files = glob.glob(os.path.join(local_directory, '**/*.ogg'), recursive=True)
        for ogg_file in ogg_files:
            if os.path.basename(ogg_file).split('.')[0] not in EXCLUDED_STEMS:
                with open(ogg_file, 'rb') as file:
                    if not is_mostly_silent(file, PERCENTAGE_OF_NON_SILENT_AUDIO_FILE):
                        print(ogg_file)
                        fs.upload(ogg_file, os.path.join(output_directory, os.path.basename(ogg_file)))


def stem_all(source_directory: str, output_directory: str):

    fs = S3FileSystem()
    files = get_whole_track_files(fs, source_directory)
    files_with_artist = [(file_path, os.path.dirname(file_path).split(os.path.sep)[-1]) for file_path in files]

    client = cast(
        Client,
        get_client(
            RUN_LOCALLY,
        ),
    )
    
    params_list: List[Tuple[S3FileSystem, str, str, str, str]] = [(fs, file_path, artist, source_directory, output_directory) for file_path, artist in files_with_artist]

    print('Stemming audio tracks')
    futures = client.map(stem, params_list, retries=2)
    progress(futures)

    return output_directory


if __name__ == '__main__':
    stem_all(get_whole_tracks_files_path(), get_original_files_path())