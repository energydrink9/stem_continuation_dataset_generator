
import fsspec.implementations.local
from stem_continuation_dataset_generator.steps.distort import distort_file

TEST_FILE_PATH_ALL = 'resources/all.ogg'


def check_augment_pitch_and_tempo() -> None:

    fs = fsspec.implementations.local.LocalFileSystem()
    distort_file(fs, TEST_FILE_PATH_ALL, f'{TEST_FILE_PATH_ALL}-distorted.ogg')

    print('Distorted file generated successfully')


if __name__ == '__main__':
    check_augment_pitch_and_tempo()