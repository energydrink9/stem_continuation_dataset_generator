from stem_continuation_dataset_generator.constants import DATASET_TAGS, get_augmented_files_path, get_distorted_files_path, get_encoded_files_path, get_merged_files_path, get_original_files_path, get_split_files_path
from stem_continuation_dataset_generator.steps.augment import augment_all
from stem_continuation_dataset_generator.steps.convert_to_ogg import convert_to_ogg
from stem_continuation_dataset_generator.steps.encode import encode_all
from stem_continuation_dataset_generator.steps.merge import assort_and_merge_all
from stem_continuation_dataset_generator.steps.split import split_all
from stem_continuation_dataset_generator.steps.uncompress import uncompress_files
from stem_continuation_dataset_generator.steps.upload import upload
from stem_continuation_dataset_generator.steps.distort import distort_all


def uncompress_step(source_dir: str):
    output_dir = uncompress_files(source_dir)
    return output_dir


def convert_to_ogg_step(source_dir):
    output_dir = convert_to_ogg(source_dir)
    return output_dir


def dataset_preparation_pipeline(source_dir: str):

    print(f'Preparing dataset. Source dir: {source_dir}')

    uncompressed_dir = uncompress_step(source_dir)
    converted_to_ogg_dir = convert_to_ogg_step(uncompressed_dir)

    print(f'Succesfully prepared dataset in directory {converted_to_ogg_dir}')


def dataset_creation_pipeline(stem_name: str):
    
    tags = DATASET_TAGS + [f'stem-{stem_name}']

    assort_and_merge_all(get_original_files_path(), get_merged_files_path(stem_name), stem_name)
    augment_all(get_merged_files_path(stem_name), get_augmented_files_path(stem_name))
    distort_all(get_augmented_files_path(stem_name), get_distorted_files_path(stem_name))
    encode_all(get_distorted_files_path(stem_name), get_encoded_files_path(stem_name))
    split_all(get_encoded_files_path(stem_name), get_split_files_path(stem_name))
    upload(get_split_files_path(stem_name), tags)
