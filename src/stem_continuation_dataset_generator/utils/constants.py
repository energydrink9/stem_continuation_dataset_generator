from typing import List

RANDOM_SEED = 42

# Constants
VOCAB_SIZE = 2048
PAD_TOKEN_ID = VOCAB_SIZE - 3  # 2045
SOS_TOKEN_ID = VOCAB_SIZE - 2  # 2046
EOS_TOKEN_ID = VOCAB_SIZE - 1  # 2047

CLEARML_PROJECT_NAME = 'stem_continuation_dataset_generator'
CLEARML_DATASET_NAME = 'stem_continuation_dataset'
CLEARML_DATASET_TAGS = ['500', 'stem-guitar']
CLEARML_DATASET_VERSION = '1.0.0'


def get_random_seed() -> int:
    return RANDOM_SEED


def get_start_of_sequence_token_id() -> int:
    return SOS_TOKEN_ID


def get_end_of_sequence_token_id() -> int:
    return EOS_TOKEN_ID


def get_pad_token_id() -> int:
    return PAD_TOKEN_ID


def get_special_tokens() -> List[int]:
    return [get_start_of_sequence_token_id(), get_end_of_sequence_token_id(), get_pad_token_id()]


def get_clearml_project_name() -> str:
    return CLEARML_PROJECT_NAME


def get_clearml_dataset_name() -> str:
    return CLEARML_DATASET_NAME


def get_clearml_dataset_tags() -> List[str]:
    return CLEARML_DATASET_TAGS


def get_clearml_dataset_version() -> str:
    return CLEARML_DATASET_VERSION
