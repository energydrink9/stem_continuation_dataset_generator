from clearml import PipelineDecorator

from stem_continuation_dataset_generator.dataset import get_remote_dataset_by_tag
from stem_continuation_dataset_generator.pipeline import AUGMENTED_FILES_DIR, DISTORTED_FILES_DIR, ENCODED_FILES_DIR, MERGED_FILES_DIR, SPLIT_FILES_DIR, dataset_creation_pipeline
from stem_continuation_dataset_generator.constants import STEM_NAME


if __name__ == '__main__':
    # parser = argparse.ArgumentParser("Create a dataset from an already pre-processed dataset")
    # parser.add_argument("dataset_id", help="Id of the pre-processed dataset.", type=str)
    # args = parser.parse_args()
   
    source_dir = get_remote_dataset_by_tag('original')

    PipelineDecorator.run_locally()
    dataset_creation_pipeline(
        STEM_NAME,
        merged_dir=MERGED_FILES_DIR,
        augmented_dir=AUGMENTED_FILES_DIR,
        distorted_dir=DISTORTED_FILES_DIR,
        encoded_dir=ENCODED_FILES_DIR,
        split_dir=SPLIT_FILES_DIR,
    )
    print('Pipeline completed')
