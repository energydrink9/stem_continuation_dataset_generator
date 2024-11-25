import argparse
from stem_continuation_dataset_generator.dataset import get_remote_dataset_by_tag
from stem_continuation_dataset_generator.pipeline import dataset_creation_pipeline


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Create a dataset from an already pre-processed dataset")
    parser.add_argument("stem_name", help="Name of the stem (musical instrument) to process", type=str)
    args = parser.parse_args()
   
    source_dir = get_remote_dataset_by_tag('original')

    dataset_creation_pipeline(args.stem_name)
    print('Pipeline completed')
