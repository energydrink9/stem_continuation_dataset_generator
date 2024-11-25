![example workflow](https://github.com/energydrink9/stem_continuation_dataset_generator/actions/workflows/python-app.yml/badge.svg)


This application implements a pipeline that can be used to create audio datasets for the generation of stem continuations of music audio files. The code uses [Dask](https://www.dask.org/) in order to scale the dataset processing on a cluster of virtual machines in the cloud. The application is configured to run on AWS EC2 and to use S3 as storage. The audio files are encoded using Meta's [Encodec](https://github.com/facebookresearch/encodec) into a discrete, compressed, tokenized representation.

The dataset generation pipeline is comprised of several steps:
- **Uncompress**. The application expects to find the stem files for a single music file (in .wav format) in a compressed zip archive. Each stem should have a predefined name in order to be identified as a guitar, bass, drum, etc.
- **Convert to ogg**. Conversion of wav files to the Ogg Opus audio format.
- **Merge**. Several different assortments of stems are generated.
- **Augment**. Audio augmentations are used to increase the dataset size, by changing pitch and tempo.
- **Distort**. Several audio distortions are applied to simulate real conditions.
- **Encode**. The audio files are encoded using **Encodec** and chunked.

### Initial setup

In order to start using the application, make sure to run the following commands that will install the package and the necessary dependencies.

```
pip install poetry
poetry install
```

#### Environment Variables

If you want to run training or dataset generation (for inference it's not needed), make sure the following environment variables are set before running the application (replace the <> tokens with the actual secrets):

```sh
export CLEARML_API_ACCESS_KEY=<YOUR_CLEARML_API_ACCESS_KEY_HERE>
export CLEARML_API_SECRET_KEY=<YOUR_CLEARML_API_SECRET_KEY_HERE>
```

Also this one in case you are running on Apple Silicon:
```sh
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

### Install

```sh
poetry install
```


### Dataset generation

Source directory should contain a folder for each artist. The artist folders should contain one compressed file for each song. The compressed file should contain one .wav file for each stem. In order to identify the different stems, the stem files should have names containing the values present at `src/stem_continuation_dataset_generator/steps/merge.py` in the `STEM_NAMES` variable.

#### Dataset preparation

To prepare the dataset, use the following command, replacing the string <SOURCE-DIRECTORY> with the path to the directory containing the source files:

```sh
poetry run python -m stem_continuation_dataset_generator.prepare <SOURCE-DIRECTORY>
```

The pipeline will uncompress the song archives and convert all the files to OGG format. The original files will be deleted and the prepared dataset will be uploaded to the ClearML repository.

#### Dataset creation

To process the files obtained at the preparation step, use the following command, replacing the string <STEM_NAME> with the name of the stem (musical instrument) for which you want to generate the dataset (i.e. **guitar** or **drum**):

```sh
poetry run python -m stem_continuation_dataset_generator.process <STEM_NAME>
```

The pipeline will augment, distort, encode and split the samples into chunks, generating three different folders for the train, validation and test sets. The result will be uploaded to ClearML into 3 different datasets.
