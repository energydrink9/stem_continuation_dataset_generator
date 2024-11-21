#!/bin/sh

cd /app/stem_continuation_dataset_generator

echo 'api { credentials {"access_key": "<KEY>", "secret_key": "<KEY>"} }' > ~/clearml.conf

# Install Poetry
pip3 install poetry || true

#poetry config virtualenvs.create false

apt-get update || true
apt-get install -y build-essential || true

echo "Installing dependencies"
poetry install --no-interaction --no-ansi || true

echo "Running script"
export ENV='prod'
poetry run --no-interaction --no-ansi python -m stem_continuation_dataset_generator.process