#!/usr/bin/env bash
ENV_NAME=fake-reviews-detector
ENV_FILE=environment.yml
MAIN_MODULE=fake_reviews_detector.main

if conda env list | grep -q "^${ENV_NAME}␈"; then
  echo "Conda environment '${ENV_NAME}' exists. Skipping creation."
else
  echo "Creating conda environment '${ENV_NAME}'..."
  conda env create -f ${ENV_FILE}
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${ENV_NAME}

if ! python -c "import ${MAIN_MODULE.split('.')[0]}" &> /dev/null; then
  echo "Installing package in editable mode..."
  pip install -e .
fi

python -m ${MAIN_MODULE}
