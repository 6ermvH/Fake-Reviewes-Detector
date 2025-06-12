#!/usr/bin/env bash
conda env create -f environment.yml --quiet || conda env update -f environment.yml --quiet
source $(conda info --base)/etc/profile.d/conda.sh
conda activate fake-reviews-detector
pip install -e .
python -m fake_reviews_detector.main
