@echo off
conda env create -f environment.yml --quiet || conda env update -f environment.yml --quiet
call conda activate fake-reviews-detector
pip install -e .
python -m fake_reviews_detector.main
