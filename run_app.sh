#!/usr/bin/env bash
conda create -n fake-reviews-detector python=3.8 -y
conda activate fake-reviews-detector
pip install -e .
python -m fake_reviews_detector.main
