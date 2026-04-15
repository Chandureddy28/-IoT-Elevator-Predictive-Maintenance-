#!/bin/bash
set -e
python generate_data.py
python train_model.py