#!/bin/bash
# create_python_env.sh
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install --upgrade pip setuptools
pip install jupyter ipykernel torch numpy pandas