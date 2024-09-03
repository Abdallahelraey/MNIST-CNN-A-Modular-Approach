# MNIST-CNN-A-modular-Approach

## Overview

MNIST-CNN-A-modular-Approach is a project that trains a Convolutional Neural Network (CNN) on the MNIST dataset and deploys it using FastAPI. The project includes data preprocessing, model training, evaluation, and deployment. It also provides a web interface for uploading images and getting predictions.

## Project Structure

```markdown
MNIST-CNN-FastAPI/
├── config/
│ └── config.yaml
├── data/
│ ├── init.py
│ ├── data_preprocessing.py
│ └── dataset_operations.py
├── models/
│ ├── init.py
│ └── model_operations.py
├── notebooks/
│ └── POC.ipynb
├── routes/
│ ├── init.py
│ └── predict.py
├── tests/
│ ├── test_api.py
│ ├── test_data.py
│ └── test_model.py
├── viewer/
│ └── mesop_view.py
├── .gitignore
├── LICENSE
├── main.py
├── README.md
├── requirements.txt
└── setup.py
```
