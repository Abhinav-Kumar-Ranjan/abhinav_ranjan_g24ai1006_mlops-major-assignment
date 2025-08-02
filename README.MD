# California Housing Linear Regression MLOps Pipeline

This repository presents a professional MLOps pipeline for Linear Regression using the California Housing dataset from `scikit-learn`. The solution encompasses model training, evaluation, quantization, containerization with Docker, and CI/CD automation via GitHub Actions.


## Project Structure

project-root/
├── .github/
│   └── workflows/
│       └── ci.yml                # CI/CD workflow
├── src/
│   ├── __init__.py
│   ├── train.py                  # Model training script
│   ├── quantize.py               # Model quantization script
│   ├── predict.py                # Inference script for Docker
│   └── utils.py                  # Utility functions
├── Dockerfile                    # Docker configuration
├── model.joblib
├── quant_params.joblib
├── unquant_params.joblib
├── .gitignore                    # Git ignore file
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation


## Features

- Model Training: Linear Regression on the California Housing dataset
- Testing: Comprehensive unit tests with `pytest`
- Quantization: Manual 8-bit quantization of model parameters
- Containerization: Docker-based deployment
- CI/CD: Automated workflows using GitHub Actions



## Requirements

- Python 3.8 or higher
- scikit-learn
- numpy
- joblib
- pytest

## Quick Start

### Installation

bash
git clone <your-repo-url>
cd <your-repo-name>
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Unix/Mac:
source venv/bin/activate
pip install -r requirements.txt


### Model Training

bash
python src/train.py

- Loads the California Housing dataset
- Trains a Linear Regression model
- Outputs R² score and loss metrics
- Saves the trained model as `model.joblib`



### Model Quantization

bash
python src/quantize.py

- Loads the trained model
- Extracts and saves original parameters as `unquant_params.joblib`
- Quantizes parameters to 8-bit unsigned integers
- Saves quantized parameters as `quant_params.joblib`
- Performs inference with de-quantized weights


### Running Tests

bash
pytest tests/ -v


## Docker Deployment

bash
docker build -t mlops-pipeline
docker run mlops-pipeline


### DockerHub Image

Pull the latest image from [DockerHub](https://https://hub.docker.com/repository/docker/abhinavkumarranjan/mlops-pipeline/tags):

bash
docker pull abhinavkumarranjan/mlops-pipeline
docker run --rm abhinavkumarranjan/mlops-pipeline


## CI/CD Pipeline

The GitHub Actions workflow (`.github/workflows/ci.yml`) includes:

1. test_suite: Executes all unit tests
2. train_and_quantize: Trains and quantizes the model
3. build_and_test_container:** Builds and tests the Docker image

The pipeline is triggered on every push to the main branch.


## Model Performance

| Metric        | Value                         |
| ------------- | ----------------------------- |
| Model Type    | Linear Regression             |
| Dataset       | California Housing (sklearn)  |
| R² Score      | ~0.60 (typical)               |
| Features      | 8 numeric features            |
| Target        | Median house value            |


## Quantization Details

- Precision: 8-bit unsigned integers (0-255)
- Parameters Quantized: Model coefficients and intercept
- Method: Manual quantization with scale and zero-point
- Storage: Separate files for original and quantized parameters


## Docker Configuration

- Python 3.9 slim base image
- Installs all required dependencies
- Includes trained model and inference script
- Runs `predict.py` on container startup


## Testing Strategy

Unit tests cover:

- Dataset loading
- Model creation and validation
- Training process
- R² score threshold validation
- Parameter extraction and
