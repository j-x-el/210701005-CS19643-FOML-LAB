# 210701005-CS19643-FOML-LAB

# Predicting Autism Spectrum Disorder Using Machine Learning

## Abstract
This project explores the application of machine learning algorithms to predict Autism Spectrum Disorder (ASD). By analyzing demographic, behavioral, and clinical data, we aim to build models that can assist in early detection of ASD. The project utilizes various machine learning techniques, including Support Vector Machines (SVM), Random Forests, and Neural Networks, to identify patterns indicative of ASD. The ultimate goal is to provide a supplementary tool for clinicians to enhance the accuracy and efficiency of ASD diagnosis.

## Code Description

### Directory Structure
- `data/`: Contains the dataset files.
- `scripts/`: Contains the Python scripts for data processing, training, and evaluation.
- `models/`: Contains the saved machine learning models.
- `notebooks/`: Jupyter notebooks for exploratory data analysis and model development.
- `results/`: Stores results and evaluation metrics.

### Key Scripts
1. `data_collection.py`: Script for collecting and aggregating data from various sources.
2. `data_preprocessing.py`: Script for cleaning and preprocessing the dataset.
3. `feature_selection.py`: Script for identifying key features for model training.
4. `train_model.py`: Script for training machine learning models.
5. `evaluate_model.py`: Script for evaluating the performance of trained models.
6. `deploy_model.py`: Script for deploying the best-performing model.

### Installation
1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/ASD-Prediction.git
    ```
2. Navigate to the project directory:
    ```sh
    cd ASD-Prediction
    ```
3. Install the required libraries:
    ```sh
    pip install -r requirements.txt
    ```

### Usage
1. Run the data collection script:
    ```sh
    python scripts/data_collection.py
    ```
2. Preprocess the data:
    ```sh
    python scripts/data_preprocessing.py
    ```
3. Train the model:
    ```sh
    python scripts/train_model.py
    ```
4. Evaluate the model:
    ```sh
    python scripts/evaluate_model.py
    ```
5. Deploy the model:
    ```sh
    python scripts/deploy_model.py
    ```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.