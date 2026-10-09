# SmartFood AI — Intelligent Food Classification & Nutrition Insights

**SmartFood AI** is a machine learning application that classifies food categories using nutritional attributes and provides model confidence, interpretable feature insights, and general nutrition awareness recommendations.

The project combines data preprocessing, imbalanced-class handling, gradient-boosted classification, model evaluation, and interactive prediction into a single workflow. It demonstrates how structured nutritional data can be transformed into useful machine learning predictions.

## Overview

Food classification based on nutritional composition can support food-data organization, nutrition analysis, and diet-related applications. SmartFood AI approaches this problem as a supervised machine learning task, using nutritional attributes to predict a food item's category.

The system is designed around three capabilities:

* **Food classification:** Predict a food category from nutritional values.
* **Prediction confidence:** Present the model's confidence score alongside its predicted class.
* **Explainability and insights:** Examine influential nutritional features and provide general health-awareness tips.

SmartFood AI is an educational machine learning project, not a medical diagnostic system or a replacement for professional dietary advice.

## Key Features

* Nutritional-data preprocessing and validation
* Feature-based food category classification
* SMOTE-based handling of class imbalance
* XGBoost classification model
* Evaluation using accuracy, precision, recall, and F1-score
* Feature-importance analysis for model interpretability
* Interactive predictions from nutritional inputs
* Confidence-score presentation
* General nutrition-awareness recommendations
* Modular workflow covering data preparation, training, evaluation, and inference

## Machine Learning Workflow

The application follows a structured supervised learning pipeline:

1. **Data preparation:** Load nutritional records and inspect the dataset.
2. **Data cleaning:** Handle missing values, inconsistent records, and unsuitable inputs.
3. **Feature preparation:** Select nutritional attributes and encode the target food category.
4. **Class imbalance handling:** Apply SMOTE to the training data where appropriate.
5. **Model training:** Train an XGBoost classifier on the prepared dataset.
6. **Model evaluation:** Assess classification performance using multiple metrics.
7. **Feature analysis:** Examine which nutritional attributes contribute to the trained model's predictions.
8. **Inference:** Accept nutritional values and generate a food category prediction.
9. **Recommendation:** Display a general nutrition-awareness tip based on the available information and implemented recommendation rules.

### Pipeline

```text
Nutritional Dataset
        |
        v
Data Validation & Cleaning
        |
        v
Feature Preparation
        |
        v
Train / Validation / Test Split
        |
        v
SMOTE on Training Data
        |
        v
XGBoost Model Training
        |
        v
Model Evaluation
        |
        v
Feature Importance Analysis
        |
        v
Interactive Prediction
        |
        v
Category + Confidence + Health Tip
```

**Data leakage prevention:** SMOTE should be applied only to the training partition, after splitting the dataset. Validation and test sets should retain their original class distributions so that evaluation better reflects real-world performance.

## Dataset & Features

The model uses structured nutritional attributes as input features.

| Feature       | Description                |
| ------------- | -------------------------- |
| Calories      | Energy content of the food |
| Fat           | Fat content                |
| Sugar         | Sugar content              |
| Protein       | Protein content            |
| Carbohydrates | Carbohydrate content       |
| Fiber         | Dietary fiber content      |

**Target variable:** Food Category

The target represents the category that the classifier learns to predict from the nutritional features.

### Data Preparation

The preprocessing workflow includes:

* Checking and handling missing values
* Cleaning inconsistent or invalid records
* Selecting relevant numerical features
* Encoding category labels for supervised learning
* Splitting the dataset before training-time resampling
* Applying SMOTE to the training partition when the class distribution warrants it

The model's reliability depends on the quality, diversity, and representativeness of the underlying dataset.

## Model Architecture

### XGBoost Classifier

SmartFood AI uses **XGBoost**, a gradient-boosted decision-tree framework suitable for structured tabular data.

The classifier builds an ensemble of decision trees sequentially, with later trees learning to improve the ensemble's predictions.

Reasons for selecting XGBoost include:

* Ability to model nonlinear relationships between nutritional attributes
* Support for multiclass classification
* Regularization options to help control overfitting
* Compatibility with structured numerical features
* Feature-importance analysis for understanding model behavior

### Class Imbalance Handling with SMOTE

Imbalanced datasets can cause a classifier to favor frequently occurring categories.

SMOTE (Synthetic Minority Over-sampling Technique) creates synthetic examples for minority classes using neighboring training examples. It can help improve minority-class learning when applied appropriately.

SMOTE does not guarantee improved performance. Its effect should be assessed using per-class precision, recall, F1-score, and the confusion matrix.

### Prediction Confidence

The prediction interface can display the model's predicted class and its associated probability when probability estimates are enabled.

A confidence score represents the model's estimated support for a prediction. It is not a guarantee of correctness, and predicted probabilities should not automatically be interpreted as calibrated real-world probabilities.

## Model Evaluation

The classifier is evaluated using multiple complementary metrics.

| Metric           | Purpose                                                        |
| ---------------- | -------------------------------------------------------------- |
| Accuracy         | Measures the proportion of correct predictions                 |
| Precision        | Measures how often predictions for a category are correct      |
| Recall           | Measures how many actual examples of a category are identified |
| F1-score         | Balances precision and recall                                  |
| Confusion matrix | Shows correct classifications and errors between categories    |

For imbalanced food categories, macro-averaged precision, recall, and F1-score can provide a more informative view of performance than accuracy alone.

Model evaluation should use held-out data that was not used for training or resampling.

## Explainability & Nutrition Insights

SmartFood AI includes feature-importance analysis to investigate which nutritional attributes contribute to the trained classifier's decisions.

The nutritional features examined include:

* Calories
* Fat
* Sugar
* Protein
* Carbohydrates
* Fiber

Feature importance describes the model's learned behavior. It does not establish that a nutrient causes a food to belong to a particular category, nor does it explain every individual prediction.

The application also provides general nutrition-awareness tips. Such tips should be based on clearly defined rules or a documented recommendation source rather than being presented as medical conclusions.

## Interactive Prediction

The prediction workflow accepts nutritional values and returns the model's output.

**Input**

* Calories
* Fat
* Sugar
* Protein
* Carbohydrates
* Fiber

**Output**

* Predicted food category
* Model confidence or class probability, where supported
* Relevant feature-importance insights
* General nutrition-awareness tip

Before inference, inputs should be validated for missing values, numeric format, and acceptable ranges. The same feature order and preprocessing conventions used during training must be preserved during prediction.

## Technology Stack

| Component                | Technology                                                       |
| ------------------------ | ---------------------------------------------------------------- |
| Programming language     | Python                                                           |
| Machine learning model   | XGBoost                                                          |
| Data processing          | pandas, NumPy                                                    |
| Class imbalance handling | imbalanced-learn / SMOTE                                         |
| Model evaluation         | scikit-learn                                                     |
| Visualization            | Matplotlib / Seaborn, where implemented                          |
| Interactive interface    | Python-based interface, according to the implemented application |

The core machine learning workflow is based on Python. Additional interface and visualization dependencies should match the actual implementation.

## Project Structure

The project is organized conceptually into the following responsibilities:

```text
SmartFood-AI/
├── data/                  # Nutritional dataset
├── notebooks/             # Data exploration and experiments
├── src/                   # Preprocessing and model logic
├── models/                # Saved trained model artifacts
├── tests/                 # Input validation and model tests
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
└── .gitignore             # Excluded files and artifacts
```

A repository's directory names should reflect the actual files in the project. Dataset files and trained model artifacts should not be committed when they contain restricted data or unnecessarily large files.

## Installation

### Prerequisites

* Python 3.10 or a compatible Python version
* pip
* Git

### Setup

Clone the repository:

```bash
git clone https://github.com/venkidev21/SmartFood-AI.git
cd SmartFood-AI
```

Create and activate a virtual environment.

**Windows**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux / macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The repository should include a `requirements.txt` containing the packages and compatible versions required by its implementation.

## Running the Application

The exact startup command depends on the interface implemented in the repository.

For a notebook-based workflow, open the project notebook in Jupyter and execute the data preparation, training, evaluation, and prediction steps in order.

For a standalone Python application, run the application's actual entry-point script after installing dependencies and configuring the dataset path.

The project should preserve the trained model and its preprocessing configuration consistently between training and inference. A saved model should only be loaded from a trusted source.

## Testing & Validation

Recommended validation checks include:

* Verify that all required nutritional columns are present.
* Reject malformed or nonnumeric prediction inputs.
* Confirm that the training and test partitions do not overlap.
* Ensure that SMOTE is applied only to training data.
* Check that the predicted class belongs to the supported label set.
* Verify that confidence values are valid probabilities when probability output is enabled.
* Evaluate performance using held-out data.
* Test inference using representative valid and invalid inputs.

## Applications

Potential application areas include:

* Food catalog categorization
* Nutrition-information platforms
* Fitness and wellness applications
* Food-data management systems
* Educational machine learning demonstrations
* Exploratory nutritional analysis

These are potential use cases; production deployment would require additional validation, data-quality controls, and application-specific testing.

## Limitations

* Classification quality depends on dataset coverage and label accuracy.
* Nutritional similarity between categories can make some predictions ambiguous.
* SMOTE may introduce synthetic samples that do not fully represent real nutritional distributions.
* Feature importance does not imply causation.
* Confidence scores can be overconfident if probabilities are not calibrated.
* General nutrition tips cannot replace individualized medical or dietary guidance.
* The classifier predicts categories from nutritional attributes; it does not independently verify a food's identity, ingredients, allergens, or preparation method.

## Future Improvements

* Expand the dataset to include more representative food categories.
* Add class-wise evaluation reports and a confusion matrix.
* Evaluate probability calibration and prediction reliability.
* Compare XGBoost against Random Forest and other suitable baseline models.
* Add automated input validation and reproducible training configurations.
* Introduce model versioning and experiment tracking.
* Build a documented REST API for application integration.
* Containerize the inference service for consistent deployment.
* Add automated tests and continuous integration.
* Improve nutrition guidance using documented and reliable nutritional references.

## Responsible Use

SmartFood AI is intended for educational, research, and exploratory food-data classification. Its predictions and recommendations should not be used as medical diagnoses or as the sole basis for dietary decisions.

## Author

**Venkatesan E**

B.Tech — Information Technology

* GitHub: https://github.com/venkidev21
* LinkedIn: https://www.linkedin.com/in/venkatesaneswaran/

