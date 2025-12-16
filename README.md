# Machine Learning Projects

This repository contains a collection of Machine Learning implementations developed using Python and Jupyter Notebook.
The project focuses on classical machine learning algorithms, feature engineering techniques, and basic neural network models.

The main goal of this repository is educational: to demonstrate end-to-end machine learning workflows, from data preprocessing to model evaluation.

---

## Repository Structure

Machine-learning/
├── classification_*.ipynb
├── regression_*.ipynb
├── feature_*.ipynb
├── neural_perceptron_adaline.ipynb
├── neural_mlp_mnist_classifier/
│   └── data/ (dataset directory – ignored in git)
├── requirements.txt
└── README.md

---

## Topics Covered

### Classification

* Logistic Regression
* k-Nearest Neighbors (KNN)
* Support Vector Machines (SVM)
* Random Forest
* AdaBoost
* XGBoost
* Naive Bayes (including NLP examples)

### Regression

* Linear Regression
* Polynomial Regression
* Feature transformations

### Feature Engineering

* Data cleaning
* Feature preprocessing
* Feature extraction techniques

### Neural Models

* Perceptron
* Adaline
* Multi-Layer Perceptron (MLP)

---

## Installation

It is recommended to use a virtual environment.

Install dependencies using:

pip install -r requirements.txt

---

## Usage

1. Open Jupyter Notebook or Jupyter Lab
2. Select any `.ipynb` file
3. Run the cells step by step:

   * Load and explore the dataset
   * Preprocess features
   * Train the model
   * Evaluate results

You may replace datasets or tune hyperparameters to perform further experiments.

---

## Model Evaluation

Depending on the notebook, models are evaluated using:

* Accuracy
* Confusion Matrix
* Precision and Recall
* Regression error metrics
* Data visualizations

---

## Datasets

Datasets are not included in this repository due to size limitations.

For experiments that require datasets (e.g., MNIST), download the dataset separately and place it inside the corresponding `data` directory.

---

## Contributing

Contributions are welcome.

1. Fork the repository
2. Create a new branch
3. Add your changes or new notebooks
4. Submit a Pull Request

Please keep notebooks clean and well-documented.

---

## License

This project is open-source.
You may add a license file if needed (e.g., MIT License).

---

## Notes

* Some notebooks require external datasets.
* Ensure dataset paths are correctly set before running the notebooks.
