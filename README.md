# Robust Hybrid Model for Credit Card Fraud Detection.
<p align="justify">This project demonstrates an automated credit card fraud detection system using Machine Learning techniques. Detecting fraudulent transactions is critical for financial institutions to prevent monetary loss and protect customers. In this project, transaction data is preprocessed, engineered, and analyzed using various classification models to identify potential fraud. This project was developed as part of my data science portfolio, focusing on real-time fraud detection in financial systems.</p>


## Publication
<p align="justify">This project has been published in the conference <b>IDC-IoT 2024 (Intelligent Data Communication Technologies and Internet of Things)</b> under the title: <i>"Robust Hybrid Machine Learning Model for Financial Fraud Detection in Credit Card Transactions"</i>. The paper presents the methodology, experimental results, and insights on hybrid machine learning approaches for accurate fraud detection.</p>

## Citation
<p align="justify">If you use this project in your research or work, please cite the paper as follows: D. Jahnavi, M. A, S. Pulata, S. Sami, B. Vakamullu and B. Mohan G, "Robust Hybrid Machine Learning Model for Financial Fraud Detection in Credit Card Transactions," 2024 2nd International Conference on Intelligent Data Communication Technologies and Internet of Things (IDCIoT), Bengaluru, India, 2024, pp. 680-686, doi: 10.1109/IDCIoT59759.2024.10467340. keywords: {Radio frequency;Logistic regression;Technological innovation;Sensitivity;Finance;Organizations;Fraud;Financialfraud;Comparisonanalysis;Hybridmodel;Contemporaryworld;Fraudulenttransactions;Flexibility},

</p>

## Table of Contents
- [Project Overview](#project-overview)
- [Why I Chose This Project](#why-i-chose-this-project)
- [Problem This Project Solves](#problem-this-project-solves)
- [Dataset](#dataset)
- [Flow of the Project](#flow-of-the-project)
- [Files in This Repository](#files-in-this-repository)
- [Tech Stack Used and Why](#tech-stack-used-and-why)
- [Usage Instructions](#usage-instructions)
- [Results and Insights](#results-and-insights)
- [Author](#author)
- [Contact](#contact)

## Project Overview
<p align="justify">The objective of this project is to automatically detect fraudulent credit card transactions. Using machine learning classification models such as Logistic Regression, Decision Tree, Random Forest, KNN, and hybrid approaches, the system analyzes transactional data to classify whether a transaction is legitimate or fraudulent. The workflow includes data preprocessing, feature engineering, model training, evaluation, and prediction, enabling accurate and efficient fraud detection.</p>

## Why I Chose This Project?
<p align="justify">Credit card fraud is a major concern for financial institutions and customers alike. I chose this project because it addresses a real-world problem where timely detection of fraud can prevent financial losses. It also allowed me to gain hands-on experience with imbalanced datasets, feature engineering, model evaluation metrics, and ensemble learning. This project strengthened my skills in data preprocessing, machine learning, and predictive analytics.</p>

## Problem This Project Solves
<p align="justify">Financial fraud can lead to significant monetary losses and customer trust issues. Manual monitoring of credit card transactions is inefficient and error-prone. This project provides an automated solution to detect fraudulent transactions in real-time, enabling financial institutions to act quickly, prevent losses, and ensure secure banking experiences for their customers.</p>

## Dataset
<p align="justify">The dataset used consists of labeled credit card transactions, categorized as <b>fraud</b> or <b>non-fraud</b>. It contains features such as transaction amount, timestamp, customer demographics, and transaction metadata.</p> <p align="justify"><b>Dataset Link:</b> <a href="https://www.kaggle.com/datasets/ealaxi/paysim1" target="_blank">Download Credit Card Fraud Dataset from Kaggle</a></p> <p align="justify">The data preprocessing steps include dropping irrelevant columns, encoding categorical variables, normalizing features, and handling class imbalance through sampling. These steps prepare the dataset for accurate model training and evaluation.</p>

## Flow of the Project
<p align="justify">The workflow of this project is designed to transform raw transaction data into actionable predictions. The steps include:</p>

<b>Load Dataset</b><br>
Transaction data is loaded from CSV files.

<b>Data Preprocessing</b><br>

Dropping irrelevant columns

Encoding categorical variables

Extracting features from dates (e.g., DOB year/month)

Handling missing values and normalizing data

<b>Dataset Sampling</b><br>
To handle class imbalance, a small fraction of non-fraud transactions and a larger fraction of fraud transactions are sampled for training.

<b>Splitting the Dataset</b><br>
The data is split into training and testing sets using K-Fold cross-validation.

<b>Model Building</b><br>
Classification models used:

Logistic Regression

Post Lasso Logistic Regression

Decision Tree

Random Forest

K-Nearest Neighbors (KNN)

Hybrid Model (Logistic Regression + Decision Tree)

<b>Model Training</b><br>
Each model is trained on the training data, with hyperparameters optimized for best performance.

<b>Evaluation</b><br>
Performance metrics including accuracy, sensitivity, specificity, G-mean, and confusion matrices are calculated.

<b>Prediction</b><br>
Trained models predict fraudulent transactions on test data, allowing identification of high-risk transactions for further investigation.

## Files in This Repository

credit_card_fraud_detection.ipynb – Jupyter Notebook containing the complete implementation

Research Paper – Published and presented in Prestegious conference IDC-IoT 2024 (Intelligent Data Communication Technologies and Internet of Things)
requirements.txt – List of Python dependencies

README.md – Project documentation

## Tech Stack Used and Why

Python: Core language for data analysis and model development

NumPy & Pandas: Numerical computations and data manipulation

Matplotlib & Seaborn: Visualization of data distributions, confusion matrices, and model performance

Scikit-learn: Machine learning models, cross-validation, and evaluation metrics

Imbalanced-learn: Handling imbalanced datasets


<p align="justify">These tools provide an end-to-end ecosystem for data preprocessing, model development, evaluation, and visualization.</p>

## Usage Instructions
<p align="justify">1. <b>Clone the repository</b><br> <code>git clone https://github.com/JAHNAVIDINGARI/CREDIT-CARD-FRAUD-DETECTION.git></p> <p align="justify">2. <b>Navigate to the project directory</b><br> <code>cd credit-card-fraud-detection</code></p> <p align="justify">3. <b>Install dependencies</b><br> <code>pip install -r requirements.txt</code></p> <p align="justify">4. <b>Run the notebook</b><br> Open <b>credit_card_fraud_detection.ipynb</b>, update dataset path if required, and execute all cells to train and evaluate the models.</p>

## Results and Insights
<p align="justify">The models trained on the credit card dataset achieved strong performance in identifying fraudulent transactions. Key observations include:</p>
Decision Tree achieved high sensitivity (~80%) and G-mean (~0.89), indicating reliable fraud detection.
Random Forest reduced false positives while maintaining strong accuracy (~97%).
Logistic Regression and Post Lasso models provided baseline comparisons with slightly lower sensitivity.
Hybrid models combining Logistic Regression and Decision Tree improved overall predictive performance.
<p align="justify">Confusion matrices demonstrated clear separation between fraud and non-fraud transactions. These results confirm that the system is robust for real-world financial applications and can assist institutions in minimizing fraudulent activity.</p>

## Authors
Jahnavi Dingari
Sandeep Pulata
Sasank Sami
Mona A
Bharadwaj V
Bharati Mohan G

## Contact
<p align="justify">For queries, collaboration, or further discussion regarding this project, please reach out via <b>LinkedIn</b> or <b>email</b>:</p>

LinkedIn: https://www.linkedin.com/in/jahnavi-dingari

Email: <a href="mailto:jahnavidingari04@gmail.com">jahnavidingari04@gmail.com
</a>
