# Credit Card Fraud Detection Using Machine Learning

A machine learning project developed to identify fraudulent credit card transactions from historical transaction data. The project uses exploratory data analysis and a Random Forest Classifier to distinguish fraudulent transactions from legitimate ones.

The primary challenge addressed in this project is the highly imbalanced nature of financial transaction data, where fraudulent transactions represent only a small percentage of the total observations.

---

## Project Overview

Credit card fraud can cause significant financial losses to customers and financial institutions. This project develops a classification model capable of recognising unusual transaction patterns and predicting whether a transaction is legitimate or fraudulent.

Instead of relying only on accuracy, the model is evaluated using precision, recall, F1-score and the Matthews Correlation Coefficient, which provide a more reliable assessment for an imbalanced dataset.

---

## Objectives

- Analyse the distribution of legitimate and fraudulent transactions
- Explore differences in transaction amounts between both classes
- Examine relationships between the available input features
- Train a machine learning model to classify transactions
- Minimise false fraud alerts through high precision
- Detect as many actual fraudulent transactions as possible
- Evaluate the model using imbalance-aware performance metrics

---

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

---

## Dataset

The dataset contains **284,807 credit card transactions** with 31 columns.

| Feature | Description |
|---|---|
| `Time` | Seconds elapsed between the transaction and the first recorded transaction |
| `V1–V28` | PCA-transformed numerical features used to protect confidential information |
| `Amount` | Monetary value of the transaction |
| `Class` | Target variable: `0` for legitimate and `1` for fraudulent |

The dataset can be downloaded from:

[Download Credit Card Fraud Dataset](https://www.geeksforgeeks.org/machine-learning/ml-credit-card-fraud-detection/)

After downloading, save the dataset as `creditcard.csv` inside the project directory.

---

## Project Structure

    Credit-Card-Fraud-Detection/
    │
    ├── creditcard.csv
    ├── credit_card_fraud_detection.ipynb
    ├── requirements.txt
    └── README.md

---

## Methodology

### 1. Data Collection

The credit card transaction dataset is loaded into a Pandas DataFrame for analysis and model development.

### 2. Exploratory Data Analysis

The dataset is examined to understand:

- The number of legitimate and fraudulent transactions
- The degree of class imbalance
- The statistical properties of transaction amounts
- Correlations between the available features
- Differences between legitimate and fraudulent transactions

### 3. Feature and Target Separation

The `Class` column is used as the target variable, while the remaining columns are used as input features.

    X = data.drop("Class", axis=1)
    y = data["Class"]

### 4. Train-Test Split

The dataset is divided into:

- **80% training data**
- **20% testing data**

A fixed random state is used to make the results reproducible.

    from sklearn.model_selection import train_test_split

    x_train, x_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

### 5. Model Training

A Random Forest Classifier is trained on the transaction data.

    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

### 6. Model Evaluation

The trained model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Matthews Correlation Coefficient
- Confusion matrix

These metrics provide a more meaningful evaluation than accuracy alone because the dataset is highly imbalanced.

---

## Why Accuracy Alone Is Misleading

Fraudulent transactions form only a small fraction of the dataset. Therefore, a model that classifies every transaction as legitimate can still obtain very high accuracy while failing to detect any fraud.

For this reason:

- **Precision** measures how many transactions predicted as fraud were actually fraudulent.
- **Recall** measures how many actual fraudulent transactions were detected.
- **F1-score** balances precision and recall.
- **MCC** measures the overall quality of binary classification, even when the classes are imbalanced.

---

## Model Performance

The Random Forest model produced the following results:

| Evaluation Metric | Score |
|---|---:|
| Accuracy | 99.96% |
| Precision | 98.73% |
| Recall | 79.59% |
| F1-score | 88.14% |
| Matthews Correlation Coefficient | 88.63% |

> The exact results may vary slightly depending on the dataset split, package versions and model parameters.

---

## Results and Observations

- The model achieved high precision, indicating that very few legitimate transactions were incorrectly flagged as fraud.
- The recall score shows that the model detected a large proportion of actual fraudulent transactions.
- The F1-score demonstrated a strong balance between fraud detection and false-alert reduction.
- MCC indicated that the model made reliable predictions despite the severe class imbalance.
- The experiment showed that accuracy should not be treated as the primary metric for fraud-detection problems.

---

## Installation

### 1. Clone the Repository

    git clone https://github.com/your-username/Credit-Card-Fraud-Detection.git
    cd Credit-Card-Fraud-Detection

Replace `your-username` with your GitHub username.

### 2. Create a Virtual Environment

    python -m venv venv

### 3. Activate the Virtual Environment

For macOS or Linux:

    source venv/bin/activate

For Windows:

    venv\Scripts\activate

### 4. Install the Dependencies

    pip install numpy pandas matplotlib seaborn scikit-learn jupyter

---

## Running the Project

Place `creditcard.csv` inside the project directory and start Jupyter Notebook:

    jupyter notebook

Open `credit_card_fraud_detection.ipynb` and run all the cells sequentially to perform the analysis, train the model and generate the evaluation results.

---

## Requirements

Add the following dependencies to `requirements.txt`:

    numpy
    pandas
    matplotlib
    seaborn
    scikit-learn
    jupyter

---

## Future Improvements

- Apply SMOTE to oversample fraudulent transactions
- Experiment with majority-class undersampling
- Tune the classification threshold to improve recall
- Perform hyperparameter optimisation using GridSearchCV
- Compare Random Forest with XGBoost and LightGBM
- Evaluate the model using Precision–Recall AUC
- Add feature-importance visualisation
- Save the trained model for future predictions
- Build a web application for real-time fraud prediction
- Deploy the model through a REST API
