# Heart-Disease-Logistic-Regression-showcase
Readme · MD
Heart Disease Prediction with Logistic Regression

A beginner-friendly machine learning project that predicts the presence of heart disease from patient health measurements, using logistic regression in scikit-learn. I built it to practice the full workflow of a binary classification problem: cleaning data, scaling features, training a model, and evaluating it properly.

Dataset

heart.csv, a widely used heart disease dataset (UCI Heart Disease data, as distributed on Kaggle). Each row is a patient, and the target column is the binary label to predict.

Features include age, sex, cp (chest pain type), trestbps (resting blood pressure), chol (cholesterol), thalach (max heart rate), oldpeak, and others.

Note on duplicates: the file has 1,025 rows, but 723 of them are exact duplicates. Only 302 rows are unique. Duplicates can end up in both the training and test sets, which leaks information and inflates accuracy, so they are removed before splitting.

Workflow
Load the data and drop duplicate rows (1,025 to 302 rows).
Split into 80% train and 20% test (random_state=100).
Scale the continuous features (age, trestbps, chol, thalach, oldpeak) with StandardScaler. The scaler is fit on the training set only and then applied to the test set, to avoid data leakage.
Train a LogisticRegression model (max_iter=1000).
Evaluate with accuracy, a confusion matrix, and a classification report.
Explore how changing the decision threshold affects false positives and false negatives, and inspect the model coefficients.
Results

Test set of 61 patients, default threshold of 0.5:

Metric	Value
Accuracy	83.6%

Confusion matrix:

	Predicted 0	Predicted 1
Actual 0	14 (TN)	6 (FP)
Actual 1	4 (FN)	37 (TP)

For class 1: precision is about 0.86, recall about 0.90, and F1 about 0.88.

What I learned
Data leakage is easy to miss. Before removing duplicates, accuracy looked higher (about 86%) because the model was partly tested on rows it had already seen. After deduplication, the honest accuracy was 83.6%.
Scaling helps convergence more than accuracy for logistic regression. Without scaling, the optimizer hit its iteration limit and printed a ConvergenceWarning. Scaling and max_iter=1000 fixed that, but accuracy barely changed.
Accuracy alone hides the kinds of mistakes a model makes. The confusion matrix and the precision/recall breakdown show which errors happen, which matters when a missed patient is costlier than a false alarm.
Logistic regression outputs probabilities. predict() applies a 0.5 cutoff, and moving that threshold trades false negatives against false positives.
Small test sets are noisy. With only 61 test rows, one patient shifts accuracy by about 1.6 percentage points, so small differences between runs are not meaningful.
How to run
bash
pip install pandas scikit-learn
python heart_disease.py

Place heart.csv in the same folder as the script.

Possible next steps
Use cross-validation instead of a single train/test split for a more reliable score.
One-hot encode the categorical columns (cp, thal, slope, restecg).
Tune the regularization strength C with GridSearchCV.
Compare against other models such as a random forest.
Bundle preprocessing and the model into a scikit-learn Pipeline.
