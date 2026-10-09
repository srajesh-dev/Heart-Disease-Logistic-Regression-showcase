import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

heart_data = pd.read_csv("heart.csv")
heart_data = heart_data.drop_duplicates()

X = heart_data.drop(columns=["target"])
y = heart_data["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=100)

continuous_features_train = X_train[[
    "age", "trestbps", "chol", "thalach", "oldpeak"]]
continuous_features_test = X_test[[
    "age", "trestbps", "chol", "thalach", "oldpeak"]]
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(continuous_features_train)
X_test_scaled = scaler.transform(continuous_features_test)
X_train[["age", "trestbps", "chol", "thalach", "oldpeak"]] = X_train_scaled
X_test[["age", "trestbps", "chol", "thalach", "oldpeak"]] = X_test_scaled

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
prediction = model.predict(X_test)

accuracy = accuracy_score(y_test, prediction)
accuracy_percentage = accuracy * 100

probs = model.predict_proba(X_test)[:, 1]

for t in [0.3, 0.5, 0.7]:
    pred = (probs >= t).astype(int)
    print("threshold", t)
    print(confusion_matrix(y_test, pred))
    print()

# --- Coefficients ---
coefs = pd.Series(model.coef_[0], index=X.columns).sort_values()
print(coefs)

print(accuracy_percentage)
print(confusion_matrix(y_test, prediction))
print(classification_report(y_test, prediction))
