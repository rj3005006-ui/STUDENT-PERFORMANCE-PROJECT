
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# 1. Load the student dataset
data = pd.read_csv("student-mat.csv", sep=";")

# 2. Select input features
features = ["studytime", "failures", "absences", "G1", "G2"]
X = data[features]

# 3. Create performance categories from final grade (G3)
def categorize_performance(grade):
    if grade < 10:
        return "Needs Improvement"
    elif grade < 15:
        return "Average"
    else:
        return "Excellent"

y = data["G3"].apply(categorize_performance)

# 4. Split the dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 5. Build and train the model
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    ))
])

model.fit(X_train, y_train)

# 6. Evaluate the model
predictions = model.predict(X_test)

print("Model Accuracy:", round(accuracy_score(y_test, predictions) * 100, 2), "%")
print("\nClassification Report:")
print(classification_report(y_test, predictions, zero_division=0))

# 7. Save the trained model and feature names
joblib.dump(
    {"model": model, "features": features},
    "student_model.joblib"
)

print("\nModel saved successfully as student_model.joblib")
