from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import os
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
from sklearn.model_selection import train_test_split

# Load environment variables from the .env file
load_dotenv()

app = Flask(__name__)

# Load dataset
dataset = pd.read_csv('healthcare-dataset-stroke-data.csv')

# Encode categorical variables
dataset['gender'] = dataset['gender'].map({'Female': 0, 'Male': 1, 'Other': 2})
dataset['ever_married'] = dataset['ever_married'].map({'Yes': 1, 'No': 0})
dataset['work_type'] = dataset['work_type'].map({'Private': 0, 'Self-employed': 1, 'Govt_job': 2, 'children': 3, 'Never_worked': 4})
dataset['Residence_type'] = dataset['Residence_type'].map({'Urban': 0, 'Rural': 1})
dataset['smoking_status'] = dataset['smoking_status'].map({'never smoked': 0, 'smokes': 1, 'formerly smoked': 2, 'Unknown': 3})


# Handle missing data
for column in dataset.columns:
    if dataset[column].isnull().any():
        if dataset[column].dtype == 'object':  # For categorical data
            dataset[column].fillna(dataset[column].mode()[0], inplace=True)
        else:  # For numerical data
            dataset[column].fillna(dataset[column].mean(), inplace=True)

# Drop the 'id' column if present
if 'id' in dataset.columns:
    dataset = dataset.drop(['id'], axis='columns')

# Define features (X) and target (Y)
X = dataset.drop('stroke', axis='columns')
Y = dataset['stroke']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.26, random_state=0)

# Initialize models
rf = RandomForestClassifier(n_estimators=100, random_state=0)       # Random Forest Classifier
gb = GradientBoostingClassifier(n_estimators=100, random_state=0)   # Gradient Boosting Classifier
knn = KNeighborsClassifier()                                        # KNeighborsClassifier
lr = LogisticRegression(max_iter=1000, random_state=0)              # Logistic Regression
gnb = GaussianNB()                                                  # Naive Bayes

# Create a voting classifier with the models
voting_clf = VotingClassifier(
    estimators=[('rf', rf), ('gb', gb), ('knn', knn), ('lr', lr), ('gnb', gnb)],
    voting='soft'
)

models = {
    "Random Forest": rf,
    "Gradient Boosting": gb,
    "KNN": knn,
    "Logistic Regression": lr,
    "GaussianNB": gnb,
    "Voting Classifier": voting_clf
}

# Train models and calculate metrics
metrics = {}
for name, model in models.items():
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate accuracy
    train_accuracy = accuracy_score(y_train, y_train_pred) * 100
    test_accuracy = accuracy_score(y_test, y_test_pred) * 100
    
    # Calculate precision
    train_precision = precision_score(y_train, y_train_pred, zero_division=0) * 100
    test_precision = precision_score(y_test, y_test_pred, zero_division=0) * 100
    
    # Calculate confusion matrix
    tn_test, fp_test, fn_test, tp_test = confusion_matrix(y_test, y_test_pred).ravel()
    tn_train, fp_train, fn_train, tp_train = confusion_matrix(y_train, y_train_pred).ravel()
    
    # Calculate Negative Predictive Value (NPV) 
    test_npv = tn_test / (tn_test + fn_test) * 100 if (tn_test + fn_test) > 0 else 0
    train_npv = tn_train / (tn_train + fn_train) * 100 if (tn_train + fn_train) > 0 else 0
    
    # Store metrics
    metrics[name] = {
        "test_accuracy": test_accuracy,
        "train_accuracy": train_accuracy,
        "test_precision": test_precision,
        "train_precision": train_precision,
        "test_npv": test_npv,
        "train_npv": train_npv
    }

# Retrieve the validation key from the .env file
VALIDATE_KEY = os.getenv('VALIDATE_KEY')

# Function to validate API key
def validate_key(key):
    return key == VALIDATE_KEY

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check for validate_key in request
        data = request.json
        if not validate_key(data.get('validate_key')):
            return jsonify({'error': 'Invalid validation key'}), 403

        # Get input data from the request
        gender = float(data['gender'])
        age = float(data['age'])
        hypertension = int(data['hypertension'])
        heart_disease = int(data['heart_disease'])
        ever_married = int(data['ever_married'])
        work_type = int(data['work_type'])
        Residence_type = int(data['Residence_type'])
        avg_glucose_level = float(data['avg_glucose_level'])
        bmi = float(data['bmi'])
        smoking_status = int(data['smoking_status'])

        # Combine input data into a NumPy array
        person = np.array([[gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type, avg_glucose_level, bmi, smoking_status]])

        # Predict the result
        result = models["Voting Classifier"].predict(person)

        # Prepare the response
        response = {
            'prediction': 'The person is likely to have a stroke. Please consult a doctor.' if result[0] == 1 else 'The person is unlikely to have a stroke. For certainty, contact a doctor.',
            'train_accuracy': metrics["Voting Classifier"]["train_accuracy"],
            'test_accuracy': metrics["Voting Classifier"]["test_accuracy"],
            'test_precision': metrics["Voting Classifier"]["test_precision"],
            'train_precision': metrics["Voting Classifier"]["train_precision"],
            'test_npv': metrics["Voting Classifier"]["test_npv"],
            'train_npv': metrics["Voting Classifier"]["train_npv"]
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/accuracy', methods=['POST'])
def accuracy():
    try:
        # Check for validate_key in request
        data = request.json
        if not validate_key(data.get('validate_key')):
            return jsonify({'error': 'Invalid validation key'}), 403

        # Return the metrics
        return jsonify(metrics)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)

