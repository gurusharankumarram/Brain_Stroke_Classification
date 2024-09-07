from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# Load dataset
dataset = pd.read_csv('healthcare-dataset-stroke-data.csv')

# Encode categorical variables
dataset['gender'] = dataset['gender'].map({'Female': 0, 'Male': 1, 'Other': 2})
dataset['ever_married'] = dataset['ever_married'].map({'Yes': 1, 'No': 0})
dataset['work_type'] = dataset['work_type'].map({'Private': 0, 'Self-employed': 1, 'Govt_job': 2, 'children': 3, 'Never_worked': 4})
dataset['Residence_type'] = dataset['Residence_type'].map({'Urban': 0, 'Rural': 1})
dataset['smoking_status'] = dataset['smoking_status'].map({'never smoked': 0, 'smokes': 1, 'formerly smoked': 2, 'Unknown': 3})

# Check for missing values after encoding
missing_values = dataset.isnull().sum()

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
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

# Initialize individual models
rf = RandomForestClassifier(n_estimators=100, random_state=0)       # Random Forest Classifier
gb = GradientBoostingClassifier(n_estimators=100, random_state=0)   # Gradient Boosting Classifier
svc = SVC(probability=True, random_state=0)                         # Support Vector Classifier
knn = KNeighborsClassifier()                                        # KNeighborsClassifier
lr = LogisticRegression(max_iter=1000, random_state=0)              # Logistic Regression
gnb = GaussianNB()                                                  # Naive Bayes

# Create a voting classifier with the models
voting_clf = VotingClassifier(
    estimators=[('rf', rf), ('gb', gb), ('svc', svc), ('knn', knn), ('lr', lr), ('gnb', gnb)],
    voting='soft'
)

# Train the ensemble model
voting_clf.fit(X_train, y_train)

# Calculate accuracy
train_accuracy = accuracy_score(y_train, voting_clf.predict(X_train))
test_accuracy = accuracy_score(y_test, voting_clf.predict(X_test))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the request
        data = request.json
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
        result = voting_clf.predict(person)

        # Prepare the response
        response = {
            'prediction': 'The person is likely to have a stroke. Please consult a doctor.' if result[0] == 1 else 'The person is unlikely to have a stroke. For certainty, contact a doctor.',
            'train_accuracy': train_accuracy * 100,
            'test_accuracy': test_accuracy * 100
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/accuracy', methods=['GET'])
def accuracy():
    response = {
        'train_accuracy': train_accuracy * 100,
        'test_accuracy': test_accuracy * 100
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)


