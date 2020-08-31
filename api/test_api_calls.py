import requests
import json

"""
The three methods below run simple tests on each of the three API endpoints using some sample data.
Changing the sample data (e.g. changing a key to an incorrect value) can ensure the tests work as intended.
Use the command "pytest" in the command line to run the tests.
"""


def test_breast_cancer_call():
    url = "http://127.0.0.1:5000/predict_bc"

    headers = {'Content-Type': 'application/json'}

    # The test data
    body = {'mean_radius': 9.029, 'mean_texture': 17.33, 'mean_perimeter': 58.79, 'mean_area': 250.5,
            'mean_smoothness': 0.1066}

    # Convert to json
    resp = requests.post(url, headers=headers, data=json.dumps(body, indent=4))

    # Validate the response
    assert resp.status_code == 200

    # Print response
    print(resp)


def test_heart_disease_call():
    url = "http://127.0.0.1:5000/predict_hd"

    headers = {'Content-Type': 'application/json'}

    body = {
        "age": 45,
        "sex": 1,
        "cp": 3,
        "trestbps": 124,
        "chol": 340,
        "fbs": 1,
        "restecg": 0,
        "thalach": 135,
        "exang": 0,
        "oldpeak": 2.5,
        "slope": 0,
        "ca": 0,
        "thal": 1
    }

    resp = requests.post(url, headers=headers, data=json.dumps(body, indent=4))

    assert resp.status_code == 200

    print(resp)


def test_diabetes_call():
    url = "http://127.0.0.1:5000/predict_diabetes"

    headers = {'Content-Type': 'application/json'}

    body = {
        "Pregnancies": 2,
        "Glucose": 90,
        "BloodPressure": 68,
        "SkinThickness": 42,
        "Insulin": 79,
        "BMI": 28.4,
        "DiabetesPedigreeFunction": 0.323,
        "Age": 34
    }

    resp = requests.post(url, headers=headers, data=json.dumps(body, indent=4))

    assert resp.status_code == 200

    print(resp)
