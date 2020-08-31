from flask import Flask, request, jsonify, render_template

import numpy as np
from flask_restful import Api, Resource
from tensorflow.keras.models import load_model

"""
This defines a Flask app that provides an interface for users to easily fill in details of a patient
to a form, which then uses the models created earlier to make a prediction based on those values.

It also provides API endpoints for making predictions directly via json values.
"""

app = Flask(__name__)
api = Api(app)

# Load the models
model_diabetes = load_model("static/model/diabetes_model.h5")
model_heart = load_model("static/model/heart_disease_model.h5")
model_breast_cancer = load_model("static/model/cancer_model.h5")


@app.route("/")
def index():
    return render_template("index.html", pred=0)


@app.route("/heart")
def heartdisease():
    return render_template("heart.html")


@app.route("/diabetes")
def diabetes():
    return render_template("diabetes.html")


@app.route("/cancer")
def breastcancer():
    return render_template("cancer.html")


# Defines the route for the form used to predict heart disease
@app.route("/formpredheart", methods=['POST'])
def predicth():
    global result_heart

    # Get values from form
    data = [request.form["age"],
            request.form["sex"],
            request.form["cp"],
            request.form["trestbps"],
            request.form["chol"],
            request.form["fbs"],
            request.form["restecg"],
            request.form["thalach"],
            request.form["exang"],
            request.form["oldpeak"],
            request.form["slope"],
            request.form["ca"],
            request.form["thal"]]

    # Convert values to numpy array
    data = np.array([np.asarray(data, dtype=float)])

    # Make prediction
    prediction = model_heart.predict_classes(data)

    # Assign prediction to value
    pred = np.where(prediction == np.amax(prediction, axis=1))[1][0]

    if pred == 0:
        result_heart = "Heart disease not present"
    elif pred == 1:
        result_heart = "Heart disease present"

    return render_template("heart.html", pred=result_heart)


# Defines the route for the form used to predict diabetes
@app.route("/formpreddiabetes", methods=['POST'])
def predictd():
    global result_diabetes

    # Get values from form
    data = [request.form["pregnancies"],
            request.form["glucose"],
            request.form["bloodpressure"],
            request.form["skinthickness"],
            request.form["insulin"],
            request.form["bmi"],
            request.form["dpf"],
            request.form["age"]]

    # Convert values to numpy array
    data = np.array([np.asarray(data, dtype=float)])

    # Make prediction
    prediction = model_diabetes.predict_classes(data)

    # Assign prediction to value
    pred = np.where(prediction == np.amax(prediction, axis=1))[1][0]

    if pred == 0:
        result_diabetes = "Diabetes not present"
    elif pred == 1:
        result_diabetes = "Diabetes present"

    return render_template("diabetes.html", pred=result_diabetes)


# Defines the route for the form used to predict breast cancer
@app.route("/formpredcancer", methods=['POST'])
def predictbc():
    global result_cancer

    # Get values from form
    data = [request.form["mean_radius"],
            request.form["mean_texture"],
            request.form["mean_perimeter"],
            request.form["mean_area"],
            request.form["mean_smoothness"]]

    # Convert values to numpy array
    data = np.array([np.asarray(data, dtype=float)])

    # Make prediction
    prediction = model_breast_cancer.predict_classes(data)

    # Assign prediction to value
    pred = np.where(prediction == np.amax(prediction, axis=1))[1][0]

    if pred == 0:
        result_cancer = "Breast cancer not present"
    elif pred == 1:
        result_cancer = "Breast cancer present"

    return render_template("cancer.html", pred=result_cancer)


# The API endpoint for making heart disease predictions
class PredictHeartDisease(Resource):

    @staticmethod
    def post():
        global predicted_hd
        data = request.get_json()

        age = data["age"]
        sex = data["sex"]
        cp = data["cp"]
        trestbps = data["trestbps"]
        chol = data["chol"]
        fbs = data["fbs"]
        restecg = data["restecg"]
        thalach = data["thalach"]
        exang = data["exang"]
        oldpeak = data["oldpeak"]
        slope = data["slope"]
        ca = data["ca"]
        thal = data["thal"]

        prediction = model_heart.predict_classes([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak,
                                                   slope, ca, thal]])

        # 0 = heart disease not predicted; 1 = heart disease predicted
        if prediction == 0:
            predicted_hd = 0
        elif prediction == 1:
            predicted_hd = 1

        return jsonify({"Prediction": predicted_hd})


# The API endpoint for making diabetes predictions
class PredictDiabetes(Resource):

    @staticmethod
    def post():

        global predicted_diabetes
        data = request.get_json()

        pregnancies = data["Pregnancies"]
        glucose = data["Glucose"]
        blood_pressure = data["BloodPressure"]
        skin_thickness = data["SkinThickness"]
        insulin = data["Insulin"]
        bmi = data["BMI"]
        dpf = data["DiabetesPedigreeFunction"]
        age = data["Age"]

        prediction = model_diabetes.predict_classes([[pregnancies, glucose, blood_pressure, skin_thickness, insulin,
                                                      bmi, dpf, age]])

        # 0 = diabetes not predicted; 1 = diabetes predicted
        if prediction == 0:
            predicted_diabetes = 0
        elif prediction == 1:
            predicted_diabetes = 1

        return jsonify({"Prediction": predicted_diabetes})


# The API endpoint for predicting breast cancer
class PredictBreastCancer(Resource):

    @staticmethod
    def post():

        global predicted_cancer
        data = request.get_json()

        mean_radius = data["mean_radius"]
        mean_texture = data["mean_texture"]
        mean_perimeter = data["mean_perimeter"]
        mean_area = data["mean_area"]
        mean_smoothness = data["mean_smoothness"]

        prediction = model_breast_cancer.predict_classes([[mean_radius, mean_texture, mean_perimeter, mean_area,
                                                           mean_smoothness]])

        # 0 = cancer not predicted; 1 = cancer predicted
        if prediction == 0:
            predicted_cancer = 0
        elif prediction == 1:
            predicted_cancer = 1

        return jsonify({"Prediction": predicted_cancer})


api.add_resource(PredictDiabetes, "/predict_diabetes")
api.add_resource(PredictHeartDisease, "/predict_hd")
api.add_resource(PredictBreastCancer, "/predict_bc")

if __name__ == '__main__':
    app.run(debug=True)
