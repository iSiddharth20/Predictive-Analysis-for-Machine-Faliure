from flask import Flask, render_template, request
from sklearn.externals import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

mul_reg = open("classification_model.pkl", "rb")
ml_model = joblib.load(mul_reg)

def calstr(num):
	if num == 0 :
		return 'No'
	elif num == 1 :
		return 'Yes'
	else :
		return 'Error'

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/predict", methods=['GET', 'POST'])
def predict():
    print("I was here 1")
    if request.method == 'POST':
        print(request.form.get('NewYork'))
        try:
            Random = float(request.form['Random'])
            Machine_nbr = float(request.form['Machine_nbr'])
            lifetime = float(request.form['lifetime'])
            pressureInd = float(request.form['pressureInd'])
            moistureInd = float(request.form['moistureInd'])
            temperatureInd= float(request.form['temperatureInd'])
            team= float(request.form['team'])
            provider= float(request.form['provider'])

            pred_args = [Random,Machine_nbr,lifetime,pressureInd,moistureInd,temperatureInd,team,provider]
            pred_args_arr = np.array(pred_args)
            pred_args_arr = pred_args_arr.reshape(1, -1)
            mul_reg = open("classification_model.pkl", "rb")
            ml_model = joblib.load(mul_reg)
            model_prediction = calstr(ml_model.predict(pred_args_arr))
        except ValueError:
            return "Please check if the values are entered correctly"
    return render_template('predict.html', prediction = model_prediction)


if __name__ == "__main__":
    app.run()
