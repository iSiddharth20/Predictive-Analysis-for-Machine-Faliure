from flask import Flask, render_template, request
from sklearn.externals import joblib
import numpy as np
from modelcode import confidence

app = Flask(__name__)

mul_reg = open("model.pkl", "rb")
ml_model = joblib.load(mul_reg)

confidence = str(round(confidence,3))
def calstr(num):
    if num == 0:
        return 'Breakdown : No || Confidence : ' + confidence + ' %'
    elif num == 1:
        return 'Breakdown : Yes || Confidence : ' + confidence + ' %'
    else:
        return 'Error!'

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
            mul_reg = open("model.pkl", "rb")
            ml_model = joblib.load(mul_reg)
            model_prediction = calstr(ml_model.predict(pred_args_arr))
        except ValueError:
            return "Please check if the values are entered correctly"
    return render_template('predict.html', prediction = model_prediction)


if __name__ == "__main__":
    app.run()
