from cv2 import normalize
from flask import Flask, request, render_template
import pandas as pd
import joblib
import numpy as np
from pytest import console_main
from sklearn import preprocessing

normalize = preprocessing.MinMaxScaler()

# Declare a Flask app
app = Flask(__name__)
# Main function here
# ------------------
@app.route('/', methods=['GET', 'POST'])
def main():
    # If a form is submitted
    price = ''
    #text = 'Patient with age'
    if request.method == "POST":
        
        # Unpickle classifier
        housePrice = joblib.load("housing.pkl")
        
        # Get values through input bars
        areaIncome = request.form.get("areaIncome")
        areaHouseAge = request.form.get("areaHouseAge")
        areaNumberOfRooms = request.form.get("areaNumberOfRooms")
        areaNumberOfBedrooms = request.form.get("areaNumberOfBedrooms")
        areaPopulation = request.form.get("areaPopulation")
        
        # Put inputs to dataframe
        X = pd.DataFrame([[areaIncome, areaHouseAge, areaNumberOfRooms, areaNumberOfBedrooms, areaPopulation]], columns = ["areaIncome", "areaHouseAge", "areaNumberOfRooms", "areaNumberOfBedrooms", "areaPopulation"])
        #df_scaled = normalize.fit_transform(X)
        # Get prediction
        prediction = housePrice.predict(X)[0]
        print(prediction)
    else:
        prediction = ""
    #sqrtCalculated = np.sqrt(price)
    return render_template("frontend.html", output = prediction)

# Running the app
if __name__ == '__main__':
    app.run(debug = True)