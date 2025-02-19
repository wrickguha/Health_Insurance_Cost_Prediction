from flask import Flask, request, render_template,jsonify
import numpy as np
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler


application = Flask(__name__)
app=application

regression = pickle.load(open('regression.pkl','rb'))
scaler = pickle.load(open('scaler.pkl','rb'))


gender_map = {"female": 0, "male": 1}
smoke_map = {"yes": 0, "no": 1}
region_map = {"southwest": 0, "southeast": 1, "northwest": 2, "northeast": 3}

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == "POST":
        try:

            Age = float(request.form.get('Age'))
            gender = request.form.get('gender').lower() 
            bmi = float(request.form.get('bmi'))
            range = float(request.form.get('range'))  
            smoke = request.form.get('smoke').lower()
            direction = request.form.get('direction').lower()

            gender_encoded = gender_map.get(gender, -1)  
            smoke_encoded = smoke_map.get(smoke, -1)
            direction_encoded = region_map.get(direction, -1)

            if -1 in [gender_encoded, smoke_encoded, direction_encoded]:
                return render_template('index.html', results="Error: Invalid input values.")
            
            new_data = np.array([[Age, gender_encoded, bmi, range, smoke_encoded, direction_encoded]])

            new_data_scaled = scaler.transform(new_data)

            result = regression.predict(new_data_scaled)

            return render_template('index.html', results=result[0])

        except Exception as e:
            print(f"Error: {e}")
            return render_template('index.html', results="Please give valid input")

    else:
        return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0")