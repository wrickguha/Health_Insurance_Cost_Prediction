from flask import Flask, request, render_template
import numpy as np
import pickle


app = Flask(__name__)

regression = pickle.load(open('regression.pkl','rb'))
scaler = pickle.load(open('scaler.pkl','rb'))

@app.route('/')
def index():
    return render_template("index.html")



if __name__ == "__main__":
    app.run(debug=True)