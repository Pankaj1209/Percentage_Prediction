import numpy as np
from flask import Flask, render_template, request
import pickle
import marks as m



app = Flask(__name__)

model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
	
	data=request.form["hours"]
	data1 = model.predict([[data]])[0][0].round(2)
	return render_template('index.html',prediction_text = data1)

if __name__ == '__main__':
	app.run(debug=True)