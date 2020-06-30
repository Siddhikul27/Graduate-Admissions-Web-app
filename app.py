# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the Random Forest CLassifier model
filename = 'model.pkl'
regressor = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        gre = int(request.form['GRE Score'])
        toefl = int(request.form['TOEFL Score'])
        uni = int(request.form['University Rating'])
        sop = float(request.form['SOP'])
        lor = float(request.form['LOR'])
        cgpa = float(request.form['CGPA'])
        research = float(request.form['Research'])
        
        data = np.array([[gre, toefl, uni, sop, lor, cgpa, research]])
        my_prediction = regressor.predict(data)
        
        return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
	app.run(debug=True)