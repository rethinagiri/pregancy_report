import pickle
from flask import Flask, render_template, request
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the pregnancy dataset
pregnancy_data = pd.read_csv('babies.csv')

# Load the pickled linear regression model
with open('lr_preg.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    # Display the first 10 rows of the pregnancy dataset in a web page
    pregnancy_data_html = pregnancy_data.head(10).to_html()
    return render_template('index.html', pregnancy_data=pregnancy_data_html)


@app.route('/result', methods=['POST'])
def index():
    gestation = float(request.form['gestation'])
    parity = float(request.form['parity'])
    age = float(request.form['age'])
    height = float(request.form['height'])
    weight = float(request.form['weight'])
    smoke = int(request.form['smoke'])

    # Perform prediction based on the user input  
    
    X = np.array([[gestation, parity, age, height, weight, smoke]])   
    result_value = model.predict(X)

    return render_template('result.html', result=result_value)


if __name__ == '__main__':
    app.run(debug=True)
