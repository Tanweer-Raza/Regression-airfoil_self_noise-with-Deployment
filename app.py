import pickle
from flask import Flask, request, app , jsonify, url_for , render_template
import numpy as np
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods = ['POST'] )   # Creating API and using it with POSTMAN
def predict_api() :

    data = request.json['data']
    print(data)
    new_data = [list(data.values())]
    output = model.predict(new_data)[0]
    return jsonify(output)



@app.route('/predict', methods = ['POST'] )   # Creating API and using HTML
def predict() :

    data = [float(x) for x in request.form.values()]
    final_features = [np.array(data)]
    print(data)

    output = model.predict(final_features)[0]
    print(output)
    #output = round(prediction[0], 2)
    return render_template('home.html', prediction_text = "Airfoil pressure is {}".format(output))


if __name__ == "__main__" :
    app.run(debug = True)


## Create Procfile for deployement in /heroku
## web : gunicorn app : app   # first app is file name and second app says from where to execute