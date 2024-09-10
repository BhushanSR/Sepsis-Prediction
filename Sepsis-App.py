from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the model and scaler
scaler = pickle.load(open('scaler.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('indexog.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the JSON data from the request
    data = request.get_json()

    # Convert the JSON data to a numpy array
    features = np.array(data['features']).reshape(1, -1)

    # Scale the features using the scaler
    scaled_features = scaler.transform(features)

    # Make a prediction using the model
    prediction = model.predict(scaled_features)[0]

    # Convert the prediction to a JSON response
    if prediction == 1:
        response = {'prediction': 'Positive(+)'}
    else:
        response = {'prediction': 'Negative(-)'}

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
